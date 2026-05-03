import copy
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import asyncio
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.pipelines import (
    build_eval_output_paths,
    get_eval_pipeline,
)
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.unity_server import UnityServer

warnings.simplefilter("ignore", category=FutureWarning)

SEED = 2025
DEFAULT_EPISODE_SELECTION_PATH = (
    "/data1/yfl_data/Dyana_data/test/"
    "eval_episode_selection_100_per_task.json"
)


def load_default_eval_trajs() -> List[int]:
    with open(DEFAULT_EPISODE_SELECTION_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    task_types = payload.get("task_types")
    if not isinstance(task_types, dict):
        raise ValueError(
            f"Invalid selection file {DEFAULT_EPISODE_SELECTION_PATH}: missing 'task_types' object."
        )

    ordered_task_names = ["circular", "linear", "harmonic"]
    trajs: List[int] = []
    for task_name in ordered_task_names:
        raw_episodes = task_types.get(task_name)
        if not isinstance(raw_episodes, list):
            raise ValueError(
                f"Invalid selection file {DEFAULT_EPISODE_SELECTION_PATH}: missing list for task_type {task_name!r}."
            )
        for episode_name in raw_episodes:
            if not isinstance(episode_name, str) or not episode_name.startswith("episode_"):
                raise ValueError(
                    f"Invalid episode entry {episode_name!r} for task_type {task_name!r}."
                )
            trajs.append(int(episode_name.split("_", 1)[1]))
    return trajs


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"Using seed: {SEED}")


@dataclass
class EvalArgsConfig:
    pipeline: str = "Local"
    """Registered eval pipeline name."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    modality_keys: List[str] = field(default_factory=lambda: ["left_hand"])
    """Modality keys to evaluate."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "Local"
    """Data config to use."""

    steps: int = 150
    """Unused rollout length override; per-trajectory dataset length is used for evaluation."""

    trajs: List[int] = field(default_factory=load_default_eval_trajs)
    """Trajectory ids to evaluate. Defaults to the selected 300 episodes."""

    repeat_num: int = 1
    """Number of times to repeat the evaluation for each episode."""

    action_horizon: int = 10
    """Action horizon to evaluate."""

    action_dim: int = 18
    """Action dimension expected by the selected eval pipeline."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options."""

    dataset_path: str = "/data1/yfl_data/Dyana_data/test"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    model_path: str | None = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/checkpoint-4000"
    """Path to the model checkpoint."""

    window_length: int = 0
    """Adjacent history frame count for pipelines that support it."""

    observe_frame_offsets: List[int] | None = None
    """Explicit history frame offsets for Local/LoGo, ordered from earliest to latest."""

    motion_hint_ratio: float = 0.25
    """Prefix ratio used by the Global pipeline."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    evaluation_output_path: str = "/data1/yfl_data/DynaHOI/scripts/evaluation_results"
    """Path to save the evaluation results."""

    residual_checkpoint: str | None = None
    """Optional Local PPO residual_policy.pt checkpoint to add on top of --model-path."""


class EvalResidualPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_horizon: int,
        action_dim: int,
        hidden_dim: int,
        residual_clip: float,
        init_log_std: float,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.residual_clip = residual_clip
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_horizon * action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_horizon, action_dim), float(init_log_std)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(features)
        raw_mean = self.mean_head(hidden).view(-1, self.action_horizon, self.action_dim)
        return torch.tanh(raw_mean) * self.residual_clip


def make_residual_feature(
    state: np.ndarray,
    sft_action: np.ndarray,
    step_count: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    state_tensor = torch.as_tensor(state[-1], dtype=torch.float32, device=device).flatten()
    action_tensor = torch.as_tensor(sft_action, dtype=torch.float32, device=device).flatten()
    progress = torch.tensor(
        [float(step_count) / max(float(steps), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([state_tensor, action_tensor, progress], dim=0)


class ResidualEvalPolicy:
    def __init__(
        self,
        base_policy: BasePolicy,
        residual_policy: EvalResidualPolicy,
        action_dim: int,
        action_horizon: int,
        device: torch.device,
    ):
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = device
        if hasattr(base_policy, "model"):
            self.model = base_policy.model

    def get_modality_config(self):
        return self.base_policy.get_modality_config()

    def get_action(self, obs: dict):
        return self.get_action_with_context(obs, step_count=0, steps=1)

    def get_action_with_context(self, obs: dict, step_count: int, steps: int):
        action_chunk = self.base_policy.get_action(obs)
        sft_action = np.asarray(action_chunk["action.left_hand"], dtype=np.float32)
        if sft_action.shape[:2] != (self.action_horizon, self.action_dim):
            raise ValueError(
                "Residual eval expects action.left_hand shape "
                f"{(self.action_horizon, self.action_dim)}, got {sft_action.shape}."
            )
        feature = make_residual_feature(
            obs["state.left_hand"],
            sft_action[:, : self.action_dim],
            step_count,
            steps,
            self.device,
        )
        with torch.no_grad():
            residual = self.residual_policy(feature.unsqueeze(0))[0].clamp(
                -self.residual_policy.residual_clip,
                self.residual_policy.residual_clip,
            )
        final_action = sft_action.copy()
        final_action[:, : self.action_dim] += residual.detach().cpu().numpy()
        output = dict(action_chunk)
        output["action.left_hand"] = final_action
        return output


def load_residual_eval_policy(
    base_policy: BasePolicy,
    checkpoint_path: str,
    args: EvalArgsConfig,
    device: torch.device,
) -> ResidualEvalPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    action_horizon = int(saved_args.get("action_horizon", args.action_horizon))
    action_dim = int(saved_args.get("action_dim", args.action_dim))
    hidden_dim = int(saved_args.get("hidden_dim", 256))
    residual_clip = float(saved_args.get("residual_clip", 0.03))
    init_log_std = float(saved_args.get("init_log_std", -4.6))
    feature_dim = action_dim + action_horizon * action_dim + 1

    residual_policy = EvalResidualPolicy(
        feature_dim=feature_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        residual_clip=residual_clip,
        init_log_std=init_log_std,
    ).to(device)
    residual_policy.load_state_dict(checkpoint["residual_policy"])
    residual_policy.eval()
    print(
        "Loaded Local PPO residual checkpoint: "
        f"{checkpoint_path}, update={checkpoint.get('update_idx')}"
    )
    return ResidualEvalPolicy(
        base_policy=base_policy,
        residual_policy=residual_policy,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device,
    )


def eval_main(args: EvalArgsConfig):
    pipeline = get_eval_pipeline(args.pipeline)
    pipeline.validate_args(args)
    if args.model_path is None:
        raise ValueError("Unified eval pipeline requires --model-path; RobotInferenceClient mode is unsupported here.")

    data_config = copy.deepcopy(DATA_CONFIG_MAP[args.data_config])
    pipeline.configure_data_config(args, data_config)

    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    pipeline.configure_transform(args, modality_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy: BasePolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=str(device),
    )
    if args.residual_checkpoint is not None:
        if args.pipeline != "Local":
            raise ValueError("--residual-checkpoint is supported only with --pipeline Local.")
        policy = load_residual_eval_policy(policy, args.residual_checkpoint, args, device)

    modality = policy.get_modality_config()
    print("Current modality config:\n", modality)

    dataset: LeRobotSingleDataset = pipeline.build_eval_dataset(args, modality)
    if len(dataset) == 0:
        raise ValueError(
            f"Eval dataset {args.dataset_path} has no available steps for pipeline {args.pipeline}."
        )
    print(len(dataset))
    obs = dataset[0]
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)

    first_traj_id, first_base_index = dataset.all_steps[0]
    for key, value in dataset.get_step_data(first_traj_id, first_base_index).items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    loop = asyncio.get_event_loop()
    server = UnityServer(host=args.host, port=args.port, resize_size=(256, 256))
    loop.run_until_complete(server.start())

    result_tag = pipeline.build_result_tag(args)
    if args.residual_checkpoint is not None:
        residual_tag = Path(args.residual_checkpoint).parent.name
        result_tag = f"{result_tag}:residual_{residual_tag}"
    traj_store_path, metrics_json_path = build_eval_output_paths(args, result_tag)
    os.makedirs(traj_store_path, exist_ok=True)
    os.makedirs(args.evaluation_output_path, exist_ok=True)

    print(f"Trajectory store path: {traj_store_path}")
    print(f"Metrics JSON path: {metrics_json_path}")

    if hasattr(policy, "model"):
        policy.model.eval()
        policy.model.action_head.set_seed(SEED)
        print(f"Random seed has been set to {SEED}, and deterministic mode is enabled")

    print("Waiting Unity client connection...")
    if not server.wait_for_connection_sync(timeout=600):
        raise RuntimeError("Unity client connection timed out, please check if the Unity client is launched.")

    print("Unity client connected, evaluation will start, results will be saved to:", metrics_json_path)
    print(f"Model used: {args.model_path}, test dataset: {args.dataset_path}")

    pipeline.run_eval_rollout(
        args,
        server,
        policy,
        dataset,
        metrics_json_path,
        traj_store_path,
    )

    print("Done")
    loop.run_until_complete(server.stop())


if __name__ == "__main__":
    eval_main(tyro.cli(EvalArgsConfig))
