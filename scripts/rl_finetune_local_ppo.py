import asyncio
import copy
import html
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal

# Avoid CuBLAS deterministic-kernel errors if any imported GR00T component
# enables PyTorch deterministic mode before inference.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions import Normal

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.pipelines import get_eval_pipeline
from gr00t.model.policy import Gr00tPolicy
from gr00t.utils.eval import evaluate_traj2
from gr00t.utils.unity_server import UnityServer


TASK_DESCRIPTIONS = {
    "circular": "Grab the object in the video that is making a circular motion",
    "linear": "Grab the object in the video that is making a straight motion",
    "harmonic": "Grab the object in the video that is doing simple harmonic motion",
}


@dataclass
class LocalPPOArgs:
    pipeline: Literal["Local"] = "Local"
    """This trainer intentionally supports only the Local pipeline."""

    data_config: Literal["Local"] = "Local"
    """This trainer intentionally supports only the Local data config."""

    model_path: str = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/checkpoint-4000"
    """Best Local SFT checkpoint used as the frozen reference policy."""

    dataset_path: str = "/data1/yfl_data/Dyana_data/test"
    """Dataset whose Unity metadata and history frames define the rollout episodes."""

    output_dir: str = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/local_ppo_residual"
    """Directory for residual PPO checkpoints and logs."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for the SFT policy and dataset."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend used when fetching Local history frames from the dataset."""

    action_horizon: int = 10
    """Number of action frames predicted and sent per Unity communication step."""

    action_dim: int = 18
    """Action dimension. The current DynaHOI hand policy uses 18."""

    window_length: int = 5
    """Local history frame count. Must match the SFT checkpoint configuration."""

    observe_frame_offsets: List[int] | None = None
    """Optional Local history offsets, ordered earliest to latest."""

    motion_hint_ratio: float = 0.25
    """Kept at the Local default; motion hints are not used here."""

    denoising_steps: int = 4
    """Denoising steps used by the frozen SFT policy."""

    trajs: List[int] | None = None
    """Explicit trajectory ids to sample. If set, train_traj_start_index is ignored."""

    train_traj_start_index: int = 1000
    """When trajs is omitted, skip the first N dataset trajectory ids for held-out final testing."""

    train_traj_end_index: int | None = None
    """Optional exclusive end index for automatically selected dataset trajectory ids."""

    max_trajs: int = 0
    """Optional cap after train_traj_start_index/train_traj_end_index slicing. 0 means no cap."""

    unity_trajectory_dir: str | None = None
    """Optional Unity PreProcessPoints directory used to preflight selected episodes."""

    validate_unity_trajectories: bool = False
    """Check selected dataset trajectories exist in Unity PreProcessPoints before training."""

    server_host: str = "0.0.0.0"
    """Host for the Python WebSocket server. Use 0.0.0.0 for remote Unity clients."""

    server_port: int = 8765
    """Port for the Python WebSocket server."""

    unity_connect_timeout: float = 600.0
    """Seconds to wait for the Unity client to connect."""

    obs_timeout: float = 600.0
    """Seconds to wait for each Unity observation."""

    metrics_timeout: float = 1200.0
    """Seconds to wait for terminal Unity metrics."""

    total_updates: int = 50
    """Number of PPO update rounds."""

    episodes_per_update: int = 4
    """Single-Unity rollout count collected before each PPO update."""

    ppo_epochs: int = 4
    """PPO epochs per update."""

    minibatch_size: int = 16
    """Chunk-level PPO minibatch size."""

    learning_rate: float = 1e-4
    """Learning rate for the residual policy and value head."""

    gamma: float = 1.0
    """Discount. With terminal-only reward, 1.0 keeps the episode score intact."""

    clip_range: float = 0.2
    """PPO ratio clipping range."""

    value_coef: float = 0.5
    """Value loss coefficient."""

    entropy_coef: float = 0.001
    """Entropy bonus coefficient."""

    reference_coef: float = 0.10
    """Penalty on residual mean magnitude to keep actions close to SFT."""

    max_grad_norm: float = 1.0
    """Gradient clipping norm."""

    residual_clip: float = 0.03
    """Per-dimension residual action clamp applied before sending to Unity."""

    init_log_std: float = -3.0
    """Initial Gaussian log standard deviation for residual sampling."""

    hidden_dim: int = 256
    """Residual MLP hidden dimension."""

    reward_success_weight: float = 6.0
    reward_score_weight: float = 2.0
    reward_dist_weight: float = 5.0
    reward_residual_weight: float = 0.05
    reward_smooth_weight: float = 0.02

    use_dense_metrics: bool = True
    """Use optional dense metrics from Unity image_and_state messages when available."""

    dense_improvement_weight: float = 3.0
    """Reward coefficient for reducing current palm-object XZ distance between chunks."""

    dense_distance_weight: float = 0.05
    """Small per-chunk penalty on current palm-object XZ distance."""

    dense_success_weight: float = 1.0
    """Per-chunk bonus when Unity reports early success."""

    save_every_updates: int = 1
    """Save residual checkpoint every N updates."""

    plot_training_curves: bool = True
    """Write training_curves.svg after PPO updates. Also writes a PNG when matplotlib is installed."""

    plot_every_updates: int = 1
    """Refresh training curves every N updates. 0 disables periodic plotting."""

    resume_residual_checkpoint: str | None = None
    """Optional residual_policy.pt checkpoint to resume."""

    seed: int = 2025
    """Random seed."""


@dataclass
class RolloutStep:
    feature: torch.Tensor
    residual: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    sft_action: torch.Tensor
    final_action: torch.Tensor
    valid_len: int
    reward: float = 0.0


@dataclass
class EpisodeRollout:
    steps: list[RolloutStep]
    reward: float
    metrics: dict
    traj_id: int
    repeat_num: int
    trajectory: np.ndarray


class ResidualPolicy(nn.Module):
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
        self.log_std = nn.Parameter(
            torch.full((action_horizon, action_dim), float(init_log_std))
        )

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, features: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        raw_mean = self.mean_head(hidden).view(-1, self.action_horizon, self.action_dim)
        mean = torch.tanh(raw_mean) * self.residual_clip
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        value = self.value_head(hidden).squeeze(-1)
        return Normal(mean, std), value, mean


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_global_determinism():
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def make_feature(
    state: np.ndarray,
    sft_action: np.ndarray,
    step_count: int,
    total_steps: int,
    device: torch.device,
) -> torch.Tensor:
    state_tensor = torch.as_tensor(state[-1], dtype=torch.float32, device=device).flatten()
    action_tensor = torch.as_tensor(sft_action, dtype=torch.float32, device=device).flatten()
    progress = torch.tensor(
        [float(step_count) / max(float(total_steps), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([state_tensor, action_tensor, progress], dim=0)


def build_local_obs(
    obs: dict,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    step_count: int,
    history_frame_count: int,
) -> dict:
    ego_cur = obs["video.ego_view"]
    if ego_cur.shape[0] != 1:
        raise ValueError(f"Local PPO expects a single Unity current frame, got {ego_cur.shape}.")

    obs_history_frames = dataset.get_adjacent_observe_frames(
        traj_id,
        "video.ego_view",
        step_count,
    )
    target_height = ego_cur.shape[1]
    target_width = ego_cur.shape[2]
    obs_history_frames = np.stack(
        [
            cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            for frame in obs_history_frames
        ],
        axis=0,
    ).astype(ego_cur.dtype)

    ego_seq = np.concatenate([obs_history_frames, ego_cur], axis=0)
    expected_frames = history_frame_count + 1
    if ego_seq.shape[0] != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} Local frames after history concatenation, got {ego_seq.shape[0]}."
        )
    obs["video.ego_view"] = ego_seq
    return obs


def action_smoothness(actions: np.ndarray) -> float:
    if actions.shape[0] < 2:
        return 0.0
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.square(diffs)))


def compute_terminal_reward(
    metrics: dict,
    trajectory: np.ndarray,
    sampled_residuals: np.ndarray,
    args: LocalPPOArgs,
) -> tuple[float, dict]:
    success = 1.0 if bool(metrics.get("success", False)) else 0.0
    score = float(metrics.get("score", 0.0) or 0.0)
    min_xz = float(metrics.get("min_XZ", 0.0) or 0.0)
    residual_norm = float(np.mean(np.square(sampled_residuals))) if sampled_residuals.size else 0.0
    smooth = action_smoothness(trajectory)

    reward = (
        args.reward_success_weight * success
        + args.reward_score_weight * score
        - args.reward_dist_weight * min_xz
        - args.reward_residual_weight * residual_norm
        - args.reward_smooth_weight * smooth
    )
    reward_terms = {
        "reward": float(reward),
        "reward_success_term": args.reward_success_weight * success,
        "reward_score_term": args.reward_score_weight * score,
        "reward_dist_term": -args.reward_dist_weight * min_xz,
        "reward_residual_term": -args.reward_residual_weight * residual_norm,
        "reward_smooth_term": -args.reward_smooth_weight * smooth,
        "residual_norm": residual_norm,
        "action_smoothness": smooth,
    }
    return float(reward), reward_terms


def compute_dense_reward(
    previous_metrics: dict | None,
    current_metrics: dict | None,
    args: LocalPPOArgs,
) -> float:
    if not args.use_dense_metrics or previous_metrics is None or current_metrics is None:
        return 0.0
    previous_distance = previous_metrics.get("current_distance_xz")
    current_distance = current_metrics.get("current_distance_xz")
    if previous_distance is None or current_distance is None:
        return 0.0
    previous_distance = float(previous_distance)
    current_distance = float(current_distance)
    if not np.isfinite(previous_distance) or not np.isfinite(current_distance):
        return 0.0

    improvement = previous_distance - current_distance
    reward = (
        args.dense_improvement_weight * improvement
        - args.dense_distance_weight * current_distance
    )
    if bool(current_metrics.get("success_early", False)):
        reward += args.dense_success_weight
    return float(reward)


def collect_episode(
    args: LocalPPOArgs,
    server: UnityServer,
    sft_policy: Gr00tPolicy,
    residual_policy: ResidualPolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    repeat_num: int,
    device: torch.device,
) -> EpisodeRollout | None:
    steps = int(dataset.trajectory_lengths[traj_id])
    start_frame_idx = dataset.get_observe_frame_start_index()
    history_frame_count = dataset.observe_frame_num
    if steps <= start_frame_idx:
        print(
            f"Skipping traj_id={traj_id}, repeat={repeat_num}: "
            f"trajectory length {steps} <= Local start index {start_frame_idx}."
        )
        return None

    unity_meta = dataset.get_unity_meta(traj_id)
    task_type = unity_meta["task_type"]
    print(
        f"Starting Local PPO rollout: traj={traj_id}, episode={unity_meta['episode']}, "
        f"task={task_type}, repeat={repeat_num}, start={start_frame_idx}, steps={steps}"
    )
    if hasattr(server, "clear_dense_metrics"):
        server.clear_dense_metrics()
    success = server.send_start_episode_sync(
        unity_meta["episode"],
        task_type,
        repeat_num,
        steps,
        start_frame_idx * 3,
        args.action_horizon,
    )
    if not success:
        raise RuntimeError(f"Failed to send start_episode for traj_id={traj_id}.")

    rollout_steps: list[RolloutStep] = []
    sent_actions: list[np.ndarray] = []
    sampled_residuals: list[np.ndarray] = []
    previous_dense_metrics: dict | None = None
    dense_reward_count = 0
    residual_policy.eval()

    for step_count in range(start_frame_idx, steps, args.action_horizon):
        obs = server.get_obs(block=True, timeout=args.obs_timeout)
        if obs is None:
            raise TimeoutError(f"Timed out waiting for Unity obs at traj={traj_id}, step={step_count}.")
        current_dense_metrics = server.get_dense_metrics_from_unity(block=False)
        if rollout_steps:
            dense_reward = compute_dense_reward(previous_dense_metrics, current_dense_metrics, args)
            rollout_steps[-1].reward += dense_reward
            if current_dense_metrics is not None:
                dense_reward_count += 1
        if current_dense_metrics is not None:
            previous_dense_metrics = current_dense_metrics

        obs = build_local_obs(obs, dataset, traj_id, step_count, history_frame_count)
        obs["annotation.human.action.task_description"] = [TASK_DESCRIPTIONS[task_type]]

        with torch.inference_mode():
            sft_action = sft_policy.get_action(obs)["action.left_hand"][:, : args.action_dim]

        if sft_action.shape != (args.action_horizon, args.action_dim):
            raise ValueError(
                "SFT action shape must match the configured action chunk before final-step slicing, "
                f"got {sft_action.shape}, expected {(args.action_horizon, args.action_dim)}."
            )

        feature = make_feature(
            obs["state.left_hand"],
            sft_action,
            step_count,
            steps,
            device,
        )
        with torch.no_grad():
            dist, value, _ = residual_policy(feature.unsqueeze(0))
            residual = dist.sample()
            log_prob = dist.log_prob(residual).sum(dim=(-1, -2))
            clipped_residual = residual.clamp(-args.residual_clip, args.residual_clip)
            sft_tensor = torch.as_tensor(sft_action, dtype=torch.float32, device=device).unsqueeze(0)
            final_action = sft_tensor + clipped_residual

        valid_len = min(args.action_horizon, steps - step_count)
        action_to_send = final_action[0, :valid_len].detach().cpu().numpy()
        success = server.send_action_data_sync(action_to_send)
        if not success:
            raise RuntimeError(f"Failed to send action_data for traj_id={traj_id}, step={step_count}.")

        rollout_steps.append(
            RolloutStep(
                feature=feature.detach().cpu(),
                residual=residual.squeeze(0).detach().cpu(),
                old_log_prob=log_prob.squeeze(0).detach().cpu(),
                old_value=value.squeeze(0).detach().cpu(),
                sft_action=torch.as_tensor(sft_action, dtype=torch.float32),
                final_action=final_action.squeeze(0).detach().cpu(),
                valid_len=valid_len,
            )
        )
        sent_actions.append(action_to_send)
        sampled_residuals.append(residual.squeeze(0).detach().cpu().numpy()[:valid_len])
        print(f"Sent residual PPO action chunk: traj={traj_id}, step={step_count}, valid_len={valid_len}")

    metrics = server.get_metrics_from_unity(block=True, timeout=args.metrics_timeout)
    if metrics is None:
        raise TimeoutError(f"Timed out waiting for Unity terminal metrics for traj_id={traj_id}.")

    episode_id = metrics.pop("episode_id")
    repeat_num_receive = metrics.pop("repeat")
    if int(unity_meta["episode"]) != int(episode_id):
        raise RuntimeError(
            f"episode_id mismatch: Python={unity_meta['episode']}, Unity={episode_id}, metrics={metrics}"
        )
    if int(repeat_num_receive) != int(repeat_num):
        raise RuntimeError(
            f"repeat mismatch: Python={repeat_num}, Unity={repeat_num_receive}, metrics={metrics}"
        )
    metrics["unity_episode"] = int(episode_id)
    metrics["unity_repeat"] = int(repeat_num_receive)

    trajectory = np.concatenate(sent_actions, axis=0) if sent_actions else np.zeros((0, args.action_dim))
    residual_array = (
        np.concatenate(sampled_residuals, axis=0)
        if sampled_residuals
        else np.zeros((0, args.action_dim))
    )
    reward, reward_terms = compute_terminal_reward(metrics, trajectory, residual_array, args)
    trajectory_metrics = evaluate_traj2(trajectory[:, :3]) if trajectory.shape[0] else {}
    if rollout_steps:
        rollout_steps[-1].reward += reward
    metrics.update(reward_terms)
    metrics.update({f"traj_{k}": v for k, v in trajectory_metrics.items()})
    metrics["dense_reward_count"] = dense_reward_count
    metrics["used_dense_metrics"] = dense_reward_count > 0
    print(
        f"Completed rollout: traj={traj_id}, repeat={repeat_num}, "
        f"success={metrics.get('success')}, reward={reward:.4f}"
    )
    return EpisodeRollout(
        steps=rollout_steps,
        reward=reward,
        metrics=metrics,
        traj_id=traj_id,
        repeat_num=repeat_num,
        trajectory=trajectory,
    )


def flatten_rollouts(episodes: list[EpisodeRollout], args: LocalPPOArgs, device: torch.device):
    rollout_steps = [step for episode in episodes for step in episode.steps]
    rewards = []
    for episode in episodes:
        running_return = 0.0
        episode_returns = []
        for step in reversed(episode.steps):
            running_return = step.reward + args.gamma * running_return
            episode_returns.append(running_return)
        rewards.extend(reversed(episode_returns))
    features = torch.stack([step.feature for step in rollout_steps]).to(device)
    residuals = torch.stack([step.residual for step in rollout_steps]).to(device)
    old_log_probs = torch.stack([step.old_log_prob for step in rollout_steps]).to(device)
    old_values = torch.stack([step.old_value for step in rollout_steps]).to(device)
    returns = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return features, residuals, old_log_probs, returns, advantages


def ppo_update(
    residual_policy: ResidualPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[EpisodeRollout],
    args: LocalPPOArgs,
    device: torch.device,
) -> dict:
    residual_policy.train()
    features, residuals, old_log_probs, returns, advantages = flatten_rollouts(
        episodes,
        args,
        device,
    )
    if features.shape[0] == 0:
        raise ValueError("Cannot run PPO update with an empty rollout buffer.")

    stats: list[dict] = []
    n = features.shape[0]
    for epoch in range(args.ppo_epochs):
        permutation = torch.randperm(n, device=device)
        for start in range(0, n, args.minibatch_size):
            indices = permutation[start : start + args.minibatch_size]
            batch_features = features[indices]
            batch_residuals = residuals[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_returns = returns[indices]
            batch_advantages = advantages[indices]

            dist, values, means = residual_policy(batch_features)
            new_log_probs = dist.log_prob(batch_residuals).sum(dim=(-1, -2))
            entropy = dist.entropy().sum(dim=(-1, -2)).mean()
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            clipped = clipped * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, batch_returns)
            reference_loss = means.square().mean()
            approx_kl = (batch_old_log_probs - new_log_probs).mean()

            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + args.reference_coef * reference_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(residual_policy.parameters(), args.max_grad_norm)
            optimizer.step()

            stats.append(
                {
                    "epoch": epoch,
                    "loss": tensor_to_float(loss),
                    "policy_loss": tensor_to_float(policy_loss),
                    "value_loss": tensor_to_float(value_loss),
                    "entropy": tensor_to_float(entropy),
                    "reference_loss": tensor_to_float(reference_loss),
                    "approx_kl": tensor_to_float(approx_kl),
                    "ratio_mean": tensor_to_float(ratio.mean()),
                }
            )

    return {
        key: float(np.mean([entry[key] for entry in stats]))
        for key in stats[0].keys()
        if key != "epoch"
    }


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _polyline_points(
    updates: list[float],
    values: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if not updates or not values:
        return ""
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-8)
    points = []
    for update, value in zip(updates, values):
        x = left + (update - x_min) / x_span * width
        y = top + height - (value - y_min) / y_span * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def write_training_curves_svg(train_log_path: Path, output_dir: Path) -> Path | None:
    rows = read_jsonl(train_log_path)
    if not rows:
        return None

    panels = [
        ("loss", ["loss"]),
        ("value_loss", ["value_loss"]),
        ("policy_loss", ["policy_loss"]),
        ("reward / success", ["reward_mean", "success_rate"]),
        ("approx_kl", ["approx_kl"]),
        ("ratio_mean", ["ratio_mean"]),
    ]
    colors = {
        "loss": "#2563eb",
        "value_loss": "#7c3aed",
        "policy_loss": "#dc2626",
        "reward_mean": "#059669",
        "success_rate": "#ea580c",
        "approx_kl": "#0891b2",
        "ratio_mean": "#4b5563",
    }
    updates = [_finite_float(row.get("update")) for row in rows]
    valid_update_values = [value for value in updates if value is not None]
    if not valid_update_values:
        return None
    x_min = min(valid_update_values)
    x_max = max(valid_update_values)

    panel_width = 520
    panel_height = 230
    margin_left = 54
    margin_right = 20
    margin_top = 42
    margin_bottom = 34
    gap_x = 34
    gap_y = 34
    cols = 2
    rows_count = math.ceil(len(panels) / cols)
    svg_width = cols * panel_width + (cols - 1) * gap_x
    svg_height = rows_count * panel_height + (rows_count - 1) * gap_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827} .axis{stroke:#d1d5db;stroke-width:1} .grid{stroke:#eef2f7;stroke-width:1} .label{font-size:12px;fill:#4b5563} .title{font-size:15px;font-weight:700}</style>',
    ]

    for idx, (title, keys) in enumerate(panels):
        col = idx % cols
        row_idx = idx // cols
        panel_x = col * (panel_width + gap_x)
        panel_y = row_idx * (panel_height + gap_y)
        left = panel_x + margin_left
        top = panel_y + margin_top
        chart_width = panel_width - margin_left - margin_right
        chart_height = panel_height - margin_top - margin_bottom

        series = {}
        all_values = []
        for key in keys:
            key_updates = []
            key_values = []
            for update, log_row in zip(updates, rows):
                value = _finite_float(log_row.get(key))
                if update is None or value is None:
                    continue
                key_updates.append(update)
                key_values.append(value)
                all_values.append(value)
            series[key] = (key_updates, key_values)
        if not all_values:
            continue
        y_min = min(all_values)
        y_max = max(all_values)
        if abs(y_max - y_min) < 1e-8:
            y_min -= 1.0
            y_max += 1.0
        else:
            pad = 0.08 * (y_max - y_min)
            y_min -= pad
            y_max += pad

        title_text = html.escape(title)
        parts.append(f'<text class="title" x="{panel_x}" y="{panel_y + 18}">{title_text}</text>')
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + chart_height * frac
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}"/>')
        parts.append(f'<text class="label" x="{panel_x}" y="{top + 4}">{y_max:.3g}</text>')
        parts.append(f'<text class="label" x="{panel_x}" y="{top + chart_height}">{y_min:.3g}</text>')
        parts.append(f'<text class="label" x="{left}" y="{top + chart_height + 24}">update {x_min:.0f}</text>')
        parts.append(f'<text class="label" text-anchor="end" x="{left + chart_width}" y="{top + chart_height + 24}">update {x_max:.0f}</text>')

        legend_x = panel_x + panel_width - margin_right - 120
        legend_y = panel_y + 18
        for legend_idx, key in enumerate(keys):
            color = colors.get(key, "#111827")
            y = legend_y + legend_idx * 16
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
            parts.append(f'<text class="label" x="{legend_x + 24}" y="{y + 4}">{html.escape(key)}</text>')

        for key, (key_updates, key_values) in series.items():
            points = _polyline_points(
                key_updates,
                key_values,
                x_min,
                x_max,
                y_min,
                y_max,
                left,
                top,
                chart_width,
                chart_height,
            )
            if points:
                color = colors.get(key, "#111827")
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                    f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
                )

    parts.append("</svg>")
    svg_path = output_dir / "training_curves.svg"
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return svg_path


def write_training_curves_png_if_available(train_log_path: Path, output_dir: Path) -> Path | None:
    rows = read_jsonl(train_log_path)
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib not available; skipped PNG training curves: {exc}")
        return None

    updates = [row["update"] for row in rows]
    panels = [
        ("loss", ["loss"]),
        ("value_loss", ["value_loss"]),
        ("policy_loss", ["policy_loss"]),
        ("reward / success", ["reward_mean", "success_rate"]),
        ("approx_kl", ["approx_kl"]),
        ("ratio_mean", ["ratio_mean"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (title, keys) in zip(axes.flatten(), panels):
        for key in keys:
            values = [row.get(key) for row in rows]
            ax.plot(updates, values, marker="o", linewidth=1.8, label=key)
        ax.set_title(title)
        ax.set_xlabel("update")
        ax.grid(True, alpha=0.25)
        ax.legend()
    png_path = output_dir / "training_curves.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def write_training_curves(train_log_path: Path, output_dir: Path):
    svg_path = write_training_curves_svg(train_log_path, output_dir)
    png_path = write_training_curves_png_if_available(train_log_path, output_dir)
    if svg_path:
        print(f"Wrote training curves: {svg_path}")
    if png_path:
        print(f"Wrote PNG training curves: {png_path}")


def save_checkpoint(
    output_dir: Path,
    update_idx: int,
    residual_policy: ResidualPolicy,
    optimizer: torch.optim.Optimizer,
    args: LocalPPOArgs,
):
    ckpt_dir = output_dir / "checkpoints" / f"update_{update_idx:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update_idx": update_idx,
            "args": asdict(args),
            "residual_policy": residual_policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "residual_policy.pt",
    )
    print(f"Saved residual PPO checkpoint: {ckpt_dir / 'residual_policy.pt'}")


def resolve_trajs(args: LocalPPOArgs, dataset: LeRobotSingleDataset) -> list[int]:
    if args.trajs:
        trajs = list(args.trajs)
    else:
        if args.train_traj_start_index < 0:
            raise ValueError(
                f"train_traj_start_index must be non-negative, got {args.train_traj_start_index}."
            )
        if args.train_traj_end_index is not None and args.train_traj_end_index < args.train_traj_start_index:
            raise ValueError(
                "train_traj_end_index must be greater than or equal to "
                f"train_traj_start_index, got {args.train_traj_end_index} < "
                f"{args.train_traj_start_index}."
            )
        trajectory_ids = list(range(len(dataset.trajectory_lengths)))
        trajs = trajectory_ids[args.train_traj_start_index : args.train_traj_end_index]
        if args.max_trajs > 0:
            trajs = trajs[: args.max_trajs]
    if not trajs:
        raise ValueError(
            "No trajectories selected for Local PPO training. "
            "Check --train-traj-start-index/--train-traj-end-index, or pass explicit --trajs."
        )
    if args.trajs:
        print(f"Using {len(trajs)} explicit Local PPO trajectory ids.")
    else:
        print(
            f"Using {len(trajs)} Local PPO trajectory ids from dataset index "
            f"{args.train_traj_start_index} to {args.train_traj_end_index or 'end'}."
        )
    return trajs


def resolve_default_unity_trajectory_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Unity_data" / "PreProcessPoints"


def validate_unity_trajectory_files(
    args: LocalPPOArgs,
    dataset: LeRobotSingleDataset,
    trajs: list[int],
):
    if not args.validate_unity_trajectories:
        return

    trajectory_dir = (
        Path(args.unity_trajectory_dir)
        if args.unity_trajectory_dir is not None
        else resolve_default_unity_trajectory_dir()
    )
    if not trajectory_dir.exists():
        raise FileNotFoundError(
            f"Unity trajectory directory does not exist: {trajectory_dir}. "
            "Set --unity-trajectory-dir to the local Unity_data/PreProcessPoints path, "
            "or disable this check with --no-validate-unity-trajectories."
        )

    missing = []
    for traj_id in trajs:
        unity_meta = dataset.get_unity_meta(traj_id)
        task_type = unity_meta["task_type"]
        episode = int(unity_meta["episode"])
        expected_path = trajectory_dir / f"{task_type}_{episode}.txt"
        if not expected_path.exists():
            missing.append(
                {
                    "traj_id": int(traj_id),
                    "episode": episode,
                    "task_type": task_type,
                    "path": str(expected_path),
                }
            )

    if missing:
        preview = "\n".join(
            f"  traj_id={item['traj_id']} episode={item['episode']} "
            f"task={item['task_type']} path={item['path']}"
            for item in missing[:10]
        )
        raise FileNotFoundError(
            "Some selected dataset trajectories do not exist in Unity PreProcessPoints. "
            "Use a dataset/split whose unity_meta episodes are present in Unity, "
            "or pass --trajs with valid ids.\n"
            f"Missing {len(missing)} / {len(trajs)} selected trajectories:\n{preview}"
        )

    print(f"Validated {len(trajs)} Unity trajectory files in {trajectory_dir}")


def build_policy_and_dataset(args: LocalPPOArgs, device: torch.device):
    pipeline = get_eval_pipeline(args.pipeline)
    pipeline.validate_args(args)

    data_config = copy.deepcopy(DATA_CONFIG_MAP[args.data_config])
    pipeline.configure_data_config(args, data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    pipeline.configure_transform(args, modality_transform)

    sft_policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=str(device),
    )
    if hasattr(sft_policy, "model"):
        sft_policy.model.eval()
        for param in sft_policy.model.parameters():
            param.requires_grad_(False)
        if hasattr(sft_policy.model.action_head, "seed"):
            sft_policy.model.action_head.seed = args.seed
        if hasattr(sft_policy.model.action_head, "deterministic"):
            sft_policy.model.action_head.deterministic = False

    dataset: LeRobotSingleDataset = pipeline.build_eval_dataset(args, sft_policy.get_modality_config())
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset {args.dataset_path} has no available Local steps. "
            "Check window_length/observe_frame_offsets."
        )
    return sft_policy, dataset


def train_main(args: LocalPPOArgs):
    if args.pipeline != "Local" or args.data_config != "Local":
        raise ValueError("rl_finetune_local_ppo.py only supports pipeline=Local and data_config=Local.")
    if args.action_dim != 18:
        raise ValueError(f"DynaHOI Local PPO expects action_dim=18, got {args.action_dim}.")
    if args.episodes_per_update <= 0:
        raise ValueError("episodes_per_update must be positive.")
    if args.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive.")

    set_seed(args.seed)
    disable_global_determinism()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_policy, dataset = build_policy_and_dataset(args, device)
    trajs = resolve_trajs(args, dataset)
    validate_unity_trajectory_files(args, dataset, trajs)

    feature_dim = args.action_dim + args.action_horizon * args.action_dim + 1
    residual_policy = ResidualPolicy(
        feature_dim=feature_dim,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        residual_clip=args.residual_clip,
        init_log_std=args.init_log_std,
    ).to(device)
    optimizer = torch.optim.AdamW(residual_policy.parameters(), lr=args.learning_rate)

    start_update = 1
    if args.resume_residual_checkpoint:
        checkpoint = torch.load(args.resume_residual_checkpoint, map_location=device)
        residual_policy.load_state_dict(checkpoint["residual_policy"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint.get("update_idx", 0)) + 1
        print(f"Resumed residual policy from {args.resume_residual_checkpoint}")

    loop = asyncio.get_event_loop()
    server = UnityServer(host=args.server_host, port=args.server_port, resize_size=(256, 256))
    loop.run_until_complete(server.start())

    print("Waiting for Unity client connection...")
    if not server.wait_for_connection_sync(timeout=args.unity_connect_timeout):
        raise RuntimeError("Unity client connection timed out.")

    rollout_log_path = output_dir / "rollout_metrics.jsonl"
    train_log_path = output_dir / "train_metrics.jsonl"
    repeat_counter = 0
    traj_cursor = 0

    try:
        for update_idx in range(start_update, args.total_updates + 1):
            episodes: list[EpisodeRollout] = []
            update_start = time.time()
            attempts = 0
            max_attempts = max(args.episodes_per_update * max(len(trajs), 1) * 2, args.episodes_per_update)
            while len(episodes) < args.episodes_per_update:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        "Could not collect enough valid Local PPO episodes. "
                        "Check selected trajs and Local history offsets."
                    )
                traj_id = trajs[traj_cursor % len(trajs)]
                traj_cursor += 1
                episode = collect_episode(
                    args=args,
                    server=server,
                    sft_policy=sft_policy,
                    residual_policy=residual_policy,
                    dataset=dataset,
                    traj_id=traj_id,
                    repeat_num=repeat_counter,
                    device=device,
                )
                repeat_counter += 1
                if episode is None:
                    continue
                episodes.append(episode)
                append_jsonl(
                    rollout_log_path,
                    {
                        "update": update_idx,
                        "traj_id": episode.traj_id,
                        "repeat_num": episode.repeat_num,
                        "num_chunks": len(episode.steps),
                        **episode.metrics,
                    },
                )

            train_stats = ppo_update(residual_policy, optimizer, episodes, args, device)
            reward_mean = float(np.mean([episode.reward for episode in episodes]))
            success_rate = float(np.mean([bool(episode.metrics.get("success", False)) for episode in episodes]))
            train_payload = {
                "update": update_idx,
                "episodes": len(episodes),
                "chunks": sum(len(episode.steps) for episode in episodes),
                "reward_mean": reward_mean,
                "success_rate": success_rate,
                "elapsed_seconds": time.time() - update_start,
                **train_stats,
            }
            append_jsonl(train_log_path, train_payload)
            print(f"PPO update {update_idx} metrics: {train_payload}")
            if (
                args.plot_training_curves
                and args.plot_every_updates > 0
                and update_idx % args.plot_every_updates == 0
            ):
                write_training_curves(train_log_path, output_dir)

            if args.save_every_updates > 0 and update_idx % args.save_every_updates == 0:
                save_checkpoint(output_dir, update_idx, residual_policy, optimizer, args)
    finally:
        loop.run_until_complete(server.stop())


if __name__ == "__main__":
    train_main(tyro.cli(LocalPPOArgs))
