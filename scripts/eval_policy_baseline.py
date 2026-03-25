# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro
import random
import torch
# 定死种子
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print(f"Using seed: {SEED}")


from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.model.transforms import GR00TTransform
from gr00t.utils.eval import get_and_send_action_baseline

warnings.simplefilter("ignore", category=FutureWarning)


def set_baseline_motion_hint(transform: ComposedModalityTransform, baseline_motion_hint: str):
    gr00t_transforms = [t for t in transform.transforms if isinstance(t, GR00TTransform)]
    if len(gr00t_transforms) != 1:
        raise ValueError(f"Expected exactly one GR00TTransform, found {len(gr00t_transforms)}.")

    gr00t_transform = gr00t_transforms[0]
    if gr00t_transform.vlm_type != "baseline":
        raise ValueError(f"Expected baseline GR00TTransform, got {gr00t_transform.vlm_type}.")

    gr00t_transform.baseline_motion_hint = baseline_motion_hint

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    modality_keys: List[str] = field(default_factory=lambda: ["left_hand"])
    """Modality keys to evaluate."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "mano_18dim_baseline"
    """Data config to use."""

    steps: int = 150
    """Number of steps to evaluate."""

    trajs: List[int] = field(default_factory=lambda: range(1000))
    """Number of trajectories to evaluate."""

    repeat_num : int = 1
    """Number of times to repeat the evaluation for each episode."""

    action_horizon: int = 16
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "/data1/yfl_data/Dyana_data/test"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    model_path: str = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/motion_hint/v1/checkpoint-8750"
    """Path to the model checkpoint."""

    baseline_motion_hint: Literal["none", "diff_map_and_crop"] = "none"
    """Optional motion hint for baseline VLM processing."""

    sample_frame_num: int = 5
    """Number of uniformly sampled observation frames for baseline evaluation."""
    
    denoising_steps: int = 4
    """Number of denoising steps to use."""

    evaluation_output_path: str = "/data1/yfl_data/DynaHOI/scripts/evaluation_results"
    """Path to save the evaluation results."""

    improve_info: str = "uniform5_baseline"

def main(args: ArgsConfig):
    data_config = DATA_CONFIG_MAP[args.data_config]

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")

    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        set_baseline_motion_hint(modality_transform, args.baseline_motion_hint)

        torch.cuda.manual_seed(2025)
        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)
    trajectory_lengths = dataset.trajectory_lengths


    from gr00t.utils.unity_server import UnityServer
    import asyncio

    loop = asyncio.get_event_loop()
    resize_size = (256, 256)
    server = UnityServer(host="127.0.0.1", port=8765, resize_size=resize_size)
    loop.run_until_complete(server.start())

    if args.sample_frame_num != 5:
        raise ValueError(f"Baseline evaluation requires sample_frame_num=5, got {args.sample_frame_num}.")

    # create directory if not exist
    additional_info_data = args.dataset_path.split("/")[-1].replace("_unity", "")
    additional_info_model = args.model_path.split("/")[-1].replace("unity", "")
    if "checkpoint" in args.model_path:
        additional_info_model_2 = args.model_path.split("/")[-2].replace("unity", "")
        additional_info_model = additional_info_model_2 + "-" + additional_info_model

    # baseline
    result_tag = f"{args.improve_info}:motionhint_{args.baseline_motion_hint}"
    traj_store_path = os.path.join(args.evaluation_output_path, "trajectories", f"{additional_info_model}:{result_tag}")
    metrics_json_path = os.path.join(args.evaluation_output_path, f"results_{additional_info_model}:{result_tag}.jsonl")


    os.makedirs(traj_store_path, exist_ok=True)
    os.makedirs(args.evaluation_output_path, exist_ok=True)

    print(f"Trajectory store path: {traj_store_path}")
    print(f"Metrics JSON path: {metrics_json_path}")

   
    policy.model.eval()
    policy.model.action_head.set_seed(2025)
    print("Random seed has been set to 2025, and deterministic mode is enabled")


    
    # Wait for Unity client connection, timeout in 60 seconds
    print("Waiting Unity client connection...")
    if not server.wait_for_connection_sync(timeout=600):
        print("❌ Unity client connection timed out, please check if the Unity client is launched")
        exit(1)

    
    print("✅ Unity client connected, evaluation will start, results will be saved to:", metrics_json_path)
    print(f"‼️‼️‼️ Model used: {args.model_path},\t Test dataset: {args.dataset_path}")

    total_time = 0
    for traj_id in args.trajs:
        start_time = time.time()
        for i in range(args.repeat_num):
            print("=============================================")
            print(f"Running trajectory: {traj_id}, repeat: {i}")
            time.sleep(0.1) # Wait 0.1 seconds after each repeat for Unity to complete cleanup tasks
            completed = get_and_send_action_baseline(
                    server,
                    policy,
                    dataset,
                    traj_id,
                    repeat_num = i,
                    modality_keys=args.modality_keys,
                    steps=int(trajectory_lengths[traj_id] * 1.2), # Expand interaction window to 1.2 * steps here
                    action_horizon=args.action_horizon,
                    metrics_json_path = metrics_json_path,
                    traj_store_path = traj_store_path,
                    sample_frame_num=args.sample_frame_num,
                )
            if not completed:
                continue
        end_time = time.time()
        print(f"🕙 Time taken: {end_time - start_time} seconds")
        total_time += end_time - start_time
    print("Done")
    # Shutdown the server
    loop.run_until_complete(server.stop())
    print(f"Total time taken: {total_time} seconds")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
