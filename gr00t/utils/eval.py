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
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

import asyncio
from gr00t.utils.unity_server import UnityServer 
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    save_plot_path=None,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []

    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )
        gt_action_across_time.append(concat_gt_action)
        try:
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)
        except KeyError as e:
            print(f"KeyError concatenating state: {e}, we will skip plotting state")

        if step_count % action_horizon == 0:
            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        plot_trajectory(info, save_plot_path)

    return mse


def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    state_joints_across_time = info["state_joints_across_time"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    mse = info["mse"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {mse:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for j in range(0, steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()





# ==================================DynaHOI==================================
from gr00t.utils.video import get_all_frames
def extract_observation_video(traj_id: int, dataset: LeRobotSingleDataset, obs_frames, sample_frames):
    """
    Extract video of the observation phase and uniformly sample sample_frames frames
    """
    video_path = dataset.get_video_path(traj_id, "ego_view")
    frames = get_all_frames(
        video_path.as_posix(),
        dataset.video_backend,
        dataset.video_backend_kwargs,
        (256, 256),
        )[:obs_frames]
    
    # 2. Uniformly sample sample_frames frames
    if len(frames) <= sample_frames:
        # Insufficient frames, return all directly
        return frames
    
    idx = np.linspace(0, len(frames)-1, sample_frames).astype(int)
    sampled_frames = frames[idx]
    return sampled_frames

def extract_observation_by_video_length(traj_id: int, dataset: LeRobotSingleDataset, video_length, video_sample_rate, sample_frames):
    """
    Extract video of the observation phase and uniformly sample sample_frames frames
    """
    video_path = dataset.get_video_path(traj_id, "ego_view")
    video_length = int(video_length * video_sample_rate)
    frames = get_all_frames(
        video_path.as_posix(),
        dataset.video_backend,
        dataset.video_backend_kwargs,
        (256, 256),
        )[:video_length]
    
    # 2. Uniformly sample sample_frames frames
    if len(frames) <= sample_frames:
        # Insufficient frames, return all directly
        return frames
    
    idx = np.linspace(0, len(frames)-1, sample_frames).astype(int)
    sampled_frames = frames[idx]
    return sampled_frames
    


def get_and_send_action(
    server: UnityServer,
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    repeat_num: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    metrics_json_path: str = "/mnt/sdc/bch/forBenchmark/Isaac-GR00T/evaluation_results",
    traj_store_path: str = "",
    do_save_traj: bool = True,
):
    """
    Retrieve and send actions to Unity client for trajectory evaluation.
    
    Args:
        server (UnityServer): Instance of Unity server connection.
        policy (BasePolicy): Policy model for action inference.
        dataset (LeRobotSingleDataset): Dataset containing trajectory metadata.
        traj_id (int): ID of the trajectory to evaluate.
        repeat_num (int): Number of repeat runs for the trajectory.
        modality_keys (list): List of modality keys for observation processing.
        steps (int): Total number of steps for evaluation (default: 300).
        action_horizon (int): Action horizon for batch action sending (default: 16).
        metrics_json_path (str): Path to save evaluation metrics JSON (default: specified path).
        traj_store_path (str): Path to save trajectory data (default: empty string).
        do_save_traj (bool): Whether to save trajectory as NPY file (default: True).
    """

    # Get unity_meta corresponding to traj_id
    unity_meta = dataset.get_unity_meta(traj_id)

    # 原先用于 预处理得到GT定位结束后的帧索引 作为 正确的steps，现在只是处理出来放在metrics里参考
    gt_action_data = dataset.get_trajectory_data(traj_id)["action"].to_numpy()
    gt_action_data = np.array([frame.tolist() for frame in gt_action_data])[:,:3]
    
    
    if not server.is_connected():
        print("❌ Unity client connection check failed")
        server.debug_connection_info()
        raise RuntimeError("Unity client not connected, cannot start evaluation")
    
    print("✅ Unity client connection is normal")
    
    # Send start episode signal using synchronous version
    success = server.send_start_episode_sync(unity_meta["episode"], unity_meta["task_type"], repeat_num, steps, 0, action_horizon)
    if not success:
        raise RuntimeError(f"Failed to send start_episode, traj_id: {traj_id}")
    
    task_dict = {
            "circular": "Grab the object in the video that is making a circular motion",
            "linear": "Grab the object in the video that is making a straight motion",
            "harmonic": "Grab the object in the video that is doing simple harmonic motion"
    }

    trajectory = []
    for step_count in range(steps):
        if step_count % action_horizon == 0:
            # === 2. Get obs from Unity (blocking to ensure state+image alignment) ===
            obs = server.get_obs(block=True)

            # Complete obs with unity metadata (not sent by Unity side)
            obs["annotation.human.action.task_description"] = [task_dict[unity_meta["task_type"]]]

            # === 3. Inference & collect predicted actions ===
            action_chunk = policy.get_action(obs)
            print(f"✅ Inference successful, current step: {step_count} / {steps}")
            action_chunk = action_chunk["action.left_hand"][:, :18] 
 
            if step_count + action_horizon > steps: action_chunk = action_chunk[:steps - step_count]
            # Send to Unity side (using synchronous version)
            success = server.send_action_data_sync(action_chunk)
            if not success: print(f"⚠️ Failed to send action data, current step: {step_count}/ {steps}")
            else: 
                print(f"✅ Action data sent successfully, current step: {step_count} / {steps}")
                trajectory.extend(action_chunk)
    
    # Get evaluation metrics (wait time, success rate, min XZ value) from Unity side
    metrics_unity = server.get_metrics_from_unity(block=True)
    episode_id = metrics_unity.pop("episode_id")
    repeat_num_receive = metrics_unity.pop("repeat")
    successIndex = metrics_unity.pop("successIndex")

    if unity_meta['episode'] != episode_id:
        print(f"❌ episode_id mismatch, Unity side: {episode_id}, Python side: {unity_meta['episode']}")
        print(f"metrics_unity: {metrics_unity}, successIndex: {successIndex}")
        raise RuntimeError("episode_id mismatch")
    if repeat_num_receive != repeat_num:
        print(f"❌ repeat_num mismatch, Unity side: {repeat_num_receive}, Python side: {repeat_num}")
        print(f"metrics_unity: {metrics_unity}, successIndex: {successIndex}")
        raise RuntimeError("repeat_num mismatch")

    trajectory_to_save = np.array(trajectory)


    # focus moving trajectory
    trajectory = trajectory_to_save[:successIndex]
    if do_save_traj:
        np.save(os.path.join(traj_store_path, f"traj_{traj_id}:repeat{repeat_num}.npy"), trajectory_to_save)

    # Evaluate trajectory smoothness between frames and trajectory linearity on Python side
    metrics = evaluate_traj2(trajectory) # 版本2：去掉零附近的，用的是successIndex之前的数据
    
    # Assemble total result
    result = {
        "traj_id": traj_id,
        "repeat_num": repeat_num,
        "episode": int(unity_meta["episode"]),
        "task_type": unity_meta["task_type"],
        **metrics,
        **metrics_unity,
        "successIndex / total_frames": f"{successIndex} / {steps}",
    }
    with open(os.path.join(metrics_json_path), "a") as f:
        f.write(json.dumps(result) + "\n")


def get_and_send_action_baseline(
    server: UnityServer,
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    repeat_num: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    metrics_json_path: str = "/mnt/sdc/bch/forBenchmark/Isaac-GR00T/evaluation_results",
    traj_store_path: str = "",
    do_save_traj: bool = True,
    sample_frame_num = 5,
):
    print(f"The current used sample_frame_num is: {sample_frame_num}")

    # Get unity_meta corresponding to traj_id
    unity_meta = dataset.get_unity_meta(traj_id)

    # Get GT action data and preprocess it for correct steps
    gt_action_data = dataset.get_trajectory_data(traj_id)["action"].to_numpy()
    gt_action_data = np.array([frame.tolist() for frame in gt_action_data])[:,:3]

    # Extract 20% of the video frames at the beginning and sample 5 frames
    obs_sample_frames = extract_observation_by_video_length(traj_id, dataset, steps, 0.2, sample_frame_num)
    obs_sample_frames = np.asarray(obs_sample_frames)

    # Important‼️: The new interface used by baseline, because the input steps is 1.2 times the total number of frames in the episode, so we need to first calculate the original total number of frames, and then calculate the starting frame index
    ori_total_frames = (steps * 5 + 5) // 6 # y = 1.2 * x；则x = (y * 5 + 5) // 6
    start_frame_idx = int(ori_total_frames * 0.2)
    
    # Send start episode signal using synchronous version
    print(f"start_frame_idx / total_frames: {start_frame_idx} / {steps}")
    success = server.send_start_episode_sync(unity_meta["episode"], unity_meta["task_type"], repeat_num, steps , start_frame_idx * 3, action_horizon) # *3是因为unity端物体移动时60FPS，手移动时20FPS
    if not success:
        raise RuntimeError(f"Failed to send start_episode, traj_id: {traj_id}")
    
    task_dict = {
            "circular": "Grab the object in the video that is making a circular motion",
            "linear": "Grab the object in the video that is making a straight motion",
            "harmonic": "Grab the object in the video that is doing simple harmonic motion"
    }
    

    trajectory = []
    for step_count in range(start_frame_idx, steps):
        if (step_count - start_frame_idx) % action_horizon == 0:
            # === 2. Get obs from Unity (blocking to ensure state+image alignment) ===
            obs = server.get_obs(block=True)

            # Put the contents of the observation frames into obs
            ego_cur = obs["video.ego_view"]          # now it is (1, 256, 256, 3)
            # Ensure dtype consistency
            obs_sample_frames_uint8 = obs_sample_frames.astype(ego_cur.dtype)
            # Get 11 or 6 frame sequence: (10 or 5 observation + 1 current)
            ego_seq = np.concatenate(
                [obs_sample_frames_uint8, ego_cur],   # (10 or 5, H, W, 3) + (1, H, W, 3)
                axis=0
            )
            # Write back to obs
            obs["video.ego_view"] = ego_seq          # shape = (11, 256, 256, 3)



            # Use unity metadata to complete obs (Unity side does not send)
            obs["annotation.human.action.task_description"] = [task_dict[unity_meta["task_type"]]]

            if dataset.add_motion_vector:
                obs["video.motion_map"] = dataset.get_motion_map(traj_id, step_count)
            # === 3. Inference & collect predicted actions ===
            action_chunk = policy.get_action(obs)
            print(f"✅ Inference successful, current step: {step_count} / {steps}")

            action_chunk = action_chunk["action.left_hand"][:, :18]
 
            if step_count + action_horizon > steps: action_chunk = action_chunk[:steps - step_count]
            success = server.send_action_data_sync(action_chunk)
            if not success: print(f"⚠️ Failed to send action data, current step: {step_count}/ {steps}")
            else: 
                print(f"✅ Action data sent successfully, current step: {step_count} / {steps}")
                trajectory.extend(action_chunk)
    
    # Get evaluation metrics (wait time, success rate, min XZ value) from Unity side
    metrics_unity = server.get_metrics_from_unity(block=True)
    episode_id = metrics_unity.pop("episode_id")
    repeat_num_receive = metrics_unity.pop("repeat")
    successIndex = metrics_unity.pop("successIndex")

    if unity_meta['episode'] != episode_id:
        print(f"❌ episode_id mismatch, Unity side: {episode_id}, Python side: {unity_meta['episode']}")
        print(f"metrics_unity: {metrics_unity}, successIndex: {successIndex}")
        raise RuntimeError("episode_id mismatch")
    if repeat_num_receive != repeat_num:
        print(f"❌ repeat_num mismatch, Unity side: {repeat_num_receive}, Python side: {repeat_num}")
        print(f"metrics_unity: {metrics_unity}, successIndex: {successIndex}")
        raise RuntimeError("repeat_num mismatch")

    trajectory_to_save = np.array(trajectory)

    # focus moving trajectory
    trajectory = trajectory_to_save[:successIndex]
    if do_save_traj:
        np.save(os.path.join(traj_store_path, f"traj_{traj_id}:repeat{repeat_num}.npy"), trajectory_to_save)

    # Evaluate trajectory smoothness between frames and trajectory linearity on Python side
    metrics = evaluate_traj2(trajectory) # version 2: remove zero nearby, use successIndex before data
    
    # Assemble total result
    result = {
        "traj_id": traj_id,
        "repeat_num": repeat_num,
        "episode": int(unity_meta["episode"]),
        "task_type": unity_meta["task_type"],
        **metrics,  # expand two metrics
        **metrics_unity,  # expand 4 metrics
        # "is_rotating": is_rotating,
        "successIndex / total_frames": f"{successIndex} / {ori_total_frames}",
    }
    with open(os.path.join(metrics_json_path), "a") as f:
        f.write(json.dumps(result) + "\n")


def evaluate_traj2(traj: np.ndarray):
    """
    Evaluate trajectory smoothness and linearity
    
    Args:
        traj (np.ndarray): N x 3 trajectory coordinates (N >= 3)
    
    Returns:
        dict: {
            "smoothness_var": smoothness score based on variance/standard deviation (越接近1越平滑),
            "linearity": linearity score based on cosine similarity (越接近1越直)
        }
    """
    N = traj.shape[0]
    # assert N >= 3, "trajectory needs at least 3 points"

    # --------------------------- 0/1 point ---------------------------
    if N <= 1:
        return {
            "smoothness_var": 1.0,   # no motion, defined as the smoothest
            "linearity": 0.0         # cannot define linearity
        }

    # --------------------------- 2 points ---------------------------
    if N == 2:
        overall_vec = traj[1] - traj[0]
        overall_norm = np.linalg.norm(overall_vec)
        if overall_norm < 1e-8:
            return {
                "smoothness_var": 1.0,  # two points coincide, equivalent to stationary
                "linearity": 0.0
            }
        else:
            return {
                "smoothness_var": 1.0,  # only one step, considered smooth
                "linearity": 1.0        # two points must be on a line
            }

    # --------------------------- N >= 3 ---------------------------
    # ---------- (1) speed (adjacent frame displacement) ----------
    diffs = np.linalg.norm(traj[1:] - traj[:-1], axis=1)  # N-1
    diffs = diffs[diffs > 1e-4]
    mean_step = np.mean(diffs)
    std_step = np.std(diffs)
    
    # Avoid division by zero: if trajectory is stationary, smoothness is set to 1
    smoothness_var = 1.0 if mean_step < 1e-8 else 1 / (1 + std_step / mean_step)

    # ---------- (2) linearity (cosine similarity) ----------
    overall_vec = traj[-1] - traj[0]
    overall_norm = np.linalg.norm(overall_vec)
    if overall_norm < 1e-8:
        linearity = 0.0  # start and end points are the same, not considered linear
    else:
        overall_unit = overall_vec / overall_norm
        segs = traj[1:] - traj[:-1]
        seg_norms = np.linalg.norm(segs, axis=1, keepdims=True)
        valid = seg_norms[:,0] > 1e-8
        segs_unit = segs[valid] / seg_norms[valid]
        cos_sims = segs_unit @ overall_unit
        linearity = np.mean(cos_sims)  # average cosine similarity
    
    return {
        "smoothness_var": float(smoothness_var),
        "linearity": float(linearity)
    }