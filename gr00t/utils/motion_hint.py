import os
from pathlib import Path

import numpy as np


MOTION_HINT_FRAME_STRIDE = 5
MOTION_HINT_MIN_PREFIX_FRAMES = MOTION_HINT_FRAME_STRIDE + 1
MOTION_HINT_CACHE_SUBDIR = "meta/motion_hint_rgb_absdiff_stride5"
MOTION_HINT_MANIFEST_FILENAME = "manifest.json"
MOTION_HINT_ALGORITHM = "rgb_absdiff_prefix_stride5_sum_v1"


def format_motion_hint_ratio_tag(motion_hint_ratio: float) -> str:
    return f"ratio_{motion_hint_ratio:.6f}".replace(".", "p")


def get_motion_hint_cache_dir(dataset_path: Path | str, motion_hint_ratio: float) -> Path:
    return Path(dataset_path) / MOTION_HINT_CACHE_SUBDIR / format_motion_hint_ratio_tag(
        motion_hint_ratio
    )


def get_motion_hint_manifest_path(cache_dir: Path | str) -> Path:
    return Path(cache_dir) / MOTION_HINT_MANIFEST_FILENAME


def get_motion_hint_image_path(cache_dir: Path | str, trajectory_id: int) -> Path:
    return Path(cache_dir) / f"episode_{trajectory_id:06d}.png"


def get_motion_hint_sample_indices(prefix_length: int) -> np.ndarray | None:
    if prefix_length <= 0:
        raise ValueError(f"prefix_length must be positive, got {prefix_length}.")
    sample_indices = np.arange(0, prefix_length, MOTION_HINT_FRAME_STRIDE, dtype=np.int64)
    if sample_indices.shape[0] < 2:
        return None
    return sample_indices


def _normalize_motion_hint_channel(channel: np.ndarray) -> np.ndarray:
    max_value = int(channel.max())
    if max_value == 0:
        return np.zeros(channel.shape, dtype=np.uint8)
    encoded = np.rint(channel.astype(np.float32) / max_value * 255.0)
    return encoded.astype(np.uint8)


def compute_motion_hint_from_frames(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"Motion hint expects video frames with shape [T, H, W, C], got {frames.shape}.")
    if frames.shape[0] < 2:
        raise ValueError(f"Motion hint expects at least 2 video frames, got {frames.shape[0]}.")
    if frames.shape[-1] != 3:
        raise ValueError(f"Motion hint expects 3-channel RGB frames, got {frames.shape}.")
    if frames.dtype != np.uint8:
        raise ValueError(f"Motion hint expects uint8 video frames, got {frames.dtype}.")

    frame_diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).astype(np.uint32)
    accumulated_diff = frame_diffs.sum(axis=0, dtype=np.uint32)
    if not np.any(accumulated_diff):
        raise ValueError("Motion hint RGB frame differences are identically zero.")

    return np.stack(
        [
            _normalize_motion_hint_channel(accumulated_diff[..., channel_idx])
            for channel_idx in range(accumulated_diff.shape[2])
        ],
        axis=-1,
    )


def resolve_motion_hint_worker_count(
    requested_num_workers: int,
    logical_cpu_count: int | None = None,
) -> int:
    if requested_num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {requested_num_workers}.")
    if requested_num_workers > 0:
        return requested_num_workers

    detected_cpus = logical_cpu_count if logical_cpu_count is not None else os.cpu_count()
    if detected_cpus is None:
        detected_cpus = 1
    return min(max(1, detected_cpus // 2), 8)
