import json
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import List, Literal

import tyro
from PIL import Image

from gr00t.data.dataset import (
    LeRobotSingleDataset,
    ModalityConfig,
    _compute_motion_hint_frame_count,
    compute_motion_hint_from_frames,
    get_motion_hint_cache_dir,
    get_motion_hint_image_path,
    get_motion_hint_manifest_path,
)
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.data.schema import EmbodimentTag
from gr00t.utils.video import get_uniform_prefix_frames


@dataclass
class PrecomputeMotionHintsArgs:
    dataset_path: str
    """Path to a single LeRobot dataset directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag used to initialize dataset metadata."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend used to decode prefix frames."""

    motion_hint_ratio: float = 0.25
    """Prefix video ratio used to compute Farneback motion hint."""

    motion_hint_num_frames: int = 6
    """Number of uniformly sampled prefix frames used to compute motion hint."""

    trajs: List[int] = field(default_factory=list)
    """Trajectory ids to precompute. Empty means all trajectories."""

    overwrite: bool = False
    """Whether to overwrite existing PNG files and manifest."""

    num_workers: int = 4
    """Number of worker processes for trajectory-level parallel precompute."""


_WORKER_DATASET: LeRobotSingleDataset | None = None
_WORKER_VIDEO_BACKEND: str | None = None
_WORKER_MOTION_HINT_RATIO: float | None = None
_WORKER_MOTION_HINT_NUM_FRAMES: int | None = None
_WORKER_CACHE_DIR: Path | None = None


def build_dataset(dataset_path: str, embodiment_tag: str, video_backend: str) -> LeRobotSingleDataset:
    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["video.ego_view"]),
    }
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        video_backend=video_backend,
    )


def _init_worker(
    dataset_path: str,
    embodiment_tag: str,
    video_backend: str,
    motion_hint_ratio: float,
    motion_hint_num_frames: int,
):
    global _WORKER_DATASET
    global _WORKER_VIDEO_BACKEND
    global _WORKER_MOTION_HINT_RATIO
    global _WORKER_MOTION_HINT_NUM_FRAMES
    global _WORKER_CACHE_DIR

    _WORKER_DATASET = build_dataset(dataset_path, embodiment_tag, video_backend)
    _WORKER_VIDEO_BACKEND = video_backend
    _WORKER_MOTION_HINT_RATIO = motion_hint_ratio
    _WORKER_MOTION_HINT_NUM_FRAMES = motion_hint_num_frames
    _WORKER_CACHE_DIR = get_motion_hint_cache_dir(dataset_path, motion_hint_ratio)


def _compute_and_save_motion_hint(traj_id: int) -> tuple[int, str]:
    if _WORKER_DATASET is None:
        raise RuntimeError("Motion hint worker dataset is not initialized.")
    if _WORKER_VIDEO_BACKEND is None:
        raise RuntimeError("Motion hint worker video backend is not initialized.")
    if _WORKER_MOTION_HINT_RATIO is None:
        raise RuntimeError("Motion hint worker ratio is not initialized.")
    if _WORKER_MOTION_HINT_NUM_FRAMES is None:
        raise RuntimeError("Motion hint worker num_frames is not initialized.")
    if _WORKER_CACHE_DIR is None:
        raise RuntimeError("Motion hint worker cache dir is not initialized.")

    prefix_start_index = _compute_motion_hint_frame_count(
        int(_WORKER_DATASET.trajectory_lengths[_WORKER_DATASET.get_trajectory_index(traj_id)]),
        _WORKER_MOTION_HINT_RATIO,
    )
    if prefix_start_index < _WORKER_MOTION_HINT_NUM_FRAMES:
        return traj_id, "skipped_short_prefix"
    video_path = _WORKER_DATASET.get_video_path(traj_id, "ego_view")
    prefix_frames, _ = get_uniform_prefix_frames(
        video_path.as_posix(),
        prefix_ratio=_WORKER_MOTION_HINT_RATIO,
        num_sampled_frames=_WORKER_MOTION_HINT_NUM_FRAMES,
        video_backend=_WORKER_VIDEO_BACKEND,
        video_backend_kwargs={},
        resize_size=None,
    )
    if prefix_frames is None:
        return traj_id, "skipped_short_prefix"
    motion_hint = compute_motion_hint_from_frames(prefix_frames)
    hint_path = get_motion_hint_image_path(_WORKER_CACHE_DIR, traj_id)
    Image.fromarray(motion_hint, mode="RGB").save(hint_path)
    return traj_id, hint_path.as_posix()


def write_manifest(
    cache_dir: Path, motion_hint_ratio: float, motion_hint_num_frames: int, overwrite: bool
) -> None:
    manifest_path = get_motion_hint_manifest_path(cache_dir)
    manifest = {
        "motion_hint_ratio": motion_hint_ratio,
        "motion_hint_num_frames": motion_hint_num_frames,
        "algorithm": "farneback_weighted_uv_magnitude_v1",
    }

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)
        if existing_manifest != manifest and not overwrite:
            raise ValueError(
                f"Existing motion hint manifest at {manifest_path} does not match requested config. "
                "Pass --overwrite to replace it."
            )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main(args: PrecomputeMotionHintsArgs) -> None:
    if not (0.0 < args.motion_hint_ratio < 1.0):
        raise ValueError(f"motion_hint_ratio must be in (0, 1), got {args.motion_hint_ratio}.")
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}.")
    if args.motion_hint_num_frames < 2:
        raise ValueError(
            f"motion_hint_num_frames must be at least 2, got {args.motion_hint_num_frames}."
        )

    dataset = build_dataset(args.dataset_path, args.embodiment_tag, args.video_backend)
    cache_dir = get_motion_hint_cache_dir(dataset.dataset_path, args.motion_hint_ratio)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(cache_dir, args.motion_hint_ratio, args.motion_hint_num_frames, args.overwrite)

    traj_ids = args.trajs if len(args.trajs) > 0 else dataset.trajectory_ids.tolist()
    if len(set(traj_ids)) != len(traj_ids):
        raise ValueError("trajs contains duplicates; each trajectory must be assigned exactly once.")
    computed = 0
    skipped_existing = 0
    skipped_short_prefix = 0
    pending_traj_ids: list[int] = []
    for traj_id in traj_ids:
        prefix_start_index = _compute_motion_hint_frame_count(
            int(dataset.trajectory_lengths[dataset.get_trajectory_index(traj_id)]),
            args.motion_hint_ratio,
        )
        if prefix_start_index < args.motion_hint_num_frames:
            print(
                f"Skipping trajectory {traj_id}: prefix_length={prefix_start_index} "
                f"is smaller than motion_hint_num_frames={args.motion_hint_num_frames}."
            )
            skipped_short_prefix += 1
            continue
        hint_path = get_motion_hint_image_path(cache_dir, traj_id)
        if hint_path.exists() and not args.overwrite:
            print(f"Skipping existing motion hint: {hint_path}")
            skipped_existing += 1
            continue
        pending_traj_ids.append(traj_id)

    if len(pending_traj_ids) == 0:
        print(
            f"Finished motion hint precompute. dataset={args.dataset_path}, "
            f"computed={computed}, skipped_existing={skipped_existing}, "
            f"skipped_short_prefix={skipped_short_prefix}, cache_dir={cache_dir}"
        )
        return

    if args.num_workers == 1:
        _init_worker(
            args.dataset_path,
            args.embodiment_tag,
            args.video_backend,
            args.motion_hint_ratio,
            args.motion_hint_num_frames,
        )
        for traj_id in pending_traj_ids:
            _, hint_path = _compute_and_save_motion_hint(traj_id)
            if hint_path == "skipped_short_prefix":
                print(
                    f"Skipping trajectory {traj_id}: prefix_length became shorter than "
                    f"motion_hint_num_frames={args.motion_hint_num_frames}."
                )
                skipped_short_prefix += 1
                continue
            print(f"Saved motion hint for trajectory {traj_id} to {hint_path}")
            computed += 1
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                args.dataset_path,
                args.embodiment_tag,
                args.video_backend,
                args.motion_hint_ratio,
                args.motion_hint_num_frames,
            ),
        ) as executor:
            future_to_traj = {
                executor.submit(_compute_and_save_motion_hint, traj_id): traj_id
                for traj_id in pending_traj_ids
            }
            for future in as_completed(future_to_traj):
                traj_id, hint_path = future.result()
                if hint_path == "skipped_short_prefix":
                    print(
                        f"Skipping trajectory {traj_id}: prefix_length became shorter than "
                        f"motion_hint_num_frames={args.motion_hint_num_frames}."
                    )
                    skipped_short_prefix += 1
                    continue
                print(f"Saved motion hint for trajectory {traj_id} to {hint_path}")
                computed += 1

    print(
        f"Finished motion hint precompute. dataset={args.dataset_path}, "
        f"computed={computed}, skipped_existing={skipped_existing}, "
        f"skipped_short_prefix={skipped_short_prefix}, cache_dir={cache_dir}"
    )


if __name__ == "__main__":
    main(tyro.cli(PrecomputeMotionHintsArgs))
