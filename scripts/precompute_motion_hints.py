import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import cv2
import tyro
from PIL import Image

from gr00t.data.dataset import (
    LeRobotSingleDataset,
    ModalityConfig,
    _compute_motion_hint_frame_count,
    get_motion_hint_cache_dir,
    get_motion_hint_image_path,
    get_motion_hint_manifest_path,
)
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.data.schema import EmbodimentTag
from gr00t.utils.motion_hint import (
    MOTION_HINT_ALGORITHM,
    compute_motion_hint_from_frames,
    get_motion_hint_sample_indices,
    resolve_motion_hint_worker_count,
)
from gr00t.utils.video import get_frames_by_indices


@dataclass(frozen=True)
class MotionHintTask:
    traj_id: int
    prefix_length: int
    video_path: str
    hint_path: str


@dataclass
class PrecomputeMotionHintsArgs:
    dataset_path: str
    """Path to a single LeRobot dataset directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag used to initialize dataset metadata."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend used to decode prefix frames."""

    motion_hint_ratio: float = 0.25
    """Prefix video ratio used to compute the motion hint."""

    trajs: List[int] = field(default_factory=list)
    """Trajectory ids to precompute. Empty means all trajectories."""

    overwrite: bool = False
    """Whether to overwrite existing PNG files and manifest."""

    num_workers: int = 0
    """Worker count. 0 means balanced auto mode."""


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


def write_manifest(cache_dir: Path, motion_hint_ratio: float, overwrite: bool) -> None:
    manifest_path = get_motion_hint_manifest_path(cache_dir)
    manifest = {
        "motion_hint_ratio": motion_hint_ratio,
        "algorithm": MOTION_HINT_ALGORITHM,
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


def build_video_backend_kwargs(video_backend: str) -> dict:
    if video_backend == "decord":
        return {"num_threads": 1}
    return {}


def build_pending_tasks(
    dataset: LeRobotSingleDataset,
    traj_ids: list[int],
    cache_dir: Path,
    motion_hint_ratio: float,
    overwrite: bool,
) -> tuple[list[MotionHintTask], int, int]:
    skipped_existing = 0
    skipped_short_prefix = 0
    pending_tasks: list[MotionHintTask] = []

    for traj_id in traj_ids:
        trajectory_length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(traj_id)])
        prefix_length = _compute_motion_hint_frame_count(trajectory_length, motion_hint_ratio)
        if prefix_length >= trajectory_length or get_motion_hint_sample_indices(prefix_length) is None:
            skipped_short_prefix += 1
            continue

        hint_path = get_motion_hint_image_path(cache_dir, traj_id)
        if hint_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        video_path = dataset.get_video_path(traj_id, "ego_view")
        pending_tasks.append(
            MotionHintTask(
                traj_id=traj_id,
                prefix_length=prefix_length,
                video_path=video_path.as_posix(),
                hint_path=hint_path.as_posix(),
            )
        )

    return pending_tasks, skipped_existing, skipped_short_prefix


def _compute_and_save_motion_hint(
    task: MotionHintTask,
    video_backend: str,
    video_backend_kwargs: dict,
) -> tuple[int, str]:
    sample_indices = get_motion_hint_sample_indices(task.prefix_length)
    if sample_indices is None:
        return task.traj_id, "skipped_short_prefix"

    frames = get_frames_by_indices(
        task.video_path,
        sample_indices,
        video_backend=video_backend,
        video_backend_kwargs=video_backend_kwargs,
    )

    try:
        motion_hint = compute_motion_hint_from_frames(frames)
    except ValueError as exc:
        if "identically zero" in str(exc):
            return task.traj_id, "skipped_zero_motion"
        raise

    Image.fromarray(motion_hint, mode="RGB").save(task.hint_path, compress_level=1)
    return task.traj_id, task.hint_path


def maybe_log_progress(
    completed_tasks: int,
    total_tasks: int,
    computed: int,
    skipped_zero_motion: int,
    progress_interval: int,
) -> None:
    if completed_tasks % progress_interval != 0 and completed_tasks != total_tasks:
        return
    print(
        f"Motion hint progress: {completed_tasks}/{total_tasks}, "
        f"computed={computed}, skipped_zero_motion={skipped_zero_motion}"
    )


def main(args: PrecomputeMotionHintsArgs) -> None:
    if not (0.0 < args.motion_hint_ratio < 1.0):
        raise ValueError(f"motion_hint_ratio must be in (0, 1), got {args.motion_hint_ratio}.")

    effective_workers = resolve_motion_hint_worker_count(args.num_workers)
    cv2.setNumThreads(1)

    dataset = build_dataset(args.dataset_path, args.embodiment_tag, args.video_backend)
    cache_dir = get_motion_hint_cache_dir(dataset.dataset_path, args.motion_hint_ratio)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(cache_dir, args.motion_hint_ratio, args.overwrite)

    traj_ids = args.trajs if len(args.trajs) > 0 else dataset.trajectory_ids.tolist()
    if len(set(traj_ids)) != len(traj_ids):
        raise ValueError("trajs contains duplicates; each trajectory must be assigned exactly once.")

    pending_tasks, skipped_existing, skipped_short_prefix = build_pending_tasks(
        dataset=dataset,
        traj_ids=traj_ids,
        cache_dir=cache_dir,
        motion_hint_ratio=args.motion_hint_ratio,
        overwrite=args.overwrite,
    )
    computed = 0
    skipped_zero_motion = 0

    if len(pending_tasks) == 0:
        print(
            f"Finished motion hint precompute. dataset={args.dataset_path}, "
            f"computed={computed}, skipped_existing={skipped_existing}, "
            f"skipped_short_prefix={skipped_short_prefix}, skipped_zero_motion={skipped_zero_motion}, "
            f"cache_dir={cache_dir}, effective_workers={effective_workers}"
        )
        return

    progress_interval = max(1, len(pending_tasks) // 10)
    video_backend_kwargs = build_video_backend_kwargs(args.video_backend)

    if effective_workers == 1:
        for completed_tasks, task in enumerate(pending_tasks, start=1):
            _, result = _compute_and_save_motion_hint(task, args.video_backend, video_backend_kwargs)
            if result == "skipped_short_prefix":
                skipped_short_prefix += 1
            elif result == "skipped_zero_motion":
                skipped_zero_motion += 1
            else:
                computed += 1
            maybe_log_progress(
                completed_tasks=completed_tasks,
                total_tasks=len(pending_tasks),
                computed=computed,
                skipped_zero_motion=skipped_zero_motion,
                progress_interval=progress_interval,
            )
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_task = {
                executor.submit(
                    _compute_and_save_motion_hint,
                    task,
                    args.video_backend,
                    video_backend_kwargs,
                ): task
                for task in pending_tasks
            }
            for completed_tasks, future in enumerate(as_completed(future_to_task), start=1):
                _, result = future.result()
                if result == "skipped_short_prefix":
                    skipped_short_prefix += 1
                elif result == "skipped_zero_motion":
                    skipped_zero_motion += 1
                else:
                    computed += 1
                maybe_log_progress(
                    completed_tasks=completed_tasks,
                    total_tasks=len(pending_tasks),
                    computed=computed,
                    skipped_zero_motion=skipped_zero_motion,
                    progress_interval=progress_interval,
                )

    print(
        f"Finished motion hint precompute. dataset={args.dataset_path}, "
        f"computed={computed}, skipped_existing={skipped_existing}, "
        f"skipped_short_prefix={skipped_short_prefix}, skipped_zero_motion={skipped_zero_motion}, "
        f"cache_dir={cache_dir}, effective_workers={effective_workers}"
    )


if __name__ == "__main__":
    main(tyro.cli(PrecomputeMotionHintsArgs))
