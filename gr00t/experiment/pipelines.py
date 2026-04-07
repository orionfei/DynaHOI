import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import torch

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import GR00TTransform
from gr00t.utils.eval import (
    get_and_send_action,
    get_and_send_action_baseline,
    get_and_send_action_baseline_motion_hint,
    get_and_send_action_motion_hint,
)


@dataclass(frozen=True)
class TrainPipelineSpec:
    name: str
    validate_args: Callable[[Any], None]
    configure_data_config: Callable[[Any, Any], None]
    configure_transform: Callable[[Any, ComposedModalityTransform], None]
    build_train_dataset: Callable[
        [Any, EmbodimentTag, dict[str, Any], ComposedModalityTransform],
        LeRobotSingleDataset | LeRobotMixtureDataset,
    ]
    configure_model_for_train: Callable[[Any, GR00T_N1_5, Any], None]
    resolve_output_dir: Callable[[Any], str]


@dataclass(frozen=True)
class EvalPipelineSpec:
    name: str
    validate_args: Callable[[Any], None]
    configure_data_config: Callable[[Any, Any], None]
    configure_transform: Callable[[Any, ComposedModalityTransform], None]
    build_eval_dataset: Callable[[Any, dict[str, Any]], LeRobotSingleDataset]
    run_eval_rollout: Callable[[Any, Any, Any, Any, str, str], None]
    build_result_tag: Callable[[Any], str]


TRAIN_PIPELINES: dict[str, TrainPipelineSpec] = {}
EVAL_PIPELINES: dict[str, EvalPipelineSpec] = {}


def register_train_pipeline(name: str):
    def wrapper(spec: TrainPipelineSpec) -> TrainPipelineSpec:
        if name in TRAIN_PIPELINES:
            raise ValueError(f"Train pipeline {name!r} is already registered.")
        if spec.name != name:
            raise ValueError(f"Train pipeline name mismatch: decorator={name!r}, spec={spec.name!r}.")
        TRAIN_PIPELINES[name] = spec
        return spec

    return wrapper


def register_eval_pipeline(name: str):
    def wrapper(spec: EvalPipelineSpec) -> EvalPipelineSpec:
        if name in EVAL_PIPELINES:
            raise ValueError(f"Eval pipeline {name!r} is already registered.")
        if spec.name != name:
            raise ValueError(f"Eval pipeline name mismatch: decorator={name!r}, spec={spec.name!r}.")
        EVAL_PIPELINES[name] = spec
        return spec

    return wrapper


def get_train_pipeline(name: str) -> TrainPipelineSpec:
    if name not in TRAIN_PIPELINES:
        available = ", ".join(sorted(TRAIN_PIPELINES))
        raise ValueError(f"Unknown train pipeline {name!r}. Available train pipelines: {available}")
    return TRAIN_PIPELINES[name]


def get_eval_pipeline(name: str) -> EvalPipelineSpec:
    if name not in EVAL_PIPELINES:
        available = ", ".join(sorted(EVAL_PIPELINES))
        raise ValueError(f"Unknown eval pipeline {name!r}. Available eval pipelines: {available}")
    return EVAL_PIPELINES[name]


def ensure_positive(value: int, name: str):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def ensure_exact(value: int, expected: int, name: str):
    if value != expected:
        raise ValueError(f"{name} must be {expected} for this pipeline, got {value}.")


def set_baseline_window_length(transform: ComposedModalityTransform, window_length: int):
    gr00t_transforms = [t for t in transform.transforms if isinstance(t, GR00TTransform)]
    if len(gr00t_transforms) != 1:
        raise ValueError(f"Expected exactly one GR00TTransform, found {len(gr00t_transforms)}.")

    gr00t_transform = gr00t_transforms[0]
    if gr00t_transform.vlm_type not in {"baseline", "baseline_motion_hint"}:
        raise ValueError(
            "Expected baseline-style GR00TTransform, got "
            f"{gr00t_transform.vlm_type}."
        )
    ensure_positive(window_length, "window_length")
    gr00t_transform.window_length = window_length


def save_model_param_info(model: torch.nn.Module, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total_trainable_params = 0.0
    total_params = 0.0

    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_name", "num_params (Million)", "requires_grad"])

        for name, param in model.named_parameters():
            num_params = param.numel() / 1e6
            total_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params
            writer.writerow([name, num_params, param.requires_grad])

        writer.writerow([])
        writer.writerow(["Total Trainable Params (Million)", total_trainable_params, ""])
        writer.writerow(["Total Params (Million)", total_params, ""])

    print(f"model info saved: {save_path}")


def sync_action_head_config(model: GR00T_N1_5, action_dim: int, action_horizon: int):
    model.action_dim = action_dim
    model.action_horizon = action_horizon
    model.action_head.config.action_dim = action_dim
    model.action_head.config.action_horizon = action_horizon
    model.action_head.config.max_action_dim = action_dim
    model.action_head.config.max_state_dim = action_dim
    model.config.action_dim = action_dim
    model.config.action_horizon = action_horizon
    model.config.action_head_cfg["action_dim"] = action_dim
    model.config.action_head_cfg["action_horizon"] = action_horizon
    model.config.action_head_cfg["max_action_dim"] = action_dim
    model.config.action_head_cfg["max_state_dim"] = action_dim

    if model.action_head.config.action_dim != action_dim:
        raise ValueError(
            f"Expected action_head.config.action_dim={action_dim}, got {model.action_head.config.action_dim}"
        )
    if model.action_head.config.action_horizon != action_horizon:
        raise ValueError(
            f"Expected action_head.config.action_horizon={action_horizon}, got {model.action_head.config.action_horizon}"
        )
    if model.action_head.config.max_action_dim != action_dim:
        raise ValueError(
            f"Expected action_head.config.max_action_dim={action_dim}, got {model.action_head.config.max_action_dim}"
        )
    if model.action_head.config.max_state_dim != action_dim:
        raise ValueError(
            f"Expected action_head.config.max_state_dim={action_dim}, got {model.action_head.config.max_state_dim}"
        )
    if model.action_dim != action_dim or model.action_horizon != action_horizon:
        raise ValueError(
            f"Expected model action dims to be ({action_dim}, {action_horizon}), got ({model.action_dim}, {model.action_horizon})"
        )
    if model.config.action_dim != action_dim or model.config.action_horizon != action_horizon:
        raise ValueError(
            "Serialized model config is out of sync with action head config: "
            f"action_dim={model.config.action_dim}, action_horizon={model.config.action_horizon}"
        )
    for key, expected_value in {
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "max_action_dim": action_dim,
        "max_state_dim": action_dim,
    }.items():
        actual_value = model.config.action_head_cfg.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Serialized action_head_cfg[{key!r}] is out of sync: expected {expected_value}, got {actual_value}"
            )


def recreate_action_head_for_horizon(
    model: GR00T_N1_5,
    action_horizon: int,
    tune_projector: bool,
    tune_diffusion_model: bool,
):
    if action_horizon == model.action_head.config.action_horizon:
        return

    print(
        f"Recreating action head with action_horizon {action_horizon} "
        f"(was {model.action_head.config.action_horizon})"
    )
    new_action_head_config = model.action_head.config
    new_action_head_config.action_horizon = action_horizon
    new_action_head = FlowmatchingActionHead(new_action_head_config)
    new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
    model.action_head = new_action_head
    model.config.action_horizon = action_horizon
    model.action_horizon = action_horizon
    model.config.action_head_cfg["action_horizon"] = action_horizon
    model.action_head.set_trainable_parameters(
        tune_projector=tune_projector,
        tune_diffusion_model=tune_diffusion_model,
    )


def replace_action_head_keep_dit(
    model: GR00T_N1_5,
    action_dim: int,
    action_horizon: int,
    tune_projector: bool,
    tune_diffusion_model: bool,
):
    new_action_head_config = model.action_head.config
    new_action_head_config.action_dim = action_dim
    new_action_head_config.action_horizon = action_horizon
    new_action_head_config.max_action_dim = action_dim
    new_action_head_config.max_state_dim = action_dim

    old_dit = model.action_head.model
    new_action_head = FlowmatchingActionHead(new_action_head_config)
    new_action_head.model = old_dit
    new_action_head.set_trainable_parameters(
        tune_projector=tune_projector,
        tune_diffusion_model=tune_diffusion_model,
    )
    model.action_head = new_action_head
    sync_action_head_config(model, action_dim, action_horizon)


def build_default_train_dataset(
    config: Any,
    embodiment_tag: EmbodimentTag,
    modality_configs: dict[str, Any],
    transforms: ComposedModalityTransform,
) -> LeRobotSingleDataset | LeRobotMixtureDataset:
    if len(config.dataset_path) == 1:
        return LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )

    single_datasets = []
    for dataset_path in config.dataset_path:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        single_datasets.append(
            LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
        )

    return LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in single_datasets],
        mode="train",
        balance_dataset_weights=config.balance_dataset_weights,
        balance_trajectory_weights=config.balance_trajectory_weights,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )


def build_baseline_train_dataset(
    config: Any,
    embodiment_tag: EmbodimentTag,
    modality_configs: dict[str, Any],
    transforms: ComposedModalityTransform,
) -> LeRobotSingleDataset:
    if len(config.dataset_path) != 1:
        raise ValueError(
            f"baseline_adjacent_window expects exactly one dataset path, got {len(config.dataset_path)}."
        )
    return LeRobotSingleDataset(
        dataset_path=config.dataset_path[0],
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        add_observe_frames=True,
        observe_frame_num=config.window_length,
    )


def build_default_eval_dataset(args: Any, modality_config: dict[str, Any]) -> LeRobotSingleDataset:
    return LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )


def build_motion_hint_train_dataset(
    config: Any,
    embodiment_tag: EmbodimentTag,
    modality_configs: dict[str, Any],
    transforms: ComposedModalityTransform,
) -> LeRobotSingleDataset | LeRobotMixtureDataset:
    if len(config.dataset_path) == 1:
        return LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            use_motion_hint=True,
            motion_hint_ratio=config.motion_hint_ratio,
            motion_hint_num_frames=config.motion_hint_num_frames,
        )

    single_datasets = []
    for dataset_path in config.dataset_path:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        single_datasets.append(
            LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                use_motion_hint=True,
                motion_hint_ratio=config.motion_hint_ratio,
                motion_hint_num_frames=config.motion_hint_num_frames,
            )
        )

    return LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in single_datasets],
        mode="train",
        balance_dataset_weights=config.balance_dataset_weights,
        balance_trajectory_weights=config.balance_trajectory_weights,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )


def build_motion_hint_eval_dataset(args: Any, modality_config: dict[str, Any]) -> LeRobotSingleDataset:
    return LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
        use_motion_hint=True,
        motion_hint_ratio=args.motion_hint_ratio,
        motion_hint_num_frames=args.motion_hint_num_frames,
    )


def build_baseline_motion_hint_train_dataset(
    config: Any,
    embodiment_tag: EmbodimentTag,
    modality_configs: dict[str, Any],
    transforms: ComposedModalityTransform,
) -> LeRobotSingleDataset:
    if len(config.dataset_path) != 1:
        raise ValueError(
            "baseline_adjacent_window_motion_hint_farneback expects exactly one dataset path, "
            f"got {len(config.dataset_path)}."
        )
    return LeRobotSingleDataset(
        dataset_path=config.dataset_path[0],
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        add_observe_frames=True,
        observe_frame_num=config.window_length,
        use_motion_hint=True,
        motion_hint_ratio=config.motion_hint_ratio,
        motion_hint_num_frames=config.motion_hint_num_frames,
    )


def build_baseline_motion_hint_eval_dataset(args: Any, modality_config: dict[str, Any]) -> LeRobotSingleDataset:
    return LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
        add_observe_frames=True,
        observe_frame_num=args.window_length,
        use_motion_hint=True,
        motion_hint_ratio=args.motion_hint_ratio,
        motion_hint_num_frames=args.motion_hint_num_frames,
    )


def _normalize_dataset_tag(dataset_path: str) -> str:
    return dataset_path.split("/")[-1].replace("_unity", "")


def _normalize_model_tag(model_path: str) -> str:
    model_tag = model_path.split("/")[-1].replace("unity", "")
    if "checkpoint" in model_path:
        parent_tag = model_path.split("/")[-2].replace("unity", "")
        model_tag = parent_tag + "-" + model_tag
    return model_tag


def build_eval_output_paths(args: Any, result_tag: str) -> tuple[str, str]:
    model_tag = _normalize_model_tag(args.model_path)
    traj_store_path = os.path.join(
        args.evaluation_output_path,
        "trajectories",
        f"{model_tag}:{result_tag}",
    )
    metrics_json_path = os.path.join(
        args.evaluation_output_path,
        f"results_{model_tag}:{result_tag}.jsonl",
    )
    return traj_store_path, metrics_json_path


def build_our_eval_result_tag(args: Any) -> str:
    dataset_tag = _normalize_dataset_tag(args.dataset_path)
    return f"{args.pipeline}:{dataset_tag}"


def build_baseline_eval_result_tag(args: Any) -> str:
    dataset_tag = _normalize_dataset_tag(args.dataset_path)
    return f"{args.pipeline}:window_{args.window_length}:{dataset_tag}"


def build_motion_hint_eval_result_tag(args: Any) -> str:
    dataset_tag = _normalize_dataset_tag(args.dataset_path)
    return f"{args.pipeline}:{dataset_tag}"


def build_baseline_motion_hint_eval_result_tag(args: Any) -> str:
    dataset_tag = _normalize_dataset_tag(args.dataset_path)
    return (
        f"{args.pipeline}:window_{args.window_length}:"
        f"ratio_{args.motion_hint_ratio}:frames_{args.motion_hint_num_frames}:{dataset_tag}"
    )


def run_standard_eval_rollout(
    args: Any,
    server: Any,
    policy: Any,
    dataset: LeRobotSingleDataset,
    metrics_json_path: str,
    traj_store_path: str,
):
    trajectory_lengths = dataset.trajectory_lengths
    total_time = 0
    for traj_id in args.trajs:
        start_time = datetime.now().timestamp()
        for repeat_idx in range(args.repeat_num):
            print("=============================================")
            print(f"Running trajectory: {traj_id}, repeat: {repeat_idx}")
            get_and_send_action(
                server,
                policy,
                dataset,
                traj_id,
                repeat_num=repeat_idx,
                modality_keys=args.modality_keys,
                steps=int(trajectory_lengths[traj_id]),
                action_horizon=args.action_horizon,
                metrics_json_path=metrics_json_path,
                traj_store_path=traj_store_path,
            )
        end_time = datetime.now().timestamp()
        print(f"Time taken: {end_time - start_time} seconds")
        total_time += end_time - start_time
    print(f"Total time taken: {total_time} seconds")


def run_baseline_eval_rollout(
    args: Any,
    server: Any,
    policy: Any,
    dataset: LeRobotSingleDataset,
    metrics_json_path: str,
    traj_store_path: str,
):
    trajectory_lengths = dataset.trajectory_lengths
    total_time = 0
    for traj_id in args.trajs:
        start_time = datetime.now().timestamp()
        for repeat_idx in range(args.repeat_num):
            print("=============================================")
            print(f"Running trajectory: {traj_id}, repeat: {repeat_idx}")
            completed = get_and_send_action_baseline(
                server,
                policy,
                dataset,
                traj_id,
                repeat_num=repeat_idx,
                modality_keys=args.modality_keys,
                steps=int(trajectory_lengths[traj_id]),
                action_horizon=args.action_horizon,
                metrics_json_path=metrics_json_path,
                traj_store_path=traj_store_path,
                window_length=args.window_length,
            )
            if not completed:
                continue
        end_time = datetime.now().timestamp()
        print(f"Time taken: {end_time - start_time} seconds")
        total_time += end_time - start_time
    print(f"Total time taken: {total_time} seconds")


def run_motion_hint_eval_rollout(
    args: Any,
    server: Any,
    policy: Any,
    dataset: LeRobotSingleDataset,
    metrics_json_path: str,
    traj_store_path: str,
):
    trajectory_lengths = dataset.trajectory_lengths
    total_time = 0
    for traj_id in args.trajs:
        if not dataset.is_motion_hint_trajectory_available(traj_id):
            print(f"Skipping trajectory {traj_id}: motion hint is unavailable.")
            continue
        start_time = datetime.now().timestamp()
        for repeat_idx in range(args.repeat_num):
            print("=============================================")
            print(f"Running trajectory: {traj_id}, repeat: {repeat_idx}")
            completed = get_and_send_action_motion_hint(
                server,
                policy,
                dataset,
                traj_id,
                repeat_num=repeat_idx,
                modality_keys=args.modality_keys,
                steps=int(trajectory_lengths[traj_id]),
                action_horizon=args.action_horizon,
                metrics_json_path=metrics_json_path,
                traj_store_path=traj_store_path,
            )
            if not completed:
                continue
        end_time = datetime.now().timestamp()
        print(f"Time taken: {end_time - start_time} seconds")
        total_time += end_time - start_time
    print(f"Total time taken: {total_time} seconds")


def run_baseline_motion_hint_eval_rollout(
    args: Any,
    server: Any,
    policy: Any,
    dataset: LeRobotSingleDataset,
    metrics_json_path: str,
    traj_store_path: str,
):
    trajectory_lengths = dataset.trajectory_lengths
    total_time = 0
    for traj_id in args.trajs:
        if not dataset.is_motion_hint_trajectory_available(traj_id):
            print(f"Skipping trajectory {traj_id}: motion hint is unavailable.")
            continue
        start_time = datetime.now().timestamp()
        for repeat_idx in range(args.repeat_num):
            print("=============================================")
            print(f"Running trajectory: {traj_id}, repeat: {repeat_idx}")
            completed = get_and_send_action_baseline_motion_hint(
                server,
                policy,
                dataset,
                traj_id,
                repeat_num=repeat_idx,
                modality_keys=args.modality_keys,
                steps=int(trajectory_lengths[traj_id]),
                action_horizon=args.action_horizon,
                metrics_json_path=metrics_json_path,
                traj_store_path=traj_store_path,
                window_length=args.window_length,
            )
            if not completed:
                continue
        end_time = datetime.now().timestamp()
        print(f"Time taken: {end_time - start_time} seconds")
        total_time += end_time - start_time
    print(f"Total time taken: {total_time} seconds")


def no_op_transform(_: Any, __: ComposedModalityTransform):
    return


def resolve_output_dir_identity(config: Any) -> str:
    return config.output_dir


def resolve_output_dir_with_timestamp(config: Any) -> str:
    cn_time = datetime.now()
    run_suffix = f"{cn_time.strftime('%m')}{cn_time.strftime('%d')}:{cn_time.strftime('%H')}"
    dataset_suffix = config.dataset_path[-1].split("/")[-1]
    return os.path.join(config.output_dir, f"{run_suffix}_{dataset_suffix}")


def validate_our_train_args(config: Any):
    ensure_exact(config.action_dim, 18, "action_dim")
    ensure_positive(config.action_horizon, "action_horizon")
    if config.window_length != 0:
        raise ValueError(f"window_length is unsupported for pipeline 'our_18d', got {config.window_length}.")
    if config.motion_hint_ratio != 0.25 or config.motion_hint_num_frames != 6:
        raise ValueError(
            "motion_hint_ratio and motion_hint_num_frames are unsupported for pipeline 'our_18d'."
        )


def validate_baseline_train_args(config: Any):
    ensure_positive(config.action_dim, "action_dim")
    ensure_positive(config.action_horizon, "action_horizon")
    ensure_positive(config.window_length, "window_length")
    if config.motion_hint_ratio != 0.25 or config.motion_hint_num_frames != 6:
        raise ValueError(
            "motion_hint_ratio and motion_hint_num_frames are unsupported for pipeline 'baseline_adjacent_window'."
        )


def validate_motion_hint_train_args(config: Any):
    ensure_exact(config.action_dim, 18, "action_dim")
    ensure_positive(config.action_horizon, "action_horizon")
    ensure_positive(config.motion_hint_num_frames, "motion_hint_num_frames")
    if not (0.0 < config.motion_hint_ratio < 1.0):
        raise ValueError(
            f"motion_hint_ratio must be in (0, 1) for pipeline 'motion_hint_farneback', got {config.motion_hint_ratio}."
        )
    if config.window_length != 0:
        raise ValueError(
            f"window_length is unsupported for pipeline 'motion_hint_farneback', got {config.window_length}."
        )


def validate_baseline_motion_hint_train_args(config: Any):
    ensure_exact(config.action_dim, 18, "action_dim")
    ensure_positive(config.action_horizon, "action_horizon")
    ensure_positive(config.window_length, "window_length")
    ensure_positive(config.motion_hint_num_frames, "motion_hint_num_frames")
    if not (0.0 < config.motion_hint_ratio < 1.0):
        raise ValueError(
            "motion_hint_ratio must be in (0, 1) for pipeline "
            f"'baseline_adjacent_window_motion_hint_farneback', got {config.motion_hint_ratio}."
        )


def validate_our_eval_args(args: Any):
    ensure_exact(args.action_dim, 18, "action_dim")
    if args.window_length != 0:
        raise ValueError(f"window_length is unsupported for pipeline 'our_18d', got {args.window_length}.")
    if args.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {args.action_horizon}.")
    if args.motion_hint_ratio != 0.25 or args.motion_hint_num_frames != 6:
        raise ValueError(
            "motion_hint_ratio and motion_hint_num_frames are unsupported for pipeline 'our_18d'."
        )


def validate_baseline_eval_args(args: Any):
    ensure_exact(args.action_dim, 18, "action_dim")
    ensure_positive(args.window_length, "window_length")
    ensure_positive(args.action_horizon, "action_horizon")
    if args.motion_hint_ratio != 0.25 or args.motion_hint_num_frames != 6:
        raise ValueError(
            "motion_hint_ratio and motion_hint_num_frames are unsupported for pipeline 'baseline_adjacent_window'."
        )


def validate_motion_hint_eval_args(args: Any):
    ensure_exact(args.action_dim, 18, "action_dim")
    ensure_positive(args.action_horizon, "action_horizon")
    ensure_positive(args.motion_hint_num_frames, "motion_hint_num_frames")
    if not (0.0 < args.motion_hint_ratio < 1.0):
        raise ValueError(
            f"motion_hint_ratio must be in (0, 1) for pipeline 'motion_hint_farneback', got {args.motion_hint_ratio}."
        )
    if args.window_length != 0:
        raise ValueError(
            f"window_length is unsupported for pipeline 'motion_hint_farneback', got {args.window_length}."
        )


def validate_baseline_motion_hint_eval_args(args: Any):
    ensure_exact(args.action_dim, 18, "action_dim")
    ensure_positive(args.action_horizon, "action_horizon")
    ensure_positive(args.window_length, "window_length")
    ensure_positive(args.motion_hint_num_frames, "motion_hint_num_frames")
    if not (0.0 < args.motion_hint_ratio < 1.0):
        raise ValueError(
            "motion_hint_ratio must be in (0, 1) for pipeline "
            f"'baseline_adjacent_window_motion_hint_farneback', got {args.motion_hint_ratio}."
        )


def configure_our_train_data_config(config: Any, data_config: Any):
    data_config.action_indices = list(range(config.action_horizon))


def configure_baseline_train_data_config(config: Any, data_config: Any):
    data_config.action_indices = list(range(config.action_horizon))


def configure_baseline_eval_data_config(args: Any, data_config: Any):
    data_config.action_indices = list(range(args.action_horizon))


def configure_motion_hint_data_config(config: Any, data_config: Any):
    data_config.action_indices = list(range(config.action_horizon))


def configure_our_model_for_train(config: Any, model: GR00T_N1_5, data_config: Any):
    data_action_horizon = len(data_config.action_indices)
    recreate_action_head_for_horizon(
        model,
        action_horizon=data_action_horizon,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )
    replace_action_head_keep_dit(
        model,
        action_dim=18,
        action_horizon=config.action_horizon,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )


def configure_baseline_model_for_train(config: Any, model: GR00T_N1_5, data_config: Any):
    data_action_horizon = len(data_config.action_indices)
    recreate_action_head_for_horizon(
        model,
        action_horizon=data_action_horizon,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )
    replace_action_head_keep_dit(
        model,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )


register_train_pipeline("our_18d")(
    TrainPipelineSpec(
        name="our_18d",
        validate_args=validate_our_train_args,
        configure_data_config=configure_our_train_data_config,
        configure_transform=no_op_transform,
        build_train_dataset=build_default_train_dataset,
        configure_model_for_train=configure_our_model_for_train,
        resolve_output_dir=resolve_output_dir_with_timestamp,
    )
)

register_train_pipeline("baseline_adjacent_window")(
    TrainPipelineSpec(
        name="baseline_adjacent_window",
        validate_args=validate_baseline_train_args,
        configure_data_config=configure_baseline_train_data_config,
        configure_transform=lambda config, transform: set_baseline_window_length(
            transform, config.window_length
        ),
        build_train_dataset=build_baseline_train_dataset,
        configure_model_for_train=configure_baseline_model_for_train,
        resolve_output_dir=resolve_output_dir_identity,
    )
)

register_train_pipeline("motion_hint_farneback")(
    TrainPipelineSpec(
        name="motion_hint_farneback",
        validate_args=validate_motion_hint_train_args,
        configure_data_config=configure_motion_hint_data_config,
        configure_transform=no_op_transform,
        build_train_dataset=build_motion_hint_train_dataset,
        configure_model_for_train=configure_our_model_for_train,
        resolve_output_dir=resolve_output_dir_with_timestamp,
    )
)

register_train_pipeline("baseline_adjacent_window_motion_hint_farneback")(
    TrainPipelineSpec(
        name="baseline_adjacent_window_motion_hint_farneback",
        validate_args=validate_baseline_motion_hint_train_args,
        configure_data_config=configure_motion_hint_data_config,
        configure_transform=lambda config, transform: set_baseline_window_length(
            transform, config.window_length
        ),
        build_train_dataset=build_baseline_motion_hint_train_dataset,
        configure_model_for_train=configure_our_model_for_train,
        resolve_output_dir=resolve_output_dir_with_timestamp,
    )
)

register_eval_pipeline("our_18d")(
    EvalPipelineSpec(
        name="our_18d",
        validate_args=validate_our_eval_args,
        configure_data_config=configure_our_train_data_config,
        configure_transform=no_op_transform,
        build_eval_dataset=build_default_eval_dataset,
        run_eval_rollout=run_standard_eval_rollout,
        build_result_tag=build_our_eval_result_tag,
    )
)

register_eval_pipeline("baseline_adjacent_window")(
    EvalPipelineSpec(
        name="baseline_adjacent_window",
        validate_args=validate_baseline_eval_args,
        configure_data_config=configure_baseline_eval_data_config,
        configure_transform=lambda args, transform: set_baseline_window_length(
            transform, args.window_length
        ),
        build_eval_dataset=lambda args, modality_config: LeRobotSingleDataset(
            dataset_path=args.dataset_path,
            modality_configs=modality_config,
            video_backend=args.video_backend,
            video_backend_kwargs=None,
            transforms=None,
            embodiment_tag=args.embodiment_tag,
            add_observe_frames=True,
            observe_frame_num=args.window_length,
        ),
        run_eval_rollout=run_baseline_eval_rollout,
        build_result_tag=build_baseline_eval_result_tag,
    )
)

register_eval_pipeline("motion_hint_farneback")(
    EvalPipelineSpec(
        name="motion_hint_farneback",
        validate_args=validate_motion_hint_eval_args,
        configure_data_config=configure_motion_hint_data_config,
        configure_transform=no_op_transform,
        build_eval_dataset=build_motion_hint_eval_dataset,
        run_eval_rollout=run_motion_hint_eval_rollout,
        build_result_tag=build_motion_hint_eval_result_tag,
    )
)

register_eval_pipeline("baseline_adjacent_window_motion_hint_farneback")(
    EvalPipelineSpec(
        name="baseline_adjacent_window_motion_hint_farneback",
        validate_args=validate_baseline_motion_hint_eval_args,
        configure_data_config=configure_motion_hint_data_config,
        configure_transform=lambda args, transform: set_baseline_window_length(
            transform, args.window_length
        ),
        build_eval_dataset=build_baseline_motion_hint_eval_dataset,
        run_eval_rollout=run_baseline_motion_hint_eval_rollout,
        build_result_tag=build_baseline_motion_hint_eval_result_tag,
    )
)
