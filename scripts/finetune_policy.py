import copy
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.pipelines import (
    get_train_pipeline,
    save_model_param_info,
)
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.utils.peft import get_lora_model

os.environ["WANDB_PROJECT"] = "GR00T-N1.5-unity"

@dataclass
class TrainArgsConfig:
    pipeline: str = "Local"
    """Registered train pipeline name."""

    dataset_path: List[str] = field(default_factory=lambda: ["/data1/yfl_data/Dyana_data/train"])
    """Path to the dataset directory or directories."""

    output_dir: str = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/W3_H10"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "Local"
    """Data configuration name from DATA_CONFIG_MAP."""

    batch_size: int = 8
    """Batch size per GPU for training."""

    max_steps: int = 4000
    """Maximum number of training steps."""

    num_gpus: int = 4
    """Number of GPUs to use for training."""

    nnodes: int = 1
    """Number of nodes to use when the script launches torchrun itself."""

    node_rank: int = 0
    """Rank of the current node when nnodes > 1 and the script launches torchrun itself."""

    master_addr: str = "127.0.0.1"
    """Master node address used by the internal torchrun launcher."""

    master_port: int = 29500
    """Master node port used by the internal torchrun launcher."""

    save_steps: int = 40000
    """Number of steps between saving checkpoints."""

    action_horizon: int = 10
    """Number of action steps predicted per inference chunk."""

    action_dim: int = 18
    """Action/state dimension used by pipeline-specific action heads."""

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = False
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 32
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA."""

    dataloader_num_workers: int = 4
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml", "none"] = "wandb"
    """Where to report training metrics."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training."""

    window_length: int = 0
    """Adjacent history frame count for pipelines that support it."""

    observe_frame_offsets: List[int] | None = None
    """Explicit history frame offsets for Local/LoGo, ordered from earliest to latest."""

    motion_hint_ratio: float = 0.25
    """Prefix ratio used by the Global pipeline."""

    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset."""

    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset."""


def train_main(config: TrainArgsConfig):
    pipeline = get_train_pipeline(config.pipeline)
    pipeline.validate_args(config)
    config.output_dir = pipeline.resolve_output_dir(config)

    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config = copy.deepcopy(DATA_CONFIG_MAP[config.data_config])
    pipeline.configure_data_config(config, data_config)
    modality_configs = data_config.modality_config()
    transforms = data_config.transform()
    pipeline.configure_transform(config, transforms)
    train_dataset = pipeline.build_train_dataset(
        config,
        embodiment_tag,
        modality_configs,
        transforms,
    )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )
    pipeline.configure_model_for_train(config, model, data_config)

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    save_model_param_info(model, os.path.join(config.output_dir, "param_info.csv"))

    if config.num_gpus == 1:
        model.to("cuda:0")

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="./scripts/ds_config.json",
        gradient_checkpointing=True,
        bf16=False,
        fp16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()


def _print_config(config: TrainArgsConfig):
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")


def launch_train(config: TrainArgsConfig):
    _print_config(config)

    if config.num_gpus <= 0:
        raise ValueError("Number of GPUs must be greater than 0.")
    if config.nnodes <= 0:
        raise ValueError("nnodes must be greater than 0.")
    if config.node_rank < 0:
        raise ValueError("node_rank must be greater than or equal to 0.")
    if config.node_rank >= config.nnodes:
        raise ValueError(
            f"node_rank ({config.node_rank}) must be smaller than nnodes ({config.nnodes})."
        )
    if config.master_port <= 0:
        raise ValueError("master_port must be greater than 0.")

    is_external_distributed = (
        os.environ.get("RANK") is not None
        or os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
        or os.environ.get("IS_TORCHRUN", "0") == "1"
    )

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not is_external_distributed and config.num_gpus > available_gpus:
        raise ValueError(
            f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})."
        )
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        if config.nnodes != 1:
            raise ValueError("Single-GPU mode only supports nnodes=1.")
        train_main(config)
        return

    if is_external_distributed:
        print("Using externally managed distributed launch")
        train_main(config)
        return

    script_path = Path(__file__).absolute()
    cmd = [
        "torchrun",
        f"--nproc_per_node={config.num_gpus}",
        f"--nnodes={config.nnodes}",
    ]
    if config.nnodes == 1:
        cmd.append("--standalone")
    else:
        cmd.extend(
            [
                f"--node_rank={config.node_rank}",
                f"--master_addr={config.master_addr}",
                f"--master_port={config.master_port}",
            ]
        )
    cmd.append(str(script_path))
    for key, value in vars(config).items():
        if isinstance(value, bool):
            cmd.append(f"--{'' if value else 'no-'}{key.replace('_', '-')}")
            continue
        cmd.append(f"--{key.replace('_', '-')}")
        if isinstance(value, list):
            for item in value:
                cmd.append(str(item))
        else:
            cmd.append(str(value))

    print("Running torchrun command:", cmd)
    env = os.environ.copy()
    env["IS_TORCHRUN"] = "1"
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    launch_train(tyro.cli(TrainArgsConfig))
