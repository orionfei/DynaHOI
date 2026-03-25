'''
用于baseline模型的微调
基于scripts/gr00t_finetune_our_18dim.py改动
模型结构不变，但是多输入10帧观察视频，并且改prompt结构支持多image输入（11 images）
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # debug用
os.environ["WANDB_PROJECT"] = "GR00T-N1.5-unity"

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, GR00TTransform
from gr00t.utils.peft import get_lora_model

from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import json

def set_baseline_motion_hint(transform: ComposedModalityTransform, baseline_motion_hint: str):
    gr00t_transforms = [t for t in transform.transforms if isinstance(t, GR00TTransform)]
    if len(gr00t_transforms) != 1:
        raise ValueError(f"Expected exactly one GR00TTransform, found {len(gr00t_transforms)}.")

    gr00t_transform = gr00t_transforms[0]
    if gr00t_transform.vlm_type != "baseline":
        raise ValueError(f"Expected baseline GR00TTransform, got {gr00t_transform.vlm_type}.")

    gr00t_transform.baseline_motion_hint = baseline_motion_hint


def save_model_param_info(model: torch.nn.Module, save_path: str):
    """
    Extract the name, parameter count, and trainable status of each parameter in the model,
    and save the information to a CSV file.

    Args:
        model (nn.Module): The model to be analyzed.
        save_path (str): Path to save the CSV file, e.g., "/path/to/output/param_info.csv".
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total_trainable_params = 0.0
    total_params = 0.0

    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['param_name', 'num_params (Million)', 'requires_grad'])  # 写表头

        for name, param in model.named_parameters():
            num_params = param.numel()/1e6
            total_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params
            writer.writerow([name, num_params, param.requires_grad])
            
        # 追加一行显示总的可训练参数量
        writer.writerow([])
        writer.writerow(['Total Trainable Params (Million)', total_trainable_params, ''])
        writer.writerow(['Total Params (Million)', total_params, ''])

    print(f"✅ model info saved: {save_path}")

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str] = field(default_factory=lambda: ["/data1/yfl_data/Dyana_data/train"])
    """Path to the dataset directory or directories"""

    output_dir: str = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/motion_hint/"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "mano_18dim_baseline" # ObAct baseline
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 8
    """Batch size per GPU for training."""

    max_steps: int = 8750
    """Maximum number of training steps."""

    num_gpus: int = 2
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "/data1/yfl_data/DynaHOI/gr00t/checkpoints/ObAct"
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

    # Advanced training parameters
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
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 2
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb" # set "none" for debug，"wandb" for train
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    baseline_motion_hint: Literal["none", "diff_map_and_crop"] = "none"
    """Optional motion hint for baseline VLM processing."""

    observe_frame_source: Literal["videos_obs", "video_prefix"] = "video_prefix"
    """Source used to construct baseline observation frames."""

    observe_frame_num: int = 5
    """Number of prepended observation frames."""

    observe_video_ratio: float = 0.2
    """Prefix ratio used when observe_frame_source == 'video_prefix'."""

    observe_frame_cache_size: int = 1024
    """Number of episodes whose sampled observation frames are cached per worker."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""

    cn_time = datetime.now(ZoneInfo("Asia/Shanghai"))
    month = cn_time.strftime("%m")
    day = cn_time.strftime("%d")
    hour = cn_time.strftime("%H")
    additional_info = config.dataset_path[-1].split("/")[-1]
    config.output_dir = config.output_dir + f"{month}{day}:{hour}_{additional_info}"

    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    set_baseline_motion_hint(transforms, config.baseline_motion_hint)

    # 1.2 data loader: we will use either single dataset or mixture dataset
    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path[0],
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend=config.video_backend,
        add_observe_frames=True,
        observe_frame_source=config.observe_frame_source,
        observe_frame_num=config.observe_frame_num,
        observe_video_ratio=config.observe_video_ratio,
        observe_frame_cache_size=config.observe_frame_cache_size,
    )
       

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    expected_action_dim = 18
    expected_action_horizon = 16

    if data_action_horizon != expected_action_horizon:
        raise ValueError(
            f"Unexpected data action horizon: {data_action_horizon}. "
            f"Expected {expected_action_horizon} for baseline continuation finetuning."
        )

    if model.action_dim != expected_action_dim:
        raise ValueError(
            f"Unexpected model action_dim: {model.action_dim}. "
            f"Expected {expected_action_dim} from ObAct checkpoint."
        )

    if model.action_horizon != expected_action_horizon:
        raise ValueError(
            f"Unexpected model action_horizon: {model.action_horizon}. "
            f"Expected {expected_action_horizon} from ObAct checkpoint."
        )

    if model.action_head.config.action_dim != expected_action_dim:
        raise ValueError(
            f"Unexpected action head config action_dim: {model.action_head.config.action_dim}. "
            f"Expected {expected_action_dim} from ObAct checkpoint."
        )

    if model.action_head.config.action_horizon != expected_action_horizon:
        raise ValueError(
            f"Unexpected action head config action_horizon: {model.action_head.config.action_horizon}. "
            f"Expected {expected_action_horizon} from ObAct checkpoint."
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"


    # save model info for debug
    save_model_param_info(model, os.path.join(config.output_dir, "param_info.csv"))

    if config.num_gpus == 1:
        model.to("cuda:0")

    if config.lora_rank > 0: # no
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        # deepspeed="",
        deepspeed="./scripts/ds_config.json",
        gradient_checkpointing=True,
        bf16=False, # for deepspeed
        fp16=True, # for deepspeed
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
        # evaluation_strategy="no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()

if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        main(config)
    else:

        if (os.environ.get("RANK") is not None or 
        os.environ.get("LOCAL_RANK") is not None or 
        os.environ.get("WORLD_SIZE") is not None or
        os.environ.get("IS_TORCHRUN", "0") == "1"):
            print("🟢 Using torchrun")
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
