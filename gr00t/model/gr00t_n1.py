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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3
ACTION_HEAD_STATE_ENCODER_KEY = "action_head.state_encoder.layer1.W"
ACTION_HEAD_ACTION_ENCODER_KEY = "action_head.action_encoder.W1.W"
ACTION_HEAD_ACTION_DECODER_KEY = "action_head.action_decoder.layer2.W"


def _iter_checkpoint_safetensor_files(checkpoint_dir: Path, tensor_name: str):
    seen_files = set()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_name = weight_map.get(tensor_name)
        if shard_name is not None:
            shard_path = checkpoint_dir / shard_name
            if shard_path.exists():
                seen_files.add(shard_path)
                yield shard_path

    single_path = checkpoint_dir / "model.safetensors"
    if single_path.exists():
        seen_files.add(single_path)
        yield single_path

    for shard_path in sorted(checkpoint_dir.glob("*.safetensors")):
        if shard_path in seen_files:
            continue
        seen_files.add(shard_path)
        yield shard_path


def _read_checkpoint_tensor_shape(checkpoint_dir: Path, tensor_name: str) -> tuple[int, ...] | None:
    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors is unavailable; skipping checkpoint config reconciliation.")
        return None

    for shard_path in _iter_checkpoint_safetensor_files(checkpoint_dir, tensor_name):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return tuple(f.get_tensor(tensor_name).shape)
    return None


def _infer_action_head_dims_from_checkpoint(checkpoint_dir: Path) -> dict[str, int]:
    state_encoder_shape = _read_checkpoint_tensor_shape(
        checkpoint_dir, ACTION_HEAD_STATE_ENCODER_KEY
    )
    action_encoder_shape = _read_checkpoint_tensor_shape(
        checkpoint_dir, ACTION_HEAD_ACTION_ENCODER_KEY
    )
    action_decoder_shape = _read_checkpoint_tensor_shape(
        checkpoint_dir, ACTION_HEAD_ACTION_DECODER_KEY
    )

    inferred_dims: dict[str, int] = {}
    if state_encoder_shape is not None and len(state_encoder_shape) == 3:
        inferred_dims["max_state_dim"] = state_encoder_shape[1]

    action_dim = None
    if action_encoder_shape is not None and len(action_encoder_shape) == 3:
        action_dim = action_encoder_shape[1]

    decoder_action_dim = None
    if action_decoder_shape is not None and len(action_decoder_shape) == 3:
        decoder_action_dim = action_decoder_shape[2]

    if action_dim is not None and decoder_action_dim is not None and action_dim != decoder_action_dim:
        raise ValueError(
            "Checkpoint action head dimensions are inconsistent: "
            f"{ACTION_HEAD_ACTION_ENCODER_KEY} implies {action_dim}, "
            f"but {ACTION_HEAD_ACTION_DECODER_KEY} implies {decoder_action_dim}."
        )

    resolved_action_dim = action_dim if action_dim is not None else decoder_action_dim
    if resolved_action_dim is not None:
        inferred_dims["action_dim"] = resolved_action_dim
        inferred_dims["max_action_dim"] = resolved_action_dim

    return inferred_dims


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path
        print("[GR00T_N1_5] Using EagleBackbone (default)")
        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def _reconcile_pretrained_config(
        cls, checkpoint_path: str | Path, config: GR00T_N1_5_Config | None = None
    ) -> GR00T_N1_5_Config:
        checkpoint_dir = Path(checkpoint_path)
        if config is None:
            config = cls.config_class.from_pretrained(str(checkpoint_dir))

        action_head_cfg = dict(getattr(config, "action_head_cfg", {}) or {})
        inferred_dims = _infer_action_head_dims_from_checkpoint(checkpoint_dir)
        if not inferred_dims or not action_head_cfg:
            return config

        updates: dict[str, tuple[Any, Any]] = {}
        if "action_dim" in inferred_dims and getattr(config, "action_dim", None) != inferred_dims["action_dim"]:
            updates["action_dim"] = (getattr(config, "action_dim", None), inferred_dims["action_dim"])
            config.action_dim = inferred_dims["action_dim"]

        for key in ("action_dim", "max_action_dim", "max_state_dim"):
            if key in inferred_dims and action_head_cfg.get(key) != inferred_dims[key]:
                updates[f"action_head_cfg.{key}"] = (action_head_cfg.get(key), inferred_dims[key])
                action_head_cfg[key] = inferred_dims[key]

        if updates:
            config.action_head_cfg = action_head_cfg
            update_summary = ", ".join(
                f"{key}: {old} -> {new}" for key, (old, new) in updates.items()
            )
            print(
                "Reconciled stale checkpoint config from saved tensor shapes at "
                f"{checkpoint_dir}: {update_summary}"
            )

        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        config = kwargs.pop("config", None)
        config = cls._reconcile_pretrained_config(local_model_path, config)

        pretrained_model = super().from_pretrained(
            local_model_path,
            local_model_path=local_model_path,
            config=config,
            **kwargs,
        )  

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
