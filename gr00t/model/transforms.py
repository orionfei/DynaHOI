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

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor

from torch.nn.utils.rnn import pad_sequence
def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]

                text_list += curr_text_list
                image_inputs += curr_image_inputs
            eagle_inputs = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )

            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)

from typing import Literal
class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    # Define model type, "base" means origin GR00T-N1.5, "baseline" means ObAct
    vlm_type: Literal["base", "baseline"] = Field(
        default="base", 
        description="Choice of VLM processing logic: 'base' or 'baseline'"
    )
    baseline_motion_hint: Literal["none", "diff_map_and_crop"] = Field(
        default="none",
        description="Optional motion hint for baseline VLM processing.",
    )

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]
        
        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs


    def _validate_baseline_images(self, images: np.ndarray) -> None:
        if images.ndim != 5:
            raise ValueError(f"Baseline VLM expects images with shape [V, T, C, H, W], got {images.shape}.")
        if images.shape[0] != 1:
            raise ValueError(f"Baseline VLM expects a single video stream, got {images.shape[0]}.")
        if images.shape[1] != 6:
            raise ValueError(f"Baseline VLM expects 6 frames (5 observation + 1 current), got {images.shape[1]}.")

    def _compute_baseline_diff_map(self, history_images: np.ndarray, current_image: np.ndarray) -> np.ndarray:
        current_float = current_image.astype(np.float32)
        history_float = history_images.astype(np.float32)
        diff_stack = np.abs(history_float - current_float[None, ...])
        aggregated_diff = np.max(diff_stack, axis=0)
        motion_energy = aggregated_diff.mean(axis=2)
        diff_map = np.repeat(np.rint(motion_energy)[..., None], 3, axis=2).astype(np.uint8)
        return diff_map

    def _compute_baseline_motion_crop(self, motion_energy: np.ndarray, current_image: np.ndarray) -> np.ndarray:
        total_energy = float(motion_energy.sum(dtype=np.float64))
        if total_energy <= 0.0:
            raise ValueError("Baseline motion hint requires positive motion energy.")

        height, width = motion_energy.shape
        ys, xs = np.indices((height, width), dtype=np.float32)
        center_x = float((motion_energy * xs).sum(dtype=np.float64) / total_energy)
        center_y = float((motion_energy * ys).sum(dtype=np.float64) / total_energy)

        dx = xs - center_x
        dy = ys - center_y
        var_x = float((motion_energy * dx * dx).sum(dtype=np.float64) / total_energy)
        var_y = float((motion_energy * dy * dy).sum(dtype=np.float64) / total_energy)
        half_extent = int(np.ceil(np.sqrt(max(var_x, var_y))))
        if half_extent < 1:
            half_extent = 1

        center_x_int = int(np.rint(center_x))
        center_y_int = int(np.rint(center_y))
        x1 = max(center_x_int - half_extent, 0)
        x2 = min(center_x_int + half_extent + 1, width)
        y1 = max(center_y_int - half_extent, 0)
        y2 = min(center_y_int + half_extent + 1, height)

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid baseline motion crop bounds: {(x1, x2, y1, y2)}.")

        motion_crop = current_image[y1:y2, x1:x2]
        resized_motion_crop = Image.fromarray(motion_crop).resize((width, height), resample=Image.BILINEAR)
        return np.asarray(resized_motion_crop, dtype=np.uint8)

    def _build_baseline_conversation(self, eagle_image_objs: list[dict], lang: str) -> list[dict]:
        if self.baseline_motion_hint == "none":
            prompt_prefix = (
                "The first five images are observation history frames uniformly sampled from the "
                "episode's early observation phase. The last image is the current frame.\n"
                "Observation history:\n"
            )
            return [
                {"type": "text", "text": prompt_prefix},
                *eagle_image_objs[:-1],
                {"type": "text", "text": "\nCurrent frame:\n"},
                eagle_image_objs[-1],
                {"type": "text", "text": f"\nTask: {lang}"},
            ]

        if self.baseline_motion_hint == "diff_map_and_crop":
            prompt_prefix = (
                "The first five images are observation history frames uniformly sampled from the "
                "episode's early observation phase. The next two images are change cues where "
                "regions with larger changes indicate the moving object's motion trajectory. "
                "The last image is the current frame.\n"
                "Observation history:\n"
            )
            return [
                {"type": "text", "text": prompt_prefix},
                *eagle_image_objs[:5],
                {"type": "text", "text": "\nChange cues showing the moving object's motion trajectory:\n"},
                eagle_image_objs[5],
                eagle_image_objs[6],
                {"type": "text", "text": "\nCurrent frame:\n"},
                eagle_image_objs[7],
                {"type": "text", "text": f"\nTask: {lang}"},
            ]

        raise ValueError(f"Invalid baseline_motion_hint: {self.baseline_motion_hint}")

    def _apply_vlm_processing_baseline(self, batch: dict) -> dict:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        images = batch["images"]
        self._validate_baseline_images(images)
        np_images = rearrange(images, "v t c h w -> (t v) c h w")

        if self.baseline_motion_hint == "diff_map_and_crop":
            history_images = np.transpose(np_images[:5], (0, 2, 3, 1))
            current_image = np.transpose(np_images[5], (1, 2, 0))
            diff_map = self._compute_baseline_diff_map(history_images, current_image)
            motion_energy = diff_map[..., 0].astype(np.float32)
            motion_crop = self._compute_baseline_motion_crop(motion_energy, current_image)
            diff_map_chw = np.transpose(diff_map, (2, 0, 1))
            motion_crop_chw = np.transpose(motion_crop, (2, 0, 1))
            np_images = np.concatenate(
                [np_images[:5], diff_map_chw[None, ...], motion_crop_chw[None, ...], np_images[5:6]],
                axis=0,
            )

        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]

        eagle_images_pil = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image_objs = [{"type": "image", "image": img} for img in eagle_images_pil]
        conversation_content = self._build_baseline_conversation(eagle_image_objs, lang)

        eagle_conversation = [
            {
                "role": "user",
                "content": conversation_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)

        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}

        if self.vlm_type != "baseline" and self.baseline_motion_hint != "none":
            raise ValueError(
                f"baseline_motion_hint={self.baseline_motion_hint} requires vlm_type='baseline', got {self.vlm_type}."
            )

        if self.vlm_type == "base":
            vlm_outputs = self._apply_vlm_processing(batch_data) 
        elif self.vlm_type == "baseline":
            vlm_outputs = self._apply_vlm_processing_baseline(batch_data)
        else:
            raise ValueError(f"Invalid vlm_type: {self.vlm_type}")
        
        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
