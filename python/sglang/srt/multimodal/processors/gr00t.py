# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_image, load_video, logger


class GR00TMultiModalProcessorOutput(BaseMultiModalProcessorOutput):
    """Output class for GR00T multimodal processor."""

    # Robot states
    states: Optional[list[Union[np.ndarray, dict]]] = None

    # Action sequences
    actions: Optional[list[Union[np.ndarray, dict]]] = None

    def organize_results(self) -> List[Tuple[Modality, any]]:
        """Organize results by modality for GR00T."""
        results = []

        # Images/Videos (for VLM backbone)
        results.extend([(Modality.IMAGE, data) for data in self.images])
        results.extend([(Modality.VIDEO, data) for data in self.videos])

        # States (for state encoder)
        if self.states:
            results.extend([(Modality.STATE, data) for data in self.states])

        # Actions (for action encoder)
        if self.actions:
            results.extend([(Modality.ACTION, data) for data in self.actions])

        return results


class GR00TMultiModalProcessor(SGLangBaseProcessor):
    """
    Multimodal processor for GR00T that handles:
    - Images/Videos (for VLM backbone)
    - Robot states (for state encoder)
    - Action sequences (for action encoder)
    """

    def __init__(
        self,
        image_token: str = "<image>",
        video_token: str = "<video>",
        state_token: str = "<state>",
        action_token: str = "<action>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Special tokens for GR00T
        self.special_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            video_token=video_token,
            state_token=state_token,
            action_token=action_token,
        )

        # Token patterns
        self.image_token_regex = re.compile(re.escape(image_token))
        self.video_token_regex = re.compile(re.escape(video_token))
        self.state_token_regex = re.compile(re.escape(state_token))
        self.action_token_regex = re.compile(re.escape(action_token))

        # Default processing parameters
        self.image_size = kwargs.get("image_size", 224)
        self.max_state_dim = kwargs.get(
            "max_state_dim", 32
        )  # Max robot state dimension
        self.max_action_seq_len = kwargs.get(
            "max_action_seq_len", 16
        )  # Max action sequence length
        self.action_dim = kwargs.get("action_dim", 7)  # Default 7DoF robot

    def _process_image(self, image_path: str, **kwargs) -> Union[Image.Image, dict]:
        """Process a single image for GR00T."""
        try:
            image = load_image(image_path)
            # Apply any GR00T-specific image preprocessing here
            # For now, just return the loaded image
            return image
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

    def _process_video(self, video_path: str, **kwargs) -> Union[torch.Tensor, dict]:
        """Process a video for GR00T."""
        try:
            video_data = load_video(video_path)
            # Apply GR00T-specific video preprocessing
            # Extract frames for VLM processing
            return video_data
        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {e}")
            return None

    def _process_state(
        self, state_data: Union[np.ndarray, list, dict], **kwargs
    ) -> Union[np.ndarray, dict]:
        """Process robot state data for GR00T."""
        try:
            if isinstance(state_data, dict):
                # Handle structured state data
                state_array = np.array(state_data.get("state", []), dtype=np.float32)
            elif isinstance(state_data, list):
                state_array = np.array(state_data, dtype=np.float32)
            elif isinstance(state_data, np.ndarray):
                state_array = state_data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported state data type: {type(state_data)}")

            # Validate state dimensions
            if state_array.shape[-1] > self.max_state_dim:
                logger.warning(
                    f"State dimension {state_array.shape[-1]} exceeds max {self.max_state_dim}, truncating"
                )
                state_array = state_array[..., : self.max_state_dim]

            return state_array

        except Exception as e:
            logger.error(f"Failed to process state data: {e}")
            return None

    def _process_action(
        self, action_data: Union[np.ndarray, list, dict], **kwargs
    ) -> Union[np.ndarray, dict]:
        """Process action sequence data for GR00T."""
        try:
            if isinstance(action_data, dict):
                # Handle structured action data
                action_array = np.array(
                    action_data.get("actions", []), dtype=np.float32
                )
            elif isinstance(action_data, list):
                action_array = np.array(action_data, dtype=np.float32)
            elif isinstance(action_data, np.ndarray):
                action_array = action_data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported action data type: {type(action_data)}")

            # Validate action dimensions
            if action_array.ndim == 1:
                # Single action, add sequence dimension
                action_array = action_array[np.newaxis, :]
            elif action_array.ndim == 2:
                # Action sequence
                seq_len, action_dim = action_array.shape
                if seq_len > self.max_action_seq_len:
                    logger.warning(
                        f"Action sequence length {seq_len} exceeds max {self.max_action_seq_len}, truncating"
                    )
                    action_array = action_array[: self.max_action_seq_len]
                if action_dim != self.action_dim:
                    logger.warning(
                        f"Action dimension {action_dim} doesn't match expected {self.action_dim}"
                    )
            else:
                raise ValueError(
                    f"Invalid action array dimensions: {action_array.shape}"
                )

            return action_array

        except Exception as e:
            logger.error(f"Failed to process action data: {e}")
            return None

    def process_inputs(
        self,
        input_ids: torch.Tensor,
        input_text: str,
        mm_inputs: Optional[List[Dict]] = None,
        **kwargs,
    ) -> GR00TMultiModalProcessorOutput:
        """
        Process multimodal inputs for GR00T.

        Args:
            input_ids: Input token IDs
            input_text: Input text with special tokens
            mm_inputs: List of multimodal inputs (images, videos, states, actions)

        Returns:
            Processed multimodal data
        """
        output = GR00TMultiModalProcessorOutput(input_text=input_text)

        if not mm_inputs:
            return output

        # Process each multimodal input
        for mm_input in mm_inputs:
            input_type = mm_input.get("type", "")
            input_data = mm_input.get("data")

            if input_type == "image":
                image = self._process_image(input_data, **kwargs)
                if image is not None:
                    output.images.append(image)

            elif input_type == "video":
                video = self._process_video(input_data, **kwargs)
                if video is not None:
                    output.videos.append(video)

            elif input_type == "state":
                state = self._process_state(input_data, **kwargs)
                if state is not None:
                    if output.states is None:
                        output.states = []
                    output.states.append(state)

            elif input_type == "action":
                action = self._process_action(input_data, **kwargs)
                if action is not None:
                    if output.actions is None:
                        output.actions = []
                    output.actions.append(action)

            else:
                logger.warning(f"Unknown input type: {input_type}")

        return output

    def process_one_image(self, image: Image.Image, **kwargs) -> torch.Tensor:
        """Process a single image into tensor format."""
        # Apply standard image preprocessing for vision models
        # This would typically include resizing, normalization, etc.
        # For GR00T, this feeds into the VLM backbone
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to expected dimensions
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Apply imagenet normalization (example)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor
