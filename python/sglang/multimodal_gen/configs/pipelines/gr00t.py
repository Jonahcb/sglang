# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.gr00t import GR00TConfig
from sglang.multimodal_gen.configs.models.encoders.llama import LlamaConfig
from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig as BaseVAEConfig
from sglang.multimodal_gen.configs.pipelines.base import ModelTaskType, PipelineConfig


def gr00t_preprocess_text(prompt: str) -> str:
    """Preprocess text for GR00T - add instruction format if needed."""
    return prompt


def gr00t_postprocess_text(outputs, text_inputs, drop_idx: int = 0):
    """Postprocess text outputs for GR00T."""
    # For GR00T, we may not need complex text processing
    # Return the hidden states directly
    if hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states[-1]
        return hidden_states
    return outputs


@dataclass
class GR00TPipelineConfig(PipelineConfig):
    """Configuration for GR00T robot control pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2V  # We'll repurpose this for robot control

    # GR00T-specific configs
    dit_config: DiTConfig = field(default_factory=GR00TConfig)

    # VAE - GR00T may not use traditional VAE, but we keep for compatibility
    vae_config: VAEConfig = field(default_factory=BaseVAEConfig)

    # Text encoding for instructions
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (LlamaConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (gr00t_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable[..., torch.Tensor], ...] = field(
        default_factory=lambda: (gr00t_postprocess_text,)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=True,
                truncation=True,
                max_length=512,
            ),
            None,
        ]
    )

    @staticmethod
    def add_cli_args(parser):
        """Add CLI arguments for GR00T pipeline configuration."""
        parser.add_argument(
            "--gr00t-embodiment-id",
            type=str,
            default="new_embodiment",
            help="Robot embodiment ID for GR00T",
        )
        parser.add_argument(
            "--gr00t-action-dim",
            type=int,
            default=7,
            help="Action dimension for the robot embodiment",
        )
        parser.add_argument(
            "--gr00t-condition-dim",
            type=int,
            default=512,
            help="Condition dimension from state encoder",
        )
        parser.add_argument(
            "--gr00t-diffusion-steps",
            type=int,
            default=4,
            help="Number of diffusion steps for GR00T",
        )

    def set_width_and_height(self, height, width, pil_image=None):
        """GR00T doesn't use traditional image dimensions."""
        # Robot control doesn't have spatial dimensions like images
        # We can set dummy values or use action dimensions
        return 1, 1  # Dummy values since robot control doesn't use spatial dimensions

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        """Prepare latent shape for GR00T action prediction."""
        # For GR00T, latents represent action vectors, not image latents
        # Shape: (batch_size, action_dim)
        action_dim = self.dit_config.arch_config.action_dim
        return (batch_size, action_dim)

    def pack_latents(self, latents, batch_size, batch):
        """Pack latents for GR00T - no special packing needed for action vectors."""
        return latents
