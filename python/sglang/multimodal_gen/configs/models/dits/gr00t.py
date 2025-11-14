# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class GR00TArchConfig(DiTArchConfig):
    """GR00T-specific architecture configuration for the Diffusion Transformer."""

    # Model dimensions
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12

    # Action dimensions (robot-specific, can be configured per embodiment)
    action_dim: int = 7  # Default: 7DoF robot arm (can be overridden)

    # Conditioning dimensions
    condition_dim: int = 512  # Encoded state dimension from state_encoder

    # Diffusion parameters
    diffusion_steps: int = 4  # GR00T uses 4 denoising steps for efficiency

    # Attention configuration
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FA3,
        }
    )

    def __post_init__(self) -> None:
        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()


@dataclass
class GR00TConfig(DiTConfig):
    """Configuration for GR00T Diffusion Transformer."""

    arch_config: GR00TArchConfig = field(default_factory=GR00TArchConfig)

    # GR00T-specific parameters
    prefix: str = "gr00t"
    quant_config: QuantizationConfig | None = None

    # Embodiment-specific settings
    embodiment_tag: str = "new_embodiment"  # Can be configured per robot type

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "gr00t-config") -> Any:
        """Add CLI arguments for GR00TConfig fields"""
        parser.add_argument(
            f"--{prefix}.prefix",
            type=str,
            dest=f"{prefix.replace('-', '_')}.prefix",
            default=GR00TConfig.prefix,
            help="Prefix for the GR00T model",
        )

        parser.add_argument(
            f"--{prefix}.embodiment-tag",
            type=str,
            dest=f"{prefix.replace('-', '_')}.embodiment_tag",
            default=GR00TConfig.embodiment_tag,
            help="Robot embodiment tag for GR00T",
        )

        parser.add_argument(
            f"--{prefix}.action-dim",
            type=int,
            dest=f"{prefix.replace('-', '_')}.arch_config.action_dim",
            default=GR00TArchConfig.action_dim,
            help="Action dimension for the robot embodiment",
        )

        parser.add_argument(
            f"--{prefix}.condition-dim",
            type=int,
            dest=f"{prefix.replace('-', '_')}.arch_config.condition_dim",
            default=GR00TArchConfig.condition_dim,
            help="Condition dimension from state encoder",
        )

        parser.add_argument(
            f"--{prefix}.diffusion-steps",
            type=int,
            dest=f"{prefix.replace('-', '_')}.arch_config.diffusion_steps",
            default=GR00TArchConfig.diffusion_steps,
            help="Number of diffusion steps for GR00T",
        )

        parser.add_argument(
            f"--{prefix}.quant-config",
            type=str,
            dest=f"{prefix.replace('-', '_')}.quant_config",
            default=None,
            help="Quantization configuration for the GR00T model",
        )

        return parser
