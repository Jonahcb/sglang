# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.mlp import MLP


class GR00TTimestepEmbedding(nn.Module):
    """
    Timestep embedding for GR00T Diffusion Transformer.

    Uses sinusoidal embeddings similar to diffusion models, with GR00T-specific
    optimizations for the 4-step diffusion process.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        act_layer: str = "silu",
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        # MLP to project timestep embeddings to model dimension
        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_size,
            act_type=act_layer,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,) with timestep values

        Returns:
            timestep embeddings of shape (batch_size, hidden_size)
        """
        # Create sinusoidal timestep embeddings
        half_dim = self.frequency_embedding_size // 2
        emb = torch.log(torch.tensor(self.max_period)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd dimensions
        if self.frequency_embedding_size % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        # Project to model dimension
        timestep_emb = self.mlp(emb)
        return timestep_emb


class GR00TConditioningEmbedding(nn.Module):
    """
    Conditioning embedding that combines timestep and state embeddings for GR00T.
    """

    def __init__(self, config):
        super().__init__()
        self.time_embed = GR00TTimestepEmbedding(
            hidden_size=config.arch_config.hidden_size,
            frequency_embedding_size=256,
        )

        # Project state conditioning to model dimension
        self.state_proj = nn.Linear(
            config.arch_config.condition_dim, config.arch_config.hidden_size
        )

        # Combine timestep and state conditioning
        self.conditioning_proj = MLP(
            config.arch_config.hidden_size * 2,
            config.arch_config.hidden_size,
            config.arch_config.hidden_size,
            act_type="silu",
        )

    def forward(
        self, timesteps: torch.Tensor, state_conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,) with diffusion timesteps
            state_conditions: Tensor of shape (batch_size, condition_dim) with encoded states

        Returns:
            conditioning embeddings of shape (batch_size, hidden_size)
        """
        # Get timestep embeddings
        time_emb = self.time_embed(timesteps)

        # Project state conditions
        state_emb = self.state_proj(state_conditions)

        # Combine timestep and state conditioning
        combined = torch.cat([time_emb, state_emb], dim=-1)
        conditioning_emb = self.conditioning_proj(combined)

        return conditioning_emb
