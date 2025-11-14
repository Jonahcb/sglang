# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T configuration and processor registration."""

from transformers import PretrainedConfig

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.srt.multimodal.processors.gr00t import GR00TMultiModalProcessor


@register_customized_processor(GR00TMultiModalProcessor)
class GR00TConfig(PretrainedConfig):
    """Configuration for GR00T model."""

    model_type = "gr00t"
    auto_map = {
        "AutoConfig": "sglang.srt.configs.gr00t.GR00TConfig",
        "AutoModelForCausalLM": "sglang.srt.models.gr00t_model.GR00TForConditionalGeneration",
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # Vision config
        vision_config=None,
        # GR00T-specific parameters
        state_dim=14,  # Robot state dimension (joint positions + velocities)
        action_dim=7,  # Robot action dimension (7DoF arm)
        condition_dim=512,  # Encoded state dimension
        action_hidden_dim=768,  # Hidden dimension for action processing
        action_num_heads=12,  # Attention heads for action processing
        action_num_layers=12,  # Layers for action processing
        max_action_seq_len=16,  # Maximum action sequence length
        diffusion_steps=4,  # GR00T diffusion steps
        num_train_timesteps=1000,  # Total diffusion timesteps
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Vision configuration
        self.vision_config = vision_config

        # GR00T-specific parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.action_hidden_dim = action_hidden_dim
        self.action_num_heads = action_num_heads
        self.action_num_layers = action_num_layers
        self.max_action_seq_len = max_action_seq_len
        self.diffusion_steps = diffusion_steps
        self.num_train_timesteps = num_train_timesteps
