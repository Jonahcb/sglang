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

"""
Implementation of GR00T model for SGLang.
Reference: https://github.com/NVIDIA/Isaac-GR00T
"""

from dataclasses import dataclass, field
from typing import Iterable, Tuple
import os
import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

logger = logging.getLogger(__name__)




# Constants
BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"

    backbone_cfg: dict = field(default_factory=dict, metadata={"help": "Backbone configuration."})
    action_head_cfg: dict = field(default_factory=lambda: {
        "action_dim": 32,
        "action_horizon": 16,
        "hidden_size": 1024,
    }, metadata={"help": "Action head configuration."})
    action_horizon: int = field(default=16, metadata={"help": "Action horizon."})
    action_dim: int = field(default=32, metadata={"help": "Action dimension."})
    hidden_size: int = field(default=1024, metadata={"help": "Hidden size."})
    model_dtype: str = field(default="float32", metadata={"help": "Model dtype."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Torch dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def text_config(self):
        """Get the text config from the Eagle backbone."""
        from sglang.srt.utils.hf_transformers_utils import get_config

        # Load config from the local eagle2_hg_model directory
        eagle_model_path = "/data/work/sglang/eagle2_hg_model"
        if os.path.exists(eagle_model_path):
            # Load config from the local eagle2_hg_model directory
            eagle_config = get_config(eagle_model_path, trust_remote_code=True)
            return eagle_config.text_config
        else:
            # Fallback to the original logic
            eagle_path = self.backbone_cfg.get("eagle_path")

            # HACK: Override the custom eagle_path with a publicly available checkpoint
            if eagle_path == "NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops":
                eagle_path = "NVIDIA/Eagle2.5-8B"  # Use publicly available Eagle checkpoint

            if eagle_path:
                # Pass token if available
                kwargs = {"trust_remote_code": True}
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if token:
                    kwargs["token"] = token

                eagle_config = get_config(eagle_path, **kwargs)
                return eagle_config.text_config
            else:
                # Fallback: try to construct from backbone_cfg
                from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLConfig
                eagle_config = Eagle2_5_VLConfig(**self.backbone_cfg)
                return eagle_config.text_config


@dataclass
class FlowmatchingActionHeadConfig:
    """Configuration for the flow matching action head."""
    action_dim: int = 32
    action_horizon: int = 16
    add_pos_embed: bool = True
    backbone_embedding_dim: int = 2048  # Will be overridden by actual backbone hidden_size
    diffusion_model_cfg: dict = field(default_factory=dict)
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    max_action_dim: int = 32
    max_state_dim: int = 64
    model_dtype: str = "float32"
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_inference_timesteps: int = 4
    num_target_vision_tokens: int = 32
    num_timestep_buckets: int = 1000
    tune_diffusion_model: bool = True
    tune_projector: bool = True
    use_vlln: bool = True
    vl_self_attention_cfg: dict = field(default_factory=dict)


class FlowmatchingActionHead(nn.Module):
    """
    Flow matching action head for GR00T model.
    Simplified implementation aligned with reference architecture.
    """

    def __init__(self, config: FlowmatchingActionHeadConfig):
        super().__init__()
        self.config = config

        # Use parameters from diffusion_model_cfg
        diffusion_cfg = config.diffusion_model_cfg
        hidden_size = config.hidden_size
        num_layers = diffusion_cfg.get("num_layers", 4)
        num_heads = diffusion_cfg.get("num_attention_heads", 8)
        attention_head_dim = diffusion_cfg.get("attention_head_dim", 64)

        # Project backbone features to action space
        self.backbone_projector = nn.Linear(config.backbone_embedding_dim, hidden_size)

        # DiT (Diffusion Transformer) for flow matching
        self.dit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Action decoder
        self.action_decoder = nn.Linear(hidden_size, config.action_horizon * config.action_dim)


    def get_action(self, backbone_outputs: BatchFeature, action_inputs: BatchFeature = None) -> BatchFeature:
        """Get action predictions during inference."""
        backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]

        # Project backbone features
        projected_features = self.backbone_projector(backbone_features)

        # Apply DiT
        dit_output = self.dit(projected_features)

        # Global average pooling
        pooled_features = dit_output.mean(dim=1)

        # Decode actions
        actions_flat = self.action_decoder(pooled_features)
        actions = actions_flat.view(-1, self.config.action_horizon, self.config.action_dim)

        # Create output BatchFeature
        output = BatchFeature({
            ACTION_KEY: actions,
        })

        return output

    def prepare_input(self, inputs: dict) -> BatchFeature:
        """Prepare inputs for action head."""
        action_inputs = BatchFeature({})

        if "action" in inputs:
            action_inputs["action"] = inputs["action"]

        return action_inputs


class GR00T_N1_5(nn.Module):
    """
    GR00T N1.5 model for SGLang runtime.
    Aligned with NVIDIA's GR00T architecture but adapted for SGLang compatibility.
    """
    config_class = GR00T_N1_5_Config
    supports_gradient_checkpointing = True

    def __init__(self, config: GR00T_N1_5_Config, quant_config=None, **kwargs):
        super().__init__()

        from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
        from sglang.srt.utils.hf_transformers_utils import get_config

        # Load backbone config from the local eagle2_hg_model directory
        eagle_model_path = "/data/work/sglang/eagle2_hg_model"
        if os.path.exists(eagle_model_path):
            # Load config from the local eagle2_hg_model directory
            backbone_config = get_config(eagle_model_path, trust_remote_code=True)
        else:
            # Fallback to the original logic
            eagle_path = config.backbone_cfg.get("eagle_path")

            # HACK: Override the custom eagle_path with a publicly available checkpoint
            if eagle_path == "NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops":
                eagle_path = "nvidia/Eagle2-2B"  # Use publicly available Eagle2-2B checkpoint

            if eagle_path:
                # Pass token if available for config loading
                config_kwargs = {"trust_remote_code": True}
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if token:
                    config_kwargs["token"] = token

                backbone_config = get_config(eagle_path, **config_kwargs)
            else:
                # Fallback: try to construct from backbone_cfg (though this may not work)
                from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLConfig
                backbone_config = Eagle2_5_VLConfig(**config.backbone_cfg)

        # Only pass the parameters that Eagle model accepts
        eagle_kwargs = {}
        if quant_config is not None:
            eagle_kwargs["quant_config"] = quant_config

        self.backbone = Eagle2_5_VLForConditionalGeneration(
            backbone_config, **eagle_kwargs
        )

        # Initialize action head with flow matching
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        # Override backbone_embedding_dim to match the actual backbone hidden_size
        action_head_cfg.backbone_embedding_dim = backbone_config.text_config.hidden_size
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.hidden_size = config.hidden_size
        self.model_dtype = config.model_dtype
        self.torch_dtype = config.torch_dtype

    # TODO: (Jonahcb) add type checking for ForwardBatch
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch
    ):
        """
        Forward pass for GR00T model - return action predictions as logits.
        """
        # Get VLM backbone embeddings
        hidden_states = self._get_vlm_embeddings(input_ids, positions, forward_batch)

        # Return action predictions as logits
        return self.compute_logits(hidden_states, forward_batch)

    def compute_logits(self, hidden_states: torch.Tensor, forward_batch):
        """Return action predictions as logits."""
        # Create backbone outputs in BatchFeature format
        backbone_outputs = BatchFeature({
            BACKBONE_FEATURE_KEY: hidden_states,
        })

        # Get action predictions
        action_outputs = self.action_head.get_action(backbone_outputs)

        # Return action predictions as "logits"
        # Shape: (batch_size, action_horizon, action_dim)
        return action_outputs[ACTION_KEY]

   # TODO: (Jonahcb) add type checking for ForwardBatch
    def _get_vlm_embeddings(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch
    ) -> torch.Tensor:
        """
        Extract embeddings from VLM backbone without final LM head processing.
        """
        from sglang.srt.managers.mm_utils import general_mm_embed_routine

        # Use the same multimodal embedding routine as backbone
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.backbone.language_model,
            multimodal_model=self.backbone,
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load model weights with proper mapping.
        First loads Eagle VLM backbone weights, then handles action head weights.
        """
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        # Remove backbone prefix from weight name before calling the backbone's load_weights method
        weights = [(name.replace("backbone.eagle_model.", ""), weight) for name, weight in weights]

        # Load Eagle VLM backbone weights
        self.backbone.load_weights(weights)

        # Handle action head weights
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            # Handle action head specific weights
            if "action_head" in name:
                # Remove the action_head prefix for parameter matching
                param_name = name.replace("action_head.", "")

                try:
                    param = params_dict[name]  # Use full name with action_head prefix
                except KeyError:
                    # Try without prefix if not found
                    try:
                        param = params_dict[param_name]
                    except KeyError:
                        continue  # Skip if parameter not found

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)



EntryClass = GR00T_N1_5


# Register with HuggingFace Transformers
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)

