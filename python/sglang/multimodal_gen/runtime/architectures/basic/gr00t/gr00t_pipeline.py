# SPDX-License-Identifier: Apache-2.0
"""
GR00T VLA diffusion pipeline implementation.

This module contains an implementation of the GR00T video diffusion pipeline
using the modular pipeline architecture.
"""


from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class GR00TPipeline(ComposedPipelineBase):

    pipeline_name = "GR00TPipeline"

    _required_config_modules = [
        "transformer",
        "scheduler",
    ]

    # Optional modules for GR00T
    _optional_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # TODO: how does this know what inputs must be present? Do we overload it?
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        # TODO: Do we overload functions that this stage calls?
        # For GR00T, text encoding is optional since we work with action tensors
        text_encoders = []
        tokenizers = []

        if "text_encoder" in self.modules:
            text_encoders.append(self.get_module("text_encoder"))
        if "text_encoder_2" in self.modules and self.modules["text_encoder_2"] is not None:
            text_encoders.append(self.get_module("text_encoder_2"))

        if "tokenizer" in self.modules:
            tokenizers.append(self.get_module("tokenizer"))
        if "tokenizer_2" in self.modules and self.modules["tokenizer_2"] is not None:
            tokenizers.append(self.get_module("tokenizer_2"))

        # Only add text encoding stage if we have both encoders and tokenizers
        if text_encoders and tokenizers:
            self.add_stage(
                stage_name="prompt_encoding_stage_primary",
                stage=TextEncodingStage(
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                ),
            )
        else:
            # For GR00T, we can skip text encoding if no models are available
            # The action tensor processing will happen in the denoising stage
            pass
        # TODO: What is conditioning in our case?
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )
        # TODO: Our denoising stage includes decoding in each iteration so do we need a decoding stage?
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # GR00T doesn't use VAE decoding since it outputs actions, not images
        # Skip the decoding stage for robot control


EntryClass = GR00TPipeline
