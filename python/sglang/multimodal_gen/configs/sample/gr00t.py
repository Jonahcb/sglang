# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.base import SamplingParams


@dataclass
class GR00TSamplingParams(SamplingParams):
    """Sampling parameters specific to GR00T robot control."""

    # Override defaults for robot control
    num_frames: int = 1  # Robot control doesn't use frames
    num_inference_steps: int = 4  # GR00T uses 4 diffusion steps
    height: int = 1  # Dummy value for robot control
    width: int = 1   # Dummy value for robot control

    # Robot-specific parameters
    embodiment_id: str = "new_embodiment"
