# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/9fa5b25a238c08fae8acf507e5dbc923f5b2e5cb/vllm/triton_utils/__init__.py
from typing import TYPE_CHECKING

from sglang.srt.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    from triton import testing
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    testing = triton.testing

__all__ = ["HAS_TRITON", "triton", "tl", "tldevice", "testing"]
