# SPDX-License-Identifier: Apache-2.0
# Adapter from https://github.com/vllm-project/vllm/blob/9fa5b25a238c08fae8acf507e5dbc923f5b2e5cb/vllm/triton_utils/importing.py

import os
import types
from importlib.util import find_spec
import logging


logger = logging.getLogger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
)
if HAS_TRITON:
    try:
        from triton.backends import backends

        # It's generally expected that x.driver exists and has
        # an is_active method.
        # The `x.driver and` check adds a small layer of safety.
        active_drivers = [
            x.driver for x in backends.values() if x.driver and x.driver.is_active()
        ]

        if len(active_drivers) != 1:
            # Strict check for non-distributed environments
            logger.info(
                "Triton is installed but %d active driver(s) found "
                "(expected 1). Disabling Triton to prevent runtime errors.",
                len(active_drivers),
            )
            HAS_TRITON = False
    except ImportError:
        # This can occur if Triton is partially installed or triton.backends
        # is missing.
        logger.warning(
            "Triton is installed, but `triton.backends` could not be imported. "
            "Disabling Triton."
        )
        HAS_TRITON = False
    except Exception as e:
        # Catch any other unexpected errors during the check.
        logger.warning(
            "An unexpected error occurred while checking Triton active drivers:"
            " %s. Disabling Triton.",
            e,
        )
        HAS_TRITON = False

if not HAS_TRITON:
    logger.info(
        "Triton not installed or not compatible; certain GPU-related"
        " functions will not be available."
    )


class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.4.0"
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.language = TritonLanguagePlaceholder()

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = lambda x: x
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.uint32 = None
        self.uint64 = None
        self.tensor = None
        self.exp = None
        self.log = None
        self.log2 = None
        self.math = self.math = types.ModuleType("triton.language.math")
        self.math.exp2 = None