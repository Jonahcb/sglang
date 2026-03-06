from typing import Any, Dict, Optional, Tuple, List

import mlx.nn as nn
import mlx.core as mx

from sglang.srt.utils import LayerFn, add_prefix

def make_layers_non_pp(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str = "",
) -> List[nn.Module]:
    # no offloader as we are working in Unified Memory.
    #from sglang.srt.utils.offloader import get_offloader

    layers = [
                layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
                for idx in range(num_hidden_layers)
        ]
        
    
    return layers

# We probably don't want to go down the rabbit hole of rewriting all the weight loading logic, we can just load the weights using the normal SGLang functions and then convert them to mlx arrays. However, this probably won't work because we defined our model classes in terms
# of MLX nn.Module, not PyTorch nn.Module
def default_mlx_weight_loader(param: mx.array, loaded_weight: mx.array) -> None:
    """Default weight loader."""
    try:
        if param.size == 1 and loaded_weight.size == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            clean_weight = loaded_weight.reshape(param.shape)
        else:
            assert param.size == loaded_weight.size, (
                f"Attempted to load weight ({loaded_weight.size}) "
                f"into parameter ({param.size})"
            )

            clean_weight = loaded_weight
        
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise