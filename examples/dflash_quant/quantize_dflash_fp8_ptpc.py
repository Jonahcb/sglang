"""Quantize the DFlash draft to FP8 W8A8, PER-CHANNEL weight + PER-TOKEN dynamic act.

This is the sglang-native "fast" fp8 scheme:
  * weight: FP8 e4m3, per-output-channel, static  (scales derived from weights)
  * activation: FP8 e4m3, per-token, DYNAMIC       (scale computed at runtime)

Why this scheme (vs per-tensor static): per-token dynamic activations let sglang
fuse the activation quant straight into the preceding RMSNorm (aiter
`add_rmsnorm_quant`, one kernel) for the norm-fed linears (qkv_proj, gate_up_proj),
eliminating the standalone cast -- and the per-token amax is computed *inside* the
fused norm kernel, so there is no separate data_to_scale/initializeScale overhead.
The GEMM uses aiter's per-token x per-channel path. sglang's QuarkW8A8Fp8 marks this
as `per_token` (input_qscheme == "per_channel" and not static).

No calibration data is needed (weight scales from weights; activations dynamic).
Output: /sgl-workspace/dflash-fp8-ptpc  (quark format; finalize before loading).
"""

import json
import os
import shutil

import torch
from transformers import AutoModel

from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.quantization.config.config import (
    Config,
    QuantizationConfig,
    FP8E4M3PerChannelSpec,
)

SRC = os.environ.get("PTPC_DRAFT_SRC", "z-lab/Qwen3.5-35B-A3B-DFlash")
OUT = os.environ.get("PTPC_OUT", "/sgl-workspace/dflash-fp8-ptpc")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PROBE = [
    "fc.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.q_proj.weight_scale",
    "layers.0.mlp.down_proj.weight",
]


def _stat(t):
    f = t.detach().float()
    return f"{str(list(t.shape)):16s} {str(t.dtype):18s} absmax={float(f.abs().max()):.4e}"


def dump(model, stage):
    print(f"\n===== {stage} =====")
    sd = model.state_dict()
    for suf in _PROBE:
        m = [k for k in sd if k.endswith(suf)]
        print(f"  {suf:42s} {_stat(sd[m[0]]) if m else '<NO MATCH>'}")


def main():
    print(f"[1/4] Loading draft {SRC} (bf16) ...")
    model = AutoModel.from_pretrained(
        SRC, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()
    dump(model, "after load")

    print("[2/4] Building FP8 per-channel-weight / per-token-dynamic-act config ...")
    # Weight: per output channel (ch_axis=0 on [out, in]), static.
    w_spec = FP8E4M3PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
    # Activation: per token (ch_axis=0 on [tokens, hidden]), DYNAMIC -> no calibration.
    a_spec = FP8E4M3PerChannelSpec(ch_axis=0, is_dynamic=True).to_quantization_spec()
    linear_cfg = QuantizationConfig(weight=w_spec, input_tensors=a_spec)
    quant_config = Config(global_quant_config=linear_cfg, exclude=["fc"])

    print("[3/4] Quantizing (no calibration: weights per-channel, acts dynamic) ...")
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, None)
    model = quantizer.freeze(model)
    dump(model, "after freeze")

    print(f"[4/4] Exporting -> {OUT} ...")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    export_safetensors(model, OUT, custom_mode="quark")

    cfg = json.load(open(os.path.join(OUT, "config.json")))
    gq = cfg.get("quantization_config", {}).get("global_quant_config", {})
    print("\nexported weight qscheme :", gq.get("weight", {}).get("qscheme"),
          " is_dynamic:", gq.get("weight", {}).get("is_dynamic"))
    print("exported input  qscheme :", gq.get("input_tensors", {}).get("qscheme"),
          " is_dynamic:", gq.get("input_tensors", {}).get("is_dynamic"))


if __name__ == "__main__":
    main()
