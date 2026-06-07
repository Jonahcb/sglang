"""Quantize the DFlash draft model to FP8 (W8A8, dynamic activations) with AMD Quark.

Weights: per-tensor FP8 e4m3 static.  Activations: per-tensor FP8 dynamic
(=> no calibration data needed).  `fc` and all norms stay bf16.
Exports a quark-format checkpoint that SGLang's QuarkConfig can load.
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
    FP8E4M3PerTensorSpec,
)

SRC = "z-lab/Qwen3.5-35B-A3B-DFlash"
OUT = "/sgl-workspace/dflash-fp8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Representative tensors to track across every stage of the pipeline. We look
# for any state-dict key that ends with these suffixes (Quark may wrap/rename
# modules, so we match by suffix rather than exact name).
_PROBE_SUFFIXES = [
    "fc.weight",  # excluded from quant -> must stay bf16 and nonzero
    "norm.weight",  # final norm
    "hidden_norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.self_attn.q_proj.weight",  # becomes fp8 after quant
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.down_proj.weight",
]


def _tstat(t: torch.Tensor) -> str:
    f = t.detach().float()
    n = f.numel()
    nz = int((f != 0).sum())
    return (
        f"{str(list(t.shape)):16s} {str(t.dtype):18s} "
        f"nonzero={nz}/{n} absmax={float(f.abs().max()):.4e}"
    )


def dump_stats(model, stage: str) -> None:
    """Print weight statistics for the in-memory model at a pipeline stage."""
    print(f"\n========== WEIGHT STATS @ {stage} ==========")
    sd = model.state_dict()
    print(f"  state_dict tensors: {len(sd)}")
    for suf in _PROBE_SUFFIXES:
        matches = [k for k in sd if k.endswith(suf) or suf in k]
        if not matches:
            print(f"  {suf:48s} <NO MATCH>")
            continue
        k = matches[0]
        print(f"  {k:48s} {_tstat(sd[k])}")
    # Global all-zero scan over real weight tensors (ignore tiny scale scalars).
    allz = [
        k
        for k, v in sd.items()
        if v.numel() > 16 and float(v.detach().float().abs().max()) == 0
    ]
    print(f"  ALL-ZERO weight tensors (numel>16): {len(allz)}")
    for k in allz[:12]:
        print(f"      zero: {k}")


def dump_stats_from_disk(out_dir: str, stage: str) -> None:
    """Reload the exported safetensors from disk and print the same stats."""
    import glob

    from safetensors import safe_open

    print(f"\n========== WEIGHT STATS @ {stage} (reloaded from disk) ==========")
    st_files = sorted(glob.glob(os.path.join(out_dir, "*.safetensors")))
    if not st_files:
        print(f"  <no .safetensors found in {out_dir}>")
        return
    # Merge keys across shards.
    handles = [safe_open(p, framework="pt", device="cpu") for p in st_files]
    keys = {k: h for h in handles for k in h.keys()}
    print(f"  on-disk tensors: {len(keys)} across {len(st_files)} file(s)")
    for suf in _PROBE_SUFFIXES:
        matches = [k for k in keys if k.endswith(suf) or suf in k]
        if not matches:
            print(f"  {suf:48s} <NO MATCH>")
            continue
        k = matches[0]
        print(f"  {k:48s} {_tstat(keys[k].get_tensor(k))}")
    allz = [
        k
        for k, h in keys.items()
        if h.get_tensor(k).numel() > 16
        and float(h.get_tensor(k).float().abs().max()) == 0
    ]
    print(f"  ALL-ZERO weight tensors (numel>16): {len(allz)}")
    for k in allz[:12]:
        print(f"      zero: {k}")


def main():
    print(f"[1/5] Loading {SRC} (trust_remote_code) on {DEVICE} ...")
    model = AutoModel.from_pretrained(
        SRC, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()
    dump_stats(model, "after load (source)")

    print("[2/5] Building FP8 W8A8 (static weight / dynamic act) config ...")
    w_spec = FP8E4M3PerTensorSpec(
        observer_method="min_max", is_dynamic=False
    ).to_quantization_spec()
    a_spec = FP8E4M3PerTensorSpec(
        observer_method="min_max", is_dynamic=True
    ).to_quantization_spec()
    linear_cfg = QuantizationConfig(weight=w_spec, input_tensors=a_spec)

    quant_config = Config(
        global_quant_config=linear_cfg,
        exclude=["fc"],  # context-feature projector stays bf16 (plain nn.Linear in SGLang)
    )

    print("[3/5] Quantizing (no calibration: weights static, acts dynamic) ...")
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, None)
    dump_stats(model, "after quantize_model")
    model = quantizer.freeze(model)
    dump_stats(model, "after freeze")

    print(f"[4/5] Exporting quark-format checkpoint -> {OUT} ...")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    export_safetensors(model, OUT, custom_mode="quark")
    dump_stats_from_disk(OUT, "after export_safetensors")

    print("[5/5] Exported files:")
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"   {f:40s} {os.path.getsize(p)/1e6:8.2f} MB")

    cfg_path = os.path.join(OUT, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        print("\nexported config.json top-level keys:", sorted(cfg.keys()))
        if "quantization_config" in cfg:
            print(
                "quantization_config keys:",
                sorted(cfg["quantization_config"].keys()),
            )


if __name__ == "__main__":
    main()
