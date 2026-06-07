"""Make the quark-mode DFlash FP8 export loadable by SGLang v0.5.12.

1. Rename weight-scale tensors  *.weight_quantizer.scale -> *.weight_scale
   (SGLang's QuarkW8A8Fp8 registers the param as `weight_scale`) and cast to fp32.
2. Restore top-level `rope_theta` / `rope_scaling` (transformers 4.57 moved
   rope_theta into `rope_parameters`, but SGLang reads getattr(config,"rope_theta")).
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

DIR = "/sgl-workspace/dflash-fp8"
ST = os.path.join(DIR, "model.safetensors")
CFG = os.path.join(DIR, "config.json")

SCALE_SUFFIX = ".weight_quantizer.scale"


def fix_safetensors():
    sd = load_file(ST)
    renamed = 0
    new_sd = {}
    for k, v in sd.items():
        # IMPORTANT: clone() every tensor. load_file() returns mmap-backed views
        # into model.safetensors; if we then save_file() over that same path, the
        # in-place overwrite corrupts any tensor still pointing at the file -> the
        # weights come back all-zero. clone() detaches each tensor into fresh
        # memory so the rewrite is safe. (The scales survived before only because
        # .to(float32) happened to copy them.)
        if k.endswith(SCALE_SUFFIX):
            nk = k[: -len(SCALE_SUFFIX)] + ".weight_scale"
            new_sd[nk] = v.to(torch.float32).clone()
            renamed += 1
        else:
            new_sd[k] = v.clone()

    # Belt-and-suspenders: write to a temp file and atomically replace, so we are
    # never reading and writing the same path simultaneously.
    tmp = ST + ".tmp"
    save_file(new_sd, tmp, metadata={"format": "pt"})
    os.replace(tmp, ST)
    print(f"renamed {renamed} scale tensors -> *.weight_scale (fp32)")


def fix_config():
    c = json.load(open(CFG))
    rp = c.get("rope_parameters") or {}
    if "rope_theta" not in c:
        c["rope_theta"] = rp.get("rope_theta", 10000000)
        print(f"set rope_theta = {c['rope_theta']}")
    if "rope_scaling" not in c:
        c["rope_scaling"] = None
        print("set rope_scaling = None")
    # Quark export sets dtype to the fp8 weight storage dtype; SGLang reads this
    # as the model's *compute* dtype and rejects fp8. The activation/compute dtype
    # must be bf16 (fp8 weights are handled by the quark scheme), so force it back.
    if c.get("dtype") not in ("bfloat16", "float16"):
        c["dtype"] = "bfloat16"
        print("set dtype = bfloat16")
    json.dump(c, open(CFG, "w"), indent=2)


if __name__ == "__main__":
    fix_safetensors()
    fix_config()
    # Verify
    from safetensors import safe_open

    f = safe_open(ST, "pt")
    sample = [
        k
        for k in sorted(f.keys())
        if k.startswith("layers.0.self_attn.q_proj")
        or k.startswith("layers.0.mlp.gate_proj")
        or k == "fc.weight"
    ]
    print("sample tensors after fix:")
    for k in sample:
        t = f.get_slice(k)
        print(f"   {k:50s} {str(t.get_dtype()):10s} {tuple(t.get_shape())}")

    # Guard against the all-zero corruption: refuse to "succeed" if any real
    # weight tensor came out all zeros after the rewrite.
    sd_check = load_file(ST)
    zeroed = [
        k
        for k, v in sd_check.items()
        if k.endswith(".weight")
        and v.numel() > 16
        and float(v.float().abs().max()) == 0.0
    ]
    if zeroed:
        raise SystemExit(
            f"FINALIZE FAILED: {len(zeroed)} weight tensors are all-zero "
            f"(e.g. {zeroed[:3]}). Checkpoint is corrupt; re-quantize and retry."
        )
    print(f"OK: all {sum(k.endswith('.weight') for k in sd_check)} weight tensors nonzero")
