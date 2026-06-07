"""Make the quark-mode STATIC DFlash FP8 export loadable by SGLang.

Same as finalize_dflash_fp8.py (for the dynamic checkpoint), plus the static
activation-scale rename:

1. *.weight_quantizer.scale -> *.weight_scale   (SGLang QuarkW8A8Fp8 param name)
2. *.input_quantizer.scale  -> *.input_scale    (NEW: static activation scale;
   SGLang reads layer.input_scale when is_static_input_scheme)
   all cast to fp32 and cloned (avoid mmap-overwrite corruption).
3. Restore top-level rope_theta / rope_scaling and force dtype=bfloat16.

Targets /sgl-workspace/dflash-fp8-static (does NOT touch the dynamic checkpoint).
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

DIR = os.environ.get("STATIC_OUT", "/sgl-workspace/dflash-fp8-static")
ST = os.path.join(DIR, "model.safetensors")
CFG = os.path.join(DIR, "config.json")

RENAMES = {
    ".weight_quantizer.scale": ".weight_scale",
    ".input_quantizer.scale": ".input_scale",
}


def fix_safetensors():
    sd = load_file(ST)
    new_sd = {}
    counts = {v: 0 for v in RENAMES.values()}
    for k, v in sd.items():
        nk = k
        for suf, repl in RENAMES.items():
            if k.endswith(suf):
                nk = k[: -len(suf)] + repl
                counts[repl] += 1
                break
        # clone() to detach from the mmap'd source before we overwrite the file.
        new_sd[nk] = (v.to(torch.float32) if nk != k else v).clone()

    tmp = ST + ".tmp"
    save_file(new_sd, tmp, metadata={"format": "pt"})
    os.replace(tmp, ST)
    for repl, n in counts.items():
        print(f"renamed {n} tensors -> *{repl} (fp32)")
    return counts


def fix_config():
    c = json.load(open(CFG))
    rp = c.get("rope_parameters") or {}
    if "rope_theta" not in c:
        c["rope_theta"] = rp.get("rope_theta", 10000000)
        print(f"set rope_theta = {c['rope_theta']}")
    if "rope_scaling" not in c:
        c["rope_scaling"] = None
        print("set rope_scaling = None")
    if c.get("dtype") not in ("bfloat16", "float16"):
        c["dtype"] = "bfloat16"
        print("set dtype = bfloat16")
    json.dump(c, open(CFG, "w"), indent=2)


if __name__ == "__main__":
    counts = fix_safetensors()
    fix_config()

    # Verify: weights nonzero, and we have the expected static scales.
    sd_check = load_file(ST)
    n_w = sum(k.endswith(".weight") for k in sd_check)
    n_ws = sum(k.endswith(".weight_scale") for k in sd_check)
    n_is = sum(k.endswith(".input_scale") for k in sd_check)
    zeroed = [
        k
        for k, v in sd_check.items()
        if k.endswith(".weight") and v.numel() > 16 and float(v.float().abs().max()) == 0.0
    ]
    print(f"\nweight tensors: {n_w}   weight_scale: {n_ws}   input_scale: {n_is}")
    sample = sorted(k for k in sd_check if k.startswith("layers.0.self_attn.q_proj"))
    for k in sample:
        v = sd_check[k]
        print(f"   {k:48s} {str(v.dtype):14s} {tuple(v.shape)}  absmax={float(v.float().abs().max()):.4e}")
    if zeroed:
        raise SystemExit(f"FINALIZE FAILED: {len(zeroed)} all-zero weight tensors {zeroed[:3]}")
    if n_is == 0:
        raise SystemExit("FINALIZE FAILED: no input_scale params -> static activation scales missing.")
    print(f"OK: {n_w} weights nonzero, {n_is} static input_scale params present.")
