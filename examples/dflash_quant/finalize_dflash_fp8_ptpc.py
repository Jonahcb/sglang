"""Make the per-channel/per-token quark export loadable by SGLang.

Renames *.weight_quantizer.scale -> *.weight_scale (per-channel vector, fp32) and
fixes config (rope_theta/rope_scaling, dtype=bfloat16). Activations are DYNAMIC, so
there are no input_scale params to rename. Targets /sgl-workspace/dflash-fp8-ptpc.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

DIR = os.environ.get("PTPC_OUT", "/sgl-workspace/dflash-fp8-ptpc")
ST = os.path.join(DIR, "model.safetensors")
CFG = os.path.join(DIR, "config.json")
SUF = ".weight_quantizer.scale"


def main():
    sd = load_file(ST)
    new_sd, n = {}, 0
    for k, v in sd.items():
        if k.endswith(SUF):
            new_sd[k[: -len(SUF)] + ".weight_scale"] = v.to(torch.float32).clone()
            n += 1
        else:
            new_sd[k] = v.clone()
    tmp = ST + ".tmp"
    save_file(new_sd, tmp, metadata={"format": "pt"})
    os.replace(tmp, ST)
    print(f"renamed {n} -> *.weight_scale")

    c = json.load(open(CFG))
    rp = c.get("rope_parameters") or {}
    c.setdefault("rope_theta", rp.get("rope_theta", 10000000))
    c.setdefault("rope_scaling", None)
    if c.get("dtype") not in ("bfloat16", "float16"):
        c["dtype"] = "bfloat16"
    json.dump(c, open(CFG, "w"), indent=2)

    chk = load_file(ST)
    n_w = sum(k.endswith(".weight") for k in chk)
    n_ws = sum(k.endswith(".weight_scale") for k in chk)
    zeroed = [k for k, v in chk.items() if k.endswith(".weight") and v.numel() > 16
              and float(v.float().abs().max()) == 0.0]
    # show a per-channel weight_scale shape (should be a vector, not scalar)
    samp = [k for k in chk if k.endswith("self_attn.q_proj.weight_scale")]
    if samp:
        print(f"  {samp[0]} shape={tuple(chk[samp[0]].shape)} (per-channel => vector)")
    if zeroed:
        raise SystemExit(f"FINALIZE FAILED: {len(zeroed)} all-zero weights {zeroed[:3]}")
    print(f"OK: {n_w} weights nonzero, {n_ws} weight_scale params (no input_scale = dynamic act).")


if __name__ == "__main__":
    main()
