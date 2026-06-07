"""Quantize the DFlash draft to FP8 W8A8 with **STATIC** activation scales (AMD Quark).

This is the upstream-correct way to produce a static-activation fp8 draft: run the
HF *reference* models offline and let Quark's ModelQuantizer observe the draft's
real activations during calibration, then export a quark checkpoint. SGLang already
loads static `input_scale` (QuarkW8A8Fp8.is_static_input_scheme), so no engine change
is needed -- this script only produces the checkpoint.

Difference vs quantize_dflash_fp8.py: activations are STATIC (is_dynamic=False), so
they need calibration data. At inference this removes the per-forward dynamic-quant
overhead (the amax reduction: aiter data_to_scale + initializeScale), leaving only
the irreducible fp8 cast.

Calibration faithfully mirrors DFLASH's data flow (reconstructed from sglang's
dflash_worker, since the reference repo ships only the model):
  1. run the TARGET on each prompt with output_hidden_states
  2. gather the target hidden states "after" layers target_layer_ids -> target_hidden
  3. build the masked draft block (bonus token in slot 0, mask_token elsewhere),
     embed it with the target embedding -> noise_embedding
  4. run the draft forward(position_ids, attention_mask, noise_embedding,
     target_hidden); Quark observes the activations.

Output: /sgl-workspace/dflash-fp8-static  (quark format, loadable by SGLang).
"""

import json
import os
import shutil

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.quantization.config.config import (
    Config,
    QuantizationConfig,
    FP8E4M3PerTensorSpec,
)

DRAFT_SRC = os.environ.get("STATIC_DRAFT_SRC", "z-lab/Qwen3.5-35B-A3B-DFlash")
TARGET_SRC = os.environ.get("STATIC_TARGET_SRC", "Qwen/Qwen3.5-35B-A3B")
OUT = os.environ.get("STATIC_OUT", "/sgl-workspace/dflash-fp8-static")
N_CALIB = int(os.environ.get("STATIC_N_CALIB", "16"))
MAX_CTX = int(os.environ.get("STATIC_MAX_CTX", "256"))  # truncate calib prompts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reuse the shared, fixed prompt set so calibration matches what we benchmark.
from accept_len_common import PROMPTS  # noqa: E402

CALIB_PROMPTS = PROMPTS[:N_CALIB]

_PROBE_SUFFIXES = [
    "fc.weight",
    "norm.weight",
    "hidden_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.q_proj.input_scale",  # NEW: static activation scale
    "layers.0.self_attn.o_proj.input_scale",
    "layers.0.mlp.down_proj.input_scale",
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


def dump_stats_from_disk(out_dir: str, stage: str) -> None:
    import glob

    from safetensors import safe_open

    print(f"\n========== WEIGHT STATS @ {stage} (reloaded from disk) ==========")
    st_files = sorted(glob.glob(os.path.join(out_dir, "*.safetensors")))
    if not st_files:
        print(f"  <no .safetensors found in {out_dir}>")
        return
    handles = [safe_open(p, framework="pt", device="cpu") for p in st_files]
    keys = {k: h for h in handles for k in h.keys()}
    n_input_scale = len([k for k in keys if k.endswith("input_scale")])
    n_weight_scale = len([k for k in keys if k.endswith("weight_scale")])
    print(f"  on-disk tensors: {len(keys)} across {len(st_files)} file(s)")
    print(f"  weight_scale params: {n_weight_scale}   input_scale params: {n_input_scale}")
    for suf in _PROBE_SUFFIXES:
        matches = [k for k in keys if k.endswith(suf) or suf in k]
        if not matches:
            print(f"  {suf:48s} <NO MATCH>")
            continue
        k = matches[0]
        print(f"  {k:48s} {_tstat(keys[k].get_tensor(k))}")


@torch.no_grad()
def build_calibration(draft_cfg):
    """Run the HF target to produce real DFLASH draft inputs for Quark.

    Returns a list of dicts; Quark's calibration loop calls draft(**dict).
    """
    dflash_cfg = getattr(draft_cfg, "dflash_config", {}) or {}
    target_layer_ids = dflash_cfg.get("target_layer_ids")
    mask_token_id = dflash_cfg.get("mask_token_id")
    block_size = int(getattr(draft_cfg, "block_size", 16))
    if not target_layer_ids or mask_token_id is None:
        raise ValueError(
            f"draft config missing dflash_config target_layer_ids/mask_token_id: {dflash_cfg}"
        )
    # HF output_hidden_states: index 0 = embeddings, index i+1 = output AFTER layer i.
    # DFlash uses hidden states "after" each selected (0-based) target layer.
    capture_idx = [int(i) + 1 for i in target_layer_ids]
    print(f"[calib] target_layer_ids={target_layer_ids} -> hidden_states idx={capture_idx}")
    print(f"[calib] mask_token_id={mask_token_id} block_size={block_size}")

    print(f"[calib] loading TARGET {TARGET_SRC} (bf16) ...")
    tok = AutoTokenizer.from_pretrained(TARGET_SRC, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_SRC,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    target.eval()
    embed = target.get_input_embeddings()

    samples = []
    for i, prompt in enumerate(CALIB_PROMPTS):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX)
        input_ids = ids["input_ids"].to(DEVICE)
        ctx = int(input_ids.shape[1])

        out = target(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len = num_layers + 1
        target_hidden = torch.cat([hs[j] for j in capture_idx], dim=-1)  # [1, ctx, 5*H]

        # Masked draft block: slot 0 = last committed (bonus) token, rest = mask token.
        block_ids = torch.full((1, block_size), int(mask_token_id), dtype=torch.long, device=DEVICE)
        block_ids[0, 0] = input_ids[0, -1]
        noise_embedding = embed(block_ids)  # [1, block, H]

        # Rotary spans the full KEY sequence (ctx context keys + block keys): the
        # reference apply_rotary_pos_emb applies cos/sin of length ctx+block to k,
        # and q takes the last `block` slots (cos[..., -q_len:, :]). So position_ids
        # = absolute positions 0..ctx-1 for the context then ctx..ctx+block-1 for
        # the block.
        position_ids = torch.arange(0, ctx + block_size, device=DEVICE).unsqueeze(0)
        # Block (q=block_size) attends to [ctx context keys + block non-causal keys].
        # Full visibility (additive zeros) matches the draft's cat([k_ctx, k_noise]).
        attention_mask = torch.zeros(
            (1, 1, block_size, ctx + block_size), dtype=torch.bfloat16, device=DEVICE
        )

        samples.append(
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "noise_embedding": noise_embedding.to(torch.bfloat16),
                "target_hidden": target_hidden.to(torch.bfloat16),
            }
        )
        print(f"[calib] sample {i:2d}: ctx={ctx:4d} target_hidden={list(target_hidden.shape)}")

    # Free the 70GB target before quantizing the (tiny) draft.
    del target, embed
    torch.cuda.empty_cache()
    return samples


def main():
    print(f"[1/6] Loading draft {DRAFT_SRC} (eager attn, bf16) ...")
    draft = AutoModel.from_pretrained(
        DRAFT_SRC,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # deterministic additive-mask path for calibration
    ).to(DEVICE)
    draft.eval()
    dump_stats(draft, "after load (draft)")

    print("[2/6] Building calibration inputs from the target ...")
    calib = build_calibration(draft.config)

    print("[3/6] Building FP8 W8A8 config with STATIC activations ...")
    w_spec = FP8E4M3PerTensorSpec(
        observer_method="min_max", is_dynamic=False
    ).to_quantization_spec()
    a_spec = FP8E4M3PerTensorSpec(
        observer_method="min_max", is_dynamic=False  # <-- STATIC (was True/dynamic)
    ).to_quantization_spec()
    linear_cfg = QuantizationConfig(weight=w_spec, input_tensors=a_spec)
    quant_config = Config(global_quant_config=linear_cfg, exclude=["fc"])

    print(f"[4/6] Calibrating + quantizing over {len(calib)} samples ...")
    quantizer = ModelQuantizer(quant_config)
    draft = quantizer.quantize_model(draft, calib)
    dump_stats(draft, "after quantize_model")
    draft = quantizer.freeze(draft)
    dump_stats(draft, "after freeze")

    print(f"[5/6] Exporting quark checkpoint -> {OUT} ...")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    export_safetensors(draft, OUT, custom_mode="quark")
    dump_stats_from_disk(OUT, "after export_safetensors")

    print("[6/6] Exported files:")
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"   {f:40s} {os.path.getsize(p)/1e6:8.2f} MB")
    cfg_path = os.path.join(OUT, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        qc = cfg.get("quantization_config", {})
        gq = qc.get("global_quant_config", {})
        it = gq.get("input_tensors", {})
        print(f"\nactivation is_dynamic in exported config: {it.get('is_dynamic')}  (expect False)")


if __name__ == "__main__":
    main()
