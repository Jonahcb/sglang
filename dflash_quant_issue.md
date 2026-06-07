# [AMD][Quantization][DFLASH] Add all popular quantization schemes to the DFlash draft model (FP8 PTPC/block/per-tensor / MXFP4 / MXFP8 / AWQ / GPTQ / INT8)

> Goal: quantize the 5 Linears (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `lm_head`) with every popular scheme, **maximally fused**.

## Current DFlash codepath (as-is, bf16) — what we are quantizing

DFlash runs the draft and target (verify) models **in the same process/thread** (`DFlashWorker`, `dflash_worker.py:113`). The draft has its own `TpModelWorker` + KV cache + attention backend (`:206`), but shares the target's KV allocator and the target's embedding/lm_head. There is no draft lm_head — the draft returns hidden states only (`dflash.py:565-568`).

The five GEMMs that quantization targets are marked **★**. Four are inside the draft layers (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`); the fifth (`lm_head`★) is the **target's** head, applied twice (draft-side greedy sampling + target-side verify).

```
 PREFILL / extend                                       forward_batch_generation  (dflash_worker.py:1190)
   ├─ target.forward_batch_generation(capture_hidden_mode=FULL)   # target prefill, bf16  (:1192)
   │     → target_hidden = per-token ctx features ;  next_token_ids = first bonus token
   └─ seed draft KV  ──────────────────────────────────┐ (calls KV-SEED below)            (:1239)
                                                        │
 ════════════════════════════════ DECODE / verify loop ════════════════════════ (dflash_worker.py:1249)
                                                        │
 _prepare_for_speculative_decoding                      │                          (dflash_worker.py:586)
   ├─ refresh draft KV (committed tokens) ──────────────┤ (calls KV-SEED below)            (:611)
   ├─ block_ids = [bonus_token, MASK, MASK, …]   (block_size cols)                         (:634)
   │    input_embeds = target.embed(block_ids)          # target embedding
   ├─ alloc draft block KV ; build ForwardBatch(TARGET_VERIFY)                             (:703)
   │     ▼
   │   DRAFT MODEL forward   draft_model_runner.forward(forward_batch)   (dflash_worker.py:722)
   │   ───────────────────   DFlashDraftModel.forward                          (dflash.py:517)
   │     input_embeds (from target embed of [bonus, MASK…])
   │       │   residual = None
   │       ▼
   │     for layer in layers (8×):                       DFlashDecoderLayer.forward  (dflash.py:390)
   │       ├─ input_layernorm (RMSNorm, fused +residual)             (dflash.py:406-410)
   │       │     ▼
   │       │   DFlashAttention.forward                                (dflash.py:264)
   │       │     ├─ qkv_proj★  (QKVParallelLinear)                    (dflash.py:271)   ◀── GEMM 1
   │       │     ├─ split q,k,v ; q_norm/k_norm (per-head RMSNorm)    (dflash.py:273-274)
   │       │     ├─ rotary_emb (RoPE)                                 (dflash.py:278)
   │       │     ├─ attn  (RadixAttention, ENCODER_ONLY / non-causal) (dflash.py:281)
   │       │     └─ o_proj★  (RowParallelLinear)                      (dflash.py:283)   ◀── GEMM 2
   │       ├─ post_attention_layernorm (RMSNorm, fused +residual)     (dflash.py:420)
   │       │     ▼
   │       │   DFlashMLP.forward                                      (dflash.py:355)
   │       │     ├─ gate_up_proj★  (MergedColumnParallelLinear)       (dflash.py:358)   ◀── GEMM 3
   │       │     ├─ act_fn = SiluAndMul()                             (dflash.py:360)
   │       │     └─ down_proj★  (RowParallelLinear)                   (dflash.py:362)   ◀── GEMM 4
   │       ▼
   │     self.norm (final RMSNorm, fused +residual)                  (dflash.py:558-562)
   │       ▼
   │     → draft_hidden  [bs, block_size, hidden]   (next_token_logits=None)  (dflash.py:565)
   │
   ├─ greedy sample  _greedy_sample_from_vocab_parallel_head(draft_hidden[:,1:], lm_head★)  (:766)
   │     → draft_next tokens   (raw matmul vs TARGET lm_head★ shard, then argmax)  (:819/835)
   └─ DFlashVerifyInput{ draft_token, positions } ; prepare_for_verify                     (:744)
         ▼
 VERIFY   target.forward_batch_generation(is_verify=True)                                  (:1279)
   ├─ target forward over drafted block  (uses lm_head★)  → logits_output
   └─ verify_input.verify()                                                                (:1292)
         → new_bonus_tokens, commit_lens, next_target_hidden
         ▼
   push committed verify tokens to draft KV ─────────────┐ (calls KV-SEED below)           (:1314)
         ▼                                               │
   batch.forward_mode = DECODE   ── loop ───────────────────────────────── (back to top of loop)
                                                         │
 ─────────────────────────────────────────────────────  ┘
 KV-SEED   _append_target_hidden_to_draft_kv                          (dflash_worker.py:940)
   target_hidden  [sum(ctx), K*hidden]
     └─ project_target_hidden()                                       (dflash.py:480)
          = hidden_norm( fc(target_hidden) ) → ctx_hidden  # fc is plain nn.Linear, NEVER quantized
     └─ for each draft layer (sequential :1079  OR fused-triton :1101, CUDA-only :293):
          k, v = attn.kv_proj_only(ctx_hidden)   # slices qkv_proj★ weight, K/V only  (dflash.py:287)
          k = apply_k_norm(k) ; k = apply_k_rope(positions, k)        (dflash.py:312-321)
          token_to_kv_pool.set_kv_buffer(layer, ctx_cache_loc, k, v)  (:1092)
```

**Key consequences for quantization** (expanded in *Common facts* below):
- The draft forward is **hand-written** — it calls `input_layernorm`/`post_attention_layernorm`/`SiluAndMul` directly and does **not** route through `communicator.prepare_attn`, so every prologue/epilogue fusion must be wired in by hand at the call sites above (`dflash.py:264-285` / `355-364` / `406-423`).
- `qkv_proj`★ is consumed **twice**: once in the draft forward (full QKV, GEMM 1) and once in `kv_proj_only` (K/V-only, path C). The KV-only fast path slices the fused weight and only works for unquantized weights (`dflash.py:296-305`) — quantized qkv falls back to full-QKV-discard-Q.
- `lm_head`★ is the **target's** vocab-parallel head, never in the draft checkpoint; it is quantized by the target's `--quantization`, and the draft-side greedy sampler does a raw `matmul` against its weight shard (`dflash_worker.py:819/835`).

## Common facts (apply to all schemes)

- 4 body Linears already receive `quant_config` (`dflash.py:215-347`); `fc` is plain `nn.Linear`, never quantized.
- **LM head is NOT in the draft** (`dflash.py:429`, draft returns `next_token_logits=None`). It is the **target model's** head, **shared**: applied to draft hidden → token proposals AND to target hidden → verify logits. ⇒ quantized via the **target** quant config (`--quantization`), NOT `--speculative-draft-model-quantization`. ⇒ **most accuracy-critical layer** for acceptance rate — quantize conservatively, benchmark acceptance hard.
- `--speculative-draft-model-quantization <scheme>`; draft inherits `--dtype bf16` (no separate draft dtype).
- DFlash uses a **custom forward** — calls norms directly, does **NOT** route through `communicator.prepare_attn` (`dflash.py:407-423`). ⇒ **every fusion must be wired explicitly** into `DFlashDecoderLayer.forward` / `DFlashMLP.forward`.
- `load_weights` (`dflash.py:595-624`) already routes stacked `weight` + scale/zero params via the generic qwen2/llama pattern — no load-path change needed.
- KV-slice fast path auto-disables for any non-`UnquantizedLinearMethod` qkv (`dflash_utils.py:400`) → full-QKV/discard-Q (correct, slightly slower).
- Block geometry clean: all N,K divide by 128 (hidden 2048, inter 6144, qkv-N 5120, o-K 4096, down-K 6144) — no padding, satisfies bpreshuffle k≥512 / m≥16.
- `SGLANG_USE_AITER=1` is the master lever (fast GEMMs + fused kernels on ROCm).
- All fusions behind one env flag (e.g. `DFLASH_FUSE_QUANT`) with per-site bf16-parity check. Strip `_dflash_dbg_*` before commit.

## Fusion-site names (used in every diagram)

- `qkv_proj-prologue` — input_layernorm (+residual) + activation quant → `qkv_proj`
- `o_proj-epilogue` — attention-output activation quant → `o_proj` (no elementwise producer; bare quant, unfused)
- `gate_up_proj-prologue` — post_attention_layernorm (+residual) + activation quant → `gate_up_proj`
- `down_proj-prologue` — SiLU·mul + activation quant → `down_proj`
- `lm_head-prologue` — final `self.norm` + activation quant → shared target LM head (target-side, optional)

**Shared enabler (blocks all prologue fusions):** the linear `apply()` quantizes activations internally. Add a **pre-quantized tuple-input path** `(q_tensor, scale)` so a fused producer kernel can feed the GEMM directly, skipping the internal quant. Reconcile scale layout/transpose with the selected GEMM variant.

---

## LM head — shared, target-side

- The LM head belongs to the **target** model.


## FP8 PTPC — per-channel weight, per-token activation

- **Reuse (no new kernels):**
  - norm prologues — `rmsnorm_quant` / `add_rmsnorm_quant` (`aiter.ops.rmsnorm`, group_size=0), already wrapped as `_fused_rmsnorm_fp8_per_token_quant` (`sglang/srt/layers/communicator.py:107`).
  - `down_proj-prologue` — `sglang_per_token_group_quant_fp8(..., fuse_silu_and_mul=True)` (`sglang/srt/layers/quantization/fp8_kernel.py:498`).
  - `o_proj-epilogue` (bare) — `sglang_per_token_quant_fp8` (`sglang/srt/layers/quantization/fp8_kernel.py:656`).
  - GEMM — `apply_fp8_linear` (`fp8_utils.py:1458`); on AMD the `_use_aiter` per-token/per-channel branch dispatches to **`gemm_a8w8_bpreshuffle`** (`from aiter`, `fp8_utils.py:~1585`). Non-aiter fallback is `torch._scaled_mm` rowwise; CUDA path is `fp8_scaled_mm`/`triton_scaled_mm`.
- **Code:** add the tuple-input branch in `apply_fp8_linear`; wire the 3 prologues into `DFlashDecoderLayer.forward`/`DFlashMLP.forward`.
- **GEMM kernel (AMD/aiter):** `gemm_a8w8_bpreshuffle` (`from aiter`); fallback `torch._scaled_mm` rowwise / `fp8_scaled_mm` (cutlass).

```
 input_LN ─[qkv_proj-prologue: fused RMSNorm+per-tok fp8 quant (CK)]→(xq,xs)→ qkv_proj GEMM ─┐
                                                                                             ▼
                                                                                           attn
                                                                                             │
 [o_proj-epilogue: per-tok fp8 quant (bare)]→(xq,xs)→ o_proj GEMM ←─────────────────────────┘
            │
 post_LN ─[gate_up_proj-prologue: fused RMSNorm+per-tok fp8 quant (CK)]→(xq,xs)→ gate_up GEMM
            │
 SiLU·mul ─[down_proj-prologue: fused silu·mul+per-tok fp8 quant]→(xq,xs)→ down_proj GEMM
   weights: per-channel fp8 + weight_scale (static)
```

## FP8 block — 128×128 weight blocks, per-1×128 dynamic activation (DeepSeek-V3 style)

- **GEMM path already wired** (`Fp8LinearMethod.apply` block branch, `sglang/srt/layers/quantization/fp8.py:769`):
  - `aiter_w8a8_block_fp8_linear` (`sglang/srt/layers/quantization/fp8_utils.py:760`, selected by `dispatch_w8a8_block_fp8_linear` :350) → `gemm_a8w8_blockscale_bpreshuffle` (`from aiter`) / `triton_gemm_a8w8_blockscale` (`aiter.ops.triton.gemm_a8w8_blockscale`), fed by `aiter_per1x128_quant` (= `get_hip_quant(aiter.QuantType.per_1x128)`, `fp8_utils.py:99`). `"fp8"` is in `rocm_supported_quantization`.
- **Reuse for the fusions (AMD triton, already in aiter):**
  - norm prologues — `fused_rms_fp8_group_quant(..., group_size=128, res1=residual)` (`aiter.ops.triton.fused_fp8_quant`).
  - `down_proj-prologue` — `act_mul_and_fp8_group_quant(x, "silu", group_size=128)` (`aiter.ops.triton.activation:131`).
  - `o_proj-epilogue` (bare) — `aiter_per1x128_quant` (same as above).
- **Code:** add the tuple-input branch in `aiter_w8a8_block_fp8_linear`; **set `transpose_scale`** to match the triton-vs-bpreshuffle GEMM picked by the aiter tuned-GEMM dispatch helper in `fp8_utils.py`.
- **GEMM kernel (AMD/aiter):** `gemm_a8w8_blockscale_bpreshuffle` (`from aiter`) or `triton_gemm_a8w8_blockscale` (`aiter.ops.triton.gemm_a8w8_blockscale`), selected by the aiter tuned-GEMM dispatch helper.

```
 input_LN ─[qkv_proj-prologue: fused_rms_fp8_group_quant g=128 (+res)]→((fp8,bs),res)→ qkv blockscale GEMM ─┐
                                                                                                            ▼
                                                                                                          attn
                                                                                                            │
 [o_proj-epilogue: aiter_per1x128_quant (bare)]→(fp8,bs)→ o_proj blockscale GEMM ←──────────────────────────┘
            │
 post_LN ─[gate_up_proj-prologue: fused_rms_fp8_group_quant g=128 (+res)]→((fp8,bs),res)→ gate_up GEMM
            │
 SiLU·mul ─[down_proj-prologue: act_mul_and_fp8_group_quant g=128]→(fp8,bs)→ down GEMM
   weights: 128×128 fp8 blocks + per-block scale (static)
```

## FP8 per-tensor static — one scale for weight, one (calibrated) scale for activation

- Cheapest scales (a single scalar per tensor); activation scale is **static/calibrated** (loaded as `input_scale`), so the "quant" is just multiply+cast — most fusable but needs calibration.
- **Reuse:** `W8A8Fp8LinearMethod.apply` (`w8a8_fp8.py:182`) → `apply_fp8_linear` (`fp8_utils.py:1458`); bare static quant `static_quant_fp8` (`fp8_kernel.py:724`).
- **To WIRE:** norm prologues — CK `rmsnorm_quant`/`add_rmsnorm_quant` (`aiter.ops.rmsnorm`, group_size=0) fed the **static** `input_scale`; `down_proj-prologue` — silu·mul + `static_quant_fp8`; `o_proj-epilogue` (bare) — `static_quant_fp8`.
- **GEMM kernel (AMD):** `torch._scaled_mm` per-tensor (the `per_tensor_weights and per_tensor_activations` branch of `apply_fp8_linear`, `fp8_utils.py:~1612`).

```
 input_LN ─[qkv_proj-prologue: RMSNorm + static-scale fp8 cast]→(fp8,s)→ qkv per-tensor GEMM ─┐
                                                                                              ▼
                                                                                            attn
                                                                                              │
 [o_proj-epilogue: static_quant_fp8 (bare)]→(fp8,s)→ o_proj GEMM ←──────────────────────────  ┘
            │
 post_LN ─[gate_up_proj-prologue: RMSNorm + static-scale fp8 cast]→(fp8,s)→ gate_up GEMM
            │
 SiLU·mul ─[down_proj-prologue: silu·mul + static-scale fp8 cast]→(fp8,s)→ down GEMM
   weights: per-tensor fp8 + 1 weight_scale; activation: 1 static input_scale (calibrated)
```

## AWQ — W4A16 (4-bit weight, bf16 activation)

- Weight-only ⇒ no prologue/epilogue quant fusion; only weight **dequant-into-GEMM**.
- **Blocker (PATCH):** `get_supported_act_dtypes()` rejects bf16 on the ROCm AWQ path → add bf16.
- **Reuse:** `AWQLinearMethod.apply` (`awq/awq.py:426`) delegates to a scheme kernel in `sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`:
  - plain — `AWQLinearKernel.apply` (`:88`) → **`awq_dequantize`** (`sgl_kernel.awq_dequantize`, or fallback `awq_dequantize_triton` from `awq/awq_triton.py:235`) + `torch.matmul`.
  - Marlin — `AWQMarlinLinearKernel.apply` (`:147`) → **`apply_awq_marlin_linear`** (`marlin_utils.py:153`), repacked via **`awq_marlin_repack`**.
  - Verify AMD/ROCm support; no new activation kernel.
- **GEMM kernel:** marlin → `gptq_marlin_gemm` (`sglang.jit_kernel.gptq_marlin`); plain → `awq_dequantize` + `torch.matmul`.

```
 (bf16 activation, never quantized) for every body Linear:
       qweight4 + scales + qzeros ─[AWQ dequant-GEMM]→ bf16 out
   input_LN / post_LN / SiLU·mul stay bf16 — nothing to fuse on the activation side
```

## GPTQ — W4A16 (4-bit weight, bf16 activation)

- Weight-only
- **Reuse:** `GPTQMarlinLinearMethod.apply` (`gptq.py:1120`) → **`apply_gptq_marlin_linear`** (`sglang/srt/layers/quantization/marlin_utils.py`), weights repacked via **`gptq_marlin_repack`** (`sglang.jit_kernel.gptq_marlin_repack`). Plain path `GPTQLinearMethod.apply` (`gptq.py:588`) → **`gptq_gemm`** + **`gptq_shuffle`** (`sgl_kernel`, `gptq.py:597`). Verify AMD/ROCm support + bf16 act dtype gate; no activation fusion.
- **GEMM kernel:** marlin → `gptq_marlin_gemm` (`sglang.jit_kernel.gptq_marlin`); plain → `gptq_gemm` (`sgl_kernel`).

```
 (bf16 activation) for every body Linear:
       qweight4 + scales + qzeros (+g_idx) ─[GPTQ/Marlin dequant-GEMM]→ bf16 out
   norms / silu stay bf16
```

## INT8 / SmoothQuant — W8A8 (per-channel weight, per-token activation)

- **Reuse:** `W8A8Int8LinearMethod.apply` (`w8a8_int8.py:205`). GPU/ROCm path = **`per_token_quant_int8`** (`int8_kernel.py:59`, `w8a8_int8.py:220`) → **`int8_scaled_mm`** (`sgl_kernel`, `w8a8_int8.py:226`) — quant is a **separate** kernel here (the fused `int8_scaled_mm_with_quant` at `:212` is **CPU-only**, intel-amx/arm64 branch; not on the GPU path).
- **To WIRE:** norm prologues — `rmsnorm_quant`/`add_rmsnorm_quant` (`aiter.ops.rmsnorm`) with an **int8 `out`** tensor (CK kernel picks dtype from `out`); `down_proj-prologue` — silu·mul + `per_token_quant_int8`; `o_proj-epilogue` (bare) — `per_token_quant_int8`. Then feed `(i8, x_scale)` straight into `int8_scaled_mm` via the tuple-input branch (folds the separate quant into the producer).
- **GEMM kernel:** `int8_scaled_mm` (`sgl_kernel`).

```
 input_LN ─[qkv_proj-prologue: RMSNorm+per-tok int8 quant]→(i8,xs)→ qkv int8 W8A8 GEMM ─┐
                                                                                        ▼
                                                                                      attn
                                                                                        │
 [o_proj-epilogue: per-tok int8 quant (bare)]→(i8,xs)→ o_proj GEMM ←─────────────────── ┘
            │
 post_LN ─[gate_up_proj-prologue: RMSNorm+per-tok int8 quant]→(i8,xs)→ gate_up GEMM
            │
 SiLU·mul ─[down_proj-prologue: silu·mul+per-tok int8 quant]→(i8,xs)→ down GEMM
   weights: per-channel int8 + scale
```

## MXFP4 — W4A4 (microscaling, 32-elem groups, e8m0 shared exponent)

- **Reuse (group=32, e8m0; already in aiter):**
  - norm prologues — `fused_rms_mxfp4_quant` (`aiter.ops.triton.fused_mxfp4_quant`, re-exported by `sglang/srt/layers/quantization/rocm_mxfp4_utils.py`).
  - `down_proj-prologue` — `act_mul_and_mxfp4_quant` (`aiter.ops.triton.activation:17`).
  - `o_proj-epilogue` (bare) — `fused_flatten_mxfp4_quant` (`aiter.ops.triton.fused_mxfp4_quant`).
  - GEMM (two options): **fused producer + `gemm_afp4wfp4`** / `gemm_afp4wfp4_preshuffle` (`aiter.ops.triton.gemm.basic.gemm_afp4wfp4`) — takes the pre-quantized fp4+e8m0 from the producer kernels above (the maximally-fused path); **or** the convenience **`gemm_afp4wfp4_pre_quant`** (`...gemm_afp4wfp4_pre_quant_atomic`) / `batched_gemm_afp4wfp4_pre_quant` which quantize the bf16 activation internally (baseline, no separate producer). All re-exported via `sglang/srt/layers/quantization/rocm_mxfp4_utils.py`. Verify AMD/ROCm support + bf16 act dtype gate.
- **GEMM kernel (AMD):** fused producer → `gemm_afp4wfp4` / `gemm_afp4wfp4_preshuffle` (`aiter.ops.triton.gemm.basic.gemm_afp4wfp4`); baseline (quant-in-GEMM) → `gemm_afp4wfp4_pre_quant` (`...gemm_afp4wfp4_pre_quant_atomic`) / `batched_gemm_afp4wfp4_pre_quant`.
- **Optional:** Hadamard rotation (R2/R4 local only, skip R1) before quant to tame outliers — extra kernel.

```
 input_LN ─[qkv_proj-prologue: RMSNorm(+res)+mxfp4 quant g=32]→(fp4,e8m0)→ qkv mxfp4 W4A4 GEMM ─┐
                                                                                                ▼
                                                                                              attn
                                                                                                │
 [o_proj-epilogue: mxfp4 quant (bare)]→(fp4,e8m0)→ o_proj GEMM ←─────────────────────────────── ┘
            │
 post_LN ─[gate_up_proj-prologue: RMSNorm(+res)+mxfp4 quant g=32]→(fp4,e8m0)→ gate_up GEMM
            │
 SiLU·mul ─[down_proj-prologue: silu·mul+mxfp4 quant g=32]→(fp4,e8m0)→ down GEMM
   weights: mxfp4 e2m1 + e8m0 block scales (g=32)   [opt: Hadamard R2/R4 pre-quant]
```

## MXFP8 — W8A8 microscaling (32-elem groups, UE8M0 shared exponent)

- Same MX family as MXFP4 but 8-bit (e4m3 elems + UE8M0 group scales) → much higher accuracy than MXFP4. Registered as `"mxfp8"` → `Fp8Config(use_mxfp8=True)` (`fp8.py:163`).
- **Reuse:** `Fp8LinearMethod.apply` mxfp8 branch (`fp8.py:177`) → `dispatch_w8a8_mxfp8_linear()` (`fp8_utils.py:368`); bare activation quant `mxfp8_group_quantize` (`fp8_utils.py:848`, `downcast_to_mxfp(x, e4m3, axis=1)`, group=32 UE8M0).
- **To WIRE:** norm prologues — CK norm + group-32 UE8M0 quant; reuse `fused_rms_fp8_group_quant(group_size=32)` **iff** its scale is UE8M0 (else norm then `mxfp8_group_quantize`); `down_proj-prologue` — `act_mul_and_fp8_group_quant(x, "silu", group_size=32)` (verify UE8M0) or silu·mul + `mxfp8_group_quantize`; `o_proj-epilogue` (bare) — `mxfp8_group_quantize`.
- **GEMM kernel (AMD):** `triton_mxfp8_blockscaled_linear` (`fp8_utils.py:1007`) → `triton_mxfp8_block_scaled_matmul` (`tl.dot_scaled`, the default non-flashinfer path; flashinfer cutlass/trtllm variants are NVIDIA-only).

```
 input_LN ─[qkv_proj-prologue: RMSNorm(+res)+mxfp8 quant g=32]→(fp8,ue8m0)→ qkv mxfp8 W8A8 GEMM ─┐
                                                                                                 ▼
                                                                                               attn
                                                                                                 │
 [o_proj-epilogue: mxfp8_group_quantize (bare)]→(fp8,ue8m0)→ o_proj GEMM ←──────────────────────  ┘
            │
 post_LN ─[gate_up_proj-prologue: RMSNorm(+res)+mxfp8 quant g=32]→(fp8,ue8m0)→ gate_up GEMM
            │
 SiLU·mul ─[down_proj-prologue: silu·mul+mxfp8 quant g=32]→(fp8,ue8m0)→ down GEMM
   weights: mxfp8 e4m3 + UE8M0 block scales (g=32)
```

---

## Checklist

### Infra (once)
- [ ] Shared **pre-quantized tuple-input** path in the fp8 (ptpc/block/per-tensor)/int8/mxfp4/mxfp8 `apply()` (skip internal quant, accept `(q, scale)`; reconcile transpose/layout per GEMM).
- [ ] Confirm `SGLANG_USE_AITER=1` documented as required.

### LM head (shared, target-side — independent axis)
- [ ] Set head quantization via `--quantization` / target config (independent of the draft flag).
- [ ] `lm_head-prologue` fusion only if activation-quantizing the head.

### Per scheme (body Linears)

**FP8 PTPC**
- [ ] `qkv_proj-prologue` — `_fused_rmsnorm_fp8_per_token_quant` (`communicator.py:107`, CK `rmsnorm_quant`).
- [ ] `gate_up_proj-prologue` — `_fused_rmsnorm_fp8_per_token_quant` (CK `add_rmsnorm_quant`, +residual).
- [ ] `down_proj-prologue` — `sglang_per_token_group_quant_fp8(fuse_silu_and_mul=True)` (`fp8_kernel.py:498`).
- [ ] `o_proj-epilogue` (bare) — `sglang_per_token_quant_fp8` (`fp8_kernel.py:656`).
- [ ] GEMM tuple-input branch in `apply_fp8_linear` → `gemm_a8w8_bpreshuffle` (`fp8_utils.py:~1585`).

**FP8 block**
- [ ] `qkv_proj-prologue` — `fused_rms_fp8_group_quant(group_size=128)` (`aiter.ops.triton.fused_fp8_quant`).
- [ ] `gate_up_proj-prologue` — `fused_rms_fp8_group_quant(group_size=128, res1=residual)`.
- [ ] `down_proj-prologue` — `act_mul_and_fp8_group_quant(x, "silu", group_size=128)` (`aiter.ops.triton.activation:131`).
- [ ] `o_proj-epilogue` (bare) — `aiter_per1x128_quant` (`fp8_utils.py:99`).
- [ ] GEMM tuple-input branch in `aiter_w8a8_block_fp8_linear` (`fp8_utils.py:760`) → `gemm_a8w8_blockscale_bpreshuffle` / `triton_gemm_a8w8_blockscale`; set `transpose_scale`.

**FP8 per-tensor static**
- [ ] `qkv_proj-prologue` — CK `rmsnorm_quant` (group_size=0) fed static `input_scale`.
- [ ] `gate_up_proj-prologue` — CK `add_rmsnorm_quant` (group_size=0, +residual) fed static `input_scale`.
- [ ] `down_proj-prologue` — silu·mul + `static_quant_fp8` (`fp8_kernel.py:724`).
- [ ] `o_proj-epilogue` (bare) — `static_quant_fp8`.
- [ ] GEMM via `apply_fp8_linear` → `torch._scaled_mm` per-tensor (`fp8_utils.py:~1612`). Requires a calibrated `input_scale`.

**AWQ** (W4A16 — no activation fusion)
- [ ] PATCH `get_supported_act_dtypes()` to accept bf16 on the ROCm AWQ path.
- [ ] Verify GEMM on AMD/ROCm — marlin: `apply_awq_marlin_linear` (`marlin_utils.py:153`) → **`gptq_marlin_gemm`** (`sglang.jit_kernel.gptq_marlin`); plain: `awq_dequantize` (`sgl_kernel`) + `torch.matmul` (`awq_kernels.py:88`).

**GPTQ** (W4A16 — no activation fusion)
- [ ] PATCH bf16 act-dtype gate.
- [ ] Verify GEMM on AMD/ROCm — marlin: `apply_gptq_marlin_linear` (`marlin_utils.py:465`) → **`gptq_marlin_gemm`** (`sglang.jit_kernel.gptq_marlin`); plain: **`gptq_gemm`** (`sgl_kernel`, `gptq.py:597`).

**INT8 / SmoothQuant**
- [ ] `qkv_proj-prologue` — `rmsnorm_quant` int8 `out` (`aiter.ops.rmsnorm`).
- [ ] `gate_up_proj-prologue` — `add_rmsnorm_quant` int8 `out` (+residual).
- [ ] `down_proj-prologue` — silu·mul + `per_token_quant_int8` (`int8_kernel.py:59`).
- [ ] `o_proj-epilogue` (bare) — `per_token_quant_int8`.
- [ ] GEMM tuple-input branch in `W8A8Int8LinearMethod.apply` (`w8a8_int8.py:205`) → `int8_scaled_mm` (`sgl_kernel`).

**MXFP4** (W4A4)
- [ ] `qkv_proj-prologue` — `fused_rms_mxfp4_quant` (`aiter.ops.triton.fused_mxfp4_quant`).
- [ ] `gate_up_proj-prologue` — `fused_rms_mxfp4_quant` (+residual).
- [ ] `down_proj-prologue` — `act_mul_and_mxfp4_quant` (`aiter.ops.triton.activation:17`).
- [ ] `o_proj-epilogue` (bare) — `fused_flatten_mxfp4_quant` (`aiter.ops.triton.fused_mxfp4_quant`).
- [ ] GEMM — `gemm_afp4wfp4` / `gemm_afp4wfp4_preshuffle` (pre-quantized in) on AMD; baseline `gemm_afp4wfp4_pre_quant`.
- [ ] (opt) Hadamard R2/R4 pre-quant rotation.

**MXFP8** (W8A8 MX, group-32 UE8M0)
- [ ] `qkv_proj-prologue` — CK norm + group-32 UE8M0 quant (`fused_rms_fp8_group_quant(group_size=32)` if UE8M0, else norm + `mxfp8_group_quantize`).
- [ ] `gate_up_proj-prologue` — same, +residual.
- [ ] `down_proj-prologue` — `act_mul_and_fp8_group_quant(x, "silu", group_size=32)` (verify UE8M0) or silu·mul + `mxfp8_group_quantize`.
- [ ] `o_proj-epilogue` (bare) — `mxfp8_group_quantize` (`fp8_utils.py:848`).
- [ ] GEMM — `triton_mxfp8_blockscaled_linear` (`fp8_utils.py:1007`) → `triton_mxfp8_block_scaled_matmul`.


### Testing
- [ ] Ensure quantized full-run output is identical to baseline output (draft + verify). The quantized version should still be lossless.


### Benchmarking
- [ ] Acceptance-rate (quantized vs. bf16)
- [ ] Throughput (quantized vs. bf16)
- [ ] Latency (quantized vs. bf16)

