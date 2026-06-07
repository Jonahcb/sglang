# profile_dflash_engine_fp8.py
#
# FP8 (Quark) variant of profile_dflash_engine.py: profiles a DFlash speculative
# run whose DRAFT model linear layers are quantized to FP8 (W8A8, dynamic acts).
# Target model is unchanged (bf16). Writes the torch profiler trace into
# TRACE_DIR (kept separate from the bf16 trace so nothing is overwritten).
import os

import sglang as sgl

TRACE_DIR = os.environ.get(
    "DFLASH_FP8_TRACE_DIR", "/sgl-workspace/dflash-fp8-trace"
)
DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)
    engine = sgl.Engine(
        model_path="Qwen/Qwen3.5-35B-A3B",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_PATH,
        speculative_draft_model_quantization="quark",  # FP8 quark draft
        speculative_num_draft_tokens=16,
        tp_size=1,
        attention_backend="triton",
        speculative_draft_attention_backend="triton",
        mem_fraction_static=0.75,
        trust_remote_code=True,
        # CUDA graph is ENABLED by default — do NOT set disable_cuda_graph.
    )

    prompt = "Write a detailed essay on the history of computing."

    # Warmup: capture CUDA graphs OUTSIDE the profiling window, so the trace
    # shows graph *replay* (steady state), not one-time graph capture.
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 16})

    engine.start_profile(
        output_dir=TRACE_DIR,
        with_stack=False,     # kills the 115M python_function events (34 GB -> hundreds of MB)
        record_shapes=True,   # keep op input shapes; set False to shrink further
        # num_steps=40,       # optional: auto-stop after N forward passes (bounds trace size)
    )
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 128})
    engine.stop_profile()

    engine.shutdown()
    print(f"\n=== PROFILE DONE === trace written under {TRACE_DIR}")


if __name__ == "__main__":
    main()
