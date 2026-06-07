# profile_dflash_engine_fp8_nocudagraph.py
#
# Identical to profile_dflash_engine_fp8.py EXCEPT CUDA graph is DISABLED
# (disable_cuda_graph=True). Everything else — model, FP8 quark draft, sampling,
# warmup, profiler config — is the same. Use this to test whether the draft FP8
# GEMMs actually execute per-iteration in eager mode (the graph-replayed draft
# was firing 0 kernels). Trace goes to a SEPARATE dir so nothing is overwritten.
import os

import sglang as sgl

TRACE_DIR = os.environ.get(
    "DFLASH_FP8_EAGER_TRACE_DIR", "/sgl-workspace/dflash-fp8-trace-eager"
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
        disable_cuda_graph=True,  # <<< ONLY difference: run eager, no graph capture/replay
    )

    prompt = "Write a detailed essay on the history of computing."

    # Warmup outside the profiling window so the trace shows steady-state eager
    # execution (weights loaded, allocator warm), not first-call overhead.
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
    print(f"\n=== PROFILE DONE (eager / no cuda graph) === trace written under {TRACE_DIR}")


if __name__ == "__main__":
    main()
