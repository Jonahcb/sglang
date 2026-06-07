# dflash_fp8_run.py
#
# Profiler-FREE runner, meant to be wrapped by an external tracer (rocprofv3 /
# rocprofiler-sdk). Same FP8 (Quark) DFlash engine + prompt + warmup as
# dflash_fp8_perfetto.py, but it does NOT start the PyTorch profiler — running
# torch.profiler (roctracer) and rocprofv3 at the same time makes the two tracers
# fight over the tracing interface. Let rocprofv3 do all the tracing:
#
#   rocprofv3 --kernel-trace --hip-trace -f pftrace csv -d ./rp_trace \
#     -- python /sgl-workspace/sglang/examples/dflash_quant/dflash_fp8_run.py
import os

import sglang as sgl

DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")


def build_engine():
    return sgl.Engine(
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


PROMPT = "Write a detailed essay on the history of computing."


def main():
    engine = build_engine()
    # Warmup also triggers CUDA-graph capture; the steady-state replay is what
    # follows. (rocprofv3 traces the whole process, so both appear in the trace —
    # filter by time if you only want replay.)
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 16})
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 128})
    engine.shutdown()
    print("\n=== DFLASH FP8 RUN DONE (trace was collected externally) ===")


if __name__ == "__main__":
    main()
