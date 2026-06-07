# dump_draft_compare.py
#
# Runs ONE deterministic DFlash generation (fp8 draft, temp=0) and dumps the
# draft model's per-iteration output to $DFLASH_DUMP_DRAFT. Toggle CUDA graph
# with DFLASH_DISABLE_CUDA_GRAPH=1. No torch profiling — this is purely to
# compare draft *values* between graph and eager mode.
#
#   # graph (default):
#   DFLASH_DUMP_DRAFT=/sgl-workspace/draft_graph.jsonl \
#     python examples/dflash_quant/dump_draft_compare.py
#   # eager:
#   DFLASH_DUMP_DRAFT=/sgl-workspace/draft_eager.jsonl DFLASH_DISABLE_CUDA_GRAPH=1 \
#     python examples/dflash_quant/dump_draft_compare.py
import os

import sglang as sgl

DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
DISABLE_GRAPH = os.environ.get("DFLASH_DISABLE_CUDA_GRAPH", "0") == "1"


def main():
    # Fresh dump file each run so iteration indices line up across runs.
    dump = os.environ.get("DFLASH_DUMP_DRAFT")
    if dump and os.path.exists(dump):
        os.remove(dump)

    engine = sgl.Engine(
        model_path="Qwen/Qwen3.5-35B-A3B",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_PATH,
        speculative_draft_model_quantization="quark",
        speculative_num_draft_tokens=16,
        tp_size=1,
        attention_backend="triton",
        speculative_draft_attention_backend="triton",
        mem_fraction_static=0.75,
        trust_remote_code=True,
        disable_cuda_graph=DISABLE_GRAPH,
    )

    prompt = "Write a detailed essay on the history of computing."
    out = engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 32})
    engine.shutdown()

    mode = "EAGER (no cuda graph)" if DISABLE_GRAPH else "CUDA GRAPH"
    print(f"\n=== {mode} done ===")
    print(f"draft dump -> {dump}")
    print("generated text:\n" + out["text"])


if __name__ == "__main__":
    main()
