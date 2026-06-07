# dump_draft_nonquant.py
#
# Control run: NON-QUANTIZED (bf16) DFlash draft, eager by default (no CUDA
# graph), with the same per-iteration draft-output dump as dump_draft_compare.py.
# This is the known-good config — if its hidden states ALSO dump as all-zero,
# the dump logic is reading the wrong tensor (not a model problem).
#
#   DFLASH_DUMP_DRAFT=/sgl-workspace/draft_nonquant_eager.jsonl \
#     python examples/dflash_quant/dump_draft_nonquant.py
#
# Set DFLASH_DISABLE_CUDA_GRAPH=0 to instead run it WITH cuda graph.
import os

import sglang as sgl

DRAFT_PATH = os.environ.get("DFLASH_DRAFT", "z-lab/Qwen3.5-35B-A3B-DFlash")
DISABLE_GRAPH = os.environ.get("DFLASH_DISABLE_CUDA_GRAPH", "1") == "1"  # eager by default


def main():
    dump = os.environ.get("DFLASH_DUMP_DRAFT")
    if dump and os.path.exists(dump):
        os.remove(dump)

    engine = sgl.Engine(
        model_path="Qwen/Qwen3.5-35B-A3B",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_PATH,
        # NOTE: no speculative_draft_model_quantization -> bf16 draft
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
    print(f"\n=== NON-QUANT (bf16) draft, {mode} done ===")
    print(f"draft dump -> {dump}")
    print("generated text:\n" + out["text"])


if __name__ == "__main__":
    main()
