# accept_len_bf16.py
#
# Acceptance-length benchmark for the NON-QUANTIZED (bf16) DFLASH draft model.
# Writes per-iteration acceptance lengths to a file and prints the average.
#
#   python accept_len_bf16.py
#
# (Pair with accept_len_fp8.py and compare the AVG acceptance length lines.)
import os

from accept_len_common import run

# bf16 draft -- the original DFLASH checkpoint (already in the HF cache).
DRAFT_PATH = os.environ.get(
    "ACCEPT_LEN_BF16_DRAFT", "z-lab/Qwen3.5-35B-A3B-DFlash"
)
OUT_PATH = os.environ.get("ACCEPT_LEN_BF16_OUT", "/sgl-workspace/accept_len_bf16.txt")

if __name__ == "__main__":
    run(
        draft_path=DRAFT_PATH,
        quantization=None,  # no quantization -> bf16 weights
        label="BF16",
        out_path=OUT_PATH,
    )
