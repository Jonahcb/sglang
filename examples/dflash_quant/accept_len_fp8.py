# accept_len_fp8.py
#
# Acceptance-length benchmark for the QUANTIZED (FP8 / quark) DFLASH draft model.
# Writes per-iteration acceptance lengths to a file and prints the average.
#
#   python accept_len_fp8.py
#
# (Pair with accept_len_bf16.py and compare the AVG acceptance length lines.)
import os

from accept_len_common import run

DRAFT_PATH = os.environ.get("ACCEPT_LEN_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
OUT_PATH = os.environ.get("ACCEPT_LEN_FP8_OUT", "/sgl-workspace/accept_len_fp8.txt")

if __name__ == "__main__":
    run(
        draft_path=DRAFT_PATH,
        quantization="quark",  # FP8 W8A8 quark draft
        label="FP8",
        out_path=OUT_PATH,
    )
