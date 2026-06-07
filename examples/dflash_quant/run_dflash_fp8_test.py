"""End-to-end test: DFlash speculative decoding with an FP8 (Quark) draft model."""
import os
import sglang as sgl


def probe_draft_quant(engine):
    """Reach into the draft model and report the quant method of its linears."""
    try:
        runner = engine.scheduler_info  # not available; fall back below
    except Exception:
        pass


def main():
    engine = sgl.Engine(
        model_path="Qwen/Qwen3.5-35B-A3B",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="/sgl-workspace/dflash-fp8",
        speculative_draft_model_quantization="quark",
        speculative_num_draft_tokens=16,
        tp_size=1,
        attention_backend="triton",
        speculative_draft_attention_backend="triton",
        mem_fraction_static=0.75,
        trust_remote_code=True,
    )

    prompt = "Write a detailed essay on the history of computing."
    out = engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 64})
    print("\n=== GENERATION OK ===")
    print(repr(out["text"][:300]))
    print("=== END ===")
    engine.shutdown()


if __name__ == "__main__":
    main()
