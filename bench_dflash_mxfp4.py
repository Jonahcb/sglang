#!/usr/bin/env python3
"""
bench_dflash_mxfp4.py
=====================
A/B throughput benchmark of the DFlash speculative-decoding *draft* model:

    bf16 (baseline)   vs   MXFP4-quantized draft

It launches one server, sweeps a set of batch sizes on a paper-style benchmark
(GSM8K by default), tears the server down, then repeats for the other config.
Everything is driven from a single process:

    python3 bench_dflash_mxfp4.py
    python3 bench_dflash_mxfp4.py --batch-sizes 1 8 16 32 --num-questions 256
    python3 bench_dflash_mxfp4.py --configs mxfp4              # only the fp4 run
    python3 bench_dflash_mxfp4.py --tasks gsm8k humaneval

Results are printed as a table at the end and written to --out (JSON).

Only the DFlash *draft* precision differs between the two configs; the target
model, attention backend (triton), and all other server args are identical, so
the throughput delta is attributable to the draft MXFP4 path.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Fixed model / server configuration (matches profile_dflash_{bf16,mxfp4}.py)
# ---------------------------------------------------------------------------
TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B"
DRAFT_MODEL = "z-lab/Qwen3.5-35B-A3B-DFlash"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GSM8K_BENCH = os.path.join(REPO_ROOT, "benchmark", "gsm8k", "bench_sglang.py")
MTBENCH_BENCH = os.path.join(REPO_ROOT, "benchmark", "mtbench", "bench_sglang.py")

# Per-config differences. `quark_mxfp4` turns on the online Quark W4A4 MXFP4
# quant of the draft Linears; the bf16 config simply omits it. SGLANG_USE_AITER
# is on for BOTH so the rest of the stack is identical (and the fp4 fusion path,
# which requires aiter, is available for the mxfp4 run).
CONFIGS = {
    "bf16": {
        "label": "bf16 (baseline)",
        "draft_quant": None,
    },
    "mxfp4": {
        "label": "mxfp4 draft",
        "draft_quant": "quark_mxfp4",
    },
}


def server_command(cfg, port, mem_fraction):
    """Build the `python -m sglang.launch_server` argv for a given config."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_MODEL,
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--speculative-num-draft-tokens",
        "16",
        "--tp-size",
        "1",
        "--attention-backend",
        "triton",
        "--speculative-draft-attention-backend",
        "triton",
        "--mem-fraction-static",
        str(mem_fraction),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if cfg["draft_quant"]:
        cmd += ["--speculative-draft-model-quantization", cfg["draft_quant"]]
    return cmd


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------
def port_is_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_health(port, proc, timeout):
    """Poll /health until the server is ready, the process dies, or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {proc.returncode} "
                f"(see its log above)"
            )
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server not healthy on port {port} after {timeout}s")


def launch_server(cfg, port, mem_fraction, log_path, ready_timeout):
    env = os.environ.copy()
    env["SGLANG_USE_AITER"] = "1"
    cmd = server_command(cfg, port, mem_fraction)
    print(f"  launching: {' '.join(cmd)}")
    print(f"  server log -> {log_path}")
    log = open(log_path, "w")
    # new session so we can signal the whole process group on teardown
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=REPO_ROOT,
        start_new_session=True,
    )
    proc._log_fh = log  # keep handle alive
    print(f"  waiting for server (pid {proc.pid}) to become healthy ...")
    wait_for_health(port, proc, ready_timeout)
    print("  server is healthy.")
    return proc


def shutdown_server(proc, port):
    if proc is None or proc.poll() is not None:
        return
    print(f"  shutting down server (pid {proc.pid}) ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("  SIGTERM timed out, sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=15)
    try:
        proc._log_fh.close()
    except Exception:
        pass
    # wait for the port to actually free up before the next launch
    for _ in range(60):
        if not port_is_open(port):
            break
        time.sleep(1)
    print("  server down, port free.")


# ---------------------------------------------------------------------------
# Benchmark runners. Each returns a record with a primary `metric`, its
# `metric_name`, and `higher_is_better` (so the summary can orient speedups):
#   gsm8k   -> output throughput tok/s (higher is better) + accuracy
#   mtbench -> end-to-end latency s    (lower is better)
# ---------------------------------------------------------------------------
def _run_capturing_stdout(cmd):
    """Run a bench subprocess, capturing stdout (for metric parsing) while
    letting stderr stream straight to our terminal — that's where the bench's
    tqdm progress bar lives, so you see live per-step progress."""
    return subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=None, text=True
    ).stdout


def run_gsm8k(port, batch_size, num_questions, max_new_tokens):
    cmd = [
        sys.executable,
        GSM8K_BENCH,
        "--port",
        str(port),
        "--parallel",
        str(batch_size),
        "--num-questions",
        str(num_questions),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    out = _run_capturing_stdout(cmd)
    tp = re.search(r"Output throughput:\s*([\d.]+)", out)
    acc = re.search(r"Accuracy:\s*([\d.]+)", out)
    rec = {
        "metric_name": "throughput_tok_s",
        "higher_is_better": True,
        "metric": float(tp.group(1)) if tp else None,
        "accuracy": float(acc.group(1)) if acc else None,
    }
    if rec["metric"] is None:
        rec["error"] = "could not parse throughput (server/bench failed)"
        rec["tail"] = out[-500:]
    return rec


def run_mtbench(port, batch_size, num_questions, max_new_tokens):
    # mtbench's own default is 80 questions, 2 turns each, 256 max_new_tokens.
    cmd = [
        sys.executable,
        MTBENCH_BENCH,
        "--port",
        str(port),
        "--parallel",
        str(batch_size),
        "--num-questions",
        str(num_questions),
    ]
    out = _run_capturing_stdout(cmd)
    lat = re.search(r"Latency:\s*([\d.]+)", out)
    rec = {
        "metric_name": "latency_s",
        "higher_is_better": False,
        "metric": float(lat.group(1)) if lat else None,
        "accuracy": None,
    }
    if rec["metric"] is None:
        rec["error"] = "could not parse latency (server/bench failed)"
        rec["tail"] = out[-500:]
    return rec


TASKS = {
    "gsm8k": run_gsm8k,
    "mtbench": run_mtbench,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="A/B DFlash draft benchmark: bf16 vs MXFP4."
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["bf16", "mxfp4"],
        choices=list(CONFIGS),
        help="which draft precisions to run (default: both)",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["gsm8k"],
        choices=list(TASKS),
        help="paper-style benchmarks to run (default: gsm8k)",
    )
    ap.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32],
        help="concurrency levels to sweep (default: 1 4 8 16 32)",
    )
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--warmup-questions",
        type=int,
        default=16,
        help="min questions for the per-batch-size warmup pass (discarded)",
    )
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="max_new_tokens for the warmup pass (kept short)",
    )
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--mem-fraction-static", type=float, default=0.9)
    ap.add_argument(
        "--ready-timeout",
        type=int,
        default=1200,
        help="seconds to wait for a server to load + warm up",
    )
    ap.add_argument("--out", type=str, default="dflash_mxfp4_bench.json")
    ap.add_argument(
        "--log-dir", type=str, default="/tmp", help="where server logs go"
    )
    args = ap.parse_args()

    if port_is_open(args.port):
        sys.exit(
            f"port {args.port} is already in use — stop the existing server "
            f"or pass --port"
        )

    results = []  # flat list of records
    t_start = time.time()

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print("=" * 70)
        print(f"CONFIG: {cfg['label']}")
        print("=" * 70)
        log_path = os.path.join(args.log_dir, f"dflash_bench_{cfg_name}.log")
        proc = None
        try:
            proc = launch_server(
                cfg, args.port, args.mem_fraction_static, log_path,
                args.ready_timeout,
            )
            for task in args.tasks:
                for bs in args.batch_sizes:
                    # Warmup pass at THIS batch size: captures the decode CUDA
                    # graph for this concurrency and primes the radix cache, so
                    # the measured run reflects steady state, not one-time costs.
                    # Result is discarded. Just enough questions to fill the
                    # batch, short generations to keep it quick.
                    wq = max(bs, args.warmup_questions)
                    print(
                        f"  [{cfg_name}] {task} batch_size={bs} warmup "
                        f"({wq}q x {args.warmup_tokens}tok) ...",
                        flush=True,
                    )
                    TASKS[task](args.port, bs, wq, args.warmup_tokens)

                    print(f"  [{cfg_name}] {task} batch_size={bs} measure ...", flush=True)
                    t0 = time.time()
                    r = TASKS[task](
                        args.port, bs, args.num_questions, args.max_new_tokens
                    )
                    dt = time.time() - t0
                    rec = {
                        "config": cfg_name,
                        "task": task,
                        "batch_size": bs,
                        "wall_time_s": round(dt, 1),
                        **r,
                    }
                    results.append(rec)
                    metric = rec.get("metric")
                    if metric is None:
                        print(f"    FAILED: {rec.get('error')}")
                    else:
                        acc = rec.get("accuracy")
                        acc_str = f"  accuracy={acc:.3f}" if acc is not None else ""
                        print(
                            f"    {rec['metric_name']}={metric:.2f}{acc_str}  "
                            f"({dt:.0f}s)"
                        )
                    # checkpoint results after every run
                    with open(args.out, "w") as f:
                        json.dump(results, f, indent=2)
        finally:
            shutdown_server(proc, args.port)

    print_summary(results, args)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}  (total {time.time() - t_start:.0f}s)")


def print_summary(results, args):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    by_task = {}
    meta = {}
    for r in results:
        by_task.setdefault(r["task"], {}).setdefault(r["config"], {})[
            r["batch_size"]
        ] = r.get("metric")
        meta[r["task"]] = (r["metric_name"], r["higher_is_better"])

    for task, by_cfg in by_task.items():
        metric_name, higher_better = meta[task]
        direction = "higher=better" if higher_better else "lower=better"
        print(f"\n[{task}]  metric={metric_name} ({direction})")
        configs = list(by_cfg)
        header = f"{'batch':>8} " + "".join(f"{c:>16}" for c in configs)
        # speedup only meaningful when comparing exactly bf16 vs mxfp4
        show_speedup = set(configs) == {"bf16", "mxfp4"}
        if show_speedup:
            header += f"{'speedup':>10}"
        print(header)
        for bs in args.batch_sizes:
            row = f"{bs:>8} "
            for c in configs:
                v = by_cfg[c].get(bs)
                row += f"{(f'{v:.2f}' if v else '-'):>16}"
            if show_speedup:
                b, m = by_cfg["bf16"].get(bs), by_cfg["mxfp4"].get(bs)
                if b and m:
                    # speedup of mxfp4 over bf16, oriented by metric direction
                    su = (m / b) if higher_better else (b / m)
                    row += f"{su:>9.2f}x"
                else:
                    row += f"{'-':>10}"
            print(row)


if __name__ == "__main__":
    main()
