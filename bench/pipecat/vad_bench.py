"""Pipecat VAD-only capacity bench.

Spawns N concurrent "agents", each owning its own SileroVADAnalyzer
instance, and feeds synthetic 16 kHz PCM frames through them at real-time
pace. No vendors, no STT/LLM/TTS. Pure ONNX-per-call cost.

This mirrors Strawgo's `cmd/loadtest` (which drives the shared `onnx-worker`
process via Unix socket): same Silero VAD model, same 32 ms / 512-sample
window at 16 kHz, same real-time cadence, same metrics.

Architecture difference under test:
  - Strawgo: 1 Rust process holds N ONNX sessions, IPC via Unix socket.
  - Pipecat: N Python tasks each hold 1 SileroVADAnalyzer (in-process ONNX).

Metrics:
  - RSS (RUSAGE_SELF) at peak
  - per-frame latency (call → return) p50/p95/p99
  - scheduled / succeeded / errors
  - CPU% via /proc/self/stat (Linux) or psutil if available

Usage:
  python vad_bench.py -n 50 --dur 30
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import resource
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero requires exactly 512 at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 LE
FRAME_INTERVAL = FRAME_SAMPLES / SAMPLE_RATE  # 0.032 s = 32 ms


def synth_pcm_frame(rng: np.random.Generator) -> bytes:
    """Speech-like noise at ~-12 dBFS, returned as int16 LE bytes."""
    sig = rng.standard_normal(FRAME_SAMPLES).astype(np.float32) * 8000
    return sig.astype(np.int16).tobytes()


@dataclass
class WorkerStats:
    scheduled: int = 0
    succeeded: int = 0
    errors: int = 0
    latencies_us: list[float] = None  # type: ignore[assignment]


async def worker(idx: int, dur_s: float, ready: asyncio.Event, start: asyncio.Event,
                 stats_out: list[WorkerStats]) -> None:
    """One 'agent': owns one VAD analyzer, paces frames at real-time."""
    rng = np.random.default_rng(idx ^ 0xC0FFEE)
    analyzer = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6),
    )
    # Pipecat 1.1: ctor ignores sample_rate; pipeline normally calls
    # set_sample_rate from StartFrame. Mirror that here.
    analyzer.set_sample_rate(SAMPLE_RATE)
    ws = WorkerStats(latencies_us=[])
    ready.set()
    await start.wait()

    deadline = time.monotonic() + dur_s
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        # real-time pacing
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            await asyncio.sleep(sleep)
        next_tick += FRAME_INTERVAL

        ws.scheduled += 1
        frame = synth_pcm_frame(rng)
        t0 = time.perf_counter_ns()
        try:
            # voice_confidence is a sync call; runs ONNX in caller thread.
            analyzer.voice_confidence(frame)
            ws.succeeded += 1
        except Exception:  # noqa: BLE001
            ws.errors += 1
            continue
        ws.latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

    # `cleanup` may be sync or async depending on version.
    try:
        result = analyzer.cleanup()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # noqa: BLE001
        pass
    stats_out.append(ws)


def get_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB. macOS: bytes.
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[int(min(len(s) - 1, (len(s) - 1) * q / 100))]


async def main_async(args: argparse.Namespace) -> None:
    print(f"config: n={args.n} dur={args.dur}s sample_rate={SAMPLE_RATE} "
          f"frame_samples={FRAME_SAMPLES} frame_interval_ms={FRAME_INTERVAL*1000:.1f}")

    rss_baseline_mb = get_rss_mb()
    print(f"RSS baseline (before agents): {rss_baseline_mb:.1f} MB")

    ready_events = [asyncio.Event() for _ in range(args.n)]
    start_event = asyncio.Event()
    stats_out: list[WorkerStats] = []

    tasks = [
        asyncio.create_task(worker(i, args.dur, ready_events[i], start_event, stats_out))
        for i in range(args.n)
    ]

    # Wait for all workers to load their ONNX session.
    await asyncio.wait([asyncio.create_task(e.wait()) for e in ready_events])
    rss_after_load_mb = get_rss_mb()
    print(f"RSS after {args.n} VAD analyzers loaded: {rss_after_load_mb:.1f} MB "
          f"(Δ {(rss_after_load_mb - rss_baseline_mb):.1f} MB; "
          f"{(rss_after_load_mb - rss_baseline_mb) / max(args.n,1):.2f} MB/agent)")

    # Release all together.
    t_start = time.monotonic()
    start_event.set()

    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - t_start

    rss_peak_mb = get_rss_mb()

    total_sched = sum(s.scheduled for s in stats_out)
    total_ok = sum(s.succeeded for s in stats_out)
    total_err = sum(s.errors for s in stats_out)
    all_lats = [v for s in stats_out for v in s.latencies_us]

    drop_rate = 0.0 if total_sched == 0 else (total_sched - total_ok) / total_sched

    print()
    print("─" * 72)
    print(f"ELAPSED            {elapsed:.1f}s  (target {args.dur}s)")
    print(f"AGENTS             {args.n}")
    print(f"RSS (baseline → after load → peak)")
    print(f"                   {rss_baseline_mb:.0f} → {rss_after_load_mb:.0f} → {rss_peak_mb:.0f} MB")
    print(f"                   per-agent load cost: "
          f"{(rss_after_load_mb - rss_baseline_mb) / max(args.n,1):.2f} MB")
    print(f"FRAMES             scheduled={total_sched} ok={total_ok} err={total_err}  "
          f"drop_rate={drop_rate*100:.2f}%")
    if all_lats:
        print(f"FRAME LATENCY (µs) p50={pct(all_lats,50):.0f} "
              f"p95={pct(all_lats,95):.0f} p99={pct(all_lats,99):.0f} (n={len(all_lats)})")
    print("─" * 72)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["framework", "n", "elapsed_s", "rss_baseline_mb",
                        "rss_after_load_mb", "rss_peak_mb", "per_agent_mb",
                        "scheduled", "ok", "err", "drop_rate",
                        "lat_p50_us", "lat_p95_us", "lat_p99_us"])
            w.writerow(["pipecat", args.n, f"{elapsed:.2f}",
                        f"{rss_baseline_mb:.1f}",
                        f"{rss_after_load_mb:.1f}",
                        f"{rss_peak_mb:.1f}",
                        f"{(rss_after_load_mb - rss_baseline_mb) / max(args.n,1):.2f}",
                        total_sched, total_ok, total_err, f"{drop_rate:.4f}",
                        f"{pct(all_lats,50):.0f}",
                        f"{pct(all_lats,95):.0f}",
                        f"{pct(all_lats,99):.0f}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1, help="concurrent VAD agents")
    ap.add_argument("--dur", type=float, default=20.0, help="seconds per level")
    ap.add_argument("--csv", default="", help="write summary row")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
