"""Compare VAD bench outputs from Strawgo (cmd/loadtest log table) and
Pipecat (vad_bench.py CSV). Run after pulling vad_results/ from the VM.
"""
from __future__ import annotations
import csv
import glob
import os
import re
import sys
from collections import defaultdict


# Strawgo loadtest stdout has a fixed-width row, e.g.:
# 2       1.50ms    2.17ms    2.31ms     312      312      0         77.2          77.5
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+(?:\.\d+)?)(ms|µs|ns)\s+(\d+(?:\.\d+)?)(ms|µs|ns)\s+(\d+(?:\.\d+)?)(ms|µs|ns)"
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
)


def parse_strawgo_log(path: str):
    with open(path) as f:
        text = f.read()
    out = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        def to_us(val: str, unit: str) -> float:
            v = float(val)
            if unit == "ms":
                return v * 1000.0
            if unit == "µs":
                return v
            if unit == "ns":
                return v / 1000.0
            return 0.0
        out.append({
            "n": n,
            "p50_us": to_us(m.group(2), m.group(3)),
            "p95_us": to_us(m.group(4), m.group(5)),
            "p99_us": to_us(m.group(6), m.group(7)),
            "scheduled": int(m.group(8)),
            "ok": int(m.group(9)),
            "err": int(m.group(10)),
            "rss_base_mb": float(m.group(11)),
            "rss_peak_mb": float(m.group(12)),
        })
    return out


def parse_pipecat_csv(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "n": int(r["n"]),
                "p50_us": float(r["lat_p50_us"]),
                "p95_us": float(r["lat_p95_us"]),
                "p99_us": float(r["lat_p99_us"]),
                "scheduled": int(r["scheduled"]),
                "ok": int(r["ok"]),
                "err": int(r["err"]),
                "rss_base_mb": float(r["rss_baseline_mb"]),
                "rss_after_load_mb": float(r["rss_after_load_mb"]),
                "rss_peak_mb": float(r["rss_peak_mb"]),
                "per_agent_mb": float(r["per_agent_mb"]),
            })
    return rows


def main(d: str) -> None:
    sg_rows = []
    for p in sorted(glob.glob(os.path.join(d, "strawgo_n*.log"))):
        sg_rows.extend(parse_strawgo_log(p))

    pc_rows = []
    for p in sorted(glob.glob(os.path.join(d, "pipecat_n*.csv"))):
        pc_rows.extend(parse_pipecat_csv(p))

    by_n = defaultdict(dict)
    for r in sg_rows:
        by_n[r["n"]]["strawgo"] = r
    for r in pc_rows:
        by_n[r["n"]]["pipecat"] = r

    Ns = sorted(by_n.keys())
    print(f"{'N':>4}  {'fw':<8}  {'sched':>6}  {'ok':>6}  {'err':>4}  "
          f"{'p50':>9}  {'p95':>9}  {'p99':>9}  {'rss_peak_mb':>11}  {'mb/agent':>9}")
    print("─" * 95)
    for n in Ns:
        for fw in ("strawgo", "pipecat"):
            r = by_n[n].get(fw)
            if not r:
                print(f"{n:>4}  {fw:<8}  (missing)")
                continue
            mb_each = (r["rss_peak_mb"] - r["rss_base_mb"]) / max(n, 1)
            print(f"{n:>4}  {fw:<8}  "
                  f"{r['scheduled']:>6}  {r['ok']:>6}  {r['err']:>4}  "
                  f"{_us(r['p50_us']):>9}  {_us(r['p95_us']):>9}  {_us(r['p99_us']):>9}  "
                  f"{r['rss_peak_mb']:>11.1f}  {mb_each:>9.2f}")
        print()


def _us(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.2f}ms"
    return f"{v:.0f}µs"


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "vad_results"
    main(d)
