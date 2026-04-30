#!/usr/bin/env bash
# VAD-only sweep: Strawgo (shared onnx-worker) vs Pipecat (per-agent ONNX).
# Run on the VM after:
#   - ~/onnx-worker/target/release/onnx-worker built
#   - ~/silero_vad.onnx + ~/smart-turn-v3.1-cpu.onnx downloaded
#   - ~/loadtest-linux + ~/vad_bench.py in place
#   - ~/bench-venv with pipecat installed
#
# Output: ~/vad_results/{strawgo,pipecat}_n<N>.csv + summary.txt

set -euo pipefail

LEVELS=(1 5 10 25 50 100 200)
DUR=20            # seconds of measurement per level
SOCKET=/tmp/onnx-worker-bench.sock
VAD_MODEL=~/silero_vad.onnx
TURN_MODEL=~/smart-turn-v3.1-cpu.onnx
WORKER=~/onnx-worker/target/release/onnx-worker
LOADTEST=~/loadtest-linux
PIPECAT_BENCH=~/vad_bench.py

OUTDIR=~/vad_results
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.txt"
echo "# VAD bench $(date -u +%FT%TZ) on $(uname -nrm)" > "$SUMMARY"
echo "# nproc=$(nproc) mem=$(free -m | awk '/^Mem:/{print $2"MB"}')" >> "$SUMMARY"
echo >> "$SUMMARY"

stop_worker() {
  pkill -f onnx-worker 2>/dev/null || true
  rm -f "$SOCKET"
  sleep 1
}

start_worker() {
  stop_worker
  echo "starting onnx-worker..."
  RUST_LOG=warn nohup "$WORKER" \
    --vad-model "$VAD_MODEL" \
    --turn-model "$TURN_MODEL" \
    --socket "$SOCKET" > "$OUTDIR/onnx-worker.log" 2>&1 &
  WORKER_PID=$!
  # wait for socket
  for i in $(seq 1 30); do
    [ -S "$SOCKET" ] && break
    sleep 0.5
  done
  if [ ! -S "$SOCKET" ]; then
    echo "onnx-worker failed to start; check $OUTDIR/onnx-worker.log"
    return 1
  fi
  echo "onnx-worker ready (pid=$WORKER_PID)"
}

worker_rss_mb() {
  if [ -n "${WORKER_PID:-}" ] && [ -d "/proc/$WORKER_PID" ]; then
    awk '/VmRSS:/ {printf "%.1f", $2/1024}' "/proc/$WORKER_PID/status"
  else
    echo "0"
  fi
}

run_strawgo() {
  local N=$1
  local OUT="$OUTDIR/strawgo_n${N}.log"
  echo "─── strawgo VAD N=$N ──────────────────────────"
  local levels_arg
  levels_arg="$N"
  "$LOADTEST" \
    -socket "$SOCKET" \
    -pcm ~/sine_440_500ms_16k.pcm \
    -dur "$DUR" \
    -levels "$levels_arg" \
    -pid "$WORKER_PID" 2>&1 | tee "$OUT" | tail -10

  local worker_rss
  worker_rss=$(worker_rss_mb)
  echo "==> strawgo VAD N=$N  worker_rss=${worker_rss}MB" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
}

run_pipecat() {
  local N=$1
  local OUT_LOG="$OUTDIR/pipecat_n${N}.log"
  local OUT_CSV="$OUTDIR/pipecat_n${N}.csv"
  echo "─── pipecat VAD N=$N ──────────────────────────"
  ~/bench-venv/bin/python3 "$PIPECAT_BENCH" -n "$N" --dur "$DUR" --csv "$OUT_CSV" 2>&1 | tee "$OUT_LOG" | tail -12
  echo "==> pipecat VAD N=$N" >> "$SUMMARY"
  tail -10 "$OUT_LOG" >> "$SUMMARY"
  echo >> "$SUMMARY"
}

# ── strawgo arm: one shared onnx-worker for all levels ─────────────────────
start_worker
for N in "${LEVELS[@]}"; do
  run_strawgo "$N"
  sleep 3
done
stop_worker

# ── pipecat arm: each level spawns its own python process ──────────────────
for N in "${LEVELS[@]}"; do
  run_pipecat "$N"
  sleep 3
done

echo
echo "==== DONE ===="
echo "Results: $OUTDIR"
echo "Summary: $SUMMARY"
