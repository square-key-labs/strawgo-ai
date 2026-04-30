#!/usr/bin/env bash
# Phase-2 sweep on the bench VM:
#   - rust-tier1: new shared-session onnx-worker + cmd/loadtest (Unix socket)
#   - cgo-embed:  loadtest-embed-linux (in-process Go ORT)
#   - tenvad:     loadtest-tenvad (in-process Go TEN-VAD)
#
# Run on VM at /home/disolaterx/strawgo-bench/.

set -euo pipefail

LEVELS=(1 5 10 25 50 100 200)
DUR=20
SOCKET=/tmp/onnx-worker-bench.sock
VAD_MODEL=/home/kartik/silero_vad.onnx
TURN_MODEL=/home/kartik/smart-turn-v3.1-cpu.onnx
PCM=/home/kartik/sine_440_500ms_16k.pcm

REPO=/home/disolaterx/strawgo-bench
WORKER=$REPO/onnx-worker/target/release/onnx-worker
LOADTEST=/home/kartik/loadtest-linux           # existing Strawgo loadtest (unchanged wire)
LOADTEST_EMBED=$REPO/loadtest-embed-linux
LOADTEST_TENVAD=$REPO/loadtest-tenvad
ORT_LIB=$REPO/lib/libonnxruntime.so.1.25.1

OUTDIR=/home/disolaterx/phase2_results
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"
echo "# Phase-2 sweep $(date -u +%FT%TZ) on $(uname -nrm)" > "$SUMMARY"
echo "# nproc=$(nproc) mem=$(free -m | awk '/^Mem:/{print $2"MB"}')" >> "$SUMMARY"
echo >> "$SUMMARY"

stop_worker() { pkill -f onnx-worker 2>/dev/null || true; rm -f "$SOCKET"; sleep 1; }
worker_rss_mb() {
  if [ -n "${WORKER_PID:-}" ] && [ -d "/proc/$WORKER_PID" ]; then
    awk '/VmRSS:/ {printf "%.1f", $2/1024}' "/proc/$WORKER_PID/status"
  else echo "0"; fi
}
start_worker() {
  stop_worker
  echo "starting onnx-worker (Tier 1)..."
  RUST_LOG=warn nohup "$WORKER" \
    --vad-model "$VAD_MODEL" --turn-model "$TURN_MODEL" --socket "$SOCKET" \
    > "$OUTDIR/onnx-worker.log" 2>&1 &
  WORKER_PID=$!
  for i in $(seq 1 30); do [ -S "$SOCKET" ] && break; sleep 0.5; done
  [ -S "$SOCKET" ] || { echo "worker failed; see $OUTDIR/onnx-worker.log"; return 1; }
  echo "onnx-worker ready (pid=$WORKER_PID)"
}

# ---------- Arm 1: Rust onnx-worker (Tier 1 fixes) ----------
echo "═══ Arm 1: rust-tier1 (Unix-socket worker, shared session) ═══"
start_worker
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/rust-tier1_n${N}.log"
  echo "── rust-tier1 N=$N ──"
  "$LOADTEST" -socket "$SOCKET" -pcm "$PCM" -dur "$DUR" -levels "$N" -pid "$WORKER_PID" \
    2>&1 | tee "$OUT" | tail -10
  echo "==> rust-tier1 N=$N  worker_rss=$(worker_rss_mb)MB" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done
stop_worker

# ---------- Arm 2: cgo embed (in-process Go ORT) ----------
echo "═══ Arm 2: cgo-embed (in-process Go ORT) ═══"
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/cgo-embed_n${N}.log"
  echo "── cgo-embed N=$N ──"
  "$LOADTEST_EMBED" -lib "$ORT_LIB" -model "$VAD_MODEL" -pcm "$PCM" -dur "$DUR" -levels "$N" \
    2>&1 | tee "$OUT" | tail -10
  echo "==> cgo-embed N=$N" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done

# ---------- Arm 3: TEN-VAD (in-process Go cgo) ----------
echo "═══ Arm 3: tenvad (in-process Go TEN-VAD) ═══"
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/tenvad_n${N}.log"
  echo "── tenvad N=$N ──"
  "$LOADTEST_TENVAD" -dur "$DUR" -levels "$N" \
    2>&1 | tee "$OUT" | tail -10
  echo "==> tenvad N=$N" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done

echo
echo "==== DONE ===="
echo "Results in: $OUTDIR"
echo "Summary:    $SUMMARY"
