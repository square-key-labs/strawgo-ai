#!/usr/bin/env bash
# A/B sweep: fp32 vs int8 on rust-tier1 + cgo-embed.
# Set the env vars below for your VM layout.

set -euo pipefail

# ── Required env (override before invocation) ─────────────────────────
BENCH_HOME=${BENCH_HOME:-$HOME/strawgo-bench}
MODELS_DIR=${MODELS_DIR:-$BENCH_HOME/testdata/models}
PCM_DIR=${PCM_DIR:-$BENCH_HOME/testdata}

LEVELS=(1 5 10 25 50 100 200)
DUR=20
SOCKET=/tmp/onnx-worker-bench.sock
VAD_FP32=${VAD_FP32:-$MODELS_DIR/silero_vad.onnx}
VAD_INT8=${VAD_INT8:-$MODELS_DIR/silero_vad_int8.onnx}
TURN_MODEL=${TURN_MODEL:-$MODELS_DIR/smart-turn-v3.1-cpu.onnx}
PCM=${PCM:-$PCM_DIR/sine_440_500ms_16k.pcm}

REPO=$BENCH_HOME
WORKER=${WORKER:-$REPO/onnx-worker/target/release/onnx-worker}
LOADTEST=${LOADTEST:-$REPO/loadtest-linux}
LOADTEST_EMBED=${LOADTEST_EMBED:-$REPO/loadtest-embed-linux}
ORT_LIB=${ORT_LIB:-$REPO/lib/libonnxruntime.so.1.25.1}

OUTDIR=${OUTDIR:-$HOME/int8_ab_results}
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"
echo "# int8 vs fp32 A/B $(date -u +%FT%TZ)" > "$SUMMARY"

stop_worker() { pkill -f onnx-worker 2>/dev/null || true; rm -f "$SOCKET"; sleep 1; }
start_worker_fp32() {
  stop_worker
  RUST_LOG=warn nohup "$WORKER" \
    --vad-model "$VAD_FP32" --turn-model "$TURN_MODEL" --socket "$SOCKET" \
    > "$OUTDIR/onnx-worker-fp32.log" 2>&1 &
  WORKER_PID=$!
  for i in $(seq 1 30); do [ -S "$SOCKET" ] && break; sleep 0.5; done
  [ -S "$SOCKET" ] || { echo "fp32 worker failed"; return 1; }
  echo "fp32 worker pid=$WORKER_PID"
}
start_worker_int8() {
  stop_worker
  RUST_LOG=warn nohup "$WORKER" \
    --vad-model "$VAD_FP32" --vad-model-int8 "$VAD_INT8" \
    --turn-model "$TURN_MODEL" --socket "$SOCKET" \
    > "$OUTDIR/onnx-worker-int8.log" 2>&1 &
  WORKER_PID=$!
  for i in $(seq 1 30); do [ -S "$SOCKET" ] && break; sleep 0.5; done
  [ -S "$SOCKET" ] || { echo "int8 worker failed"; return 1; }
  echo "int8 worker pid=$WORKER_PID"
}

# ── Arm 1: rust-tier1 fp32 (re-run for clean A/B) ─────────────
echo "═══ rust-tier1 fp32 ═══"
start_worker_fp32
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/rust-fp32_n${N}.log"
  echo "── rust-fp32 N=$N ──"
  "$LOADTEST" -socket "$SOCKET" -pcm "$PCM" -dur "$DUR" -levels "$N" -pid "$WORKER_PID" \
    2>&1 | tee "$OUT" | tail -5
  echo "==> rust-fp32 N=$N" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done
stop_worker

# ── Arm 2: rust-tier1 int8 ─────────────
echo "═══ rust-tier1 int8 ═══"
start_worker_int8
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/rust-int8_n${N}.log"
  echo "── rust-int8 N=$N ──"
  "$LOADTEST" -socket "$SOCKET" -pcm "$PCM" -dur "$DUR" -levels "$N" -pid "$WORKER_PID" \
    2>&1 | tee "$OUT" | tail -5
  echo "==> rust-int8 N=$N" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done
stop_worker

# ── Arm 3: cgo-embed fp32 (re-run) ─────────────
echo "═══ cgo-embed fp32 ═══"
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/cgo-fp32_n${N}.log"
  echo "── cgo-fp32 N=$N ──"
  "$LOADTEST_EMBED" -lib "$ORT_LIB" -model "$VAD_FP32" -pcm "$PCM" -dur "$DUR" -levels "$N" \
    2>&1 | tee "$OUT" | tail -5
  echo "==> cgo-fp32 N=$N" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done

# ── Arm 4: cgo-embed int8 ─────────────
echo "═══ cgo-embed int8 ═══"
for N in "${LEVELS[@]}"; do
  OUT="$OUTDIR/cgo-int8_n${N}.log"
  echo "── cgo-int8 N=$N ──"
  "$LOADTEST_EMBED" -lib "$ORT_LIB" -model "$VAD_INT8" -pcm "$PCM" -dur "$DUR" -levels "$N" \
    2>&1 | tee "$OUT" | tail -5
  echo "==> cgo-int8 N=$N" >> "$SUMMARY"
  tail -8 "$OUT" >> "$SUMMARY"
  echo >> "$SUMMARY"
  sleep 3
done

echo "==== DONE ===="
echo "Results: $OUTDIR"
echo "Summary: $SUMMARY"
