#!/usr/bin/env bash
# Pipecat full-pipeline sweep (denoise → VAD → smart-turn) on the bench VM.
# Mirrors run_phase2_sweep.sh / run_vad_sweep.sh in structure so output
# slots into bench/compare_vad.py without changes.
#
# Prereqs on the VM (~/...):
#   - bench-venv with pipecat-ai==1.1.0 + onnxruntime + numpy + loguru
#   - silero_vad.onnx (bundled with pipecat — auto-loaded)
#   - smart-turn-v3.1-cpu.onnx
#   - gtcrn_simple.onnx (16 kHz streaming denoiser; see PIPELINE_PIPECAT_REPORT.md)
#   - bench/pipecat/pipeline_bench.py copied to ~/pipeline_bench.py
#
# Usage:
#   ./run_pipeline_sweep.sh           # full sweep N ∈ {1,5,10,25,50,100,200}
#   LEVELS="1 5 10" ./run_pipeline_sweep.sh   # custom levels

set -euo pipefail

LEVELS=(${LEVELS:-1 5 10 25 50 100 200})
DUR=${DUR:-20}                   # seconds per level
TURN_MODEL=${TURN_MODEL:-$HOME/smart-turn-v3.1-cpu.onnx}
DENOISE_MODEL=${DENOISE_MODEL:-$HOME/gtcrn_simple.onnx}
TURN_CADENCE_MS=${TURN_CADENCE_MS:-2000}    # smart-turn every 2 s/agent
PIPELINE_BENCH=${PIPELINE_BENCH:-$HOME/pipeline_bench.py}
VENV_PY=${VENV_PY:-$HOME/bench-venv/bin/python3}

OUTDIR=${OUTDIR:-$HOME/pipeline_results}
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.txt"
echo "# pipecat-pipeline bench $(date -u +%FT%TZ) on $(uname -nrm)" > "$SUMMARY"
echo "# nproc=$(nproc) mem=$(free -m 2>/dev/null | awk '/^Mem:/{print $2"MB"}')" >> "$SUMMARY"
echo "# denoise=$DENOISE_MODEL  turn=$TURN_MODEL  cadence=${TURN_CADENCE_MS}ms" >> "$SUMMARY"
echo >> "$SUMMARY"

for f in "$DENOISE_MODEL" "$TURN_MODEL" "$PIPELINE_BENCH" "$VENV_PY"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: missing $f" >&2
    exit 2
  fi
done

run_pipeline() {
  local N=$1
  local OUT_LOG="$OUTDIR/pipecat_pipeline_n${N}.log"
  local OUT_CSV="$OUTDIR/pipecat_pipeline_n${N}.csv"
  echo "─── pipecat_pipeline N=$N ──────────────────────────"
  "$VENV_PY" "$PIPELINE_BENCH" \
    -n "$N" --dur "$DUR" --csv "$OUT_CSV" \
    --denoise-model "$DENOISE_MODEL" \
    --turn-model "$TURN_MODEL" \
    --turn-cadence-ms "$TURN_CADENCE_MS" \
    2>&1 | tee "$OUT_LOG" | tail -14
  echo "==> pipecat_pipeline N=$N" >> "$SUMMARY"
  tail -10 "$OUT_LOG" >> "$SUMMARY"
  echo >> "$SUMMARY"
}

for N in "${LEVELS[@]}"; do
  run_pipeline "$N"
  sleep 3
done

echo
echo "==== DONE ===="
echo "Results: $OUTDIR"
echo "Summary: $SUMMARY"
