#!/usr/bin/env bash
# fetch-pipeline-models — download ONNX models needed by:
#
#   - src/audio/vad/pipeline_embed/    (integration tests)
#   - cmd/loadtest-pipeline/           (3-stage pipeline load harness)
#   - cmd/bench-denoise-cost/          (denoiser cost microbench)
#
# Models are gitignored (*.onnx). Run this once to populate
# testdata/models/. Idempotent — skips files already present.
#
# All downloads are small enough to fetch directly from public mirrors.
# Sources:
#   - silero_vad.onnx        snakers4/silero-vad
#   - smart-turn-v3.1-cpu    pipecat-ai/smart-turn (HF)
#   - gtcrn_simple.onnx      yuyun2000/SpeechDenoiser (MIT, 16 kHz streaming)
#   - nsnet2-20ms.onnx       microsoft/DNS-Challenge baseline (HF mirror)
#   - rnnoise.onnx           ailia-models export (BSD/Apache)
#   - dfn3_{enc,erb_dec,df_dec}.onnx  Rikorose/DeepFilterNet (Apache/MIT)
#
# Usage:
#   ./scripts/fetch-pipeline-models.sh

set -euo pipefail

cd "$(dirname "$0")/.."

DEST=testdata/models
mkdir -p "$DEST"

fetch() {
  local url=$1
  local out="$DEST/$2"
  if [ -f "$out" ]; then
    echo "✓ $2 already present"
    return 0
  fi
  echo "→ fetching $2"
  curl -fsSL -o "$out" "$url" || {
    echo "ERROR: download failed: $url" >&2
    return 1
  }
}

# Silero VAD (Phase-2 default, used everywhere).
fetch "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
      "silero_vad.onnx"

# Smart-turn v3.1 (turn-end detector).
fetch "https://huggingface.co/pipecat-ai/smart-turn-v3.1/resolve/main/smart-turn-v3.1-cpu.onnx" \
      "smart-turn-v3.1-cpu.onnx"

# GTCRN streaming denoiser (current production default).
fetch "https://github.com/yuyun2000/SpeechDenoiser/raw/main/onnx_model/gtcrn_simple.onnx" \
      "gtcrn_simple.onnx"

# NSNet2 (Microsoft DNS-Challenge baseline, 20 ms hop).
fetch "https://huggingface.co/niobures/NSNet2/resolve/main/models/NSNet2-ONNX/nsnet2-20ms-baseline.onnx" \
      "nsnet2-20ms.onnx"

# RNNoise (ailia-models export, 1-second batch — bench cost only, not streaming).
fetch "https://huggingface.co/niobures/RNNoise/resolve/main/models/ailia-models/rnn_model.onnx" \
      "rnnoise.onnx"

# DFN3 split into 3 sub-graphs (encoder + ERB decoder + DF decoder).
DFN3_TGZ="$DEST/.dfn3_tmp.tar.gz"
if [ ! -f "$DEST/dfn3_enc.onnx" ] || [ ! -f "$DEST/dfn3_erb_dec.onnx" ] || [ ! -f "$DEST/dfn3_df_dec.onnx" ]; then
  echo "→ fetching DeepFilterNet 3 ONNX bundle"
  curl -fsSL -o "$DFN3_TGZ" \
    "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3_onnx.tar.gz"
  tar -xzf "$DFN3_TGZ" -C "$DEST"
  # Layout inside tarball: tmp/export/{enc,erb_dec,df_dec}.onnx
  if [ -d "$DEST/tmp/export" ]; then
    for f in enc erb_dec df_dec; do
      mv "$DEST/tmp/export/${f}.onnx" "$DEST/dfn3_${f}.onnx"
    done
    rm -rf "$DEST/tmp" "$DFN3_TGZ"
    echo "✓ extracted dfn3_{enc,erb_dec,df_dec}.onnx"
  else
    echo "WARN: unexpected DFN3 layout — leaving tarball at $DFN3_TGZ" >&2
  fi
fi

echo
echo "=== models in $DEST ==="
ls -lh "$DEST"/*.onnx 2>/dev/null || echo "(none)"
