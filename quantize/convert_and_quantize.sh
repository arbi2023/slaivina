#!/usr/bin/env bash
# Convert a merged Hugging Face slaivina checkpoint to GGUF and quantize it
# to several candidate levels for CPU/commodity-hardware inference.
#
# See docs: skills/quantize/SKILL.md and PLAN.md#quantization.
#
# Usage:
#   quantize/convert_and_quantize.sh <merged-model-dir> [output-dir]
#
# Example:
#   quantize/convert_and_quantize.sh training/output/qwen3_4b_qlora/merged
#
# Requires a llama.cpp checkout with:
#   - convert_hf_to_gguf.py runnable via its own Python venv (see
#     LLAMA_CPP_DIR / LLAMA_CPP_CONVERT_PYTHON below -- do NOT run it with
#     this repo's .venv, its `transformers` pin conflicts with the training
#     stack; see skills/quantize/SKILL.md#setup).
#   - a built `llama-quantize` binary (cmake --build build --target
#     llama-quantize) new enough to know the model architecture
#     (Qwen3ForCausalLM here) -- the llama-quantize shipped by some package
#     managers (e.g. Homebrew) can lag behind and lack this.
#
# Environment overrides:
#   LLAMA_CPP_DIR             path to the llama.cpp checkout
#                             (default: ~/tools/llama.cpp)
#   LLAMA_CPP_CONVERT_PYTHON  python to run convert_hf_to_gguf.py with
#                             (default: $LLAMA_CPP_DIR/.venv-convert/bin/python)
#   LLAMA_QUANTIZE_BIN        path to the llama-quantize binary
#                             (default: $LLAMA_CPP_DIR/build/bin/llama-quantize)
#   QUANT_LEVELS              space-separated quant types to produce
#                             (default: "Q4_K_M Q5_K_M Q8_0")
#   MODEL_LABEL               short name used in output filenames
#                             (default: "slaivina-4b", see
#                             PLAN.md#target-model-selection)

set -euo pipefail

MERGED_MODEL_DIR="${1:?Usage: convert_and_quantize.sh <merged-model-dir> [output-dir]}"
OUTPUT_DIR="${2:-quantize/output}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/tools/llama.cpp}"
LLAMA_CPP_CONVERT_PYTHON="${LLAMA_CPP_CONVERT_PYTHON:-$LLAMA_CPP_DIR/.venv-convert/bin/python}"
LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
QUANT_LEVELS="${QUANT_LEVELS:-Q4_K_M Q5_K_M Q8_0}"
MODEL_LABEL="${MODEL_LABEL:-slaivina-4b}"

if [[ ! -d "$MERGED_MODEL_DIR" ]]; then
  echo "error: merged model dir not found: $MERGED_MODEL_DIR" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_CPP_CONVERT_PYTHON" ]]; then
  echo "error: convert python not found/executable: $LLAMA_CPP_CONVERT_PYTHON" >&2
  echo "       see skills/quantize/SKILL.md#setup to create it." >&2
  exit 1
fi
if [[ ! -x "$LLAMA_QUANTIZE_BIN" ]]; then
  echo "error: llama-quantize binary not found/executable: $LLAMA_QUANTIZE_BIN" >&2
  echo "       see skills/quantize/SKILL.md#setup to build it." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

F16_GGUF="$OUTPUT_DIR/${MODEL_LABEL}-f16.gguf"

echo "== Converting $MERGED_MODEL_DIR -> $F16_GGUF (f16) =="
"$LLAMA_CPP_CONVERT_PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_MODEL_DIR" \
  --outtype f16 \
  --outfile "$F16_GGUF"

for LEVEL in $QUANT_LEVELS; do
  # llama-quantize wants lowercase in the filename by convention, but the
  # type argument itself is uppercase (e.g. Q4_K_M).
  LEVEL_LOWER="$(echo "$LEVEL" | tr '[:upper:]' '[:lower:]')"
  OUT_FILE="$OUTPUT_DIR/${MODEL_LABEL}-${LEVEL_LOWER}.gguf"
  echo "== Quantizing -> $OUT_FILE ($LEVEL) =="
  "$LLAMA_QUANTIZE_BIN" "$F16_GGUF" "$OUT_FILE" "$LEVEL"
done

echo "== Done. Outputs in $OUTPUT_DIR: =="
ls -lh "$OUTPUT_DIR"
