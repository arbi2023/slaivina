---
name: slaivina-quantize
description: Use when converting a merged slaivina checkpoint to GGUF (or AWQ/GPTQ) and producing quantized model variants for commodity hardware.
---

# Quantization

Reference: [PLAN.md — Quantization](../../PLAN.md#quantization).

## Steps
1. Convert the merged HF model directory to GGUF with `llama.cpp`'s
   `convert_hf_to_gguf.py`.
2. Quantize with `llama-quantize` to at least:
   - `Q4_K_M` (default balance),
   - `Q5_K_M` (higher quality),
   - `Q8_0` (near-lossless baseline for comparison).
3. Optionally export AWQ/GPTQ 4-bit variants for comparison.
4. Wire this up in `quantize/convert_and_quantize.sh <merged-model-dir>`.
5. Record tokens/sec and perplexity per quant level (feeds into the
   `slaivina-eval` skill) so size/quality tradeoffs are concrete.

## Setup

The `llama.cpp` toolchain needs to exist outside this repo (it's a separate
C++/Python project, not a slaivina dependency):

1. Clone it somewhere stable, e.g. `~/tools/llama.cpp` (this is the default
   `LLAMA_CPP_DIR` `quantize/convert_and_quantize.sh` looks for):
   ```
   git clone https://github.com/ggerganov/llama.cpp.git ~/tools/llama.cpp
   ```
   Prefer a fresh/recent clone over a distro package (e.g. Homebrew's
   `llama.cpp` formula can lag behind upstream and fail to recognize newer
   architectures like `Qwen3ForCausalLM`).
2. `convert_hf_to_gguf.py` needs its **own** Python venv, separate from this
   repo's `.venv` -- its pinned `transformers==4.57.6` conflicts with the
   training stack's `transformers==5.5.0`. If your `pip`/`uv` is configured
   with a private/corporate index that can't reach this checkout's deps,
   pass `--index-url https://pypi.org/simple` explicitly.
   ```
   cd ~/tools/llama.cpp
   python3 -m venv .venv-convert
   .venv-convert/bin/pip install -r requirements/requirements-convert_hf_to_gguf.txt
   ```
3. Build `llama-quantize` (and, for benchmarking, `llama-cli`,
   `llama-perplexity`, `llama-bench`) from that same checkout, CPU-only is
   fine (no GPU needed for quantization/inference of these small models):
   ```
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
   cmake --build build --target llama-quantize llama-cli llama-perplexity llama-bench -j"$(nproc)"
   ```
4. `quantize/convert_and_quantize.sh` picks up `$LLAMA_CPP_DIR`,
   `$LLAMA_CPP_CONVERT_PYTHON`, and `$LLAMA_QUANTIZE_BIN` from the
   environment (with sensible defaults matching the layout above) -- see
   the script's header comment for the full list of overrides.

## Smoke-testing a quantized GGUF

After running `quantize/convert_and_quantize.sh`, sanity-check that the
GGUF actually loads and generates before trusting any benchmark numbers:

```
~/tools/llama.cpp/build/bin/llama-cli \
  -m quantize/output/slaivina-4b-q4_k_m.gguf \
  -no-cnv -st -c 2048 -n 30 -p "Il mare"
```

Notes on the flags (all were hit as real failures while first setting this
up, not just cargo-culted):
- `-c 2048` (`--ctx-size`) is required -- the default is the model's full
  native context (Qwen3 supports up to 262144 tokens), and allocating that
  much KV-cache on CPU triggered an **OOM kill** on a 32 GB machine.
- `-no-cnv` (`--no-conversation`) plus `-st` (`--single-turn`) are both
  required to get a single one-shot completion and a clean exit. `-no-cnv`
  alone still drops into an interactive prompt loop after generating (which
  then spins forever printing empty `>` prompts if stdin isn't a TTY);
  `-st` is what actually makes it exit after the first turn.
- Without `-n` (`--n-predict`) it defaults to `-1` (generate until EOS or
  context limit), which is fine for real use but makes smoke tests
  unpredictable in length -- pin it to something small like `30`.

For a benchmarking (not just smoke-test) run, use `llama-perplexity` on a
held-out text file and `llama-bench` for tokens/sec -- see
`quantize/bench/` and the "Quantization benchmark" entry in `eval/EVAL.md`
for a worked example and results table.

## Guardrails
- If training happened via `mlx-lm` on Apple Silicon, merge back into a
  standard HF checkpoint first — `convert_hf_to_gguf.py` expects that format
  regardless of training framework.
- Name outputs `slaivina-<size>-<quant>.gguf` using the base model's size
  (e.g. `slaivina-4b-q4_k_m.gguf` for the current default — see
  [PLAN.md — Target model selection](../../PLAN.md#target-model-selection-sota-small-open-weight)),
  not a hardcoded size that can go stale if the base model changes.
- Don't commit GGUF/quantized weights to git; ship via release assets or HF Hub.
