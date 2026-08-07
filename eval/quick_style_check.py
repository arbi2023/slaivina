#!/usr/bin/env python3
"""Manual, qualitative milestone check for the fine-tuning phase.

Not a rigorous eval (see PLAN.md#evaluation for perplexity/style-similarity,
still to be scaffolded) -- this is a cheap "did continued-pretraining
actually move the model toward the corpus's voice" sanity check you can run
by eye right after training, before spending time on quantization. Runs
against the merged model (default: 4-bit NF4 via bitsandbytes/Unsloth for
lower VRAM use; --full-precision for fp16), or the base model for
comparison.

**GPU-only, dev-side tool -- requires Unsloth (CUDA)**: this script (like
training/pretrain.py) is for the repo owner to sanity-check a training run
before quantizing, and needs a CUDA GPU. This is *not* a constraint on the
shipped end product: the actual distributed artifact (the GGUF Q4_K_M
quant, served via llama.cpp/Ollama -- see quantize/convert_and_quantize.sh
and skills/quantize/SKILL.md) is explicitly built for CPU-only inference
and already runs fine on a GPU-less laptop (benchmarked ~13.8 tok/s
generation, see eval/EVAL.md). --load-in-4bit here is a *different*
quantization (bitsandbytes NF4, training-toolchain-side) than the shipped
GGUF quant -- don't conflate the two. Final validation before shipping a
new version should test the actual GGUF via llama-cli/llama-server, not
this script.

Generates a few short completions from a fixed persona + few-shot prompt
(the same style of prompt PLAN.md#deployment-and-serving describes using
at inference time) and prints them so you can compare base vs fine-tuned
output for whether tone/brevity/imagery start to resemble the blog.
Generation stops at the first newline (a fragment/post boundary in this
corpus) rather than running to --max-new-tokens, which is just a fallback
cap.

Caveat: SEED_WORDS are used simplistically as a literal continuation
prefix (`f"\n{seed_word}"` appended straight after the prompt) -- the model
is completing "...mare" as the start of a sentence, not writing "about"
the sea as a theme. Don't over-read thematic relevance in the samples;
they're testing voice/style continuation, not topical steering. A model
that free-associates away from the seed word's meaning is not necessarily
failing.

TODO: try generating longer (e.g. --max-new-tokens 200+) and then
truncating post-hoc at a good sentence/clause boundary, rather than
relying on stop_strings=["\n"] alone -- the current newline-stop can end a
fragment on a very short/incomplete-feeling beat (see min_new_tokens
guard) when a slightly longer generation would have produced a more
satisfying complete thought.

Recommended generation params (defaults below): temperature=0.8,
repetition_penalty=1.15. Found via a small manual sweep (2026-08-07, see
eval/EVAL.md) after user feedback that output was too long/repetitive/
rambling -- notably, lowering temperature *alone* (no repetition penalty)
made repetition/looping *worse*, not better, since sampling at a lower
temperature narrows toward the highest-probability continuation, which can
reinforce a self-referential loop rather than escape it. repetition_penalty
is the targeted fix for repetition; temperature is a separate axis
(overall randomness/creativity) -- don't conflate the two. This same
combination is worth carrying over to any downstream serving config
(Ollama Modelfile / llama-server request params), not just this script.

Output is always printed to stdout, and (unless --no-log) also appended to
--log-file (default eval/EVAL.md) with a timestamp/model header, per
skills/eval/SKILL.md's "log everything qualitatively in eval/EVAL.md"
guardrail -- otherwise these samples only exist in a terminal scrollback.

Usage
--------------------------------------------------------------------
    # fine-tuned merged model (default: 4-bit NF4) -- prints to stdout and
    # appends to eval/EVAL.md
    uv run eval/quick_style_check.py

    # test the fp16 intermediate itself (e.g. right after a training
    # change, before quantizing) -- needs more VRAM
    uv run eval/quick_style_check.py --full-precision

    # base (non-fine-tuned) model, for comparison
    uv run eval/quick_style_check.py --model unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit

    # skip the log-file append (stdout only)
    uv run eval/quick_style_check.py --no-log

    # override generation params (defaults are already the recommended
    # temperature=0.8/repetition_penalty=1.15 combo)
    uv run eval/quick_style_check.py --temperature 0.9 --repetition-penalty 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PERSONA = (
    "Sei l'autore del blog 'Come me su una slavina': scrivi in italiano, "
    "con tono intimo, immagini oniriche/malinconiche, frasi brevi."
)

# NB: these are literal continuation prefixes ("<prompt>\n<seed_word>"), not
# topical prompts -- the model completes the seed word as the start of a
# sentence/fragment. See the module docstring's caveat before reading too
# much thematic meaning into a given seed's output.
SEED_WORDS = ["mare", "autobus", "insonnia", "specchio", "citofono"]


def load_fewshot(posts_path: Path, n: int, seed: int) -> list[str]:
    with posts_path.open(encoding="utf-8") as f:
        posts = [json.loads(line)["text"] for line in f if line.strip()]
    return random.Random(seed).sample(posts, n)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="training/output/qwen3_4b_qlora/merged",
        help="path or HF repo id of the model to test (default: the merged "
        "fine-tuned model)",
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        default=True,
        help="load via bitsandbytes NF4 (default: on). This is a "
        "training-toolchain-side quantization for fitting VRAM during "
        "this pre-quantization check -- it is NOT the shipped GGUF Q4_K_M "
        "quant (see quantize/convert_and_quantize.sh and "
        "skills/quantize/SKILL.md). Final validation before shipping "
        "should test the actual GGUF via llama-cli/llama-server, not "
        "this script.",
    )
    parser.add_argument(
        "--full-precision",
        dest="load_in_4bit",
        action="store_false",
        help="load the fp16 merged model instead of NF4 (needs more "
        "VRAM; rarely necessary -- use to test the pre-quantization "
        "intermediate itself, e.g. after a training change).",
    )
    parser.add_argument("--fewshot-n", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=12,
        help="floor on generated tokens before the newline stop condition "
        "is allowed to end generation, so fragments aren't just one word",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature (default: 0.8 -- lower than the "
        "original 0.9. NB: lowering temperature alone (without a "
        "repetition penalty) was found to *worsen* repetition/looping, "
        "since do_sample narrows toward the top continuation, which can "
        "reinforce a loop rather than escape it -- see PLAN.md#path-to-"
        "further-improve-output-quality-before-publishing)",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="discourage repeated tokens (1.0 = disabled/no penalty, "
        ">1.0 penalizes repeats). Default 1.15 chosen from a small "
        "sweep: repetition_penalty=1.15 + temperature=0.8 gave the "
        "shortest, least rambling/repetitive fragments among the "
        "configs tried (see eval/EVAL.md sweep entries and PLAN.md#path-"
        "to-further-improve-output-quality-before-publishing).",
    )
    parser.add_argument(
        "--posts-path", type=Path, default=Path("data/processed/posts.jsonl")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("eval/EVAL.md"),
        help="append generated samples here as a qualitative eval log "
        "(default: eval/EVAL.md)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="print to stdout only, skip appending to --log-file",
    )
    args = parser.parse_args()

    from unsloth import FastLanguageModel  # isort: skip
    import torch

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    fewshot = load_fewshot(args.posts_path, args.fewshot_n, args.seed)

    results = []
    for seed_word in SEED_WORDS:
        prompt = PERSONA + "\n\n" + "\n".join(fewshot) + f"\n{seed_word}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                # These are meant to be single short fragments (like the
                # few-shot examples), not free-running text -- stop as
                # soon as the model emits a newline (i.e. tries to start
                # a "next post") rather than always spending the full
                # max_new_tokens budget and cutting a fragment off
                # mid-sentence.
                stop_strings=["\n"],
                tokenizer=tokenizer,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        results.append((seed_word, completion))
        print(f"--- seed: {seed_word!r} ---")
        print(completion)
        print()

    if not args.no_log:
        gen_params = (
            f"temperature={args.temperature}, top_p={args.top_p}, "
            f"repetition_penalty={args.repetition_penalty}"
        )
        append_log(args.log_file, args.model, gen_params, fewshot, results)
        print(f"Appended results to {args.log_file}", file=sys.stderr)

    return 0


def append_log(
    log_file: Path,
    model_name: str,
    gen_params: str,
    fewshot: list[str],
    results: list[tuple[str, str]],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_file.exists()
    with log_file.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "# Qualitative eval log\n\n"
                "Samples from `eval/quick_style_check.py`, appended per run "
                "-- see skills/eval/SKILL.md and PLAN.md#evaluation.\n"
            )
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        f.write(f"\n## {timestamp} -- model: `{model_name}`\n\n")
        f.write(f"Generation params: {gen_params}\n\n")
        f.write("Few-shot examples used:\n")
        for post in fewshot:
            f.write(f"- {post}\n")
        f.write("\n")
        for seed_word, completion in results:
            f.write(f"**seed: `{seed_word}`**\n\n> {completion}\n\n")


if __name__ == "__main__":
    sys.exit(main())
