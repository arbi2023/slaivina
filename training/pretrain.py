#!/usr/bin/env python3
"""Single-stage QLoRA continued-pretraining for the slaivina style model.

Loads a 4-bit base model via Unsloth, attaches LoRA adapters, and trains
causal-LM style (no prompt/response split) on the cleaned blog corpus --
see PLAN.md#fine-tuning and PLAN.md#why-no-sft-stage for why this is the
only training stage. Hyperparameters live in a YAML config
(training/configs/qwen3_4b_qlora.yaml by default); this script is meant to
stay generic across configs/models.

Usage
--------------------------------------------------------------------
    uv run training/pretrain.py
    uv run training/pretrain.py --config training/configs/qwen3_4b_qlora.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from datasets import Dataset

# scripts/split_dataset.py owns the canonical post-loading/splitting logic
# (post boundaries are only unambiguous in posts.jsonl, not in the derived
# pretrain_train.txt/pretrain_val.txt -- see that module's docstring).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.split_dataset import load_posts, split_posts

logger = logging.getLogger("pretrain")


def build_datasets(
    posts_path: Path, val_fraction: float, seed: int
) -> tuple[Dataset, Dataset]:
    posts = load_posts(posts_path)
    train_posts, val_posts = split_posts(posts, val_fraction, seed)
    train_ds = Dataset.from_dict({"text": train_posts})
    val_ds = Dataset.from_dict({"text": val_posts})
    return train_ds, val_ds


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/qwen3_4b_qlora.yaml"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 1
    cfg = load_config(args.config)

    posts_path = Path(cfg.get("posts_path", "data/processed/posts.jsonl"))
    if not posts_path.exists():
        logger.error(
            "posts file not found: %s (run scripts/clean.py first)", posts_path
        )
        return 1

    # Deferred, heavy imports: unsloth patches torch/transformers/trl on
    # import, so (a) keep this out of the module-level import path
    # (argparse --help, config errors, etc. shouldn't pay this cost), and
    # (b) import unsloth *first* -- importing trl/transformers/torch before
    # unsloth leaves trl's SFTConfig unpatched, which silently breaks
    # SFTTrainer's args handling (e.g. a bogus `eos_token` sentinel value).
    from unsloth import FastLanguageModel  # isort: skip
    import torch
    from trl import SFTConfig, SFTTrainer

    logger.info("loading base model: %s", cfg["model_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    train_ds, val_ds = build_datasets(posts_path, cfg["val_fraction"], cfg["seed"])
    logger.info("train=%d val=%d posts", len(train_ds), len(val_ds))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bf16_ok = torch.cuda.is_bf16_supported()
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 5),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 20),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 2),
        optim=cfg.get("optim", "adamw_8bit"),
        seed=cfg["seed"],
        report_to=cfg.get("report_to", "none"),
        bf16=bf16_ok,
        fp16=not bf16_ok,
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        packing=cfg.get("packing", True),
        # This tiny corpus overfits fast (train_loss keeps dropping while
        # eval_loss turns upward well before num_train_epochs is reached --
        # see PLAN.md#method) -- always restore the checkpoint with the
        # best held-out eval_loss rather than the last one.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = []
    if cfg.get("early_stopping_patience"):
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        callbacks=callbacks or None,
    )

    trainer_stats = trainer.train()
    logger.info("training complete: %s", trainer_stats)

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("saved LoRA adapter to %s", adapter_dir)

    if cfg.get("merge_and_save", True):
        merged_dir = output_dir / "merged"
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )
        logger.info("saved merged model to %s", merged_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
