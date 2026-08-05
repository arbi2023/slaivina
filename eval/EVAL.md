# Qualitative eval log

Samples from `eval/quick_style_check.py`, appended per run -- see skills/eval/SKILL.md and PLAN.md#evaluation.

## 2026-08-03 13:49 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> e nebbia, una stella in più, una macchia di latte… come se l'universo si fosse rotto e ti stesse cercando il cazzo.

**seed: `autobus`**

> 17, a pochi metri dalla stazione: se non la trovi non c'è.

**seed: `insonnia`**

> e merda.

**seed: `specchio`**

> specchio. E tu mi fai ridere, mi fai piangere, mi fai pensare che sono un cazzo di coglione. E se mi rido… ti rido? Tipo, io ridi e tu ridi, ridiamo insieme? Ma no. Tu… ridi da solo… io… piango. E il gatto… dorme.

**seed: `citofono`**

> : 'Come stai?' Io: 'Sto bene… Sto bene… Sto bene…' Finché ti stanco, allora mi chiami: 'Sto sull'albero… sull'albero!'


## 2026-08-03 13:51 UTC -- model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> a mezza luna e un cappello di pelle

**seed: `autobus`**

> non c'è nessuno.

**seed: `insonnia`**

> di notte, pelle fredda come ghiaccio.

**seed: `specchio`**

> di una notte, con un fiore che non si apre mai.

**seed: `citofono`**

> : non risponde.

## 2026-08-05 -- Quantization benchmark

Ran `quantize/convert_and_quantize.sh training/output/qwen3_4b_qlora/merged`
(see skills/quantize/SKILL.md) to produce GGUF variants of the merged
fine-tuned model, then benchmarked each with `llama-perplexity` (held-out
`data/processed/pretrain_val.txt`, `-c 256` -- the val set is tiny, only
~881 tokens, so treat these as directional, not statistically tight) and
`llama-bench` (`-p 128 -n 64`, CPU-only, 10 threads, no GPU offload).

| Quant  | Size     | Perplexity        | Prompt proc. (t/s) | Generation (t/s) |
|--------|----------|--------------------|---------------------|-------------------|
| Q4_K_M | 2.32 GiB | 33.29 +/- 5.50     | 72.17 +/- 1.57      | 13.77 +/- 0.50    |
| Q5_K_M | 2.69 GiB | 32.19 +/- 5.26     | 19.56 +/- 1.31      | 11.75 +/- 0.40    |
| Q8_0   | 3.98 GiB | 31.51 +/- 5.15     | 23.60 +/- 1.88      | 8.85 +/- 0.47     |

**Decision: keep Q4_K_M** as the shipped quant. It's the smallest and
fastest, and the perplexity gap vs. Q8_0 (near-lossless baseline) is only
~5.6% relative -- not worth 1.7x the size or the generation-speed hit.
Q5_K_M's prompt-processing number looks anomalously low relative to Q4_K_M
and Q8_0 (non-monotonic) -- likely bench noise from the very short `-p 128`
prompt / thread contention on this machine rather than a real regression;
worth re-running with a larger prompt if this ever needs to be revisited.
The Q5_K_M/Q8_0 GGUF files were deleted after this benchmark to reclaim
disk space (regenerable any time via `quantize/convert_and_quantize.sh`).

