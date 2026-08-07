# Slaivina: A Small "Come me su una slavina"-Style Language Model

Learning project to fine-tune, quantize, and RAG-augment a small open-weights
LLM so it can generate text in the style of the blog
`https://comemesuunaslavina.retrocog.org/`, running on a laptop / commodity
hardware (CPU or a single consumer GPU with 6–12GB VRAM).

> **Site note:** the URL is served by **Ghost 3.2** (static HTTrack mirror),
> not Hugo — `generator` meta tag and Ghost asset paths confirm this. Content
> is Italian, short poetic/aphoristic posts (often a single sentence or
> short paragraph + a photo), ~51 paginated index pages. This shapes the
> scraper and the data strategy below (small corpus, style-transfer framing
> rather than knowledge-heavy fine-tuning).

---

## Learning goals (why each phase exists)

| Phase | Ecosystem skill practiced |
|---|---|
| Data prep | Web scraping, text cleaning, dataset curation, train/val split |
| Fine-tuning | PEFT/LoRA/QLoRA, instruction vs. continued-pretraining framing |
| Quantization | GGUF/AWQ/GPTQ, precision tradeoffs, llama.cpp toolchain |
| Evaluation | Perplexity, stylistic similarity, human eval |
| RAG | Embeddings, vector DBs, retrieval-augmented generation |
| Deployment | Local inference servers (Ollama/llama.cpp), simple UI |

---

## Target model selection (SOTA, small, open-weight)

Pick one **base** model to fine-tune. All are runnable on a laptop after
quantization; pick based on available RAM/VRAM and Italian fluency needs.

> **Updated 2026 (revised):** the project's content is text-first (occasional
> photos, not something generation or RAG need to understand), so a
> multimodal base buys nothing here. The entire **Qwen3.5** generation
> (0.8B/2B/4B/9B/122B) is natively vision-language via early fusion — the
> vision component is baked into pretraining, not a separable bolt-on module
> — so it was dropped in favor of the newest **pure-text** small model,
> **Qwen3-4B-Instruct-2507**, which also scores higher than Qwen3.5-2B on
> most published benchmarks (MMLU-Pro, GPQA, IFEval, etc.). Kept the original
> 2024/2025-era table as history.

| Model | Params | Notes |
|---|---|---|
| **Qwen3-4B-Instruct-2507** (recommended default, 2026) | 4B | Newest pure-text-generation small Qwen (dense, no vision tower); Apache-2.0; outperforms Qwen3.5-2B on most published benchmarks; mature Unsloth/llama.cpp/GGUF support; CPU-feasible after 4-bit quant |
| Qwen3-1.7B | 1.7B | Smaller/faster pure-text fallback if 4B is too heavy for available hardware; noticeably weaker on reasoning/instruction-following than the 4B |
| Qwen3.5-2B | 2B | Multimodal (vision-language, early fusion) alternative — only worth it if the project later needs to *understand* post photos (e.g. image-grounded RAG/eval), not just generate text |
| ~~Qwen2.5-1.5B-Instruct~~ | 1.5B | *(superseded)* original default |
| ~~Qwen2.5-3B-Instruct~~ | 3B | *(superseded)* |
| ~~Llama-3.2-3B-Instruct~~ | 3B | *(superseded)* Meta community license |
| ~~Gemma-2-2B-it~~ | 2B | *(superseded)* |
| ~~Phi-3.5-mini-instruct~~ | 3.8B | *(superseded)* |

**Recommendation:** start with **Qwen3-4B-Instruct-2507** for fast iteration
on a laptop (16GB+ RAM, or any 6GB+ GPU after 4-bit quant); drop to
**Qwen3-1.7B** only if hardware is too constrained for the 4B. Being a plain
text-generation model, no vision-specific loader/handling is needed —
Unsloth's standard `FastLanguageModel` path applies directly.

Given the corpus will be small (a single blog, likely a few hundred short
posts), **do not attempt full fine-tuning** — parameter-efficient fine-tuning
(LoRA/QLoRA) is the only sane approach; full fine-tuning would overfit/forget
catastrophically on so little data.

---

## Data preparation

### Acquisition
- The mirror is static HTML (HTTrack). Options:
  1. **Best**: get raw access to the Ghost content export (Admin → Labs →
     Export content JSON) if you manage the blog — gives clean structured
     JSON (title, html/mobiledoc, tags, published_at) directly, no scraping.
  2. **Fallback** (current reality — export access was lost): a generic,
     config-driven crawler, `scripts/scrape.py`, walks the static mirror by
     following each listing page's `rel=next` link (not a hardcoded
     `page/N/` template, so it isn't tied to this site's current page
     count), collects post permalinks, and pulls fields out of each post
     page via CSS-selector-based queries defined per-site in
     `configs/sites/*.yaml`. See the script's module docstring for the
     query mini-language/config schema, and `configs/sites/README.md` for
     how to adapt it to another site.
  Prefer option 1 — cleaner, includes drafts/tags, avoids HTML boilerplate
  entirely.

### Cleaning
Implemented in `scripts/clean.py`. Strip HTML tags and Ghost `kg-card`
comments, drop embedded images (no alt text/captions on this blog worth
keeping), unescape entities and NFC-normalize unicode, preserve paragraph
breaks (matters for the ~7% of posts that are multi-paragraph dialogue),
and de-dupe exact-duplicate normalized text (keeping the earliest
`published_at`). Output: `data/processed/posts.jsonl` with
`{slug, title, date, tags, text}`.

> **Corpus reality check (from the actual scrape, 403 raw posts, 397 kept
> after cleaning):** this blog is genuinely aphoristic — median post length
> is ~11 words, max ~42. Tags are present on only 14/403 posts, too sparse
> to use for anything (filtering, stratification). Two posts are image-only
> (old Tumblr imports with no text at all, just an `<img>`) — excluded. Four
> more are legacy Tumblr-import "stub" posts whose only visible text lives
> inside an `<a>` tag (a bare cross-reference like "autocit." back to the
> original Tumblr post, no actual aphorism) — also excluded. One exact
> normalized-text duplicate (a repost under a different slug/date) was
> found and de-duped, keeping the earlier instance. Brevity is the style,
> not noise — no length-based filtering of short posts.
- Output is a single dataset shape: a **continued-pretraining corpus** (one
  long text file / JSONL of cleaned post bodies), for causal-LM style
  absorption (captures voice, rhythm, imagery).

> **Decided against synthetic instruction/SFT pairs (see below).** An
> earlier version of this plan called for generating synthetic
> `(prompt, response)` pairs per post to instruction-tune the model. On
> reflection, given this corpus's nature, that was dropped.

### Why no SFT stage
Supervised fine-tuning (SFT) trains a model on explicit `(input, output)`
pairs, teaching it to produce *this* output when given *this* input — it's
how you turn a plain text-completion engine into something that responds to
a request ("write me something about X") instead of just continuing
whatever text it's fed.

The posts here are **spontaneous, self-contained expression, not replies to
a prompt or event** — there is no natural "instruction" a post is answering.
Fabricating one (e.g. asking a model to guess "what theme could have caused
this 11-word fragment") is reverse-rationalization, not recovered fact, and
introduces real risk for a corpus this size and this short:
- Dozens of equally plausible "themes" fit any given aphorism, so the label
  is close to arbitrary/noisy by construction.
- The fine-tuned model would learn the *fabricator's* narrow prompt-phrasing
  style/register (all generated by one model, one template) rather than
  genuine variety.
- It teaches a false causal structure — that these aphorisms are triggered
  by explicit themes — which isn't how the blog actually works, and risks
  making outputs feel more explained/essay-like than the source material's
  unexplained, spontaneous quality.
- Confirmed via the actual scrape: over half the posts (223/403) have
  `title == text` (Ghost auto-derives the title when none is set), so even
  the title can't stand in as a "real" instruction without leaking the
  answer.

**Instead, on-demand generation is handled entirely at inference time**, not
via training data: a fixed persona system prompt (see below) plus a
few-shot handful of real posts included directly in context. Continued
pretraining alone is responsible for teaching voice, rhythm, and imagery;
no fabricated data is introduced. See
[Fine-tuning](#fine-tuning) for how this simplifies training to a single
stage, and [Deployment and serving](#deployment-and-serving) for how the
persona/few-shot prompt is assembled at serve time.

A fixed persona/voice system prompt, used consistently at inference (not
train) time, e.g.:
`"Sei l'autore del blog 'Come me su una slavina': scrivi in italiano, con
tono intimo, immagini oniriche/malinconiche, frasi brevi."`

### Splits & size expectations
- 90/10 or 95/5 train/val split, plain random split with a fixed seed —
  tags are too sparse (14/403 posts) to stratify by, and year-stratification
  wasn't found to add enough value to justify the complexity.
- With a small corpus (few hundred posts), expect val set to be tiny —
  supplement evaluation with held-out qualitative review (see
  [Evaluation](#evaluation)).
- Deduplicate any post reused across CTA/footer boilerplate.

### Deliverables in repo
```
data/
  raw/            # scraped/exported original HTML or JSON, untouched
  processed/
    posts.jsonl        # {slug, title, date, tags, text}
    pretrain.txt        # concatenated cleaned bodies for CPT
    pretrain_train.txt  # after split_dataset.py
    pretrain_val.txt
scripts/
  scrape.py
  clean.py
  split_dataset.py
```

---

## Fine-tuning

### Tooling
**Use Unsloth** as the training framework. It's built on PyTorch +
Hugging Face `transformers`/`peft`/`trl` under the hood (same concepts,
same checkpoint format), but patches attention/kernels for ~2x faster
training and significantly lower memory use — the difference between
"fits on a laptop GPU or even CPU in reasonable time" and "doesn't".
Same Python API style as vanilla `transformers`, so nothing conceptual
is lost by starting here instead of the lower-level stack.

- `pip install unsloth` (pulls in compatible `torch`, `transformers`,
  `peft`, `trl`, `bitsandbytes` versions).
- Load base model via `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`.
- Attach LoRA via `FastLanguageModel.get_peft_model(...)`.
- Train with `trl.SFTTrainer` in its plain-text/CPT mode (no chat template
  or prompt/response split needed — see [Method](#method) for why there's
  only one training stage now).
- Mac/Apple Silicon note: Unsloth currently targets NVIDIA/CUDA (and
  experimental AMD/Intel); on a Mac use `mlx-lm` LoRA fine-tuning instead
  (see
  [Hardware expectations](#hardware-expectations)).

### Method
- **QLoRA**: load base model in 4-bit (`bitsandbytes` nf4), attach LoRA
  adapters (rank 16–32, alpha 32–64, target `q_proj,k_proj,v_proj,o_proj,
  gate_proj,up_proj,down_proj`), train only adapters.
- **Single stage: continued pretraining** on `pretrain_train.txt` (causal
  LM, no prompt/response split), 1–3 epochs, low LR (~1e-4), to soak up
  vocabulary/imagery/rhythm. There is no separate SFT/instruction-tuning
  stage — see
  [Why no SFT stage](#why-no-sft-stage) for the reasoning: this corpus is
  spontaneous, self-contained expression, not replies to a prompt, so
  fabricating instruction/response pairs would introduce reverse-
  rationalized, noisy training data rather than teach a real skill. The
  target experience isn't a chatbot answering requests — it's generating
  novel fragments *in the corpus's style*, optionally seeded with a short
  random word/phrase rather than a themed instruction (see
  [Deployment and serving](#deployment-and-serving)).
- Batch size 1–4 with gradient accumulation to fit laptop VRAM/RAM; sequence
  length 512–1024 is plenty (posts are short).
- Track loss/eval with Weights & Biases or plain CSV logging — keep it
  simple for a learning project.

### Hardware expectations
- Any NVIDIA GPU with 6GB+ VRAM: Unsloth QLoRA, minutes-to-tens-of-minutes
  per epoch at 1.5B–3B — the sweet spot for this project.
- CPU-only: Unsloth requires CUDA, so on a CPU-only laptop fall back to
  plain `transformers`+`peft`+`bitsandbytes` (slow, hours per epoch, but
  workable for a learning project at 1.5B).
- Apple Silicon (MPS): use `mlx-lm` LoRA fine-tuning instead of Unsloth —
  notably faster than PyTorch/MPS on Mac hardware.

### Fine-tuning deliverable
- LoRA adapter weights (small, tens of MB) + merged full-precision model
  checkpoint (`merge_and_unload`) ready for quantization.

### Path to further improve output quality (before publishing)

Current qualitative checks (`eval/EVAL.md`, `eval/quick_style_check.py`)
show the model is not yet satisfying: with a small corpus (~397 posts) and
too many epochs, the model tips from *generalizing style* into
*memorizing/overfitting specific fragments*, and even at reasonable epoch
counts, generation quality is inconsistent. Before publishing anywhere
(even as a learning artifact), it's worth deliberately working through the
levers available, roughly in order of effort/risk:

1. **Re-tune the cheap knobs first**: epochs, learning rate, LoRA rank/alpha,
   and generation-time sampling params (temperature, repetition penalty,
   `min_new_tokens`/`max_new_tokens`, stop strings). These are free to
   iterate on (no new data, no new training code) and the current setup
   hasn't been swept systematically — only a handful of epoch counts were
   tried. Track perplexity **and** qualitative samples per setting in
   `eval/EVAL.md`, since perplexity alone can mask overfitting-driven
   memorization (a model can get *better* perplexity on training-adjacent
   text while getting *worse* at genuinely novel generation).
   - **Done (2026-08-07), generation-side**: swept temperature/
     repetition_penalty in `eval/quick_style_check.py` (samples in
     `eval/EVAL.md`). Finding: lowering temperature *alone* (no repetition
     penalty) made rambling/looping *worse*, not better — sampling at
     lower temperature narrows toward the highest-probability
     continuation, which can reinforce a self-referential loop instead of
     escaping it. `repetition_penalty=1.15` combined with
     `temperature=0.8` gave the most coherent, least repetitive fragments
     among configs tried, and is now the script's default. Still pending:
     training-side sweep (LoRA rank/alpha, learning rate).
2. **Data-side levers before touching the objective**: with such a small
   corpus, per-epoch overfitting risk is largely a data-volume problem.
   Options: augment via paraphrasing/back-translation (risk: dilutes the
   very voice being targeted — use cautiously), dedupe/near-dedupe existing
   posts more aggressively so no near-identical passage gets over-weighted,
   or simply accept a data ceiling and treat this project as bumping into
   the "how much style can you learn from ~400 short texts" limit — itself
   a useful, honest learning-project finding to document.
3. **Supervised refinement loop (rate/pass/reject → curated corpus)**: once
   1–2 are exhausted, generate a batch of fragments from the current
   fine-tuned model, have the blog author rate/pass/reject each one, and
   accumulate an approved corpus over time (store as
   `data/processed/rated_fragments.jsonl`, alongside `posts.jsonl`, with a
   `label: pass|reject` field and generation metadata for provenance).
   Two ways to use this corpus once it's large enough:
   - **Continued SFT on approved fragments only** — simplest: just add the
     pass-rated fragments to the training set for another CPT pass. Treats
     "passed" generations as more of the same distribution to imitate. Easy
     to reason about, but doesn't use the *rejected* examples at all, so it
     wastes half the signal collected.
   - **Preference optimization (DPO/ORPO)** on pass/reject pairs treated as
     chosen/rejected — makes explicit use of *both* signals: pushes the
     model's output distribution away from what got rejected, not just
     toward what got approved. More principled for a "author knows it when
     they see it" quality signal that's hard to write down as a training
     objective otherwise, but needs enough paired examples per
     prompt/context to be stable, and is one more training method to learn/
     debug (worth it here specifically *because* this is a learning
     project).
   Realistically this only becomes worth doing once dozens–low-hundreds of
   rated examples exist — building the rating tool and rated-corpus format
   is cheap and can start immediately, even if actual retraining on it
   waits.
4. **Reconsider model size only as a last resort**: if none of the above
   meaningfully improves felt quality, the ceiling may be intrinsic to
   using a 4B model with a QLoRA adapter on this little data. Moving to a
   full fine-tune (rank-permitting) or a larger base model are valid next
   experiments, but expensive relative to 1–3, so worth trying last.

The guiding principle: don't reach for a new training *method* (DPO,
bigger model) before exhausting cheaper *tuning* and *data* levers on the
current method — the failure mode observed so far (repetitive/rambling,
memorized-feeling output) looks like a hyperparameter/data-volume problem
first, not necessarily an objective-function problem.

---

## Quantization

Goal: produce a model runnable comfortably on commodity hardware (CPU
inference at interactive-ish speed).

1. Convert merged HF model → **GGUF** using `llama.cpp`'s
   `convert_hf_to_gguf.py`.
2. Quantize to a few candidate levels with `llama-quantize`:
   - `Q4_K_M` (best default balance of size/quality),
   - `Q5_K_M` (higher quality, still small),
   - `Q8_0` (near-lossless, for comparison/eval baseline).
3. Optionally also produce an **AWQ** or **GPTQ** 4-bit export if you want
   to compare against GGUF and learn the differences (AutoAWQ / AutoGPTQ).
4. Benchmark tokens/sec and perplexity at each quant level on val set to
   make the size/quality tradeoff concrete (this is a core "learning"
   deliverable — a small table/plot of quant level vs. perplexity vs. speed).

Deliverable: `slaivina-<size>-q4_k_m.gguf` (and siblings) small enough to ship
in the repo release or via Hugging Face Hub model card.

**Cross-platform note:** GGUF is architecture-agnostic — the exact same
file runs on x86 (Linux/Windows) and Apple Silicon via `llama.cpp`/Ollama
(Metal backend on Mac, AVX2/CUDA on x86), so training only needs to happen
once on whichever machine is available. If training was done on Mac via
`mlx-lm` (see [Hardware expectations](#hardware-expectations)), merge the
LoRA adapter back into a standard Hugging
Face checkpoint first — `convert_hf_to_gguf.py` expects that format
regardless of which machine/framework produced it.

---

## Evaluation

Given the subjective/stylistic goal, combine automatic + human eval:
- **Perplexity** on held-out val text (base vs. fine-tuned vs. each quant
  level) — sanity check that fine-tuning helped and quantization didn't
  destroy it.
- **Style-similarity**: embed generated samples and real posts with a
  sentence-embedding model (e.g. `intfloat/multilingual-e5-small`), compare
  cosine similarity distributions (generated-vs-corpus vs. base-model-vs-
  corpus) as a proxy for "sounds like the blog".
- **Human/blind eval**: generate N samples from base vs. fine-tuned model
  given the same prompts; have the blog author (you) or readers blind-rate
  which sound authentic — this is the real ground truth for a style-transfer
  task.
- **Qualitative log**: keep a `EVAL.md`/notebook with side-by-side samples
  across training stages and quant levels.

---

## RAG system (extending content knowledge beyond style)

Style fine-tuning teaches *voice*; RAG lets the model *recall or reference*
actual post content/facts (dates, recurring themes/characters, specific
imagery) without needing it memorized in weights — and lets it stay current
as new posts are published.

### RAG pipeline
1. **Chunking**: each post is already a natural short chunk; optionally
   also chunk by paragraph for longer posts.
2. **Embeddings**: `intfloat/multilingual-e5-small` or
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (both small,
   Italian-capable, CPU-friendly).
3. **Vector store**: start with **ChromaDB** (simplest, embedded, file-
   based, zero-infra) or **FAISS** (lighter, no server) — both fine for a
   few hundred–thousand chunks on a laptop. Qdrant is a good "next step" if
   you want to learn a proper vector DB server.
4. **Retrieval**: top-k (k=3–5) similarity search per user query, optionally
   with a re-ranker (`bge-reranker-base`) if quality needs a boost.
5. **Generation**: feed retrieved chunks + system persona prompt + user
   query into the fine-tuned model (via llama.cpp server or Ollama) to
   produce a grounded, in-style answer.
6. **Orchestration**: keep it simple and inspectable — plain Python
   (`llama-index` or `langchain` optional convenience, or hand-rolled ~150
   lines) rather than a heavy framework, since this is a learning project.

### RAG deliverable
```
rag/
  build_index.py     # embeds posts.jsonl -> chroma/faiss index
  query.py            # CLI: ask a question, get retrieved+generated answer
  server.py           # optional small FastAPI wrapper
```

### Fine-tuning vs RAG: the honest takeaway for a corpus this small

Fine-tuning and RAG aren't really competing solutions to the same problem —
they answer different questions:

- "Should this text **sound like** the author?" → weights (fine-tuning),
  because voice is a distributed property.
- "Should this text **contain specific true content** from the corpus?" →
  context (RAG or full-stuffing), because facts are point properties, not
  distributed ones, and don't survive compression into a model this size
  trained on this little data.

Given that our corpus is small enough to fully fit in context, the marginal
value of RAG's retrieval mechanism specifically (versus just stuffing
everything) is arguably close to zero right now — its main advantage
(scaling past context limits) isn't yet needed. What retrieval buys us
today is smaller prompts/faster inference, at the cost of the
relevance-quality risk already observed in practice (see
`skills/rag/SKILL.md` guardrails). That's a genuine engineering tradeoff,
not a fundamental ML one.

---

## Deployment and serving

The target experience is **not a chatbot** — it's a generator of novel
fragments in the corpus's style, most naturally triggered by something
closer to a "shuffle"/seed button than a conversational turn. Two modes,
simplest first:
- **Unconditional sampling**: empty (or persona-only) prompt, higher
  temperature, let the continued-pretrained model sample freely from what
  it absorbed. This is the most honest reflection of the corpus's
  spontaneous, non-dialogic nature.
- **Seeded sampling**: prepend a short word/phrase as a loose nudge — drawn
  either from the corpus's own vocabulary (e.g. a random noun/phrase pulled
  from `posts.jsonl`) or typed by a user — and let the model continue from
  there. This is a real text-continuation task (not an invented
  instruction/response pair), so it's consistent with the
  [Why no SFT stage](#why-no-sft-stage) decision.
- The persona system prompt (see
  [Why no SFT stage](#why-no-sft-stage)) plus a small number of real posts
  as few-shot context can be included at inference time to anchor
  formatting/register, without needing any training-time instruction data.
- **Serving mechanics are unaffected by this framing** — Ollama/llama.cpp
  still expose a chat-style API underneath; the point is that the
  UI/CLI built on top presents a single generate action, not multi-turn chat.

> **RAG's role needs reconsidering under this framing** (deferred until
> that phase): the RAG pipeline below was designed around "user asks a
> question → grounded answer," which doesn't fit "generate a fragment,
> optionally seeded." It likely still has value — e.g. pulling a real seed
> word/theme from the corpus, or checking a generated fragment against
> similar real posts for eval — but the query-answering framing needs
> revisiting when that phase starts, not assumed as-is.

- **Ollama**: easiest path — `Modelfile` wrapping the GGUF + system prompt,
  `ollama create slaivina -f Modelfile`, then `ollama run slaivina`.
- **llama.cpp server** (`llama-server`): OpenAI-compatible local API,
  useful for the RAG script to call over HTTP.
- **UI options** (pick one, optional):
  - `open-webui` pointed at Ollama for a full chat UI with zero code,
  - a minimal Gradio/Streamlit app for prompt → generated post, with a
    toggle for "RAG-grounded" vs "pure generation".

### Docker versus Hugging Face Hub for distribution

Question: should the scraper and/or the trained model be packaged and
shipped as a Docker image, instead of (or in addition to) a Hugging Face
Hub upload? Conclusion: **use each for what it's good at, don't pick one
for everything.**

- **Model weights → Hugging Face Hub, not Docker.** Weights (LoRA adapters,
  merged checkpoints, GGUF quants) are large binary artifacts that need
  versioning, dedup, and easy fetch by whatever inference stack someone
  chooses (`transformers`, `llama.cpp`, Ollama, vLLM...). HF Hub is built
  for exactly this (git-lfs storage, model cards, per-file quant variants —
  see how `unsloth/*-GGUF` repos already do this). Baking weights into a
  Docker image instead would bloat the image, duplicate what HF already
  solves, and lock consumers into pulling a whole container just to get a
  file. This matches what §4 Quantization already recommends (ship GGUF via
  release assets or HF Hub).
- **Docker → for *serving*, not for *distributing weights*.** A
  `deployment/docker-compose.yml` bundling `llama-server` (or Ollama) + the
  optional RAG API (`rag/server.py`) is genuinely useful: one-command,
  reproducible local deployment, independent of the host's Python/OS setup.
  The compose file would reference/download the model from HF Hub at
  container start (or mount a locally-quantized file), rather than embedding
  weights in the image itself.
- **The scraper doesn't need Docker either.** It's a small pure-Python
  script with light dependencies (`requests`, `beautifulsoup4`, `PyYAML`,
  `python-dateutil`); a venv managed via `uv` (`pyproject.toml`/`uv.lock`)
  is already fully portable.
  Docker would only earn its keep here if run unattended/scheduled outside
  a CI runner that already provides Python (e.g. a bare-metal cron box) —
  a secondary concern, not a reason to containerize the whole project.

### CI and CD plan

**Constraint that shapes this plan**: no self-hosted runner is available —
only GitHub-hosted runners (no GPU, ~14GB free disk on `ubuntu-latest`).
Training and quantization (7.5GB f16 GGUF intermediate alone) must
therefore stay a **local, manual step on this machine**, same as today.
CI/CD's job is to pick up cleanly *after* a quantized artifact already
exists locally, not to reproduce the training/quantization pipeline itself.

**Why not conventional per-push CI**: this repo has sparse commit cadence
and no dense, fast-changing test surface — Dependabot already covers
dependency/vulnerability scanning, and GitHub code scanning (if enabled)
covers static analysis. Inventing a synthetic test suite just to have
something for a CI workflow to run would be process for its own sake. A
cheap `ruff check` lint gate is reasonable to keep (near-zero cost, catches
real style/error issues), but there's no separate `pytest` step planned
unless real tests emerge naturally out of future script changes.

**Why not semantic-release for versioning**: `semantic-release` infers
version bumps from conventional-commit messages across *all* commits since
the last release — but commits here mix unrelated concerns (RAG glue,
docs, scraper fixes, training config) that have nothing to do with "did the
model change." A model's meaningful version axis is "did the weights
change, and how much" (new base model = major, retrain/new data = minor,
requant only = patch) — a judgment call tied to an actual training/quant
event, not something safely inferred from commit prose. A stray `fix:`
commit to `rag/query.py` would falsely trigger a "new model version" under
default semantic-release rules.

**Chosen design**: a **GitHub Release** (manually tagged, e.g.
`model-v1.2.0`) is both the artifact-storage mechanism *and* the versioning
trigger, published by hand right after a local quantization run you're
happy with. This keeps versioning deliberate (matches how sparse/manual
training runs already are) while staying fully GitHub-hosted-runner-safe,
since the workflow only ever needs to download an already-produced release
asset (~2.3GB GGUF), never train or quantize anything itself.

**Conventional lint-only CI was considered and dropped**: Dependabot
already covers dependency/vuln scanning, and a `ruff check` gate adds
little given the sparse, low-risk commit cadence here — not worth a
dedicated workflow. Run `uv run ruff check .` locally/on-demand instead if
desired.

**Scheduled data-refresh automation was considered and dropped**: a
weekly-cron `refresh-data.yml` (re-running `scripts/scrape.py --refresh` +
`clean.py` + `make_sft_pairs.py`, pushing results to a private HF dataset
repo) is technically feasible on GitHub-hosted runners, but doesn't fit
this project's actual cadence — the blog is not updated frequently enough
to justify scheduled re-scraping, and training itself only happens as a
deliberate, infrequent, manual decision (per the versioning convention
above), so pre-fetching data on a fixed schedule has no consumer to serve.
It would add a recurring workflow, an extra `HF_TOKEN` exposure, and a
second HF dataset repo to maintain, for a problem a one-line local command
already solves. Instead: run
`scripts/scrape.py --refresh && scripts/clean.py && scripts/make_sft_pairs.py`
locally as step 0 whenever a new training run is starting. Revisit
automation only if the blog's publishing pace picks up enough to make
manual refresh genuinely burdensome.

#### Model publish (GitHub-hosted, triggered by GitHub Release) — `.github/workflows/publish-model.yml`
- **Local step (manual, on this machine, scripted)**: train →
  merge → `quantize/convert_and_quantize.sh` → benchmark
  (`eval/EVAL.md`) → when happy with the result, run
  `quantize/create_release.sh <version>` (new script, see below) to tag
  and publish the GitHub Release with the chosen GGUF quant(s) attached.
  This is the deliberate "is this good enough to publish" gate — same
  judgment call as today, just now also the versioning decision, and
  scripted rather than done by hand through the GitHub UI each time.
- **Automated step, triggered `on: release: types: [published]`**: the
  workflow downloads its own release's assets (`gh release download` or
  the `release` event payload's asset URLs — no repo-clone-sized checkout
  needed beyond `eval/EVAL.md`/templates), renders the model card (base
  model, LoRA config, quant benchmark table, licensing/ethics note per
  [Licensing, ethics, attribution](#licensing-ethics-attribution),
  release tag as the HF Hub revision/version), and uploads the GGUF(s) +
  LoRA adapter (also attached to the release, or fetched from wherever it
  lives) + model card to the HF Hub model repo via `huggingface_hub`
  (`HF_TOKEN` stored as a **GitHub Actions repo secret** — safe here since
  the job only reads a release asset, no model weights or GPU work happen
  on GitHub's side beyond a file transfer).
- Net effect: creating the GitHub Release *is* "ship this version" — no
  separate manual HF upload step, no self-hosted runner, no
  commit-message-driven guessing about whether a new model version exists.

#### Release creation script/procedure — `quantize/create_release.sh`
Rather than tagging and attaching assets by hand through the GitHub UI
each time, a small script wraps the `gh` CLI so releasing a model version
is a single repeatable command:
- Usage: `quantize/create_release.sh <version> [gguf-path...]` (e.g.
  `quantize/create_release.sh v1.2.0 quantize/output/slaivina-4b-q4_k_m.gguf`).
- Steps the script performs: validate the version string (semver-like,
  `vMAJOR.MINOR.PATCH`), confirm the given GGUF file(s) exist and print
  their size for a sanity check, `git tag model-<version>` +
  `git push origin model-<version>`, then
  `gh release create model-<version> <gguf-path...> --title ... --notes ...`
  (notes can pull the latest benchmark line from `eval/EVAL.md`
  automatically, or accept a `--notes-file`).
- Requires the `gh` CLI authenticated locally (already the case for a repo
  owner) — this script only ever runs on this machine, never in CI; the
  GitHub Release it creates is what *triggers* `publish-model.yml`.
- Version-numbering convention (manual judgment call, not automated): major
  = new base model, minor = retrain/new data/method (e.g. a DPO pass),
  patch = requantization only (same underlying weights).

#### Deployment stack build (GitHub-hosted, on tag/release) — optional, later
- Build/publish the `deployment/docker-compose.yml` serving stack
  (`llama-server`/Ollama + optional RAG API) — safe on GitHub-hosted
  runners since weights are pulled from HF Hub at container start rather
  than baked into the image (see
  [Docker versus Hugging Face Hub for distribution](#docker-versus-hugging-face-hub-for-distribution)).
  Lower priority; only worth doing once the serving stack itself exists.

#### Summary of what runs where
| Step | Runner | Trigger | Touches model weights? |
|---|---|---|---|
| `quantize/create_release.sh` | local (this machine) | manual, run after a happy benchmark | reads local GGUF, no training |
| `publish-model.yml` | GitHub-hosted | GitHub Release published (via the script above) | reads a release asset, no training/quantization |
| deployment stack build | GitHub-hosted | on tag (later) | no (pulls from HF Hub) |

---

## Repository layout (proposed)

```
slaivina/
  PLAN.md                  <- this file
  README.md                <- quickstart for contributors
  data/                     (raw/ processed/, gitignored except samples)
  scripts/                  (scrape, clean, dataset building)
  training/
    stage_a_pretrain.py
    stage_b_sft.py
    configs/*.yaml
  quantize/
    convert_and_quantize.sh
  eval/
    perplexity.py
    style_similarity.py
    EVAL.md
  rag/
    build_index.py
    query.py
    server.py
  deployment/
    Modelfile
    docker-compose.yml (optional: llama-server + open-webui)
  pyproject.toml / uv.lock
```

---

## Contributor how-to (quickstart, to expand in README.md)

1. **Setup**: `uv sync` (installs from `pyproject.toml`/`uv.lock` into a
   `.venv`; add heavier phase-specific deps such as transformers/peft/trl/
   bitsandbytes/accelerate/datasets/sentence-transformers/chromadb to
   `pyproject.toml` as those phases are implemented). Run scripts with
   `uv run python scripts/...` or activate `.venv` directly.
2. **Get data**: either drop a Ghost content-export JSON into `data/raw/`,
   or run `scripts/scrape.py` against the mirror URL.
3. **Clean & build datasets**: `python scripts/clean.py` then
   `python scripts/make_sft_pairs.py`.
4. **Fine-tune**: `python training/stage_a_pretrain.py --config
   training/configs/qwen2.5-1.5b.yaml` then `stage_b_sft.py`.
5. **Merge + quantize**: `bash quantize/convert_and_quantize.sh
   <merged-model-dir>`.
6. **Evaluate**: `python eval/perplexity.py` and
   `python eval/style_similarity.py`.
7. **Build RAG index**: `python rag/build_index.py`.
8. **Run locally**: `ollama create slaivina -f deployment/Modelfile &&
   ollama run slaivina`, or `python rag/query.py "tema: la nebbia"`.
9. Contributions welcome: more posts (as the blog grows), better synthetic
   prompts, additional eval prompts, alternate base models.

---

## Licensing, ethics, attribution

- You manage the blog, so using its content for this fine-tune is your
  call; still worth stating clearly in the repo:
  - Data license/terms for `data/` (e.g. "content © the blog author, used
    with permission for this personal ML project, not for redistribution
    as training data by third parties without consent").
  - Model card noting it's a small personal-style-transfer experiment, not
    a factual/authoritative source, and generations are AI-authored
    pastiche, not the real author's words — label outputs accordingly if
    ever shared publicly.
  - Respect the base model's license (Apache-2.0 for Qwen2.5/Gemma;
    Meta's community license for Llama) when redistributing fine-tuned
    weights.

---

## Suggested milestones

1. Data export/scrape + cleaning pipeline working end-to-end (small win,
   validates whole pipeline early).
2. Stage A continued-pretraining run + qualitative check ("does it sound
   Italian/poetic at all now?").
3. Stage B SFT run + first side-by-side base-vs-tuned comparison.
4. Quantize to GGUF, benchmark size/speed/perplexity tradeoffs. **Done**
   2026-08-05: converted the merged fine-tune to GGUF and quantized to
   Q4_K_M/Q5_K_M/Q8_0 via `quantize/convert_and_quantize.sh`; benchmarked
   perplexity + tokens/sec for each (see `eval/EVAL.md`); kept Q4_K_M as
   the shipped default (best size/speed, ~5.6% perplexity gap vs. Q8_0).
5. RAG index + retrieval-grounded generation demo. **Done** 2026-08-06:
   `rag/build_index.py` embeds `data/processed/posts.jsonl` (one chunk per
   post) with `intfloat/multilingual-e5-small` into a persistent ChromaDB
   collection (`rag/index/`, gitignored); `rag/query.py` retrieves top-k
   chunks and generates a grounded answer via a running `llama-server`
   serving the Q4_K_M quant. End-to-end smoke-tested successfully (see
   skills/rag/SKILL.md for the run commands).
6. Package as Ollama model + simple UI; write up README + learnings.
7. CI/CD: `quantize/create_release.sh` to tag/publish a GitHub Release with
   the quantized GGUF, triggering a GitHub-hosted `publish-model.yml`
   workflow that uploads it + a rendered model card to HF Hub — see
   [CI and CD plan](#ci-and-cd-plan). (Lint-only CI and scheduled data
   refresh were considered and deliberately dropped — see that section.)
8. Publish: HF Hub model page (model card with usage instructions,
   benchmark table, licensing/ethics note) + a repo-level review pass on
   instructions for using the other tools in this repo (scraper, training,
   quantization, eval, RAG) so the project is usable end-to-end by someone
   else, not just as a personal working log.
