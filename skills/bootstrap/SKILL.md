---
name: repo-bootstrap
description: Use when setting up agent-facing documentation and skills for a new or existing repository — producing/reviewing a plan file, an AGENTS.md, and phase-based skill scaffolds from scratch.
---

# Repo bootstrap (meta-skill)

Generalized checklist for bootstrapping AI-agent-facing documentation in any
repository, distilled from bootstrapping this one (see `../../PLAN.md` /
`../../AGENTS.md` for the resulting artifacts).

This is a **meta-skill**: it describes how to run the bootstrap procedure
itself, not a project phase. Prefer it when starting a new repo, or when an
existing repo has ad-hoc docs that need to be reorganized into this pattern.

## How-to

1. **Scan before writing anything.** Check the repo root (and `.github/`)
   for files that collide with conventions AI agents auto-load as
   instructions: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
   `.github/copilot-instructions.md`, `.github/instructions/**/*.instructions.md`.
   If an existing file has that name but different intent (e.g. a plan/spec
   mistakenly named `agents.md`), rename it before adding new content —
   don't let a spec and agent-instructions content mix in one file.

2. **Produce/confirm the plan file** (`PLAN.md` or similar, *not* one of the
   reserved names above):
   - Capture goals, starting point, constraints, and a phased roadmap.
   - Use plain descriptive `##`/`###` headers with **no leading numbers**
     and **no slashes, ampersands, or other punctuation that makes anchor
     slugs ambiguous** (e.g. write "Deployment and serving", not
     "Deployment / serving") — this keeps `#anchor` links stable even as
     content is reordered or reworded elsewhere.
   - Review it once drafted: does the name match the content's actual
     purpose? Does it reference itself consistently (e.g. its own proposed
     repo-layout section should name itself correctly)?

3. **Author a short, separate `AGENTS.md`.** Keep it operational, not
   narrative:
   - One-line project summary + link to the plan file for full rationale.
   - Target repo layout (mark clearly what doesn't exist yet).
   - Hard conventions (languages, what never gets committed, preferred
     techniques, default choices) with anchor links back to the plan file's
     relevant section for rationale — never embed section *numbers* in the
     cross-reference text.
   - A git-usage rule if the agent should not perform mutating git actions
     (state explicitly which commands are read-only-allowed vs. forbidden).
   - A working agreement: the plan file is living and must be updated
     alongside behavior/approach changes; skills hold reusable per-phase
     instructions.

4. **Scaffold `skills/`.** One `SKILL.md` per reusable/repeatable phase of
   work (not per file or per tiny task):
   - `skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`
     written in third person, stating *when* to invoke it) followed by
     concrete imperative steps and guardrails.
   - Each skill links back to its plan-file section via an anchor link with
     descriptive text (`[PLAN.md — Section title](../../PLAN.md#anchor)`),
     not a section number.
   - Add a `skills/README.md` documenting the convention and a table
     mapping skills to plan-file sections/anchors.
   - Only add a skill once a phase has enough concrete, repeatable steps to
     justify it — don't scaffold speculative skills with no content.
   - Distinguish **structural facts** (file paths, script names, order of
     operations) from **volatile decisions** (model/library names, specific
     hyperparameters, dataset choices). Structural facts are safe to state
     directly in a skill; volatile decisions should only ever be linked to
     their plan-file section, never restated — they're exactly the content
     most likely to change and hardest to keep in sync across files once
     duplicated. (Caught in this repo: a model name and several
     hyperparameters/embedding-model choices had drifted out of sync across
     three skill files after being restated instead of linked.)

5. **Validate discovery, don't assume it.** Confirm the agent environment
   actually picks up `AGENTS.md` and the new skills (e.g. via `/env` or
   `/skills` in Copilot CLI, or the equivalent for the agent in use) before
   treating the scaffold as functional. If verification isn't possible
   in-session, say so explicitly rather than asserting it works.

6. **Rely on git for history, not a side-log.** Don't maintain a separate
   `bootstrap.md`/process-log file — commit messages and PR descriptions
   are git's job and avoid a second, driftable record. If a durable lesson
   generalizes beyond this one repo, fold it back into this meta-skill
   (see Guardrails) instead of narrating it in prose elsewhere.

7. **Hand off, don't commit.** If the agent's git permissions are
   read-only (see `AGENTS.md`'s git-usage rule, if present), leave all
   changes unstaged for human review — do not `git add`/`commit`/`push` on
   the user's behalf.

## Guardrails

- Don't duplicate the plan file's content into `AGENTS.md` or skills —
  link to it. Duplication drifts; links stay accurate as long as anchors
  are stable (see step 2's header-naming rule).
- Don't scope discovery/search commands beyond the repo when investigating
  environment or tooling conventions — stay in-repo or ask the user first.
- Treat this meta-skill itself as living: if a bootstrap run surfaces a new
  generalizable lesson, fold it back in here.
