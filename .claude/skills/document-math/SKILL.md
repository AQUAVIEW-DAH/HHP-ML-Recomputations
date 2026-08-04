---
name: document-math
description: Write or update the mathematical documentation for an experiment or analysis script in docs/math/, then commit it. Use whenever an experiment/analysis lands (new script, new figure family, new evaluation), when the math of an existing pipeline changes, or when the user runs /document-math.
---

# Math documentation skill

Purpose: the user runs sessions in bypass-permissions mode and does not see the
math details as they are implemented. Every experiment's mathematics must land
in `docs/math/` in the same commit series as the code, so the user (and the
mentor) can review the exact formulas on GitHub, where `$...$` / `$$...$$`
LaTeX renders natively.

## Workflow

1. Identify the script(s) whose math needs documenting. Read the actual code —
   document what is computed, not what was intended.
2. Write or update `docs/math/<topic>.md` with this structure:
   - **Provenance header**: script path(s), output locations, date, and the
     figure/CSV names the math produces.
   - **Setup**: data subset, filters, sample sizes, model and hyperparameters,
     fold/protocol definition. State every constant with its value.
   - **The math**: each computation as a display equation (`$$...$$`) with all
     symbols defined in plain language, in the order the pipeline applies them.
     Derived quantities show their formula, not just their name.
   - **Interpretation notes**: what each quantity does and does not mean;
     known caveats (e.g., greedy path-dependence, training-distribution
     conditioning).
3. Keep one file per experiment/analysis family; update in place when the math
   changes (git history is the version trail). Never fork dated copies.
4. Update the index table in `docs/math/README.md` (file, one-line scope,
   status).
5. Commit `docs/math/` together with (or immediately after) the code commit it
   documents, and push.

## Style rules

- GitHub-flavored Markdown with native LaTeX math. No raw `.tex` files, no
  build pipeline.
- Define every symbol at first use. Prefer full words over single letters for
  project-specific quantities (e.g. `MAE(A)`).
- State units everywhere (m, kJ/cm², s⁻²).
- Plain language between equations: each equation gets a sentence saying what
  it is for.
- Where the code and a cited paper differ (thresholds, criteria), say so
  explicitly.
