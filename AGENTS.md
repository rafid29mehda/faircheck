# AGENTS.md — orientation for any session working on FairCheck

Read this first, then `PLAN.md` (the agreed plan and milestone checklist) and
`docs/DECISIONS.md` (why things are the way they are). `CHANGELOG.md` records what
changed per milestone.

---

## What this project is

**FairCheck** — a web app plus pure-Python library that takes a CSV of classification
predictions and returns a correct, uncertainty-aware fairness report.

It is an **applied portfolio project** for CS PhD applications (bias and fairness in ML).
The audience is a professor who opens it for two minutes. That makes **correctness and
honest communication rank above visual polish and feature count**. Concretely:

- never print a number the code did not compute;
- never state a disparity more confidently than the confidence interval supports;
- never show a legal-sounding conclusion (the four-fifths rule is a screening heuristic).

## Current state

| | |
|---|---|
| Phase | 2 (execution). Plan approved 2026-09-21. |
| Milestone | **M1–M5 complete** (core, report/CLI, four bundled examples). **M6 next** (Streamlit app). |
| Tests | 239 passing, 98% coverage on `faircheck/` |
| Quality gate | `ruff check` + `ruff format --check` + `mypy` + `pytest` all green |
| Not started | `app.py`, `README.md`, `CITATION.cff` |
| Deployment | **Deliberately deferred** — the user asked not to think about hosting or a GitHub remote yet. Do not create a remote or deploy without being asked. |

## How to work here

**One milestone at a time.** After each milestone: run the tests, show the user real
output, tick the box in `PLAN.md`, add a `CHANGELOG.md` entry, commit, and **stop for
their go-ahead**. Do not run ahead to the next milestone unprompted.

If research during execution shows the plan is wrong, stop, explain, propose a revision.

## Commands

```bash
# The dev environment already exists (Python 3.12.7, gitignored).
.venv/bin/python -m pytest -q                  # tests
.venv/bin/python -m pytest --cov=faircheck     # tests + coverage
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                                 # strict-ish: untyped defs are errors

# Recreate the environment from scratch if needed. NOTE: `python -m venv` inside the
# workspace needs to run outside the agent sandbox (it creates symlinks).
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e . --no-deps
```

Tests must pass **offline**. Never add a test that downloads data; example CSVs are
committed for exactly this reason.

## Architecture invariants

These are enforced by tests, not just convention — breaking one turns CI red.

1. **`faircheck/` contains no UI code.** No `import streamlit` / `gradio` / `altair` /
   `matplotlib` anywhere under it. `tests/test_core_has_no_ui.py` greps for this.
   `app.py` and the CLI are thin callers with zero metric logic.
2. **Fairlearn is a test oracle, not a runtime dependency.** It appears only in
   `requirements-dev.txt` and in tests. See DECISIONS.md D2.
3. **Undefined rates are `NaN`, rendered `n/a`.** A rate with a zero denominator (TPR for
   a group with no positives) must never be `0.0`, and a gap involving a `NaN` must stay
   `NaN`. This is the single most important behavioural rule in the project; Fairlearn
   returns `0.0` here and we deliberately differ. See DECISIONS.md D3.
   *Rendering trap:* pandas ignores `float_format` for `NaN` and uses `na_rep`, so a table
   built with `to_string`/`to_markdown` prints "NaN" unless `na_rep="n/a"` is passed.
4. **Metrics are shape-generic.** Every rate takes an array whose last axis is the four
   confusion cells `(TP, FP, FN, TN)`; every gap reduces over the group axis. So the same
   formulas serve the point estimate `(n_groups, 4)` and the bootstrap
   `(n_boot, n_groups, 4)`. Do not write a second copy of any metric for the bootstrap.
5. **Registries, not repetition.** `metrics.RATES` and `metrics.GAPS` carry each metric's
   label, formula, reference and inputs. The report and UI iterate over these; they must
   not restate a formula or a citation inline.
6. **Verdicts come from signed contrasts, never from max−min gaps.** A max−min gap is
   non-negative by construction, so its CI excludes zero even under no disparity. Use
   `BootstrapResult.has_clear_contrast`, which can return `None`. See DECISIONS.md D11 —
   this is the second most important behavioural rule after invariant 3.

## Conventions

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`.
- Type hints everywhere; `mypy` has `disallow_untyped_defs = true` (tests included).
- Line length 100. `pytest` runs with `filterwarnings = ["error"]`, so a new warning
  fails the suite — fix the cause rather than filtering it.
- Comments explain *why* / state constraints. Do not add comments narrating what the next
  line does.
- Tests are built from stated confusion-matrix cells wherever possible, so expected values
  can be verified by hand and a failure points at the formula rather than the fixture.
- Error messages are user-facing product copy: say what is wrong, in which column, and
  what to do. `tests/test_validate.py` asserts on their content.

## Verified facts (checked 2026-09-21, don't re-derive from memory)

- Pinned and confirmed installing together on Python 3.12 **and** 3.13: `fairlearn==0.14.0`,
  `streamlit==1.64.0`, `numpy==2.5.3`, `pandas==3.0.6`, `scikit-learn==1.9.1`, `altair==6.3.0`,
  `fpdf2==2.8.8` (pure-Python PDF, D6).
- `numpy` 2.5.x needs Python >= 3.12, so the CI matrix is **3.12 and 3.13** (not 3.11).
- Fairlearn's `MetricFrame` reproduces the hand-computed case in `tests/conftest.py` exactly.
- Fairlearn's `true_positive_rate` returns `0.0` for a group with no positives. This is the
  documented reason for invariant 3.
- Hugging Face Spaces now requires a **paid plan** for Gradio/Docker (compute) Spaces, and
  the native Streamlit SDK was deprecated in April 2025. Streamlit Community Cloud is the
  free path. Relevant only when deployment is un-deferred.
- Chouldechova's identity `FPR = p/(1-p) · (1-PPV)/PPV · (1-FNR)` is asserted in
  `tests/test_identities.py`; it is also the engine for the impossibility explainer.

## Key files

| Path | Role |
|---|---|
| `PLAN.md` | Agreed plan, research notes with links, milestone checklist. The contract. |
| `docs/DECISIONS.md` | Numbered decisions with rationale. Cite as `D<n>` in code comments. |
| `CHANGELOG.md` | What shipped per milestone. |
| `faircheck/types.py` | Cell order constants, `ColumnMapping`, `ValidatedData`, `GroupCounts`. |
| `faircheck/metrics.py` | Confusion counts, per-group rates, gaps, `RATES`/`GAPS` registries. |
| `faircheck/validate.py` | Input checks and the user-facing error/warning copy. |
| `faircheck/calibration.py` | Per-group ROC-AUC, reliability bins, positive-class ECE, threshold sweep. |
| `faircheck/impossibility.py` | Chouldechova identity + equal-PPV / equal-error counterfactuals. |
| `faircheck/guidance.py` | Metric-selection helper + hedged plain-language summary (D7, D11). |
| `faircheck/report.py` | Audit orchestrator; Markdown report; derived PDF via fpdf2 (D6). |
| `faircheck/__main__.py` | CLI: `python -m faircheck report ...` — thin caller, no metric logic. |
| `examples/` | Seeded generators + committed CSVs (ACS Income, Adult, COMPAS-like, near-fair). |
| `tests/conftest.py` | `cells_to_frame` + the hand-computed case. Build fixtures from cells. |
