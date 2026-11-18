# AGENTS.md: rules for EVERY coding agent (Claude Code, Antigravity/Gemini, Codex, OpenCode, Cursor, local models)

You are working on **FoodLens**: an ML model that estimates how long raw chicken has been at room temperature from photos taken under red/green/blue LED light in a dark box. The **ML model plus an honest evaluation is the graded deliverable**. Apps are optional and come later.

These rules are mandatory. If a rule conflicts with speed, the rule wins.

## 0. Start of every session (do these in order, do not skip)
1. Read `plan/PROGRESS.md` (where we are) and `plan/LESSONS.md` (mistakes to avoid).
2. Read **only** the phase file named in the handoff prompt: `plan/phases/P<n>_*.md`.
3. Read the skill files that phase lists in "Skills to load" (`.claude/skills/<name>/SKILL.md`). Agents other than Claude must open these files manually.
4. Run `git status` and `git log --oneline -5`. If the tree is dirty with work you did not do, **stop and ask the user**.
5. Run `bash scripts/check.sh` (once it exists, from Phase 0) and confirm it passes **before** changing anything. If it fails, fixing it is your first task.
6. Write a 5-line plan for this session in your reply. Then work.

Do **not** read the whole repo, the raw images, or docs you were not told to read. Context is limited and expensive.

## 1. Scientific rules (never break these)
- **Never modify files in `experiment/`** (raw data, read-only). All derived files go to `data/` (gitignored).
- **No data leakage:** train/test splits are always grouped by `session_id` (and `piece_id` once known) using `foodlens.evaluation.splits`. Never use a random frame-level split except inside the clearly labelled leakage demo.
- Feature scaling, feature selection, hyper-parameter tuning, thresholds and OOD statistics are fit **inside each training fold only** (use sklearn `Pipeline`).
- Never tune on the test fold. Nested CV or a fixed config only.
- Every reported metric must come with: the grouped CV scheme, n groups, 95 % CI (bootstrap over groups), and a baseline. Numbers in docs and reports are produced by code into `reports/`, never typed by hand.
- Do not claim bacteria are counted. Bright spots are **glints** (specular reflections) used as moisture features. See `docs/01_RESEARCH_AND_RATIONALE.md` §3.
- If a result looks too good (>98 %), assume leakage and investigate before reporting.

## 2. Engineering rules
- Python 3.12, environment managed by **uv** in `.venv` (`uv sync`). Never `pip install` globally. Never install a model into `.venv` that already exists on the machine (see `docs/06_APP_DESIGN_OPTIONAL.md` §3).
- Code lives in `src/foodlens/` (package). Notebooks are generated from `notebooks/src/*.py` (jupytext percent format) by `scripts/build_notebooks.sh`. **Never hand-edit `.ipynb`.**
- **File size limits:** a module ≤ 300 lines, a function ≤ 50 lines, a notebook source ≤ 400 lines. If you need more, split the module. If a file you must write is long, write it in **several smaller edits/chunks**, never one giant generation.
- One responsibility per module. Pure functions where possible; config passed in, not read from globals. Type hints on public functions. Docstrings state units (seconds, °C, 0–255).
- All paths, thresholds and parameters come from `configs/default.yaml` via `foodlens.config`. No magic numbers in code; add them to config with a comment.
- Errors: raise typed errors (`foodlens.errors`) with a code like `E_DARK_FRAME`. **Never** `except Exception: pass`. Never return fake/default values silently. Log a reason for every rejected frame.
- Determinism: set seeds from config; sort file lists; record versions in outputs.
- Every new function with logic gets a **pytest** test using tiny synthetic images (`tests/fixtures/`), not the real dataset. Tests must run in < 60 s total.
- Do not add a dependency without adding it to `pyproject.toml` via `uv add` and noting why in the session log.

## 3. Long-running commands: WAIT for them, but watch for errors
- Commands such as `scripts/check.sh`, feature extraction or training may take **5–20 minutes. That is normal. Do not skip them, do not assume they pass, do not start the next step before they finish.**
- Run long jobs with output to a log file: `bash scripts/check.sh 2>&1 | tee logs/check_latest.log`.
- If your tool returns before the command ends, **poll** the log (`tail -n 30 logs/<file>.log`) every 30–60 s until you see the final marker line (`CHECK RESULT: PASS`, `CHECK RESULT: FAIL`, or `STAGE DONE`).
- While waiting, scan the new log lines for `Traceback`, `Error`, `FAILED`, `nan`, `Killed`. If one appears, **stop the job**, fix the root cause, and rerun from the start of that stage.
- A command counts as successful only with exit code 0 **and** the success marker. Report the actual last lines.
- Before running a stage on all 944+ images, run it with `--limit 20` first.

## 4. End of every session (mandatory checklist)
1. `bash scripts/check.sh` passes (ruff, pytest, notebook build if touched). Paste the final 5 lines in your reply.
2. If you changed a pipeline stage, run it with `--limit 20` and check the output file exists and looks sane (row counts, no NaN columns).
3. Update `plan/PROGRESS.md`: tick tasks, update "Current state", and add a session entry (format inside the file). Keep the **last 5 sessions in full**; compress older ones to **one sentence** each under "Older sessions".
4. Add any new pitfall or insight to `plan/LESSONS.md` (one line each, dated).
5. Write the full session log to `plan/sessions/S<NN>_<yyyy-mm-dd>_<phase>.md` (what was done, files touched, commands run with results, open issues).
6. `git add -A && git commit -m "<phase>: <summary>"` (conventional message). **Commit messages and titles must contain NO reference to Claude, Anthropic, AI tools or agents: no `Co-Authored-By` trailers, no "Generated with" lines.** Push only if the user asked for it.
7. Generate the **next-session prompt** from `plan/HANDOFF.md` with the placeholders filled in, and print it as the last thing in your reply.

## 5. When unsure
- If the task is ambiguous or needs a scientific decision (label definition, thresholds that change results), write the question to `plan/PROGRESS.md` → "Open questions" and ask the user. Do not guess silently.
- Never delete data, models or reports without asking.
- Prefer the simplest approach that is correct. This is a 7th-semester B.Tech project: no Kubernetes, no microservices, no MLOps platforms.
