# P0: Foundation (1 session)

**Goal:** a reproducible Python project skeleton so that every later phase only adds modules and tests.
**Skills to load:** `session-protocol`, `long-running-commands`, `notebook-authoring`.
**Inputs:** none besides the repo. **Do not touch `experiment/` or `data-collection-code/`.**

## Tasks

### P0.T1: Environment with uv
- Check `uv --version` (install only if missing: `brew install uv`, and record it in `scripts/uninstall_global_tools.sh`).
- `uv init --package --name foodlens --python 3.12` (keep the existing README; if `uv init` overwrites anything, restore it from git).
- `.python-version` = `3.12`.
- `uv add numpy pandas pyarrow pillow opencv-python-headless scikit-image scikit-learn scipy statsmodels matplotlib seaborn pydantic pyyaml joblib tqdm requests typer rich`
- `uv add --dev pytest pytest-cov ruff mypy jupytext nbclient ipykernel nbformat nbstripout`
- Do **not** add torch/timm yet (P7A adds them).
- **Accept:** `uv run python -c "import foodlens, cv2, sklearn; print('ok')"` prints ok.

### P0.T2: Package layout
Create the empty modules with docstrings exactly as in `docs/03_ARCHITECTURE.md` §6 (`io/`, `qc/`, `vision/`, `features/`, `labels/`, `evaluation/`, `models/`, `inference/`, plus `config.py`, `errors.py`, `logging_utils.py`, `viz.py`, `cli.py`). Each sub-package gets `__init__.py`.
- `errors.py`: `class FoodLensError(Exception)` with attributes `code: str`, `detail: str`; subclasses `DataError`, `QCError`, `ModelError`, `ConfigError`. Codes are string constants in one place (`E_DECODE`, `E_NO_TIME`, `E_DARK_FRAME`, `E_SCENE_CHANGE`, `E_NO_MEAT_FOUND`, `E_OOD`, `E_TEMP_UNAVAILABLE`, `E_MODEL_CAPABILITY`, `E_CONFIG`).
- `logging_utils.py`: `get_logger(name)` using `rich` handler; level from env `FOODLENS_LOG`.
- `viz.py`: `save_fig(fig, name)` → `reports/figures/name.png` (300 dpi) + `.svg`; `LIGHT_COLORS = {"RED":"#d62728","GREEN":"#2ca02c","BLUE":"#1f77b4"}`.

### P0.T3: Config
- `configs/default.yaml` with sections: `paths` (raw_dir=experiment, data_dir=data, reports_dir=reports, models_dir=models), `seed: 42`, `n_jobs: -1`, `location` (lat 30.765, lon 76.786, tz Asia/Kolkata), `qc`, `roi`, `glint`, `vision`, `triplet`, `cv`, `labels`, `thermal` (T_min −8.0, T_ref 25.0). Copy the defaults from `.claude/skills/image-pipeline/SKILL.md`.
- `configs/samples.yaml`: one entry per session folder:
  ```yaml
  sessions:
    exp1_D1A: {experiment: EXP-1, piece_id: null, storage_day: null, temp: {mode: weather}, notes: "blue LED off until ~13:34"}
    ...
  ```
  `piece_id`/`storage_day` stay `null` until the data team confirms (code must handle null: group falls back to session_id).
- `config.py`: pydantic models for each section; `load_config(path=None) -> Config`; `config_hash(cfg, sections) -> str` (sha1 of the canonical JSON, first 10 chars).
- **Tests:** loading the default config works; an invalid value (e.g. negative `roi.min_frac`) raises `ConfigError`; the hash is stable across runs.

### P0.T4: CLI skeleton
`foodlens` entry point (typer) with commands `ingest`, `qc`, `roi`, `features`, `labels`, `train`, `evaluate`, `predict`, `eda-sheets`. Each takes `--config`, `--limit`, `--force` and prints "not implemented yet" for now. Every implemented stage must end by printing `STAGE DONE <stage> rows=<n>`.

### P0.T5: `scripts/check.sh` (the gate every session runs)
```bash
#!/usr/bin/env bash
set -uo pipefail
mkdir -p logs
fail=0
step(){ echo "== $1"; shift; "$@" || { echo "!! FAILED: $*"; fail=1; }; }
step "ruff"        uv run ruff check src tests
step "ruff-format" uv run ruff format --check src tests
step "pytest"      uv run pytest -q --maxfail=5
step "size-limits" uv run python scripts/check_sizes.py
if [ -n "$(ls notebooks/src/*.py 2>/dev/null)" ]; then step "notebooks" bash scripts/build_notebooks.sh --check; fi
[ $fail -eq 0 ] && echo "CHECK RESULT: PASS" || { echo "CHECK RESULT: FAIL"; exit 1; }
```
- `scripts/check_sizes.py`: fail if any `src/**/*.py` > 300 lines or `notebooks/src/*.py` > 400 lines; warn on functions > 50 lines (use `ast`).
- `scripts/build_notebooks.sh`: jupytext → `notebooks/*.ipynb`, execute with nbclient (timeout from env, default 900 s); `--check` mode executes into a temp dir.
- `scripts/uninstall_global_tools.sh`: prints the removal commands for anything installed outside `.venv` (initially only uv, if installed now); executes only with `--yes`.
- ruff config in `pyproject.toml`: line-length 100, rules `E,F,I,B,UP,SIM`.

### P0.T6: Tests and commit
- `tests/conftest.py` with fixtures that make synthetic images: `make_meat_image(light="RED", size=(400,300), glints=10, seed=0)` returns a dark background, a bright ellipse in the chosen channel, and small saturated dots. Also `make_dark_image()`.
- Tests for config and errors. `bash scripts/check.sh` → PASS.
- Update `README.md` "Setup" (uv sync, check.sh).
- **Accept:** check.sh PASS; `uv run foodlens --help` lists the commands.

## Session end
Follow the AGENTS.md §4 checklist. Next phase: P1.
