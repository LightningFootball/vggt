# Repository Guidelines

## Project Structure & Module Organization
- Source: `vggt/` (models, heads, utils), training code in `training/`, examples in `examples/`.
- Demos: `demo_gradio.py`, `demo_viser.py`, `demo_colmap.py` (+ optimized variants).
- Scripts & reports: `scripts/`, `report/`, docs in `docs/`.
- Tests: `tests/` (pytest). Config/package metadata: `pyproject.toml`.

## Build, Test, and Development Commands
- Install core: `pip install -r requirements.txt` (Python 3.10+).
- Demo extras: `pip install -r requirements_demo.txt`.
- Training extras: `pip install -r requirements_training.txt`.
- Editable install: `pip install -e .` (uses setuptools from `pyproject.toml`).
- Run tests: `pytest -q` (from repo root).
- Demos:
  - `python demo_gradio.py`
  - `python demo_viser.py --image_folder path/to/images`
  - `python demo_colmap.py --scene_dir /PATH/TO/SCENE [--use_ba]`
- Benchmarks: `python scripts/benchmark_vram_optimized.py [--device cuda] ...`
- Example training: `python examples/simple_trainer.py default --data_dir /DATA --result_dir /OUT`.

## Coding Style & Naming Conventions
- Language: Python (>=3.10). Use 4-space indentation and type hints where practical.
- Names: files/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Imports: standard lib, third-party, local (grouped). Prefer Black-compatible formatting; keep lines ≤ 88 chars.
- Keep APIs stable; add docstrings for public functions/classes.

## Testing Guidelines
- Framework: pytest. Place tests in `tests/` named `test_*.py`.
- Keep tests CPU-friendly by default; gate GPU-heavy paths behind flags.
- Add unit tests for new modules and edge cases. Run `pytest -q` before opening a PR.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood. Common prefixes observed: `fix:`, `conf:`, `training:` when relevant.
- PRs must include:
  - Summary of changes and rationale
  - Related issues (e.g., `Closes #123`)
  - Usage/testing instructions; screenshots for demo/UI changes
  - Notes on perf/VRAM impact if touching models/inference

## Security & Configuration Tips
- Don’t commit large binaries or secrets. Use `.gitignore` patterns already present.
- Pin new runtime deps in the appropriate `requirements_*.txt` and update `pyproject.toml` only if packaging changes.

## Agent-Specific Instructions
- Follow this file’s scope across the repo; make minimal, focused changes.
- Preserve existing file structure and naming; update docs and tests alongside code changes.
- Avoid broad refactors in unrelated areas without prior discussion.
