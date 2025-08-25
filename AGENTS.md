# Repository Guidelines

## Project Structure & Module Organization
- `vggt/`: Core library code (models, heads, layers, utils, dependency).
  - `vggt/models/vggt.py`: Main model entry.
- `training/`: Fine-tuning pipeline (Hydra configs under `training/config/`, `launch.py`, `trainer.py`).
- `examples/`: Sample scenes and inputs for demos.
- `docs/`: Extra installation/packaging notes.
- Top-level demos: `demo_gradio.py`, `demo_viser.py`, `demo_colmap.py`.

## Build, Test, and Development Commands
- Install (library only): `pip install -r requirements.txt && pip install -e .`
- Demo extras: `pip install -r requirements_demo.txt`
- Quick import check: `python -c "import vggt; print('ok')"`
- Run Gradio demo: `python demo_gradio.py`
- Run Viser viewer: `python demo_viser.py --image_folder path/to/images`
- Export COLMAP: `python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR [--use_ba]`
- Train/finetune (DDP example): `torchrun --nproc_per_node=4 training/launch.py`

## Coding Style & Naming Conventions
- Language: Python ≥ 3.10; follow PEP 8, 4-space indents, 120-char soft limit.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Types: add type hints where practical (public APIs, data structures).
- Imports: standard → third-party → local; avoid unused imports.
- Config: add new options under `training/config/*.yaml` with clear, minimal defaults.

## Testing Guidelines
- No unit test suite is present. For contributions:
  - Add targeted tests under `tests/` (e.g., `tests/test_geometry.py`).
  - Prefer small, synthetic tensors to exercise shapes/dtypes/device.
  - Provide runnable repro snippets for demo/training changes.
  - Validate demos end-to-end with a tiny image set in `examples/`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; optional scope, e.g. `train: fix DDP seed sync`.
- Group related changes; avoid large, mixed commits.
- PRs must include:
  - Summary, rationale, and before/after behavior.
  - Usage notes (commands/config diffs) and, if UI/demo related, a screenshot or short log.
  - Linked issues (`Fixes #123`) and limitations.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or secrets (e.g., HF tokens).
- Large downloads occur via Hugging Face; allow offline fallback by documenting local paths.
- GPU memory varies; document flags like `max_img_per_gpu`, `accum_steps`, and learning-rate changes in PRs.
