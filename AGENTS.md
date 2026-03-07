## Cursor Cloud specific instructions

### Overview

This is a Python ML research codebase (RL Curriculum / IF-RLVR) for training language models with Reinforcement Learning and Verifiable Rewards. It uses `uv` as the package manager. Two git submodules (`open-instruct`, `IFBench`) must be initialized before dependency installation.

### Services

| Service | Description | How to Run |
|---------|-------------|------------|
| Data generation | Synthetic IFEval constraint dataset creation | `uv run python create_data.py --num-prompts N --num-constraints M` |
| GRPO Training | Main RL training pipeline (requires GPU) | `source configs/<model>.sh && source configs/<gpus>.sh && bash scripts/train_if_rlvr.sh` |
| IFBench Evaluation | Model evaluation suite (requires GPU) | See `IFBench/README.md` |

### Lint / Test / Build

- **Lint (open-instruct):** `uv run ruff check` from `/workspace/open-instruct`
- **Lint (root scripts):** `uv run ruff check create_data.py extract_prompts.py` from `/workspace`
- **Tests:** `uv run pytest tests/` from `/workspace/open-instruct` — 25 unit tests, all pass without GPU
- Ruff config is in `open-instruct/pyproject.toml` under `[tool.ruff]`

### Key gotchas

- **flash-attn build requires torch pre-installed.** The `[tool.uv.extra-build-dependencies]` in `open-instruct/pyproject.toml` is intended to handle this, but in practice `uv sync` alone may fail on a fresh environment. The workaround is: (1) `uv pip install 'torch>=2.9.0,<2.10' --index-url https://download.pytorch.org/whl/cu129`, then (2) `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv sync --no-build-isolation`.
- **No GPU in Cloud VM.** Training scripts require NVIDIA GPUs. Data processing scripts (`create_data.py`, `extract_prompts.py`, `scripts/convert_if_datasets_to_rlvr.py`) and tests run without GPU.
- **NLTK data** (`punkt`, `punkt_tab`) must be downloaded: `uv run python -m nltk.downloader punkt punkt_tab`.
- **Submodules** must be initialized before `uv sync`: `git submodule update --init --recursive`.
