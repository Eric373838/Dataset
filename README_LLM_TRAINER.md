# Own LLM Pretraining Studio (Windows + CUDA/cuDNN)

This setup is for **training your own model from scratch** (pretraining), not fine-tuning a pretrained checkpoint.

## What you get

- `app.py`: Gradio web UI to configure full pretraining runs.
- `trainer_core.py`: Scratch-pretraining pipeline (new tokenizer + new GPT model).
- `run_app.bat`: One-click Windows launcher with robust auto-setup checks.
- `requirements-llm-trainer.txt`: Dependency list.
- `outputs/`: Saved tokenizer, checkpoints, and final model.

## Core behavior

- Upload one `.txt` dataset in the UI.
- Train a **new ByteLevel BPE tokenizer** from your dataset.
- Build a **new GPT architecture from scratch** (choose layers/heads/embedding size).
- Pretrain on your NVIDIA GPU with CUDA (or block run when CUDA is missing).
- Save artifacts to `outputs/<run_name>/`.

---

## 1) Windows requirements

1. Python 3.10+
2. NVIDIA GPU + current driver
3. CUDA-capable environment
4. Enough disk + VRAM for your architecture

---

## 2) Start from BAT file (recommended)

Double-click:

- `run_app.bat`

The BAT script now automatically:

1. Detects Python launcher (`python` or `py -3`).
2. Creates and activates `.venv`.
3. Upgrades `pip/setuptools/wheel`.
4. Installs CUDA-enabled PyTorch (`cu124`, fallback `cu121`).
5. Installs all required Python packages.
6. Verifies imports (`torch`, `transformers`, `gradio`, `datasets`, `tokenizers`).
7. Launches `app.py`.
8. Shows clear error messages and **keeps the window open** if anything fails.

---

## 3) Use the app

1. Upload your training data (`.txt`).
2. Set a run name.
3. Choose architecture: vocab size, layers, heads, embedding dim.
4. Set training hyperparameters.
5. Keep **Require CUDA GPU** checked for GPU-only training.
6. Click **Start Pretraining**.

Outputs are written to:

- `outputs/<run_name>/tokenizer`
- `outputs/<run_name>/model`
- `outputs/<run_name>/checkpoint-*`

---

## 4) Important note on "full training"

This pipeline creates both:

- tokenizer from your own data
- model weights from random initialization

So it is true **training from scratch / pretraining**, not loading a pretrained base model.
