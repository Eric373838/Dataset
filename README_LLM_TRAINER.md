# Own LLM Pretraining Studio (Windows + CUDA/cuDNN)

This setup is for **training your own model from scratch** (pretraining), not fine-tuning a pretrained checkpoint.

## What you get

- `app.py`: Gradio web UI to configure full pretraining runs.
- `trainer_core.py`: Scratch-pretraining pipeline (new tokenizer + new GPT model).
- `run_app.bat`: One-click Windows launcher.
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

## 2) Start from BAT file

Double-click:

- `run_app.bat`

The BAT script will:

1. Create `.venv` if needed.
2. Install **CUDA-enabled PyTorch (cu121)**.
3. Install app dependencies.
4. Launch the web app.

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
