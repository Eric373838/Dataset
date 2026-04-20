# GPU LLM Trainer (Windows + CUDA/cuDNN)

This repository now includes a complete **local LLM training app** that you can launch from a `.bat` file.

## What you get

- `app.py`: Gradio web UI to configure and run training.
- `trainer_core.py`: Training pipeline using Hugging Face Transformers.
- `run_app.bat`: One-click launcher for Windows.
- `requirements-llm-trainer.txt`: Python dependencies.
- `outputs/`: Folder where trained models are saved.

## Features

- Upload a `.txt` dataset directly in the UI.
- Choose base model, sequence length, epochs, batch size, learning rate, and precision.
- Uses your **NVIDIA GPU** automatically when CUDA is available.
- Streams logs in the UI while training runs.
- Saves model and tokenizer to `outputs/<your_run_name>/`.

---

## 1) System requirements (Windows)

1. **Python 3.10+**
2. **NVIDIA GPU** with recent drivers
3. **CUDA + cuDNN** correctly installed (or a PyTorch build that bundles CUDA runtime)
4. Enough VRAM for the selected model

---

## 2) Install

From repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-llm-trainer.txt
```

> If needed, install a CUDA-enabled PyTorch build from the official PyTorch selector first.

---

## 3) Run the app

Double-click:

- `run_app.bat`

Or manually:

```bash
python app.py
```

Open the shown local URL (usually `http://127.0.0.1:7860`).

---

## 4) Train a model

1. Upload your `.txt` dataset in the app.
2. Set base model (example: `gpt2`, `distilgpt2`, `EleutherAI/pythia-70m`).
3. Set run/output name.
4. Tune hyperparameters.
5. Click **Start Training**.

Saved artifacts:

- `outputs/<run_name>/`
  - model weights
  - tokenizer
  - trainer state/checkpoints (if enabled)

---

## 5) Tips for better results

- Start with a small model (`distilgpt2`) to validate your pipeline.
- Keep `max_seq_length` moderate (e.g. 256/512) for lower VRAM.
- Increase gradient accumulation for limited VRAM.
- Use `fp16` on supported NVIDIA GPUs for faster training.
- Clean your text data before training.

---

## 6) Notes

- This setup is for **causal language model fine-tuning on plain text**.
- For large-scale production LLM training, consider distributed frameworks and dataset streaming.
