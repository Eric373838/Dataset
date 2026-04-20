from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch

from trainer_core import TrainConfig, stream_training


OUTPUT_ROOT = Path("outputs")
OUTPUT_ROOT.mkdir(exist_ok=True)


def _cuda_summary() -> str:
    if torch.cuda.is_available():
        return (
            f"✅ CUDA available | GPU: {torch.cuda.get_device_name(0)} | "
            f"CUDA: {torch.version.cuda}"
        )
    return "⚠️ CUDA not detected. Training will run on CPU."


def start_training(
    dataset_file,
    model_name,
    run_name,
    epochs,
    learning_rate,
    train_batch_size,
    grad_accum,
    max_seq_length,
    save_steps,
    logging_steps,
    warmup_ratio,
    weight_decay,
    precision,
    seed,
):
    if dataset_file is None:
        yield "ERROR: Please upload a .txt dataset file."
        return

    dataset_path = dataset_file.name
    if not dataset_path.lower().endswith(".txt"):
        yield "ERROR: Only .txt datasets are currently supported."
        return

    fp16 = precision == "fp16"
    bf16 = precision == "bf16"

    config = TrainConfig(
        dataset_txt_path=dataset_path,
        model_name=model_name.strip(),
        output_dir=str(OUTPUT_ROOT),
        run_name=run_name.strip(),
        epochs=float(epochs),
        learning_rate=float(learning_rate),
        train_batch_size=int(train_batch_size),
        gradient_accumulation_steps=int(grad_accum),
        max_seq_length=int(max_seq_length),
        save_steps=int(save_steps),
        logging_steps=int(logging_steps),
        warmup_ratio=float(warmup_ratio),
        weight_decay=float(weight_decay),
        fp16=fp16,
        bf16=bf16,
        seed=int(seed),
    )

    for log_text in stream_training(config):
        yield log_text


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Professional LLM Trainer") as demo:
        gr.Markdown("# Professional LLM Trainer (CUDA/cuDNN Ready)")
        gr.Markdown(
            "Upload a `.txt` dataset, configure training parameters, and fine-tune a causal LLM on your NVIDIA GPU."
        )
        cuda_box = gr.Textbox(value=_cuda_summary(), label="Hardware status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                dataset = gr.File(label="Dataset .txt", file_types=[".txt"])
                model_name = gr.Textbox(value="distilgpt2", label="Base model (HF model id)")
                run_name = gr.Textbox(value="my_run", label="Output run name")

                epochs = gr.Slider(minimum=1, maximum=10, step=1, value=2, label="Epochs")
                learning_rate = gr.Number(value=2e-5, label="Learning rate")
                train_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=2, label="Train batch size")
                grad_accum = gr.Slider(minimum=1, maximum=32, step=1, value=8, label="Gradient accumulation")
                max_seq_length = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Max sequence length")
                precision = gr.Radio(
                    choices=["fp16", "bf16", "fp32"],
                    value="fp16",
                    label="Precision",
                    info="Use fp16 for most CUDA GPUs. Use bf16 only on supported hardware.",
                )

                warmup_ratio = gr.Slider(minimum=0.0, maximum=0.2, step=0.01, value=0.03, label="Warmup ratio")
                weight_decay = gr.Slider(minimum=0.0, maximum=0.2, step=0.01, value=0.01, label="Weight decay")
                save_steps = gr.Slider(minimum=50, maximum=2000, step=50, value=200, label="Save steps")
                logging_steps = gr.Slider(minimum=1, maximum=200, step=1, value=10, label="Logging steps")
                seed = gr.Number(value=42, precision=0, label="Random seed")

                start_btn = gr.Button("Start Training", variant="primary")

            with gr.Column(scale=1):
                logs = gr.Textbox(label="Training logs", lines=30, interactive=False)
                gr.Markdown("Saved models will appear under `outputs/<run_name>/`.")

        start_btn.click(
            fn=start_training,
            inputs=[
                dataset,
                model_name,
                run_name,
                epochs,
                learning_rate,
                train_batch_size,
                grad_accum,
                max_seq_length,
                save_steps,
                logging_steps,
                warmup_ratio,
                weight_decay,
                precision,
                seed,
            ],
            outputs=logs,
        )

        demo.load(lambda: _cuda_summary(), outputs=cuda_box)

    return demo


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ui = build_ui()
    ui.queue(default_concurrency_limit=1)
    ui.launch(server_name="127.0.0.1", server_port=7860)
