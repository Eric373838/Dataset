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
            f"CUDA runtime: {torch.version.cuda}"
        )
    return "❌ CUDA not detected. Install CUDA-enabled PyTorch and NVIDIA drivers."


def start_training(
    dataset_file,
    run_name,
    epochs,
    learning_rate,
    train_batch_size,
    grad_accum,
    max_seq_length,
    vocab_size,
    n_layer,
    n_head,
    n_embd,
    save_steps,
    logging_steps,
    warmup_ratio,
    weight_decay,
    precision,
    require_cuda,
    seed,
):
    if dataset_file is None:
        yield "ERROR: Please upload a .txt dataset file."
        return

    dataset_path = dataset_file.name
    if not dataset_path.lower().endswith(".txt"):
        yield "ERROR: Only .txt datasets are currently supported."
        return

    if not run_name.strip():
        yield "ERROR: Output run name cannot be empty."
        return

    fp16 = precision == "fp16"
    bf16 = precision == "bf16"

    config = TrainConfig(
        dataset_txt_path=dataset_path,
        output_dir=str(OUTPUT_ROOT),
        run_name=run_name.strip(),
        epochs=float(epochs),
        learning_rate=float(learning_rate),
        train_batch_size=int(train_batch_size),
        gradient_accumulation_steps=int(grad_accum),
        max_seq_length=int(max_seq_length),
        vocab_size=int(vocab_size),
        n_layer=int(n_layer),
        n_head=int(n_head),
        n_embd=int(n_embd),
        save_steps=int(save_steps),
        logging_steps=int(logging_steps),
        warmup_ratio=float(warmup_ratio),
        weight_decay=float(weight_decay),
        fp16=fp16,
        bf16=bf16,
        require_cuda=bool(require_cuda),
        seed=int(seed),
    )

    for log_text in stream_training(config):
        yield log_text


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Own LLM Pretraining Studio") as demo:
        gr.Markdown("# Own LLM Pretraining Studio (Train from Scratch)")
        gr.Markdown(
            "Upload a `.txt` dataset and **pretrain your own GPT-style model from scratch** on your NVIDIA GPU."
        )
        cuda_box = gr.Textbox(value=_cuda_summary(), label="Hardware status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                dataset = gr.File(label="Dataset .txt", file_types=[".txt"])
                run_name = gr.Textbox(value="my_own_model", label="Output run name")

                epochs = gr.Slider(minimum=1, maximum=30, step=1, value=3, label="Epochs")
                learning_rate = gr.Number(value=3e-4, label="Learning rate")
                train_batch_size = gr.Slider(minimum=1, maximum=32, step=1, value=4, label="Train batch size")
                grad_accum = gr.Slider(minimum=1, maximum=64, step=1, value=8, label="Gradient accumulation")
                max_seq_length = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Context length")

                gr.Markdown("### New model architecture (from scratch)")
                vocab_size = gr.Slider(minimum=4096, maximum=65536, step=512, value=16384, label="Tokenizer vocab size")
                n_layer = gr.Slider(minimum=2, maximum=48, step=1, value=8, label="Number of transformer layers")
                n_head = gr.Slider(minimum=2, maximum=32, step=1, value=8, label="Attention heads")
                n_embd = gr.Slider(minimum=128, maximum=2048, step=64, value=512, label="Embedding dimension")

                precision = gr.Radio(
                    choices=["fp16", "bf16", "fp32"],
                    value="fp16",
                    label="Precision",
                    info="fp16 is recommended on most NVIDIA CUDA GPUs.",
                )
                require_cuda = gr.Checkbox(value=True, label="Require CUDA GPU (recommended)")

                warmup_ratio = gr.Slider(minimum=0.0, maximum=0.2, step=0.01, value=0.03, label="Warmup ratio")
                weight_decay = gr.Slider(minimum=0.0, maximum=0.2, step=0.01, value=0.01, label="Weight decay")
                save_steps = gr.Slider(minimum=50, maximum=10000, step=50, value=500, label="Save steps")
                logging_steps = gr.Slider(minimum=1, maximum=500, step=1, value=10, label="Logging steps")
                seed = gr.Number(value=42, precision=0, label="Random seed")

                start_btn = gr.Button("Start Pretraining", variant="primary")

            with gr.Column(scale=1):
                logs = gr.Textbox(label="Pretraining logs", lines=30, interactive=False)
                gr.Markdown("Your trained model and tokenizer are saved in `outputs/<run_name>/`.")

        start_btn.click(
            fn=start_training,
            inputs=[
                dataset,
                run_name,
                epochs,
                learning_rate,
                train_batch_size,
                grad_accum,
                max_seq_length,
                vocab_size,
                n_layer,
                n_head,
                n_embd,
                save_steps,
                logging_steps,
                warmup_ratio,
                weight_decay,
                precision,
                require_cuda,
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
