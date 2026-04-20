from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    dataset_txt_path: str
    model_name: str
    output_dir: str
    run_name: str
    epochs: float
    learning_rate: float
    train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    save_steps: int
    logging_steps: int
    warmup_ratio: float
    weight_decay: float
    fp16: bool
    bf16: bool
    seed: int


class UILogger:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._done = False
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        with self._lock:
            self._lines.append(message)

    def mark_done(self) -> None:
        with self._lock:
            self._done = True

    def snapshot(self) -> tuple[str, bool]:
        with self._lock:
            return "\n".join(self._lines), self._done


def _read_txt_lines(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Dataset .txt is empty after removing blank lines.")
    return lines


def _tokenize_dataset(lines: list[str], tokenizer, max_seq_length: int) -> Dataset:
    ds = Dataset.from_dict({"text": lines})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def run_training(config: TrainConfig, logger: UILogger) -> None:
    if config.fp16 and config.bf16:
        raise ValueError("Enable either fp16 or bf16, not both.")

    os.makedirs(config.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.write(f"Device detected: {device}")

    if device == "cuda":
        logger.write(f"CUDA device count: {torch.cuda.device_count()}")
        logger.write(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.write(f"CUDA version (torch): {torch.version.cuda}")
    else:
        logger.write("WARNING: CUDA not available. Training will run on CPU and be much slower.")

    logger.write(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.write(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    lines = _read_txt_lines(config.dataset_txt_path)
    logger.write(f"Loaded {len(lines)} text rows from dataset.")

    tokenized_dataset = _tokenize_dataset(lines, tokenizer, config.max_seq_length)
    logger.write(f"Tokenized dataset rows: {len(tokenized_dataset)}")

    run_output_path = Path(config.output_dir) / config.run_name
    run_output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_output_path),
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_pin_memory=device == "cuda",
        report_to=[],
        seed=config.seed,
        run_name=config.run_name,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class ProgressCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logger.write(f"step={state.global_step} logs={logs}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ProgressCallback()],
    )

    logger.write("Starting training...")
    trainer.train()

    logger.write("Saving model + tokenizer...")
    trainer.save_model(str(run_output_path))
    tokenizer.save_pretrained(str(run_output_path))

    logger.write(f"Training complete. Artifacts saved at: {run_output_path}")
    logger.mark_done()


def stream_training(config: TrainConfig) -> Generator[str, None, None]:
    logger = UILogger()

    def _target():
        try:
            run_training(config, logger)
        except Exception as exc:  # pragma: no cover
            logger.write(f"ERROR: {exc}")
            logger.mark_done()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    last_payload = ""
    while True:
        payload, done = logger.snapshot()
        if payload != last_payload:
            last_payload = payload
            yield payload
        if done:
            break
