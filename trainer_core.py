from __future__ import annotations

import os
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import torch
from datasets import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    dataset_txt_path: str
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
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    require_cuda: bool


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


def _read_dataset_text(path: str) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    compact = raw.strip()
    if not compact:
        raise ValueError("Dataset .txt is empty.")
    return compact


def _train_tokenizer(dataset_path: str, tokenizer_dir: Path, vocab_size: int, logger: UILogger) -> GPT2TokenizerFast:
    logger.write(f"Training ByteLevel BPE tokenizer from dataset (vocab_size={vocab_size})...")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[dataset_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    )
    bpe.save_model(str(tokenizer_dir))

    tokenizer = GPT2TokenizerFast(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )

    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.write(f"Tokenizer saved at: {tokenizer_dir}")
    return tokenizer


def _tokenize_and_chunk(text: str, tokenizer: GPT2TokenizerFast, max_seq_length: int, logger: UILogger) -> Dataset:
    logger.write("Tokenizing dataset text...")
    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]

    if len(tokens) < max_seq_length:
        raise ValueError(
            f"Not enough tokens ({len(tokens)}) for max_seq_length={max_seq_length}. "
            "Use a larger dataset or lower max_seq_length."
        )

    chunks: list[list[int]] = []
    for i in range(0, len(tokens) - max_seq_length + 1, max_seq_length):
        chunks.append(tokens[i : i + max_seq_length])

    logger.write(f"Created {len(chunks)} training blocks of {max_seq_length} tokens.")

    return Dataset.from_dict(
        {
            "input_ids": chunks,
            "attention_mask": [[1] * max_seq_length for _ in chunks],
        }
    )


def run_training(config: TrainConfig, logger: UILogger) -> None:
    if config.fp16 and config.bf16:
        raise ValueError("Enable either fp16 or bf16, not both.")

    os.makedirs(config.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.write(f"Device detected: {device}")

    if config.require_cuda and device != "cuda":
        raise RuntimeError("CUDA is required by configuration, but no CUDA GPU was detected.")

    if device == "cuda":
        logger.write(f"CUDA device count: {torch.cuda.device_count()}")
        logger.write(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.write(f"CUDA version (torch): {torch.version.cuda}")
    else:
        logger.write("WARNING: CUDA not available. Training runs on CPU.")

    run_output_path = Path(config.output_dir) / config.run_name
    run_output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = run_output_path / "tokenizer"
    text = _read_dataset_text(config.dataset_txt_path)
    tokenizer = _train_tokenizer(config.dataset_txt_path, tokenizer_dir, config.vocab_size, logger)

    train_dataset = _tokenize_and_chunk(text, tokenizer, config.max_seq_length, logger)

    logger.write(
        "Building GPT model from scratch with config: "
        f"layers={config.n_layer}, heads={config.n_head}, embd={config.n_embd}, "
        f"ctx={config.max_seq_length}, vocab={tokenizer.vocab_size}"
    )

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=config.max_seq_length,
        n_ctx=config.max_seq_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(model_config)

    # Keep compatibility across multiple Transformers versions by passing only
    # supported TrainingArguments fields at runtime.
    training_kwargs = {
        "output_dir": str(run_output_path),
        "overwrite_output_dir": True,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 3,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "dataloader_pin_memory": device == "cuda",
        "report_to": [],
        "seed": config.seed,
        "run_name": config.run_name,
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in supported}

    # Some older versions may not support `run_name`/`report_to`; that is fine.
    missing = sorted(set(training_kwargs.keys()) - set(filtered_kwargs.keys()))
    if missing:
        logger.write(f"INFO: Skipping unsupported TrainingArguments keys: {missing}")

    training_args = TrainingArguments(**filtered_kwargs)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logger.write(f"step={state.global_step} logs={logs}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ProgressCallback()],
    )

    logger.write("Starting pretraining from scratch...")
    trainer.train()

    model_dir = run_output_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.write("Saving model + tokenizer...")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    logger.write(f"Pretraining complete. Artifacts saved at: {run_output_path}")
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
