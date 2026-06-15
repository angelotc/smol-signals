"""Remote training worker used by `modal_gemma12b_qlora.py`."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def is_gemma4(model_name: str) -> bool:
    normalized = model_name.lower()
    return "gemma-4" in normalized or "gemma4" in normalized


def text_block_messages(messages: list[dict[str, str]]) -> list[dict]:
    formatted = []
    for message in messages:
        formatted.append({
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        })
    return formatted


def message_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def gemma4_chat_text(messages: list[dict], *, add_generation_prompt: bool) -> str:
    parts = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        content = message_content_text(message.get("content", ""))
        parts.append(f"<|turn>{role}\n{content}<turn|>")
    if add_generation_prompt:
        parts.append("<|turn>model\n")
    return "\n".join(parts)


class JsonlChatDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int,
        *,
        chat_renderer=None,
        chat_template: str = "auto",
        use_text_blocks: bool = False,
        enable_thinking: bool | None = None,
    ):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chat_renderer = chat_renderer or tokenizer
        self.chat_template = chat_template
        self.use_text_blocks = use_text_blocks
        self.enable_thinking = enable_thinking
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        messages = self.rows[idx]["messages"]
        prompt_messages = messages[:-1]
        full_messages = messages
        if self.use_text_blocks:
            prompt_messages = text_block_messages(prompt_messages)
            full_messages = text_block_messages(full_messages)
        prompt_text = self._render(prompt_messages, add_generation_prompt=True)
        full_text = self._render(full_messages, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]

        if len(full_ids) > self.max_seq_len:
            overflow = len(full_ids) - self.max_seq_len
            full_ids = full_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def _render(self, messages: list[dict], *, add_generation_prompt: bool) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        if self.chat_template == "gemma4":
            return gemma4_chat_text(messages, add_generation_prompt=add_generation_prompt)
        return self.chat_renderer.apply_chat_template(messages, **kwargs)


class DataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def download_dataset(repo_id: str, data_dir: Path) -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: hf_hub_download(
            repo_id,
            name,
            repo_type="dataset",
            token=token,
            local_dir=data_dir,
        )
        for name in ["train.jsonl", "val.jsonl"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--base-model", default="unsloth/gemma-4-12b")
    parser.add_argument("--output-dir", default="/runs/gemma4-12b-smol-signals")
    parser.add_argument("--push-model", default="")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_paths = download_dataset(args.dataset_repo, run_dir / "data")

    gemma4 = is_gemma4(args.base_model)
    processor = None
    if gemma4:
        processor = AutoProcessor.from_pretrained(args.base_model, token=token, trust_remote_code=True)
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if "bnb-4bit" not in args.base_model.lower():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs = {}
    if quant is not None:
        model_kwargs["quantization_config"] = quant
    model_cls = AutoModelForMultimodalLM if gemma4 else Gemma3ForConditionalGeneration
    model = model_cls.from_pretrained(
        args.base_model,
        token=token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        **model_kwargs,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = JsonlChatDataset(
        data_paths["train.jsonl"],
        tokenizer,
        args.max_seq_len,
        chat_renderer=processor or tokenizer,
        chat_template="gemma4" if gemma4 else "auto",
        use_text_blocks=not gemma4,
        enable_thinking=False if gemma4 else None,
    )
    val_ds = JsonlChatDataset(
        data_paths["val.jsonl"],
        tokenizer,
        args.max_seq_len,
        chat_renderer=processor or tokenizer,
        chat_template="gemma4" if gemma4 else "auto",
        use_text_blocks=not gemma4,
        enable_thinking=False if gemma4 else None,
    )
    collator = DataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if processor is not None:
        processor.save_pretrained(str(final_dir))

    metrics = trainer.evaluate()
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.push_model:
        api = HfApi(token=token)
        api.create_repo(args.push_model, repo_type="model", private=True, exist_ok=True)
        model.push_to_hub(args.push_model, token=token, private=True)
        if processor is not None:
            processor.push_to_hub(args.push_model, token=token, private=True)
        else:
            tokenizer.push_to_hub(args.push_model, token=token, private=True)
        print(f"Pushed adapter to https://huggingface.co/{args.push_model}")


if __name__ == "__main__":
    main()
