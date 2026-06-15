"""Remote PEFT merge worker used by `modal_merge_gemma4_lora.py`."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


def get_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/gemma-4-12b")
    parser.add_argument("--adapter-model", required=True)
    parser.add_argument("--push-model", required=True)
    parser.add_argument("--output-dir", default="/runs/gemma4-12b-smol-signals-merged")
    parser.add_argument("--processor-source", default="")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--revision", default="")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = get_token()
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite-output-dir to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor_source = args.processor_source or args.adapter_model
    print(f"Loading processor from {processor_source}", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(
            processor_source,
            token=token,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(
            f"Processor load from {processor_source} failed: {exc}; falling back to {args.base_model}",
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(
            args.base_model,
            token=token,
            trust_remote_code=True,
        )

    model_kwargs = {
        "token": token,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if args.revision:
        model_kwargs["revision"] = args.revision

    print(f"Loading base model {args.base_model}", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = True

    print(f"Loading adapter {args.adapter_model}", flush=True)
    peft_model = PeftModel.from_pretrained(
        model,
        args.adapter_model,
        token=token,
        is_trainable=False,
    )

    print("Merging adapter into base model", flush=True)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    merged_model.config.use_cache = True

    if getattr(merged_model, "generation_config", None) is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            merged_model.generation_config.pad_token_id = tokenizer.pad_token_id
            merged_model.generation_config.eos_token_id = tokenizer.eos_token_id

    print(f"Saving merged model to {output_dir}", flush=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(output_dir)

    provenance = {
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        "processor_source": processor_source,
        "dtype": "bfloat16",
        "merge_method": "peft.PeftModel.merge_and_unload",
    }
    (output_dir / "small_signals_merge.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    readme = output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "---",
                    "license: gemma",
                    "base_model: unsloth/gemma-4-12b",
                    "tags:",
                    "- gemma",
                    "- lora",
                    "- peft",
                    "- merged",
                    "---",
                    "",
                    "# Small Signals Gemma 4 12B SFT Merged",
                    "",
                    "Merged checkpoint created from `unsloth/gemma-4-12b` and ",
                    f"`{args.adapter_model}` using `peft.PeftModel.merge_and_unload()`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    print(f"Uploading merged model to {args.push_model}", flush=True)
    api = HfApi(token=token)
    api.create_repo(
        args.push_model,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.push_model,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Merge Gemma 4 LoRA adapter into base model",
    )
    info = api.model_info(args.push_model, token=token)
    print(
        json.dumps(
            {
                "pushed_model": args.push_model,
                "sha": info.sha,
                "private": info.private,
                "siblings": sorted(s.rfilename for s in info.siblings),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
