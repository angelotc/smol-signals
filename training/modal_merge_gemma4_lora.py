"""Modal job to merge the Smol Signals Gemma 4 LoRA adapter into a full model.

Run locally with:

    modal run training/modal_merge_gemma4_lora.py \
      --push-model ajc426/small-signals-gemma-4-12b-sft-merged

Expected Modal secret:
    huggingface-secret with HF_TOKEN
"""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "smol-signals-gemma4-12b-merge"
BASE_MODEL = "unsloth/gemma-4-12b"
ADAPTER_MODEL = "ajc426/small-signals-gemma-4-12b-sft-lora"
DEFAULT_PUSH_MODEL = "ajc426/small-signals-gemma-4-12b-sft-merged"

VOLUME_PATH = PurePosixPath("/runs")
HF_CACHE = PurePosixPath("/hf-cache")
REMOTE_MERGE_SCRIPT = PurePosixPath("/opt/remote_merge_gemma4_lora.py")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate>=1.2.0",
        "huggingface_hub>=0.25.0",
        "hf_transfer>=0.1.8",
        "peft>=0.14.0",
        "librosa",
        "pillow",
        "protobuf",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
        "soundfile",
        "torch>=2.5.0",
        "torchvision",
        "transformers==5.12.0",
    )
    .env({
        "HF_HOME": str(HF_CACHE),
        "HF_HUB_CACHE": str(HF_CACHE / "hub"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_file(
        Path(__file__).parent / "remote_merge_gemma4_lora.py",
        remote_path=str(REMOTE_MERGE_SCRIPT),
    )
)

app = modal.App(APP_NAME, image=image)
runs = modal.Volume.from_name("smol-signals-gemma4-12b-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("smol-signals-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    timeout=6 * 60 * 60,
    memory=98304,
    cpu=8,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={
        str(VOLUME_PATH): runs,
        str(HF_CACHE): hf_cache,
    },
)
def merge(
    push_model: str = DEFAULT_PUSH_MODEL,
    base_model: str = BASE_MODEL,
    adapter_model: str = ADAPTER_MODEL,
    processor_source: str = "",
    output_name: str = "gemma4-12b-smol-signals-merged",
    max_shard_size: str = "4GB",
    private: bool = True,
    overwrite_output_dir: bool = True,
) -> None:
    output_dir = VOLUME_PATH / output_name
    cmd = [
        "python",
        str(REMOTE_MERGE_SCRIPT),
        "--base-model",
        base_model,
        "--adapter-model",
        adapter_model,
        "--push-model",
        push_model,
        "--output-dir",
        str(output_dir),
        "--max-shard-size",
        max_shard_size,
    ]
    if processor_source:
        cmd.extend(["--processor-source", processor_source])
    if not private:
        cmd.append("--no-private")
    if overwrite_output_dir:
        cmd.append("--overwrite-output-dir")
    subprocess.run(cmd, check=True)
    runs.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    push_model: str = DEFAULT_PUSH_MODEL,
    base_model: str = BASE_MODEL,
    adapter_model: str = ADAPTER_MODEL,
    processor_source: str = "",
    output_name: str = "gemma4-12b-smol-signals-merged",
    max_shard_size: str = "4GB",
    private: bool = True,
    overwrite_output_dir: bool = True,
) -> None:
    merge.remote(
        push_model=push_model,
        base_model=base_model,
        adapter_model=adapter_model,
        processor_source=processor_source,
        output_name=output_name,
        max_shard_size=max_shard_size,
        private=private,
        overwrite_output_dir=overwrite_output_dir,
    )
