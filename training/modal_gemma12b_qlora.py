"""Modal QLoRA trainer for Smol Signals Gemma 12B SFT data.

Run locally with:

    modal run training/modal_gemma12b_qlora.py --dataset-repo ajc426/small-signals-gemma12b-sft

Expected Modal secret:
    huggingface-secret with HF_TOKEN
"""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "smol-signals-gemma4-12b-posttrain"
BASE_MODEL = "unsloth/gemma-4-12b"
VOLUME_PATH = PurePosixPath("/runs")
HF_CACHE = PurePosixPath("/hf-cache")
REMOTE_TRAIN_SCRIPT = PurePosixPath("/opt/remote_gemma12b_train.py")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.2.0",
        "huggingface_hub>=0.25.0",
        "librosa",
        "peft>=0.14.0",
        "pillow",
        "protobuf",
        "sentencepiece>=0.2.0",
        "soundfile",
        "torch>=2.5.0",
        "torchvision",
        "transformers>=4.50.0",
        "wandb",
    )
    .env({
        "HF_HOME": str(HF_CACHE),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_file(
        Path(__file__).parent / "remote_gemma12b_train.py",
        remote_path=str(REMOTE_TRAIN_SCRIPT),
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("smol-signals-gemma4-12b-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("smol-signals-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={
        str(VOLUME_PATH): volume,
        str(HF_CACHE): hf_cache,
    },
)
def train(
    dataset_repo: str,
    push_model: str = "",
    base_model: str = BASE_MODEL,
    max_steps: int = -1,
    epochs: float = 1.0,
    max_seq_len: int = 8192,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
) -> None:
    output_dir = VOLUME_PATH / "gemma4-12b-smol-signals"
    cmd = [
        "python",
        str(REMOTE_TRAIN_SCRIPT),
        "--dataset-repo",
        dataset_repo,
        "--base-model",
        base_model,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--max-steps",
        str(max_steps),
        "--max-seq-len",
        str(max_seq_len),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
    ]
    if push_model:
        cmd.extend(["--push-model", push_model])
    subprocess.run(cmd, check=True)
    volume.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    dataset_repo: str,
    push_model: str = "",
    base_model: str = BASE_MODEL,
    max_steps: int = -1,
    epochs: float = 1.0,
    max_seq_len: int = 8192,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
) -> None:
    train.remote(
        dataset_repo=dataset_repo,
        push_model=push_model,
        base_model=base_model,
        max_steps=max_steps,
        epochs=epochs,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
