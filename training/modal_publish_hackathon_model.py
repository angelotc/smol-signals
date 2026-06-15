"""Publish the merged Gemma 4 model from Modal volume to the hackathon org.

Run locally with:

    modal run training/modal_publish_hackathon_model.py

Expected Modal secret:
    huggingface-secret with HF_TOKEN
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath

import modal

APP_NAME = "smol-signals-publish-hackathon-model"
SOURCE_DIR = PurePosixPath("/runs/gemma4-12b-smol-signals-merged")
TARGET_MODEL = "build-small-hackathon/small-signals-gemma-4-12b-sft-merged"
HF_CACHE = PurePosixPath("/hf-cache")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "hf_transfer>=0.1.8",
        "huggingface_hub>=0.25.0",
    )
    .env({
        "HF_HOME": str(HF_CACHE),
        "HF_HUB_CACHE": str(HF_CACHE / "hub"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

app = modal.App(APP_NAME, image=image)
runs = modal.Volume.from_name("smol-signals-gemma4-12b-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("smol-signals-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    timeout=3 * 60 * 60,
    memory=8192,
    cpu=4,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={
        str(SOURCE_DIR.parent): runs,
        str(HF_CACHE): hf_cache,
    },
)
def publish(
    target_model: str = TARGET_MODEL,
    source_dir: str = str(SOURCE_DIR),
    private: bool = False,
) -> None:
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    folder = Path(source_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Missing merged model folder: {folder}")

    api = HfApi(token=token)
    api.create_repo(target_model, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=target_model,
        repo_type="model",
        folder_path=str(folder),
        commit_message="Publish Small Signals merged model for Build Small",
    )
    info = api.model_info(target_model, token=token)
    print(
        json.dumps(
            {
                "target_model": target_model,
                "sha": info.sha,
                "private": info.private,
                "siblings": sorted(s.rfilename for s in info.siblings),
            },
            indent=2,
        ),
        flush=True,
    )


@app.local_entrypoint()
def main(
    target_model: str = TARGET_MODEL,
    source_dir: str = str(SOURCE_DIR),
    private: bool = False,
) -> None:
    publish.remote(
        target_model=target_model,
        source_dir=source_dir,
        private=private,
    )
