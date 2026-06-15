"""Smoke-test Gemma 4 + LoRA on Modal with vLLM.

This is an experimental serving path. Production stays on
``modal_gemma4_transformers.py`` unless this path matches the grounding quality
of the verified endpoint.

Run:

    modal run serving/modal_gemma4_vllm_dev.py --limit 1
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

import modal

MINUTES = 60
PORT = 8000

APP_NAME = "smol-signals-gemma4-vllm-dev"
BASE_MODEL = "unsloth/gemma-4-12b"
ADAPTER_MODEL = "ajc426/small-signals-gemma-4-12b-sft-lora"
SERVED_MODEL = "small-signals-gemma-4-12b"
REQUEST_MODEL = "small-signals-gemma-4-12b:signals"

GPU = "L40S"
N_GPUS = 1
CONTEXT_LENGTH = 8192
MAX_LORA_RANK = 16

HF_CACHE_PATH = PurePosixPath("/root/.cache/huggingface")
VLLM_CACHE_PATH = PurePosixPath("/root/.cache/vllm")
CHAT_TEMPLATE_PATH = PurePosixPath("/opt/gemma4_chat_template.jinja")

_LOCAL_DIR = Path(__file__).parent

hf_cache = modal.Volume.from_name("smol-signals-vllm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("smol-signals-vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.12.0")
    .env({
        "HF_HOME": str(HF_CACHE_PATH),
        "HF_HUB_CACHE": str(HF_CACHE_PATH / "hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_file(
        _LOCAL_DIR / "gemma4_chat_template.jinja",
        remote_path=str(CHAT_TEMPLATE_PATH),
    )
)

app = modal.App(APP_NAME, image=vllm_image)


def _start_server() -> subprocess.Popen:
    lora_modules = json.dumps({
        "name": REQUEST_MODEL,
        "path": ADAPTER_MODEL,
        "base_model_name": BASE_MODEL,
    })
    limit_mm = json.dumps({"image": 0, "video": 0, "audio": 0})
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(CONTEXT_LENGTH),
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(N_GPUS),
        "--enforce-eager",
        "--chat-template",
        str(CHAT_TEMPLATE_PATH),
        "--chat-template-content-format",
        "openai",
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--limit-mm-per-prompt",
        limit_mm,
        "--enable-lora",
        "--lora-modules",
        lora_modules,
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--max-loras",
        "1",
        "--lora-target-modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "--uvicorn-log-level",
        "info",
    ]
    print("Starting vLLM Gemma 4 dev server:", flush=True)
    print(" ".join(cmd), flush=True)
    return subprocess.Popen(cmd, start_new_session=True)


def _check_running(process: subprocess.Popen) -> None:
    if (return_code := process.poll()) is not None:
        raise subprocess.CalledProcessError(return_code, process.args)


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES) -> None:
    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{PORT}/health"
    last_error = ""
    while time.time() < deadline:
        try:
            _check_running(process)
            with urllib.request.urlopen(health_url, timeout=10) as response:
                if response.status >= 400:
                    raise urllib.error.HTTPError(
                        health_url,
                        response.status,
                        response.reason,
                        response.headers,
                        None,
                    )
            return
        except (
            subprocess.CalledProcessError,
            TimeoutError,
            OSError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = str(exc)
            time.sleep(5)
    raise TimeoutError(
        f"vLLM server was not ready within {timeout} seconds. Last error: {last_error}"
    )


@app.function(
    image=vllm_image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={
        str(HF_CACHE_PATH): hf_cache,
        str(VLLM_CACHE_PATH): vllm_cache,
    },
    max_containers=1,
    scaledown_window=2 * MINUTES,
    timeout=30 * MINUTES,
)
@modal.web_server(PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    process = _start_server()
    wait_ready(process)


@app.local_entrypoint()
def main(
    dataset_path: str = "training/data/sft_gemma12b/val.jsonl",
    limit: int = 1,
    offset: int = 0,
    timeout: int = 1200,
    max_tokens: int = 512,
    structured_json_schema: bool = True,
    output_path: str = "training/output/vllm_gemma4_dev_eval.json",
) -> None:
    from serving.eval_sglang_endpoint import evaluate_examples

    base_url = serve.get_web_url()
    if not base_url:
        raise RuntimeError("Modal did not return a web URL for the vLLM server.")
    print(f"vLLM Gemma 4 dev URL: {base_url}", flush=True)
    summary = evaluate_examples(
        base_url,
        dataset_path,
        limit=limit,
        offset=offset,
        timeout=timeout,
        max_tokens=max_tokens,
        request_model=REQUEST_MODEL,
        structured_json_schema=structured_json_schema,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation output to {out}", flush=True)
