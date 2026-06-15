"""Smoke-test the repaired merged Smol Signals Gemma 4 model on Modal with SGLang.

This serves the merged HF checkpoint directly. It intentionally does not use
SGLang runtime LoRA flags or the experimental LoRA shape patches from
`modal_gemma4_sglang_dev.py`.

Run:

    modal run serving/modal_gemma4_sglang_merged_repair_dev.py --limit 1
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path, PurePosixPath

import modal
import requests

MINUTES = 60
PORT = 8000

APP_NAME = "smol-signals-gemma4-sglang-merged-repair-dev"
MERGED_MODEL = "ajc426/small-signals-gemma-4-12b-sft-merged-repair"
SERVED_MODEL = "small-signals-gemma-4-12b-signals"

GPU = "L40S"
N_GPUS = 1
CONTEXT_LENGTH = 8192

HF_CACHE_PATH = PurePosixPath("/cache/huggingface")
CHAT_TEMPLATE_PATH = PurePosixPath("/opt/gemma4_chat_template.jinja")

_LOCAL_DIR = Path(__file__).parent

hf_cache = modal.Volume.from_name("smol-signals-sglang-hf-cache", create_if_missing=True)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-gemma-4-12B")
    .entrypoint([])
    .pip_install(
        "requests>=2.31",
        "git+https://github.com/huggingface/transformers.git@1423d22f7a3b62e8c70ad67b58ec25cd9b675897",
    )
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

app = modal.App(APP_NAME, image=sglang_image)


def _start_server() -> subprocess.Popen:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        MERGED_MODEL,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp",
        str(N_GPUS),
        "--trust-remote-code",
        "--skip-server-warmup",
        "--dtype",
        "bfloat16",
        "--context-length",
        str(CONTEXT_LENGTH),
        "--mem-fraction-static",
        "0.82",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--chat-template",
        str(CHAT_TEMPLATE_PATH),
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
    ]
    print("Starting repaired merged SGLang Gemma 4 dev server:", flush=True)
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
            response = requests.get(health_url, timeout=10)
            response.raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
        ) as exc:
            last_error = str(exc)
            time.sleep(5)
    raise TimeoutError(
        f"SGLang server was not ready within {timeout} seconds. Last error: {last_error}"
    )


@app.function(
    image=sglang_image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={str(HF_CACHE_PATH): hf_cache},
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
    structured_json_schema: bool = False,
    output_path: str = "training/output/sglang_gemma4_merged_repair_eval.json",
) -> None:
    from serving.eval_sglang_endpoint import evaluate_examples

    base_url = serve.get_web_url()
    if not base_url:
        raise RuntimeError("Modal did not return a web URL for the merged SGLang server.")
    print(f"Repaired merged SGLang Gemma 4 dev URL: {base_url}", flush=True)
    summary = evaluate_examples(
        base_url,
        dataset_path,
        limit=limit,
        offset=offset,
        timeout=timeout,
        max_tokens=max_tokens,
        request_model=SERVED_MODEL,
        structured_json_schema=structured_json_schema,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation output to {out}", flush=True)
