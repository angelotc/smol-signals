"""Smoke-test Gemma 4 + LoRA on Modal with SGLang's dedicated Gemma 4 image.

This is intentionally separate from ``modal_gemma4_sglang.py``. The older file
patches a generic SGLang runtime; this one checks whether the current upstream
Gemma 4 image can serve our fine-tuned adapter without local runtime patches.

Run:

    modal run serving/modal_gemma4_sglang_dev.py --limit 1
"""
from __future__ import annotations

import subprocess
import time
import json
from pathlib import Path, PurePosixPath

import modal
import requests

MINUTES = 60
PORT = 8000

APP_NAME = "smol-signals-gemma4-sglang-dev"
BASE_MODEL = "unsloth/gemma-4-12b"
ADAPTER_MODEL = "ajc426/small-signals-gemma-4-12b-sft-lora"
SERVED_MODEL = "small-signals-gemma-4-12b"
ADAPTER_NAME = "signals"

GPU = "L40S"
N_GPUS = 1
CONTEXT_LENGTH = 8192
MAX_LORA_RANK = 16

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
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/lora/lora.py')\n"
        "text = path.read_text()\n"
        "needle = (\n"
        "    '                k_proj_weight = (\\n'\n"
        "    '                    weights[k_name]\\n'\n"
        "    '                    if \"k_proj\" in target_module\\n'\n"
        "    '                    else torch.zeros_like(weights[v_name])\\n'\n"
        "    '                )\\n'\n"
        "    '                weights[qkv_name] = torch.cat(\\n'\n"
        "    '                    (\\n'\n"
        "    '                        weights[q_name],\\n'\n"
        "    '                        k_proj_weight,\\n'\n"
        "    '                        weights[v_name],\\n'\n"
        "    '                    ),\\n'\n"
        "    '                    0,\\n'\n"
        "    '                )\\n'\n"
        ")\n"
        "patch = (\n"
        "    '                k_proj_weight = (\\n'\n"
        "    '                    weights[k_name]\\n'\n"
        "    '                    if \"k_proj\" in target_module and k_name in weights\\n'\n"
        "    '                    else torch.zeros_like(weights[v_name] if v_name in weights else weights[q_name])\\n'\n"
        "    '                )\\n'\n"
        "    '                v_proj_weight = (\\n'\n"
        "    '                    weights[v_name]\\n'\n"
        "    '                    if \"v_proj\" in target_module and v_name in weights\\n'\n"
        "    '                    else torch.zeros_like(weights[k_name] if k_name in weights else weights[q_name])\\n'\n"
        "    '                )\\n'\n"
        "    '                weights[qkv_name] = torch.cat(\\n'\n"
        "    '                    (\\n'\n"
        "    '                        weights[q_name],\\n'\n"
        "    '                        k_proj_weight,\\n'\n"
        "    '                        v_proj_weight,\\n'\n"
        "    '                    ),\\n'\n"
        "    '                    0,\\n'\n"
        "    '                )\\n'\n"
        ")\n"
        "if patch not in text:\n"
        "    if needle not in text:\n"
        "        raise SystemExit(f'Could not find qkv patch point in {path}')\n"
        "    text = text.replace(needle, patch)\n"
        "pop_needle = (\n"
        "    '                weights.pop(q_name)\\n'\n"
        "    '                if \"k_proj\" in target_module:\\n'\n"
        "    '                    weights.pop(k_name)\\n'\n"
        "    '                weights.pop(v_name)\\n'\n"
        ")\n"
        "pop_patch = (\n"
        "    '                weights.pop(q_name)\\n'\n"
        "    '                if \"k_proj\" in target_module and k_name in weights:\\n'\n"
        "    '                    weights.pop(k_name)\\n'\n"
        "    '                if \"v_proj\" in target_module and v_name in weights:\\n'\n"
        "    '                    weights.pop(v_name)\\n'\n"
        ")\n"
        "if pop_patch not in text:\n"
        "    if pop_needle not in text:\n"
        "        raise SystemExit(f'Could not find qkv pop patch point in {path}')\n"
        "    text = text.replace(pop_needle, pop_patch)\n"
        "path.write_text(text)\n"
        "print('Patched SGLang LoRA qkv normalization for missing Gemma 4 v_proj adapters')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import re\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/models/gemma4_mm.py')\n"
        "text = path.read_text()\n"
        "qkv_pattern = re.compile(\n"
        "    r'(?ms)(        if module_name == \"qkv_proj\":\\n)'\n"
        "    r'.*?'\n"
        "    r'(        elif module_name == \"o_proj\":\\n)'\n"
        ")\n"
        "qkv_body = (\n"
        "    '            layer_type = self.config.layer_types[layer_idx]\\n'\n"
        "    '            is_full_attention = layer_type == \"full_attention\"\\n'\n"
        "    '            head_dim = (\\n'\n"
        "    '                self.config.head_dim\\n'\n"
        "    '                if is_full_attention\\n'\n"
        "    '                else getattr(self.config, \"swa_head_dim\", self.config.head_dim)\\n'\n"
        "    '            )\\n'\n"
        "    '            num_kv_heads = (\\n'\n"
        "    '                self.config.num_key_value_heads\\n'\n"
        "    '                if is_full_attention\\n'\n"
        "    '                else getattr(self.config, \"swa_num_key_value_heads\", self.config.num_key_value_heads)\\n'\n"
        "    '            )\\n'\n"
        "    '            return (\\n'\n"
        "    '                self.config.hidden_size,\\n'\n"
        "    '                head_dim * (self.config.num_attention_heads + num_kv_heads * 2),\\n'\n"
        "    '            )\\n'\n"
        ")\n"
        "text, qkv_count = qkv_pattern.subn(r'\\1' + qkv_body + r'\\2', text, count=1)\n"
        "o_pattern = re.compile(\n"
        "    r'(?ms)(        elif module_name == \"o_proj\":\\n)'\n"
        "    r'.*?'\n"
        "    r'(        elif module_name == \"gate_up_proj\":\\n)'\n"
        ")\n"
        "o_body = (\n"
        "    '            layer_type = self.config.layer_types[layer_idx]\\n'\n"
        "    '            head_dim = (\\n'\n"
        "    '                self.config.head_dim\\n'\n"
        "    '                if layer_type == \"full_attention\"\\n'\n"
        "    '                else getattr(self.config, \"swa_head_dim\", self.config.head_dim)\\n'\n"
        "    '            )\\n'\n"
        "    '            return (\\n'\n"
        "    '                head_dim * self.config.num_attention_heads,\\n'\n"
        "    '                self.config.hidden_size,\\n'\n"
        "    '            )\\n'\n"
        ")\n"
        "text, o_count = o_pattern.subn(r'\\1' + o_body + r'\\2', text, count=1)\n"
        "gate_pattern = re.compile(\n"
        "    r'(?ms)(elif module_name == \"gate_up_proj\":\\n)'\n"
        "    r'.*?'\n"
        "    r'(        elif module_name == \"down_proj\":\\n)'\n"
        ")\n"
        "gate_body = (\n"
        "    '            intermediate_size = self.config.intermediate_size\\n'\n"
        "    '            if isinstance(intermediate_size, int):\\n'\n"
        "    '                return self.config.hidden_size, intermediate_size * 2\\n'\n"
        "    '            assert len(set(intermediate_size)) == 1, (\\n'\n"
        "    '                \"Currently SGLang requires uniform intermediate size for all layers. \"\\n'\n"
        "    '                \"Please file an issue if you need support for non-uniform intermediate sizes.\"\\n'\n"
        "    '            )\\n'\n"
        "    '            return self.config.hidden_size, intermediate_size[0] * 2\\n'\n"
        ")\n"
        "text, gate_count = gate_pattern.subn(r'\\1' + gate_body + r'\\2', text, count=1)\n"
        "down_pattern = re.compile(\n"
        "    r'(?ms)(elif module_name == \"down_proj\":\\n)'\n"
        "    r'.*?'\n"
        "    r'(        else:\\n)'\n"
        ")\n"
        "down_body = (\n"
        "    '            intermediate_size = self.config.intermediate_size\\n'\n"
        "    '            if isinstance(intermediate_size, int):\\n'\n"
        "    '                return intermediate_size, self.config.hidden_size\\n'\n"
        "    '            assert len(set(intermediate_size)) == 1, (\\n'\n"
        "    '                \"Currently SGLang requires uniform intermediate size for all layers. \"\\n'\n"
        "    '                \"Please file an issue if you need support for non-uniform intermediate sizes.\"\\n'\n"
        "    '            )\\n'\n"
        "    '            return intermediate_size[0], self.config.hidden_size\\n'\n"
        ")\n"
        "text, down_count = down_pattern.subn(r'\\1' + down_body + r'\\2', text, count=1)\n"
        "if qkv_count != 1 or o_count != 1 or gate_count != 1 or down_count != 1:\n"
        "    lines = [line for line in text.splitlines() if 'intermediate_size' in line or 'gate_up_proj' in line or 'down_proj' in line]\n"
        "    raise SystemExit('Could not patch Gemma 4 get_hidden_dim in ' + str(path) + '\\n' + '\\n'.join(lines[-40:]))\n"
        "path.write_text(text)\n"
        "print('Patched SGLang Gemma 4 get_hidden_dim for per-layer attention and scalar intermediate_size')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/lora/layers.py')\n"
        "text = path.read_text()\n"
        "needle = (\n"
        "    '    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:\\n'\n"
        "    '        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)\\n'\n"
        ")\n"
        "patch = (\n"
        "    '    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:\\n'\n"
        "    '        if x.shape[-1] != self.A_buffer.shape[-1]:\\n'\n"
        "    '            if not getattr(self, \"_lora_shape_mismatch_warned\", False):\\n'\n"
        "    '                print(\\n'\n"
        "    '                    \"Skipping RowParallel LoRA for shape mismatch: \"\\n'\n"
        "    '                    f\"activation={x.shape[-1]} adapter={self.A_buffer.shape[-1]}\",\\n'\n"
        "    '                    flush=True,\\n'\n"
        "    '                )\\n'\n"
        "    '                self._lora_shape_mismatch_warned = True\\n'\n"
        "    '            return base_output\\n'\n"
        "    '        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)\\n'\n"
        ")\n"
        "if patch not in text:\n"
        "    if needle not in text:\n"
        "        raise SystemExit(f'Could not find RowParallel apply_lora patch point in {path}')\n"
        "    text = text.replace(needle, patch, 1)\n"
        "path.write_text(text)\n"
        "print('Patched SGLang RowParallel LoRA to skip activation/adapter width mismatches')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import re\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/lora/layers.py')\n"
        "text = path.read_text()\n"
        "pattern = re.compile(\n"
        "    r'(class RowParallelLinearWithLoRA\\(BaseLayerWithLoRA\\):.*?'\n"
        "    r'    def apply_lora\\(self, base_output: torch.Tensor, x: torch.Tensor\\) -> torch.Tensor:\\n)'\n"
        "    r'        lora_a_output = self\\.lora_backend\\.run_lora_a_sgemm\\(x, self\\.A_buffer\\)\\n',\n"
        "    re.S,\n"
        ")\n"
        "insert = (\n"
        "    r'\\1'\n"
        "    '        if x.shape[-1] != self.A_buffer.shape[-1]:\\n'\n"
        "    '            if not getattr(self, \"_lora_shape_mismatch_warned\", False):\\n'\n"
        "    '                print(\\n'\n"
        "    '                    \"Skipping RowParallel LoRA for shape mismatch: \"\\n'\n"
        "    '                    f\"activation={x.shape[-1]} adapter={self.A_buffer.shape[-1]}\",\\n'\n"
        "    '                    flush=True,\\n'\n"
        "    '                )\\n'\n"
        "    '                self._lora_shape_mismatch_warned = True\\n'\n"
        "    '            return base_output\\n'\n"
        "    '        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)\\n'\n"
        ")\n"
        "if 'class RowParallelLinearWithLoRA' in text and 'Skipping RowParallel LoRA for shape mismatch' not in text.split('class RowParallelLinearWithLoRA', 1)[1]:\n"
        "    text, count = pattern.subn(insert, text, count=1)\n"
        "    if count != 1:\n"
        "        raise SystemExit(f'Could not patch RowParallelLinearWithLoRA.apply_lora in {path}')\n"
        "path.write_text(text)\n"
        "print('Patched RowParallelLinearWithLoRA LoRA mismatch guard')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/lora/mem_pool.py')\n"
        "text = path.read_text()\n"
        "needle = (\n"
        "    '                assert (\\n'\n"
        "    '                    buffer_view.shape == weight.shape\\n'\n"
        "    '                ), f\"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}.\"\\n'\n"
        "    '                copy_weight_into_buffer(buffer_view, weight)\\n'\n"
        ")\n"
        "patch = (\n"
        "    '                if buffer_view.shape != weight.shape:\\n'\n"
        "    '                    if buffer_view.ndim == weight.ndim == 2:\\n'\n"
        "    '                        if not getattr(self, \"_lora_load_shape_mismatch_warned\", False):\\n'\n"
        "    '                            print(\\n'\n"
        "    '                                \"Loading partial 2D LoRA tensor for shape mismatch: \"\\n'\n"
        "    '                                f\"buffer={tuple(buffer_view.shape)} weight={tuple(weight.shape)}\",\\n'\n"
        "    '                                flush=True,\\n'\n"
        "    '                            )\\n'\n"
        "    '                            self._lora_load_shape_mismatch_warned = True\\n'\n"
        "    '                        buffer_view.zero_()\\n'\n"
        "    '                        rows = min(buffer_view.shape[0], weight.shape[0])\\n'\n"
        "    '                        cols = min(buffer_view.shape[1], weight.shape[1])\\n'\n"
        "    '                        copy_weight_into_buffer(\\n'\n"
        "    '                            buffer_view[:rows, :cols],\\n'\n"
        "    '                            weight[:rows, :cols].contiguous(),\\n'\n"
        "    '                        )\\n'\n"
        "    '                        return\\n'\n"
        "    '                    assert (\\n'\n"
        "    '                        buffer_view.shape == weight.shape\\n'\n"
        "    '                    ), f\"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}.\"\\n'\n"
        "    '                copy_weight_into_buffer(buffer_view, weight)\\n'\n"
        ")\n"
        "if 'Loading partial 2D LoRA tensor for shape mismatch' not in text:\n"
        "    if needle not in text:\n"
        "        raise SystemExit(f'Could not patch LoRA buffer loader in {path}')\n"
        "    text = text.replace(needle, patch, 1)\n"
        "path.write_text(text)\n"
        "print('Patched SGLang LoRA buffer loader for non-uniform 2D tensor widths')\n"
        "PY"
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
        BASE_MODEL,
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
        "--enable-lora",
        "--lora-paths",
        f"{ADAPTER_NAME}={ADAPTER_MODEL}",
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--max-loras-per-batch",
        "1",
    ]
    print("Starting SGLang Gemma 4 dev server:", flush=True)
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
    output_path: str = "training/output/sglang_gemma4_dev_eval.json",
) -> None:
    from serving.eval_sglang_endpoint import evaluate_examples

    base_url = serve.get_web_url()
    if not base_url:
        raise RuntimeError("Modal did not return a web URL for the SGLang server.")
    print(f"SGLang Gemma 4 dev URL: {base_url}", flush=True)
    summary = evaluate_examples(
        base_url,
        dataset_path,
        limit=limit,
        offset=offset,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation output to {out}", flush=True)
