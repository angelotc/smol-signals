"""Serve and evaluate the fine-tuned Smol Signals Gemma 4 adapter on Modal.

Examples:

    modal run serving/modal_gemma4_sglang.py --limit 8
    modal deploy serving/modal_gemma4_sglang.py

The web server exposes SGLang's OpenAI-compatible API. The app should call:

    {MODAL_SGLANG_URL}/v1/chat/completions

with model `small-signals-gemma-4-12b:signals`.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import requests

MINUTES = 60
PORT = 8000

APP_NAME = "smol-signals-gemma4-sglang"
BASE_MODEL = "unsloth/gemma-4-12b"
ADAPTER_MODEL = "ajc426/small-signals-gemma-4-12b-sft-lora"
SERVED_MODEL = "small-signals-gemma-4-12b"
ADAPTER_NAME = "signals"
REQUEST_MODEL = f"{SERVED_MODEL}:{ADAPTER_NAME}"

GPU = "L40S"
N_GPUS = 1
TARGET_INPUTS = 1
CONTEXT_LENGTH = 8192
MAX_LORA_RANK = 16

HF_CACHE_PATH = PurePosixPath("/root/.cache/huggingface")
CHAT_TEMPLATE_PATH = PurePosixPath("/opt/gemma4_chat_template.jinja")

_LOCAL_DIR = Path(__file__).parent
_SERVER_PROCESS: subprocess.Popen | None = None

hf_cache = modal.Volume.from_name("smol-signals-sglang-hf-cache", create_if_missing=True)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.10.post1-cu130-runtime")
    .entrypoint([])
    .pip_install(
        "requests>=2.31",
        "transformers==5.12.0",
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/models/transformers.py')\n"
        "text = path.read_text()\n"
        "needle = 'self.ignore_unexpected_prefixes.extend([\"classifier.\", \"score.\"])'\n"
        "patch = (\n"
        "    'self.ignore_unexpected_prefixes.extend([\"classifier.\", \"score.\"])\\n'\n"
        "    '        self.ignore_unexpected_prefixes.extend([\\n'\n"
        "    '            \"model.embed_vision.\",\\n'\n"
        "    '            \"model.embed_audio.\",\\n'\n"
        "    '            \"model.embed_video.\",\\n'\n"
        "    '            \"model.multi_modal_projector.\",\\n'\n"
        "    '            \"model.vision_tower.\",\\n'\n"
        "    '            \"model.vision_embedder.\",\\n'\n"
        "    '            \"model.audio_embedder.\",\\n'\n"
        "    '            \"model.video_embedder.\",\\n'\n"
        "    '            \"model.image_embedder.\",\\n'\n"
        "    '            \"model.mm_projector.\",\\n'\n"
        "    '        ])'\n"
        ")\n"
        "if patch not in text:\n"
        "    if needle not in text:\n"
        "        raise SystemExit(f'Could not find patch point in {path}')\n"
        "    text = text.replace(needle, patch)\n"
        "    path.write_text(text)\n"
        "print('Patched SGLang Transformers backend to ignore Gemma 4 multimodal-only weights')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/models/transformers.py')\n"
        "text = path.read_text()\n"
        "needle = 'def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> nn.Module:\\n'\n"
        "patch = needle + '    return rms_norm\\n'\n"
        "if patch not in text:\n"
        "    if needle not in text:\n"
        "        raise SystemExit(f'Could not find RMSNorm patch point in {path}')\n"
        "    text = text.replace(needle, patch)\n"
        "    path.write_text(text)\n"
        "print('Patched SGLang Transformers backend to keep native Gemma 4 RMSNorm modules')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/models/transformers.py')\n"
        "text = path.read_text()\n"
        "top_needle = (\n"
        "    '        num_heads = self.text_config.num_attention_heads\\n'\n"
        "    '        num_kv_heads = getattr(self.text_config, \"num_key_value_heads\", num_heads)\\n'\n"
        "    '        hidden_size = self.text_config.hidden_size\\n'\n"
        "    '        head_dim = getattr(self.text_config, \"head_dim\", hidden_size // num_heads)\\n'\n"
        ")\n"
        "top_patch = (\n"
        "    '        num_heads = self.text_config.num_attention_heads\\n'\n"
        "    '        default_num_kv_heads = getattr(self.text_config, \"num_key_value_heads\", num_heads)\\n'\n"
        "    '        hidden_size = self.text_config.hidden_size\\n'\n"
        "    '        default_head_dim = getattr(self.text_config, \"head_dim\", hidden_size // num_heads)\\n'\n"
        "    '        global_head_dim = getattr(self.text_config, \"global_head_dim\", None)\\n'\n"
        "    '        num_global_kv_heads = getattr(\\n'\n"
        "    '            self.text_config, \"num_global_key_value_heads\", default_num_kv_heads\\n'\n"
        "    '        )\\n'\n"
        "    '        attention_k_eq_v = getattr(self.text_config, \"attention_k_eq_v\", False)\\n'\n"
        ")\n"
        "loop_needle = (\n"
        "    '        for idx in range(self.start_layer, self.end_layer):\\n'\n"
        "    '            # Per-layer sliding window (e.g. Gemma2, Cohere)\\n'\n"
        "    '            per_layer_sliding_window = -1\\n'\n"
        "    '            if (\\n'\n"
        "    '                layer_types is not None\\n'\n"
        "    '                and idx < len(layer_types)\\n'\n"
        "    '                and layer_types[idx] == \"sliding_attention\"\\n'\n"
        "    '                and global_sliding_window is not None\\n'\n"
        "    '            ):\\n'\n"
        "    '                per_layer_sliding_window = global_sliding_window\\n'\n"
        ")\n"
        "loop_patch = (\n"
        "    '        for idx in range(self.start_layer, self.end_layer):\\n'\n"
        "    '            layer_type = (\\n'\n"
        "    '                layer_types[idx]\\n'\n"
        "    '                if layer_types is not None and idx < len(layer_types)\\n'\n"
        "    '                else None\\n'\n"
        "    '            )\\n'\n"
        "    '            is_sliding_layer = layer_type == \"sliding_attention\"\\n'\n"
        "    '            # Per-layer sliding window (e.g. Gemma2, Cohere)\\n'\n"
        "    '            per_layer_sliding_window = -1\\n'\n"
        "    '            if is_sliding_layer and global_sliding_window is not None:\\n'\n"
        "    '                per_layer_sliding_window = global_sliding_window\\n'\n"
        "    '\\n'\n"
        "    '            layer_head_dim = default_head_dim\\n'\n"
        "    '            layer_num_kv_heads = default_num_kv_heads\\n'\n"
        "    '            if attention_k_eq_v and not is_sliding_layer and global_head_dim:\\n'\n"
        "    '                layer_head_dim = global_head_dim\\n'\n"
        "    '                layer_num_kv_heads = num_global_kv_heads\\n'\n"
        ")\n"
        "attn_needle = (\n"
        "    '                head_dim=head_dim,\\n'\n"
        "    '                scaling=head_dim**-0.5,\\n'\n"
        "    '                num_kv_heads=divide(num_kv_heads, tp_size),\\n'\n"
        ")\n"
        "attn_patch = (\n"
        "    '                head_dim=layer_head_dim,\\n'\n"
        "    '                scaling=layer_head_dim**-0.5,\\n'\n"
        "    '                num_kv_heads=divide(layer_num_kv_heads, tp_size),\\n'\n"
        ")\n"
        "if top_patch not in text:\n"
        "    if top_needle not in text:\n"
        "        raise SystemExit(f'Could not find attention metadata patch point in {path}')\n"
        "    text = text.replace(top_needle, top_patch)\n"
        "if loop_patch not in text:\n"
        "    if loop_needle not in text:\n"
        "        raise SystemExit(f'Could not find attention loop patch point in {path}')\n"
        "    text = text.replace(loop_needle, loop_patch)\n"
        "if attn_patch not in text:\n"
        "    if attn_needle not in text:\n"
        "        raise SystemExit(f'Could not find RadixAttention constructor patch point in {path}')\n"
        "    text = text.replace(attn_needle, attn_patch)\n"
        "path.write_text(text)\n"
        "print('Patched SGLang Transformers backend for Gemma 4 per-layer KV head dimensions')\n"
        "PY"
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
        "    path.write_text(text)\n"
        "print('Patched SGLang LoRA qkv normalization for missing Gemma 4 v_proj adapters')\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/sgl-workspace/sglang/python/sglang/srt/lora/layers.py')\n"
        "text = path.read_text()\n"
        "column_needle = (\n"
        "    '    def forward(self, input_: torch.Tensor):\\n'\n"
        "    '        # duplicate the logic in ColumnParallelLinear\\n'\n"
        "    '        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None\\n'\n"
        "    '        output_parallel = self.base_layer.quant_method.apply(\\n'\n"
        "    '            self.base_layer, input_, bias\\n'\n"
        "    '        )\\n'\n"
        "    '\\n'\n"
        "    '        if self.set_lora:\\n'\n"
        "    '            output_parallel = self.apply_lora(output_parallel, input_)\\n'\n"
        "    '\\n'\n"
        "    '        if self.base_layer.gather_output:\\n'\n"
        "    '            output = tensor_model_parallel_all_gather(output_parallel)\\n'\n"
        "    '        else:\\n'\n"
        "    '            output = output_parallel\\n'\n"
        "    '        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None\\n'\n"
        "    '        return output, output_bias\\n'\n"
        ")\n"
        "column_patch = (\n"
        "    '    def forward(self, input_: torch.Tensor):\\n'\n"
        "    '        orig_shape = input_.shape\\n'\n"
        "    '        restore_shape = input_.ndim > 2\\n'\n"
        "    '        if restore_shape:\\n'\n"
        "    '            input_ = input_.reshape(-1, input_.shape[-1]).contiguous()\\n'\n"
        "    '\\n'\n"
        "    '        # duplicate the logic in ColumnParallelLinear\\n'\n"
        "    '        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None\\n'\n"
        "    '        output_parallel = self.base_layer.quant_method.apply(\\n'\n"
        "    '            self.base_layer, input_, bias\\n'\n"
        "    '        )\\n'\n"
        "    '\\n'\n"
        "    '        if self.set_lora:\\n'\n"
        "    '            output_parallel = self.apply_lora(output_parallel, input_)\\n'\n"
        "    '\\n'\n"
        "    '        if self.base_layer.gather_output:\\n'\n"
        "    '            output = tensor_model_parallel_all_gather(output_parallel)\\n'\n"
        "    '        else:\\n'\n"
        "    '            output = output_parallel\\n'\n"
        "    '        if restore_shape:\\n'\n"
        "    '            output = output.reshape(*orig_shape[:-1], output.shape[-1])\\n'\n"
        "    '        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None\\n'\n"
        "    '        if getattr(self.base_layer, \"parent_cls\", None) is not None:\\n'\n"
        "    '            return output\\n'\n"
        "    '        return output, output_bias\\n'\n"
        ")\n"
        "row_needle = (\n"
        "    '    def forward(self, input_: torch.Tensor, skip_all_reduce=False):\\n'\n"
        "    '        if self.base_layer.input_is_parallel:\\n'\n"
        "    '            input_parallel = input_\\n'\n"
        "    '        else:\\n'\n"
        "    '            tp_rank = get_tensor_model_parallel_rank()\\n'\n"
        "    '            splitted_input = split_tensor_along_last_dim(\\n'\n"
        "    '                input_, num_partitions=self.base_layer.tp_size\\n'\n"
        "    '            )\\n'\n"
        "    '            input_parallel = splitted_input[tp_rank].contiguous()\\n'\n"
        "    '\\n'\n"
        "    '        bias_ = (\\n'\n"
        "    '            None\\n'\n"
        "    '            if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)\\n'\n"
        "    '            else self.base_layer.bias\\n'\n"
        "    '        )\\n'\n"
        "    '        output_parallel = self.base_layer.quant_method.apply(\\n'\n"
        "    '            self.base_layer, input_parallel, bias=bias_\\n'\n"
        "    '        )\\n'\n"
        "    '\\n'\n"
        "    '        should_reduce = (\\n'\n"
        "    '            self.base_layer.reduce_results\\n'\n"
        "    '            and self.base_layer.tp_size > 1\\n'\n"
        "    '            and not skip_all_reduce\\n'\n"
        "    '        )\\n'\n"
        "    '\\n'\n"
        "    '        if self.set_lora and should_reduce:\\n'\n"
        "    '            lora_a_output = self.lora_backend.run_lora_a_sgemm(\\n'\n"
        "    '                input_parallel, self.A_buffer\\n'\n"
        "    '            )\\n'\n"
        "    '            output_ = tensor_model_parallel_all_reduce(output_parallel)\\n'\n"
        "    '            lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)\\n'\n"
        "    '            output_ = self.lora_backend.run_lora_b_sgemm(\\n'\n"
        "    '                x=lora_a_output,\\n'\n"
        "    '                weights=self.B_buffer,\\n'\n"
        "    '                output_offset=self.output_offset,\\n'\n"
        "    '                base_output=output_,\\n'\n"
        "    '            )\\n'\n"
        "    '        else:\\n'\n"
        "    '            if self.set_lora:\\n'\n"
        "    '                output_parallel = self.apply_lora(output_parallel, input_parallel)\\n'\n"
        "    '            if should_reduce:\\n'\n"
        "    '                output_ = tensor_model_parallel_all_reduce(output_parallel)\\n'\n"
        "    '            else:\\n'\n"
        "    '                output_ = output_parallel\\n'\n"
        "    '\\n'\n"
        "    '        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None\\n'\n"
        "    '        return output_, output_bias\\n'\n"
        ")\n"
        "row_patch = (\n"
        "    '    def forward(self, input_: torch.Tensor, skip_all_reduce=False):\\n'\n"
        "    '        orig_shape = input_.shape\\n'\n"
        "    '        restore_shape = input_.ndim > 2\\n'\n"
        "    '        if restore_shape:\\n'\n"
        "    '            input_ = input_.reshape(-1, input_.shape[-1]).contiguous()\\n'\n"
        "    '\\n'\n"
        "    '        if self.base_layer.input_is_parallel:\\n'\n"
        "    '            input_parallel = input_\\n'\n"
        "    '        else:\\n'\n"
        "    '            tp_rank = get_tensor_model_parallel_rank()\\n'\n"
        "    '            splitted_input = split_tensor_along_last_dim(\\n'\n"
        "    '                input_, num_partitions=self.base_layer.tp_size\\n'\n"
        "    '            )\\n'\n"
        "    '            input_parallel = splitted_input[tp_rank].contiguous()\\n'\n"
        "    '\\n'\n"
        "    '        bias_ = (\\n'\n"
        "    '            None\\n'\n"
        "    '            if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)\\n'\n"
        "    '            else self.base_layer.bias\\n'\n"
        "    '        )\\n'\n"
        "    '        output_parallel = self.base_layer.quant_method.apply(\\n'\n"
        "    '            self.base_layer, input_parallel, bias=bias_\\n'\n"
        "    '        )\\n'\n"
        "    '\\n'\n"
        "    '        should_reduce = (\\n'\n"
        "    '            self.base_layer.reduce_results\\n'\n"
        "    '            and self.base_layer.tp_size > 1\\n'\n"
        "    '            and not skip_all_reduce\\n'\n"
        "    '        )\\n'\n"
        "    '\\n'\n"
        "    '        if self.set_lora and should_reduce:\\n'\n"
        "    '            lora_a_output = self.lora_backend.run_lora_a_sgemm(\\n'\n"
        "    '                input_parallel, self.A_buffer\\n'\n"
        "    '            )\\n'\n"
        "    '            output_ = tensor_model_parallel_all_reduce(output_parallel)\\n'\n"
        "    '            lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)\\n'\n"
        "    '            output_ = self.lora_backend.run_lora_b_sgemm(\\n'\n"
        "    '                x=lora_a_output,\\n'\n"
        "    '                weights=self.B_buffer,\\n'\n"
        "    '                output_offset=self.output_offset,\\n'\n"
        "    '                base_output=output_,\\n'\n"
        "    '            )\\n'\n"
        "    '        else:\\n'\n"
        "    '            if self.set_lora:\\n'\n"
        "    '                output_parallel = self.apply_lora(output_parallel, input_parallel)\\n'\n"
        "    '            if should_reduce:\\n'\n"
        "    '                output_ = tensor_model_parallel_all_reduce(output_parallel)\\n'\n"
        "    '            else:\\n'\n"
        "    '                output_ = output_parallel\\n'\n"
        "    '\\n'\n"
        "    '        if restore_shape:\\n'\n"
        "    '            output_ = output_.reshape(*orig_shape[:-1], output_.shape[-1])\\n'\n"
        "    '        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None\\n'\n"
        "    '        if getattr(self.base_layer, \"parent_cls\", None) is not None:\\n'\n"
        "    '            return output_\\n'\n"
        "    '        return output_, output_bias\\n'\n"
        ")\n"
        "if column_patch not in text:\n"
        "    if column_needle not in text:\n"
        "        raise SystemExit(f'Could not find ColumnParallelLinearWithLoRA patch point in {path}')\n"
        "    text = text.replace(column_needle, column_patch)\n"
        "if row_patch not in text:\n"
        "    if row_needle not in text:\n"
        "        raise SystemExit(f'Could not find RowParallelLinearWithLoRA patch point in {path}')\n"
        "    text = text.replace(row_needle, row_patch)\n"
        "path.write_text(text)\n"
        "print('Patched SGLang LoRA linear wrappers for Gemma 4 Transformers 3D hidden states')\n"
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
        "python",
        "-m",
        "sglang.launch_server",
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
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.82",
        "--enable-metrics",
        "--chat-template",
        str(CHAT_TEMPLATE_PATH),
        "--enable-lora",
        "--lora-paths",
        f"{ADAPTER_NAME}={ADAPTER_MODEL}",
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--max-loras-per-batch",
        "1",
    ]
    print("Starting SGLang server:")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def _check_running(process: subprocess.Popen) -> None:
    if (return_code := process.poll()) is not None:
        raise subprocess.CalledProcessError(return_code, process.args)


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES) -> None:
    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        try:
            _check_running(process)
            requests.get(health_url, timeout=10).raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
        ):
            time.sleep(5)
    raise TimeoutError(f"SGLang server was not ready within {timeout} seconds")


@app.function(
    image=sglang_image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={str(HF_CACHE_PATH): hf_cache},
    max_containers=1,
    scaledown_window=10 * MINUTES,
    timeout=30 * MINUTES,
)
@modal.web_server(PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    global _SERVER_PROCESS
    _SERVER_PROCESS = _start_server()
    wait_ready(_SERVER_PROCESS)


def _safe_json_parse(text: str) -> dict[str, Any]:
    trimmed = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed, re.IGNORECASE)
    if fenced:
        trimmed = fenced.group(1)
    if not trimmed.startswith("{"):
        brace = re.search(r"\{[\s\S]*\}", trimmed)
        if brace:
            trimmed = brace.group(0)
    parsed = json.loads(trimmed)
    if not isinstance(parsed, dict):
        raise ValueError("model output was not a JSON object")
    return parsed


def _norm_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _extract_transcript(messages: list[dict[str, str]]) -> str:
    user_content = messages[-1]["content"] if messages else ""
    marker = "\nTranscript:\n"
    if marker in user_content:
        return user_content.split(marker, 1)[1]
    return user_content


def _target_payload(row: dict[str, Any]) -> dict[str, Any]:
    return _safe_json_parse(row["messages"][-1]["content"])


def _ticker_set(payload: dict[str, Any]) -> set[str]:
    tickers = set()
    for asset in payload.get("assets") or []:
        if isinstance(asset, dict) and asset.get("ticker"):
            tickers.add(str(asset["ticker"]).upper())
    for rotation in payload.get("rotations") or []:
        if not isinstance(rotation, dict):
            continue
        from_ticker = rotation.get("fromTicker")
        to_ticker = rotation.get("toTicker")
        if from_ticker and to_ticker:
            tickers.add(f"{str(from_ticker).upper()}->{str(to_ticker).upper()}")
    return tickers


def _grounded(payload: dict[str, Any], transcript: str) -> bool:
    haystack = _norm_ws(transcript)
    calls = list(payload.get("assets") or []) + list(payload.get("rotations") or [])
    for call in calls:
        if not isinstance(call, dict):
            return False
        quote = call.get("evidenceQuote")
        if not isinstance(quote, str) or _norm_ws(quote) not in haystack:
            return False
        why = call.get("whyThisIsCurrentCall")
        if not isinstance(why, str) or len(why.strip()) < 8:
            return False
    return True


def _post_chat(base_url: str, messages: list[dict[str, str]], timeout: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": "Bearer EMPTY",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REQUEST_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    print(f"SGLang status {response.status_code}", flush=True)
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text[:1000]}",
            response=response,
        )
    return response.json()


def _load_rows(path: str, limit: int, offset: int) -> list[dict[str, Any]]:
    if limit == 0:
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[offset: offset + limit if limit > 0 else None]


def evaluate_examples(
    base_url: str,
    dataset_path: str,
    *,
    limit: int,
    offset: int,
    timeout: int,
) -> dict[str, Any]:
    rows = _load_rows(dataset_path, limit, offset)
    print(f"Loaded {len(rows)} eval rows from {dataset_path}", flush=True)
    results = []
    for index, row in enumerate(rows, start=offset):
        messages = row["messages"][:-1]
        target = _target_payload(row)
        transcript = _extract_transcript(messages)
        started = time.time()
        raw_response = _post_chat(base_url, messages, timeout)
        content = raw_response["choices"][0]["message"].get("content") or ""
        parsed = _safe_json_parse(content)
        elapsed = time.time() - started

        item = {
            "index": index,
            "video_id": row.get("meta", {}).get("video_id"),
            "channel_title": row.get("meta", {}).get("channel_title"),
            "valid_json": True,
            "grounded": _grounded(parsed, transcript),
            "target_signal": str(target.get("signal") or "").upper(),
            "pred_signal": str(parsed.get("signal") or "").upper(),
            "target_tickers": sorted(_ticker_set(target)),
            "pred_tickers": sorted(_ticker_set(parsed)),
            "latency_s": round(elapsed, 2),
            "raw": parsed,
        }
        item["signal_match"] = item["target_signal"] == item["pred_signal"]
        item["ticker_overlap"] = bool(set(item["target_tickers"]) & set(item["pred_tickers"]))
        results.append(item)
        print(
            f"[{index}] {item['channel_title']} {item['video_id']} "
            f"signal {item['target_signal']}->{item['pred_signal']} "
            f"grounded={item['grounded']} tickers={item['pred_tickers']} "
            f"latency={item['latency_s']}s",
            flush=True,
        )

    total = len(results)
    summary = {
        "model": REQUEST_MODEL,
        "dataset_path": dataset_path,
        "total": total,
        "valid_json": sum(1 for r in results if r["valid_json"]),
        "grounded": sum(1 for r in results if r["grounded"]),
        "signal_matches": sum(1 for r in results if r["signal_match"]),
        "non_none_ticker_overlaps": sum(
            1
            for r in results
            if r["target_signal"] != "NONE" and r["ticker_overlap"]
        ),
        "results": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    dataset_path: str = "training/data/sft_gemma12b/val.jsonl",
    limit: int = 8,
    offset: int = 0,
    timeout: int = 600,
    output_path: str = "training/output/sglang_gemma4_eval.json",
) -> None:
    base_url = serve.get_web_url()
    if not base_url:
        raise RuntimeError("Modal did not return a web URL for the SGLang server.")
    print(f"SGLang URL: {base_url}", flush=True)
    summary = evaluate_examples(
        base_url,
        dataset_path,
        limit=limit,
        offset=offset,
        timeout=timeout,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation output to {out}", flush=True)
