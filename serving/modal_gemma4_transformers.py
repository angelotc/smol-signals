"""Serve the fine-tuned Smol Signals Gemma 4 adapter on Modal with Transformers.

This is the production fallback for Gemma 4 while SGLang's generic
Transformers backend does not support Gemma 4's non-uniform KV-cache shapes.

Deploy:

    modal deploy serving/modal_gemma4_transformers.py

The web app exposes:

    {MODAL_GEMMA4_URL}/v1/chat/completions

with model `small-signals-gemma-4-12b:signals`.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import PurePosixPath
from typing import Any

import modal

MINUTES = 60

APP_NAME = "smol-signals-gemma4-transformers"
BASE_MODEL = "unsloth/gemma-4-12b"
ADAPTER_MODEL = "ajc426/small-signals-gemma-4-12b-sft-lora"
REQUEST_MODEL = "small-signals-gemma-4-12b:signals"
GPU = "L40S"
CONTEXT_LENGTH = 8192
DEFAULT_MAX_NEW_TOKENS = 1500

HF_CACHE_PATH = PurePosixPath("/root/.cache/huggingface")

hf_cache = modal.Volume.from_name("smol-signals-sglang-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.10.post1-cu130-runtime")
    .entrypoint([])
    .pip_install(
        "accelerate>=1.2.0",
        "fastapi[standard]>=0.115.0",
        "peft>=0.14.0",
        "sentencepiece>=0.2.0",
        "transformers==5.12.0",
    )
    .run_commands("python -m pip uninstall -y torchao")
    .env({
        "HF_HOME": str(HF_CACHE_PATH),
        "HF_HUB_CACHE": str(HF_CACHE_PATH / "hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App(APP_NAME, image=image)


def _message_content_text(content: Any) -> str:
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


def _gemma4_chat_text(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    parts = []
    for message in messages:
        role = "model" if message.get("role") == "assistant" else message.get("role", "user")
        content = _message_content_text(message.get("content", ""))
        parts.append(f"<|turn>{role}\n{content}<turn|>")
    if add_generation_prompt:
        parts.append("<|turn>model\n")
    return "\n".join(parts)


def _clean_output(text: str) -> str:
    cleaned = (text or "").strip()
    for stop in ("<turn|>", "<|turn>"):
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0].strip()
    return cleaned


def _looks_like_complete_json_object(text: str) -> bool:
    cleaned = _clean_output(text).strip()
    if not cleaned:
        return False
    if not cleaned.startswith("{"):
        brace_index = cleaned.find("{")
        if brace_index < 0:
            return False
        cleaned = cleaned[brace_index:]
    try:
        parsed, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and not cleaned[end:].strip()


@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={str(HF_CACHE_PATH): hf_cache},
    max_containers=1,
    scaledown_window=2 * MINUTES,
    timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve():
    import torch
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Loading processor for {BASE_MODEL}", flush=True)
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        token=token,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {BASE_MODEL}", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL,
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    )
    print(f"Loading adapter {ADAPTER_MODEL}", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, token=token)
    model.eval()
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    web_app = FastAPI(title="Smol Signals Gemma 4")

    @web_app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": REQUEST_MODEL,
            "backend": "transformers",
            "base_model": BASE_MODEL,
            "adapter_model": ADAPTER_MODEL,
        }

    @web_app.post("/v1/chat/completions")
    async def chat_completions(body=Body(...)) -> JSONResponse:
        messages = body.get("messages") or []
        max_new_tokens = int(body.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
        max_new_tokens = max(1, min(max_new_tokens, DEFAULT_MAX_NEW_TOKENS))

        prompt = _gemma4_chat_text(messages, add_generation_prompt=True)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=CONTEXT_LENGTH,
        ).to("cuda:0")
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        class JsonObjectStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, prompt_tokens:]
                if generated.shape[-1] < 8:
                    return False
                text = tokenizer.decode(generated, skip_special_tokens=False)
                return _looks_like_complete_json_object(text)

        started = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([JsonObjectStoppingCriteria()]),
            )
        generated_ids = output_ids[0, prompt_tokens:]
        content = _clean_output(tokenizer.decode(generated_ids, skip_special_tokens=False))
        completion_tokens = int(generated_ids.shape[-1])
        torch.cuda.empty_cache()

        response = {
            "id": f"chatcmpl-modal-{int(started * 1000)}",
            "object": "chat.completion",
            "created": int(started),
            "model": body.get("model") or REQUEST_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        print(
            f"Generated {completion_tokens} tokens in {time.time() - started:.2f}s",
            flush=True,
        )
        return JSONResponse(response)

    return web_app
