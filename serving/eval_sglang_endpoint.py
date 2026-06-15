"""Evaluate a deployed OpenAI-compatible chat endpoint against local SFT rows."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from signals import SIGNAL_JSON_SCHEMA

REQUEST_MODEL = "small-signals-gemma-4-12b-signals"


def _signal_json_schema() -> dict[str, Any]:
    return SIGNAL_JSON_SCHEMA


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
    requested_signal = str(payload.get("signal") or payload.get("overallSignal") or "").upper()
    if requested_signal != "NONE":
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


def _effective_signal(payload: dict[str, Any]) -> str:
    requested = str(payload.get("signal") or payload.get("overallSignal") or "").strip().upper()
    if requested == "NONE" and payload.get("rotations"):
        return "HOLD"
    return requested or "NONE"


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


def _post_chat(
    base_url: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
    request_model: str,
    structured_json_schema: bool,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": "Bearer EMPTY",
        "Content-Type": "application/json",
    }
    payload = {
        "model": request_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if structured_json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "signal_extraction",
                "schema": _signal_json_schema(),
                "strict": True,
            },
        }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    print(f"Endpoint status {response.status_code}", flush=True)
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
    max_tokens: int,
    request_model: str = REQUEST_MODEL,
    structured_json_schema: bool = False,
) -> dict[str, Any]:
    rows = _load_rows(dataset_path, limit, offset)
    print(f"Loaded {len(rows)} eval rows from {dataset_path}", flush=True)
    results = []
    for index, row in enumerate(rows, start=offset):
        messages = row["messages"][:-1]
        target = _target_payload(row)
        transcript = _extract_transcript(messages)
        started = time.time()
        item = {
            "index": index,
            "video_id": row.get("meta", {}).get("video_id"),
            "channel_title": row.get("meta", {}).get("channel_title"),
            "valid_json": False,
            "grounded": False,
            "target_signal": str(target.get("signal") or "").upper(),
            "pred_signal": "",
            "target_tickers": sorted(_ticker_set(target)),
            "pred_tickers": [],
            "latency_s": 0.0,
            "error": "",
        }
        try:
            raw_response = _post_chat(
                base_url,
                messages,
                timeout,
                max_tokens,
                request_model,
                structured_json_schema,
            )
            content = raw_response["choices"][0]["message"].get("content") or ""
            parsed = _safe_json_parse(content)
            item.update({
                "valid_json": True,
                "grounded": _grounded(parsed, transcript),
                "raw_pred_signal": str(parsed.get("signal") or "").upper(),
                "pred_signal": _effective_signal(parsed),
                "pred_tickers": sorted(_ticker_set(parsed)),
                "raw": parsed,
            })
        except Exception as exc:
            item["error"] = str(exc)[:1000]
            if "content" in locals() and content:
                item["raw_content_preview"] = content[:4000]
        item["latency_s"] = round(time.time() - started, 2)
        item["signal_match"] = item["target_signal"] == item["pred_signal"]
        item["ticker_overlap"] = bool(set(item["target_tickers"]) & set(item["pred_tickers"]))
        results.append(item)
        print(
            f"[{index}] {item['channel_title']} {item['video_id']} "
            f"signal {item['target_signal']}->{item['pred_signal']} "
            f"grounded={item['grounded']} tickers={item['pred_tickers']} "
            f"latency={item['latency_s']}s"
            f"{' error=' + item['error'][:160] if item['error'] else ''}",
            flush=True,
        )

    total = len(results)
    summary = {
        "model": request_model,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", default="training/data/sft_gemma12b/val.jsonl")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-model", default=REQUEST_MODEL)
    parser.add_argument("--structured-json-schema", action="store_true")
    parser.add_argument("--output-path", default="training/output/sglang_gemma4_eval.json")
    args = parser.parse_args()

    summary = evaluate_examples(
        args.base_url,
        args.dataset_path,
        limit=args.limit,
        offset=args.offset,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        request_model=args.request_model,
        structured_json_schema=args.structured_json_schema,
    )
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation output to {out}", flush=True)


if __name__ == "__main__":
    main()
