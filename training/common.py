"""Shared helpers for Smol Signals fine-tuning data.

These helpers intentionally reuse the app's schema and prompt text so the
student model is trained on the same task the Space serves at inference time.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from signals import SCHEMA_AND_RULES, SYSTEM_PROMPT, _build_prompt
from tickers import canonical_ticker, normalize_signal, truncate_for_prompt

TARGET_CHANNEL_TITLES = (
    "Defiant Gatekeeper",
    "Chicken Genius Singapore",
    "Financial Education",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_published(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_user_prompt(
    channel_title: str,
    video_title: str,
    published_at: str | None,
    transcript: str,
    *,
    max_transcript_chars: int = 40000,
) -> str:
    if max_transcript_chars == 40000:
        return _build_prompt(channel_title, video_title, parse_published(published_at), transcript)

    published = parse_published(published_at)
    published_text = published.isoformat() if published else "unknown"
    return "\n".join([
        f"Channel: {channel_title}",
        f"Video: {video_title}",
        f"Published at: {published_text}",
        "",
        SCHEMA_AND_RULES,
        "",
        "Transcript:",
        truncate_for_prompt(transcript, max_transcript_chars),
    ])


def compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def quote_in_transcript(quote: str, transcript: str) -> bool:
    quote_norm = normalize_ws(quote)
    if len(quote_norm) < 8:
        return False
    return quote_norm in normalize_ws(transcript)


def canonical_assistant_output(video: dict[str, Any]) -> dict[str, Any]:
    """Convert a stored VideoResult into the JSON object the extractor emits.

    The HF bucket stores normalized/scored app results, not the raw model JSON.
    This reconstructs a clean assistant target from the accepted calls.
    """
    assets: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []

    for call in video.get("calls") or []:
        signal = str(call.get("signal") or "").strip().lower()
        evidence = str(call.get("evidence_quote") or call.get("evidenceQuote") or "").strip()
        why = str(call.get("why_current") or call.get("whyThisIsCurrentCall") or "").strip()
        summary = str(call.get("summary") or video.get("summary") or "").strip()
        if len(evidence) < 8 or len(why) < 8:
            continue

        ticker_text = str(call.get("ticker") or "")
        if signal == "rotate" or "→" in ticker_text:
            parts = re.split(r"\s*→\s*", ticker_text, maxsplit=1)
            if len(parts) != 2:
                continue
            from_ticker = canonical_ticker(parts[0])
            to_ticker = canonical_ticker(parts[1])
            if not from_ticker or not to_ticker or from_ticker == to_ticker:
                continue
            rotations.append({
                "fromLabel": from_ticker,
                "fromTicker": from_ticker,
                "toLabel": to_ticker,
                "toTicker": to_ticker,
                "summary": summary,
                "evidenceQuote": evidence,
                "whyThisIsCurrentCall": why,
            })
            continue

        ticker = canonical_ticker(ticker_text)
        if not ticker:
            continue
        asset = {
            "ticker": ticker,
            "signal": normalize_signal(signal),
            "summary": summary,
            "evidenceQuote": evidence,
            "whyThisIsCurrentCall": why,
        }
        company_name = call.get("company_name") or call.get("companyName")
        if company_name:
            asset["companyName"] = str(company_name)
        assets.append(asset)

    if assets or rotations:
        top_signal = assets[0]["signal"] if assets else "hold"
        return {
            "signal": top_signal,
            "assets": assets,
            "rotations": rotations,
            "summary": str(video.get("summary") or "Current market call extracted.").strip(),
        }

    return {
        "signal": "NONE",
        "assets": [],
        "rotations": [],
        "summary": str(video.get("summary") or "No explicit current forward-looking call.").strip(),
    }


def is_grounded(label: dict[str, Any], transcript: str) -> bool:
    for asset in label.get("assets") or []:
        if not quote_in_transcript(str(asset.get("evidenceQuote") or ""), transcript):
            return False
    for rotation in label.get("rotations") or []:
        if not quote_in_transcript(str(rotation.get("evidenceQuote") or ""), transcript):
            return False
    return True


def make_messages(record: dict[str, Any], *, max_transcript_chars: int) -> list[dict[str, str]]:
    prompt = build_user_prompt(
        record["channel_title"],
        record["video_title"],
        record.get("published_at"),
        record["transcript"],
        max_transcript_chars=max_transcript_chars,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": compact_json(record["assistant_json"])},
    ]
