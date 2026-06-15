"""Extract directional stock calls from a transcript using a small (<=32B) model.

Ported from src/lib/youtube/summarize.ts. In production this can call the
fine-tuned Gemma 4 12B Modal endpoint; otherwise it falls back to Hugging Face
Inference so local/dev deployments still work without Modal.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import requests
from huggingface_hub import InferenceClient

from tickers import canonical_ticker, normalize_signal, truncate_for_prompt

HF_MODEL = os.environ.get("HF_MODEL", "google/gemma-3-27b-it")
MODAL_GEMMA4_URL = os.environ.get("MODAL_GEMMA4_URL", "").rstrip("/")
MODAL_GEMMA4_MODEL = os.environ.get(
    "MODAL_GEMMA4_MODEL",
    "small-signals-gemma-4-12b-signals",
)
DEFAULT_MODEL = MODAL_GEMMA4_MODEL if MODAL_GEMMA4_URL else HF_MODEL


@dataclass
class AssetCall:
    ticker: str
    signal: str
    summary: str
    evidence_quote: str
    why_current: str
    company_name: str | None = None


@dataclass
class RotationCall:
    from_label: str
    from_ticker: str
    to_label: str
    to_ticker: str
    summary: str
    evidence_quote: str
    why_current: str


@dataclass
class ExtractedSignal:
    has_call: bool
    signal: str
    summary: str
    assets: list[AssetCall] = field(default_factory=list)
    rotations: list[RotationCall] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    raw: object = None


SYSTEM_PROMPT = (
    "You read YouTube transcripts and extract explicit directional stock-market "
    "calls. You output JSON only. Do not output any numeric fields."
)

SCHEMA_AND_RULES = """Schema:
{
  "signal": "buy" | "sell" | "hold" | "NONE",
  "assets": [{
    "ticker": "AAPL",
    "signal": "buy" | "sell" | "hold",
    "summary": "short description of the current call",
    "evidenceQuote": "short exact transcript quote proving the current call",
    "whyThisIsCurrentCall": "why this is a current forward-looking recommendation or forecast"
  }],
  "rotations": [{
    "fromLabel": "growth stocks", "fromTicker": "QQQ",
    "toLabel": "defensive healthcare stocks", "toTicker": "XLV",
    "summary": "rotate away from growth into defensive healthcare",
    "evidenceQuote": "short exact transcript quote proving the rotation call",
    "whyThisIsCurrentCall": "why this is a current forward-looking recommendation"
  }],
  "summary": "one short sentence, or why this is NONE"
}
Rules:
- Do not output confidence, probability, price target, return, or rating score.
- A measurable call requires a current, forward-looking recommendation or forecast made by the creator as of the publish date.
- Only return buy/sell/hold for clear directional calls about publicly traded stocks, the broad US stock market, or major cryptocurrencies (e.g. BTC, ETH, SOL).
- Every asset call MUST include evidenceQuote and whyThisIsCurrentCall. Without a direct evidence quote, drop the ticker.
- assets must contain only exchange tickers like AAPL, NVDA, TSLA, BRK.B, SPY, or major crypto symbols like BTC, ETH, SOL (use the bare symbol, e.g. BTC not BTC-USD). Stick to widely-traded coins; drop obscure altcoins. Never put themes, products, sectors, or company names in assets.
- assets must use unique tickers. Do not repeat a ticker. Return only the strongest direct calls, normally 1-3 assets.
- Do not create calls for companies mentioned only as examples, customers, suppliers, competitors, lawsuit counterparties, acquisition targets, or sector peers. Use the ticker of the stock the creator is actually recommending or forecasting.
- rotations must be an array; use [] when there is no explicit rotation call.
- Rotation proxies: growth/tech=QQQ; defensive/healthcare=XLV; value=IVE; utilities=XLU; staples=XLP; energy=XLE; small caps=IWM; bonds/Treasuries=TLT; cash/T-bills=SGOV.
- Do not count educational, historical, hypothetical, or old past-purchase mentions as calls.
- A disclosed current trade counts as a call when the creator says they are buying/selling now, recently closed one position to buy another, or still intend to hold/sell a named asset.
- A valuation or price-target statement can imply a buy/sell/hold call, but do not output a price target field.
- "companies like Tesla", "Tesla is an example", "I bought Tesla years ago" are NOT current calls.
- If the video is about collectibles, personal stories, entertainment, education, or any non-stock topic, return NONE with [] assets and [] rotations.
- Use SPY only for direct broad US stock-market calls. Generic portfolio language like "hold my investments" or "be careful with new investments" is NONE unless the creator explicitly calls the broad market or SPY itself.

Examples that must return NONE:
Transcript: "Tesla is a good example of narrative investing. People who bought it early did well."
Output: {"signal":"NONE","assets":[],"rotations":[],"summary":"Educational example, not a current call."}
Transcript: "I will hold my current investments and be careful making new investments."
Output: {"signal":"NONE","assets":[],"rotations":[],"summary":"Generic portfolio caution without a named asset or explicit market call."}
Transcript: "I am buying Tesla now because I think it will outperform over the next year."
Output: {"signal":"buy","assets":[{"ticker":"TSLA","signal":"buy","summary":"Current buy call on TSLA.","evidenceQuote":"I am buying Tesla now because I think it will outperform over the next year.","whyThisIsCurrentCall":"The speaker says they are buying now and expects future outperformance."}],"rotations":[],"summary":"Current buy call on TSLA."}
Transcript: "I closed my GM short position to buy more Tesla stock."
Output: {"signal":"buy","assets":[{"ticker":"TSLA","signal":"buy","summary":"Current disclosed buy of TSLA.","evidenceQuote":"I closed my GM short position to buy more Tesla stock.","whyThisIsCurrentCall":"The speaker says they closed another position to buy more TSLA."}],"rotations":[],"summary":"Current disclosed buy of TSLA."}
"""

SIGNAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["buy", "sell", "hold", "NONE"]},
        "assets": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "signal": {"type": "string", "enum": ["buy", "sell", "hold"]},
                    "summary": {"type": "string"},
                    "evidenceQuote": {"type": "string"},
                    "whyThisIsCurrentCall": {"type": "string"},
                },
                "required": [
                    "ticker",
                    "signal",
                    "summary",
                    "evidenceQuote",
                    "whyThisIsCurrentCall",
                ],
                "additionalProperties": False,
            },
        },
        "rotations": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "fromLabel": {"type": "string"},
                    "fromTicker": {"type": "string"},
                    "toLabel": {"type": "string"},
                    "toTicker": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidenceQuote": {"type": "string"},
                    "whyThisIsCurrentCall": {"type": "string"},
                },
                "required": [
                    "fromLabel",
                    "fromTicker",
                    "toLabel",
                    "toTicker",
                    "summary",
                    "evidenceQuote",
                    "whyThisIsCurrentCall",
                ],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["signal", "assets", "rotations", "summary"],
    "additionalProperties": False,
}


def _build_prompt(channel_title, video_title, published_at, transcript) -> str:
    published = published_at.isoformat() if published_at else "unknown"
    return "\n".join([
        f"Channel: {channel_title}",
        f"Video: {video_title}",
        f"Published at: {published}",
        "",
        SCHEMA_AND_RULES,
        "",
        "Transcript:",
        truncate_for_prompt(transcript),
    ])


def _safe_json_parse(text: str):
    trimmed = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed, re.IGNORECASE)
    if fenced:
        trimmed = fenced.group(1)
    # Fall back to the first {...} block if the model added prose around it.
    if not trimmed.startswith("{"):
        brace = re.search(r"\{[\s\S]*\}", trimmed)
        if brace:
            trimmed = brace.group(0)
    return json.loads(trimmed)


def _clean_required(value) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if len(cleaned) >= 8 else None


def _modal_chat_completion(messages: list[dict[str, str]], model: str) -> str:
    timeout = int(os.environ.get("MODAL_GEMMA4_TIMEOUT", "1200"))
    token = os.environ.get("MODAL_GEMMA4_API_KEY", "EMPTY")
    response = requests.post(
        f"{MODAL_GEMMA4_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "signal_extraction",
                    "schema": SIGNAL_JSON_SCHEMA,
                    "strict": True,
                },
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _hf_chat_completion(messages: list[dict[str, str]], model: str) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    client = InferenceClient(token=token)
    completion = client.chat_completion(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


def summarize_transcript_to_signal(channel_title, video_title, published_at, transcript,
                                   model: str | None = None) -> ExtractedSignal:
    model = model or DEFAULT_MODEL
    prompt = _build_prompt(channel_title, video_title, published_at, transcript)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if MODAL_GEMMA4_URL:
        text = _modal_chat_completion(messages, model)
    else:
        text = _hf_chat_completion(messages, model)
    parsed = _safe_json_parse(text)
    return _normalize(parsed, model, transcript)


def _norm_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _quote_in_transcript(quote: str, transcript: str | None) -> bool:
    if transcript is None:
        return True
    quote_norm = _norm_ws(quote)
    return len(quote_norm) >= 8 and quote_norm in _norm_ws(transcript)


def _normalize(parsed: dict, model: str, transcript: str | None = None) -> ExtractedSignal:
    requested = str(parsed.get("signal") or parsed.get("overallSignal") or "").strip().upper()
    top_signal = "hold" if requested == "NONE" else normalize_signal(
        parsed.get("overallSignal") or parsed.get("signal") or "hold")

    raw_assets = parsed.get("assets")
    if not isinstance(raw_assets, list):
        raw_assets = []

    assets: list[AssetCall] = []
    seen = set()
    if requested != "NONE":
        for a in raw_assets:
            if not isinstance(a, dict):
                continue
            ticker = canonical_ticker(a.get("ticker") or a.get("symbol") or "")
            evidence = _clean_required(a.get("evidenceQuote"))
            why = _clean_required(a.get("whyThisIsCurrentCall"))
            if (
                not ticker
                or not evidence
                or not why
                or ticker in seen
                or not _quote_in_transcript(evidence, transcript)
            ):
                continue
            seen.add(ticker)
            assets.append(AssetCall(
                ticker=ticker,
                signal=normalize_signal(a.get("signal") or a.get("stance") or top_signal),
                summary=a.get("summary") or a.get("thesis") or parsed.get("summary") or "",
                evidence_quote=evidence,
                why_current=why,
                company_name=a.get("name") or a.get("companyName") or a.get("company"),
            ))

    rotations: list[RotationCall] = []
    seen_rot = set()
    for r in parsed.get("rotations") or []:
        if not isinstance(r, dict):
            continue
        ft = canonical_ticker(r.get("fromTicker") or r.get("sellTicker") or "")
        tt = canonical_ticker(r.get("toTicker") or r.get("buyTicker") or "")
        evidence = _clean_required(r.get("evidenceQuote"))
        why = _clean_required(r.get("whyThisIsCurrentCall"))
        key = f"{ft}->{tt}"
        if (
            not ft
            or not tt
            or ft == tt
            or not evidence
            or not why
            or key in seen_rot
            or not _quote_in_transcript(evidence, transcript)
        ):
            continue
        seen_rot.add(key)
        rotations.append(RotationCall(
            from_label=r.get("fromLabel") or r.get("from") or ft,
            from_ticker=ft,
            to_label=r.get("toLabel") or r.get("to") or tt,
            to_ticker=tt,
            summary=r.get("summary") or parsed.get("summary") or "",
            evidence_quote=evidence,
            why_current=why,
        ))

    has_call = bool(assets or rotations)
    return ExtractedSignal(
        has_call=has_call,
        signal=top_signal if has_call else "hold",
        summary=parsed.get("summary") or parsed.get("noCallReason") or parsed.get("noSignalReason") or "",
        assets=assets,
        rotations=rotations,
        model=model,
        raw=parsed,
    )
