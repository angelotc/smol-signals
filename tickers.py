"""Ticker / signal normalization ported from src/lib/youtube/utils.ts."""
import re

SIGNALS = ("buy", "sell", "hold")
MARKET_TICKERS = {"MARKET", "THE MARKET", "S&P", "S&P 500", "SP500", "SPY", "INDEX"}

_TICKER_LIKE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z])?$")
_CHANNEL_ID_RE = re.compile(r"UC[\w-]{20,}")
_VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")


def is_ticker_like(value: str) -> bool:
    return bool(_TICKER_LIKE.match(value))


def canonical_ticker(value: str) -> str:
    ticker = (value or "").strip().upper().lstrip("$")
    if ticker == "NONE":
        return ""
    if ticker in MARKET_TICKERS:
        return "SPY"
    sanitized = re.sub(r"[^A-Z.-]", "", ticker)
    return sanitized if is_ticker_like(sanitized) else ""


def normalize_signal(value: str) -> str:
    signal = (value or "").strip().lower()
    return signal if signal in SIGNALS else "hold"


def extract_channel_id(value: str):
    m = _CHANNEL_ID_RE.search(value or "")
    return m.group(0) if m else None


def extract_video_id(value: str):
    trimmed = (value or "").strip()
    if _VIDEO_ID_RE.match(trimmed):
        return trimmed
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})", trimmed)
    return m.group(1) if m else None


def extract_handle(value: str):
    trimmed = (value or "").strip()
    if trimmed.startswith("@"):
        return trimmed[1:]
    m = re.search(r"/@([\w.-]+)", trimmed)
    return m.group(1) if m else None


def truncate_for_prompt(value: str, max_chars: int = 40000) -> str:
    if len(value) <= max_chars:
        return value
    head = value[: int(max_chars * 0.72)]
    tail = value[len(value) - int(max_chars * 0.24):]
    return f"{head}\n\n[... transcript truncated ...]\n\n{tail}"
