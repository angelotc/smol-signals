"""Benchmark-relative, vol-adjusted scoring of stock calls vs SPY.

Ported from src/lib/youtube/market.ts. Uses the public Yahoo Finance chart
endpoint for daily adjusted closes. A horizon only scores once enough calendar
time has elapsed AND a price bar exists at/after the target date.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

BENCHMARK_TICKER = "SPY"
BROAD_MARKET_TICKER = "SPY"
CALL_SCORE_CAP = 25
SEVEN_DAY_WEIGHT = 0.25
FULL_CONFIDENCE_CALLS = 20
NEUTRAL_REPUTATION_SCORE = 50
HOLD_WEIGHT = 0.25
SCORE_SCALE = 4
MATURITY_TOLERANCE_DAYS = 4
MAX_VOL_RATIO = 4
MIN_VOL_SAMPLES = 5

_DAY_MS = 24 * 60 * 60 * 1000
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; small-signals/1.0)"}

# Crypto has no bare symbol on Yahoo Finance — it lives under a "-USD" pair
# (BTC -> BTC-USD). Map the coins creators actually call out so their calls score
# against SPY like any stock. Coins Yahoo doesn't cover just stay "unavailable"
# (graceful no-op), same as an unknown stock ticker. Note: Nano rebranded to XNO.
_CRYPTO_YAHOO_SYMBOLS = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD", "DOGE": "DOGE-USD", "AVAX": "AVAX-USD",
    "DOT": "DOT-USD", "MATIC": "MATIC-USD", "LINK": "LINK-USD", "LTC": "LTC-USD",
    "NANO": "XNO-USD",
}


def _yahoo_symbol(ticker: str) -> str:
    """Resolve a call ticker to the symbol Yahoo's chart endpoint expects."""
    return _CRYPTO_YAHOO_SYMBOLS.get(ticker.upper(), ticker)


@dataclass
class PricePoint:
    date: datetime
    close: float


@dataclass
class MarketOutcome:
    verdict: str  # pending | correct | incorrect | neutral | unavailable
    return_pct: float | None = None
    benchmark_return_pct: float | None = None
    alpha_pct: float | None = None
    vol_ratio: float | None = None
    weighted_score: float | None = None
    start_price: float | None = None
    end_price: float | None = None
    note: str = ""


def _clamp(v, lo, hi):
    return min(hi, max(lo, v))


def _now_ms():
    return int(time.time() * 1000)


def _fetch_yahoo_daily_bars(ticker: str, start: datetime, end: datetime):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1d",
        "includeAdjustedClose": "true",
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=20)
    except requests.RequestException as e:
        return [], f"yahoo request failed: {e}"
    if not resp.ok:
        return [], f"yahoo HTTP {resp.status_code}"

    data = resp.json()
    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        return [], "yahoo returned no result"
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    closes = (indicators.get("quote") or [{}])[0].get("close") or []
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []

    points = []
    for i, ts in enumerate(timestamps):
        close = adj[i] if i < len(adj) and adj[i] is not None else (closes[i] if i < len(closes) else None)
        if isinstance(close, (int, float)) and math.isfinite(close):
            points.append(PricePoint(datetime.fromtimestamp(ts, tz=timezone.utc), float(close)))
    return points, ""


def _resolve_window(points: list[PricePoint], target_ms: int):
    start = points[0]
    end = next((p for p in points if p.date.timestamp() * 1000 >= target_ms), points[-1])
    tolerance = MATURITY_TOLERANCE_DAYS * _DAY_MS
    matured = end.date.timestamp() * 1000 >= target_ms - tolerance
    return start, end, matured


def _measure_return_pct(points, target_ms):
    if len(points) < 2:
        return None
    start, end, _ = _resolve_window(points, target_ms)
    return ((end.close - start.close) / start.close) * 100


def _window_volatility(points, target_ms):
    in_window = [p for p in points if p.date.timestamp() * 1000 <= target_ms]
    returns = [in_window[i].close / in_window[i - 1].close - 1 for i in range(1, len(in_window))]
    if len(returns) < MIN_VOL_SAMPLES:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _vol_ratio(ticker_pts, bench_pts, target_ms):
    tv = _window_volatility(ticker_pts, target_ms)
    bv = _window_volatility(bench_pts, target_ms)
    if tv is None or bv is None or bv == 0:
        return 1.0
    return _clamp(tv / bv, 1, MAX_VOL_RATIO)


def _classify(signal, active_return):
    if signal == "sell":
        return "correct" if active_return < 0 else "incorrect"
    return "correct" if active_return > 0 else "incorrect"


def _score(signal, active_return):
    if signal == "buy":
        return active_return
    if signal == "sell":
        return -active_return
    return active_return * HOLD_WEIGHT


def measure_ticker_outcome(ticker: str, signal: str, published_at: datetime | None,
                           horizon_days: int) -> MarketOutcome:
    if not published_at:
        return MarketOutcome("unavailable", note="missing publish date")

    target_ms = int(published_at.timestamp() * 1000) + horizon_days * _DAY_MS
    if _now_ms() < target_ms:
        return MarketOutcome("pending", note=f"{horizon_days}d horizon not elapsed")

    start_window = published_at
    end_window = published_at + timedelta(days=horizon_days + 5)
    benchmark_ticker = None if ticker.upper() == BROAD_MARKET_TICKER else BENCHMARK_TICKER

    measured, merr = _fetch_yahoo_daily_bars(_yahoo_symbol(ticker), start_window, end_window)
    if len(measured) < 2:
        return MarketOutcome("unavailable", note=merr or "not enough price data")

    start, end, matured = _resolve_window(measured, target_ms)
    if not matured:
        return MarketOutcome("pending", note="price bars not matured")

    return_pct = ((end.close - start.close) / start.close) * 100
    benchmark_return_pct = 0.0
    bench_pts = []
    if benchmark_ticker:
        bench_pts, _ = _fetch_yahoo_daily_bars(benchmark_ticker, start_window, end_window)
        benchmark_return_pct = _measure_return_pct(bench_pts, target_ms)
        if benchmark_return_pct is None:
            return MarketOutcome("unavailable", note="not enough benchmark data")

    alpha_pct = return_pct - benchmark_return_pct if benchmark_ticker else None
    score_base = alpha_pct if alpha_pct is not None else return_pct
    vol_ratio = _vol_ratio(measured, bench_pts, target_ms) if benchmark_ticker else 1.0
    risk_adjusted = score_base / vol_ratio

    return MarketOutcome(
        verdict=_classify(signal, score_base),
        return_pct=return_pct,
        benchmark_return_pct=benchmark_return_pct if benchmark_ticker else None,
        alpha_pct=alpha_pct,
        vol_ratio=vol_ratio,
        weighted_score=_score(signal, risk_adjusted),
        start_price=start.close,
        end_price=end.close,
    )


def score_call(outcomes_by_horizon: dict[int, MarketOutcome]):
    """Combine 30d (primary) and 7d outcomes into one capped call score."""
    primary = outcomes_by_horizon.get(30)
    if not primary or primary.weighted_score is None:
        return None
    seven = outcomes_by_horizon.get(7)
    seven_score = seven.weighted_score * SEVEN_DAY_WEIGHT if seven and seven.weighted_score is not None else 0
    score = _clamp(primary.weighted_score + seven_score, -CALL_SCORE_CAP, CALL_SCORE_CAP)
    return {"score": score, "verdict": primary.verdict}


def calculate_channel_score(call_scores: list[dict]):
    """call_scores: list of {"score": float, "verdict": str} from score_call."""
    measured = [c for c in call_scores if c]
    n = len(measured)
    wins = sum(1 for c in measured if c["verdict"] in ("correct", "neutral"))
    avg = sum(c["score"] for c in measured) / n if n else 0
    confidence = min(1, n / FULL_CONFIDENCE_CALLS)
    score = _clamp(NEUTRAL_REPUTATION_SCORE + avg * SCORE_SCALE * confidence, 0, 100)
    return {
        "score": score,
        "measured_calls": n,
        "win_rate": (wins / n) if n else 0,
        "avg_weighted_score": avg,
    }
