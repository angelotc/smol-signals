"""Channel/video analysis orchestration, decoupled from any UI.

`run_analysis` pulls transcripts, extracts calls with the small model, scores
matured calls against SPY, and returns plain dataclasses. Both the Gradio UI
handler and the REST/`gr.Server` endpoints call this and format the result
themselves (markdown rows vs. JSON).
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime

import storage
from market import calculate_channel_score, measure_ticker_outcome, score_call
from signals import DEFAULT_MODEL, summarize_transcript_to_signal
from youtube_source import (
    Video,
    fetch_transcript,
    list_channel_videos,
    resolve_channel,
    single_video,
)

HORIZONS = (7, 30)

# Videos are processed concurrently — each is almost entirely network wait
# (transcript fetch + LLM call + Yahoo lookups), so threads overlap that I/O.
# Tune with ANALYZE_CONCURRENCY (lower it if the HF router rate-limits you).
_DEFAULT_CONCURRENCY = 6


def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("ANALYZE_CONCURRENCY", _DEFAULT_CONCURRENCY)))
    except ValueError:
        return _DEFAULT_CONCURRENCY


# Large channels can have thousands of uploads; analyzing every one would make an
# interactive run take hours (each video = transcript + LLM + market lookups). By
# default we analyze a bounded, representative subset: the latest few uploads plus
# an evenly-spaced historical sample. Full backfill belongs in a separate batch
# mode, not the interactive request. Tune with the ANALYSIS_* env vars below.
_DEFAULT_MAX_VIDEOS = 125
_DEFAULT_LATEST_VIDEOS = 25


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _stable_hash(value: str) -> int:
    """Process-independent hash (Python's str hash is salted per run)."""
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def _has_pending_calls(stored: dict) -> bool:
    """True if a stored video still has calls awaiting their 30d outcome."""
    return any(c.get("score") is None and c.get("signal") != "rotate"
               for c in stored.get("calls", []))


def _select_videos(videos: list[Video], stored_map: dict[str, dict],
                   channel_id: str | None) -> list[Video]:
    """Bound a run to a deterministic, representative subset of uploads.

    Returns all videos when the channel is within budget. Otherwise keeps the
    latest ANALYSIS_LATEST_VIDEOS, fills the remaining ANALYSIS_MAX_VIDEOS budget
    with evenly-spaced historical picks (deterministic per channel via
    ANALYSIS_SAMPLE_SEED, defaulting to the channel id), and additionally keeps
    any cached video with still-pending calls so their 30d outcomes keep maturing
    across reruns (cheap — those are reused, not re-transcribed). The subset is
    returned in the input's chronological order.
    """
    max_videos = max(1, _env_int("ANALYSIS_MAX_VIDEOS", _DEFAULT_MAX_VIDEOS))
    if len(videos) <= max_videos:
        return videos

    latest_n = min(max(0, _env_int("ANALYSIS_LATEST_VIDEOS", _DEFAULT_LATEST_VIDEOS)),
                   max_videos)
    latest = videos[:latest_n]          # uploads come newest-first
    older = videos[latest_n:]
    budget = max_videos - len(latest)

    if budget >= len(older):
        sampled = older
    else:
        # Evenly-spaced picks across history. step > 1 guarantees `budget`
        # distinct indices; a per-channel phase shifts them within each stride so
        # the sample is stable for a channel without clamping past the end.
        seed = (os.environ.get("ANALYSIS_SAMPLE_SEED") or channel_id or "").strip()
        step = len(older) / budget
        phase = (_stable_hash(seed) % 1000) / 1000 * step if seed else 0.0
        sampled = [older[int(i * step + phase)] for i in range(budget)]

    selected_ids = {v.video_id for v in latest} | {v.video_id for v in sampled}
    for v in older:
        if v.video_id in selected_ids:
            continue
        stored = stored_map.get(v.video_id)
        if stored and _has_pending_calls(stored):
            selected_ids.add(v.video_id)

    return [v for v in videos if v.video_id in selected_ids]


@dataclass
class CallResult:
    ticker: str
    signal: str  # buy | sell | hold | rotate
    summary: str
    evidence_quote: str = ""
    why_current: str = ""
    company_name: str | None = None
    # Outcome of the 30d horizon; verdict is "pending" until the call matures.
    verdict: str = "pending"
    alpha_pct: float | None = None
    return_pct: float | None = None
    score: float | None = None
    horizon_days: int = 30


@dataclass
class VideoResult:
    video_id: str
    title: str
    url: str
    published_at: str | None  # ISO 8601 or None
    has_call: bool
    summary: str
    calls: list[CallResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class ChannelResult:
    channel_id: str | None
    title: str
    model: str
    reputation: float
    measured_calls: int
    win_rate: float
    avg_weighted_score: float
    videos: list[VideoResult] = field(default_factory=list)
    # Per-call scores ({"score","verdict"}) that fed reputation. Kept for the
    # persistence layer to merge into a channel's accumulated history; omitted
    # from the public API payload via to_dict().
    call_scores: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("call_scores", None)
        return data


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _analyze_video(video: Video, channel_title: str) -> tuple[VideoResult, list[dict]]:
    """Returns (VideoResult, call_scores) for one video."""
    base = dict(video_id=video.video_id, title=video.title, url=video.url,
                published_at=_iso(video.published_at))
    try:
        transcript = fetch_transcript(video.video_id)
    except Exception as e:  # noqa: BLE001 - surface as a result, keep going
        return VideoResult(**base, has_call=False, summary="", error=str(e)), []

    extracted = summarize_transcript_to_signal(
        channel_title, video.title, video.published_at, transcript)

    if not extracted.has_call:
        return VideoResult(**base, has_call=False, summary=extracted.summary or "no call"), []

    calls: list[CallResult] = []
    call_scores: list[dict] = []
    for asset in extracted.assets:
        outcomes = {h: measure_ticker_outcome(asset.ticker, asset.signal,
                                              video.published_at, h) for h in HORIZONS}
        sc = score_call(outcomes)
        if sc:
            call_scores.append(sc)
        o30 = outcomes[30]
        calls.append(CallResult(
            ticker=asset.ticker, signal=asset.signal, summary=asset.summary,
            evidence_quote=asset.evidence_quote, why_current=asset.why_current,
            company_name=asset.company_name,
            verdict=(sc["verdict"] if sc else o30.verdict),
            alpha_pct=o30.alpha_pct, return_pct=o30.return_pct,
            score=(sc["score"] if sc else None)))

    for rot in extracted.rotations:
        calls.append(CallResult(
            ticker=f"{rot.from_ticker}→{rot.to_ticker}", signal="rotate",
            summary=rot.summary, evidence_quote=rot.evidence_quote,
            why_current=rot.why_current, verdict="n/a"))

    return VideoResult(**base, has_call=True, summary=extracted.summary, calls=calls), call_scores


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _rebuild_video(stored: dict) -> VideoResult:
    """Reconstruct a VideoResult (with CallResult objects) from a stored dict.

    Fields are filtered to the current dataclass schema so older stored docs
    (missing newly-added fields, or carrying removed ones) still load.
    """
    call_keys = {f.name for f in dataclass_fields(CallResult)}
    calls = [CallResult(**{k: v for k, v in c.items() if k in call_keys})
             for c in stored.get("calls", [])]
    return VideoResult(
        video_id=stored["video_id"], title=stored.get("title", ""),
        url=stored.get("url", ""), published_at=stored.get("published_at"),
        has_call=stored.get("has_call", False), summary=stored.get("summary", ""),
        calls=calls, error=stored.get("error"))


def _reuse_stored(stored: dict) -> tuple[VideoResult, list[dict]]:
    """Reuse a previously-analyzed video, re-scoring only its pending calls.

    Skips the expensive transcript fetch + LLM extraction (the call text won't
    change). Matured calls keep their stored score; calls that were still pending
    are re-measured against SPY in case their 30d horizon has now elapsed.
    """
    vr = _rebuild_video(stored)
    pub = _parse_iso(vr.published_at)
    call_scores: list[dict] = []
    for c in vr.calls:
        if c.signal == "rotate":
            continue
        if c.score is None and pub is not None:
            outcomes = {h: measure_ticker_outcome(c.ticker, c.signal, pub, h)
                        for h in HORIZONS}
            sc = score_call(outcomes)
            o30 = outcomes[30]
            c.verdict = sc["verdict"] if sc else o30.verdict
            c.alpha_pct = o30.alpha_pct
            c.return_pct = o30.return_pct
            c.score = sc["score"] if sc else None
        if c.score is not None:
            call_scores.append({"score": c.score, "verdict": c.verdict})
    return vr, call_scores


def iter_analysis(target: str) -> Iterator[tuple[str, object]]:
    """Generator form of the analysis, for streaming progress to a UI.

    Analyzes a bounded, representative subset of the channel's uploads (see
    `_select_videos`): all of them when within budget, else the latest few plus
    an evenly-spaced historical sample. Source is the full uploads playlist with
    a YOUTUBE_API_KEY, or the ~15 most recent via RSS without one.

    Incremental: if the channel has been analyzed before, videos already in the
    store are reused (transcript + LLM skipped) and only their still-pending
    calls are re-scored. Genuinely new videos — and any that errored last run —
    are analyzed fresh. So re-running a channel "diffs" to the new uploads.

    Yields `(kind, payload)` events as work proceeds:
      * ("status", {fraction, label, ...})  — about to do something (e.g. a video)
      * ("video",  {video, videos_done, videos_total, fraction})  — one video done
      * ("result", ChannelResult)  — final result (always the last event)

    `run_analysis` drains this for the simple callback API; the streaming endpoint
    forwards the events to the frontend over Gradio's queue/SSE.
    Raises ValueError/RuntimeError on resolution failures so callers can surface them.
    """
    target = (target or "").strip()
    if not target:
        raise ValueError("Paste a channel or video URL to start.")

    if "watch?v=" in target or "youtu.be/" in target:
        vid = single_video(target)
        if not vid:
            raise ValueError(f"Could not parse a video ID from: {target}")
        channel_id, channel_title, videos = None, vid.title, [vid]
    else:
        channel = resolve_channel(target)
        channel_id, channel_title = channel.channel_id, channel.title
        yield "status", {"fraction": 0.02,
                         "label": f"Resolved {channel_title} — listing videos…",
                         "channel_id": channel_id, "title": channel_title}
        videos = list_channel_videos(channel)

    # Prior analysis for this channel, keyed by video id. Videos found here are
    # reused (see _reuse_stored) instead of being re-transcribed and re-extracted.
    stored_map: dict[str, dict] = {}
    if channel_id:
        existing = storage.get_channel(channel_id)
        if existing:
            stored_map = {v["video_id"]: v for v in existing.get("videos", [])
                          if v.get("video_id")}

    # Bound the run to a representative subset before the heavy per-video loop.
    upload_count = len(videos)
    videos = _select_videos(videos, stored_map, channel_id)
    sampled = len(videos) < upload_count

    if not videos:
        yield "result", ChannelResult(channel_id, channel_title, DEFAULT_MODEL,
                                      reputation=50.0, measured_calls=0, win_rate=0.0,
                                      avg_weighted_score=0.0)
        return

    total = len(videos)
    workers = min(_concurrency(), total)

    def _is_fresh(video: Video) -> bool:
        # New, or errored last run (retry those from scratch).
        stored = stored_map.get(video.video_id)
        return stored is None or bool(stored.get("error"))

    fresh = sum(1 for v in videos if _is_fresh(v))
    if stored_map and fresh < total:
        label = (f"Refreshing {channel_title}: {fresh} new, "
                 f"{total - fresh} cached ({workers} at a time)…")
    elif sampled:
        label = (f"Analyzing {total} sampled videos from {upload_count} uploads "
                 f"({workers} at a time)…")
    else:
        label = f"Analyzing {total} videos ({workers} at a time)…"
    yield "status", {
        "fraction": 1 / (total + 1),
        "label": label,
        "videos_done": 0,
        "videos_total": total,
    }

    def _safe(video: Video) -> tuple[VideoResult, list[dict]]:
        try:
            if not _is_fresh(video):
                return _reuse_stored(stored_map[video.video_id])
            return _analyze_video(video, channel_title)
        except Exception as e:  # noqa: BLE001 - surface as a per-video error, keep going
            return (VideoResult(video.video_id, video.title, video.url,
                                _iso(video.published_at), has_call=False, summary="",
                                error=str(e)), [])

    # Run videos concurrently; stream each as it finishes (completion order).
    results: dict[str, tuple[VideoResult, list[dict]]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_safe, v): v for v in videos}
        for fut in as_completed(futures):
            video = futures[fut]
            vr, scores = fut.result()
            results[video.video_id] = (vr, scores)
            done += 1
            yield "video", {
                "video": asdict(vr),
                "videos_done": done,
                "videos_total": total,
                "fraction": done / (total + 1),
            }

    # Reassemble in the channel's original (chronological) order for the stored
    # and returned result, regardless of which finished first.
    video_results = [results[v.video_id][0] for v in videos]
    all_scores = [s for v in videos for s in results[v.video_id][1]]

    rep = calculate_channel_score(all_scores)
    yield "result", ChannelResult(
        channel_id=channel_id, title=channel_title, model=DEFAULT_MODEL,
        reputation=rep["score"], measured_calls=rep["measured_calls"],
        win_rate=rep["win_rate"], avg_weighted_score=rep["avg_weighted_score"],
        videos=video_results, call_scores=all_scores)


def run_analysis(target: str, progress=None) -> ChannelResult:
    """Analyze a channel or a single video, returning the final ChannelResult.

    `progress`, if given, is called as progress(fraction, label) for UI feedback.
    Thin wrapper over `iter_analysis` that drains its events and returns the result.
    """
    result: ChannelResult | None = None
    for kind, payload in iter_analysis(target):
        if kind == "result":
            result = payload  # type: ignore[assignment]
        elif progress:
            data = payload  # type: ignore[assignment]
            label = data.get("label") or (
                f"Scored {data.get('videos_done', '')}/{data.get('videos_total', '')}")
            progress(data.get("fraction", 0.0), label)
    assert result is not None, "iter_analysis must end with a result event"
    return result
