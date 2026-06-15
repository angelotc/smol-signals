"""Durable store backed by a Hugging Face Dataset repo.

The Space is the only writer (single-writer principle). Layout in the repo:

    channels/<channel_id>.json   accumulated scores + latest videos per channel
    videos/<video_id>.json       last analysis of each video
    leaderboard.json             ranked channel summaries
    runs.jsonl                   append-only log, one line per analyze run

Config via env: HF_BUCKET_NAME (e.g. "org/small-signals") and HF_TOKEN.
When HF_BUCKET_NAME is unset the module runs in a no-op mode so the app still
works locally / in a Space without a dataset attached.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from market import calculate_channel_score

_LOCK = threading.Lock()
_repo_ready = False
_leaderboard_cache: list[dict] | None = None


def _repo() -> str | None:
    return (os.environ.get("HF_BUCKET_NAME") or "").strip() or None


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def is_enabled() -> bool:
    return _repo() is not None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api() -> HfApi:
    return HfApi(token=_token())


def _ensure_repo() -> None:
    global _repo_ready
    if _repo_ready:
        return
    _api().create_repo(_repo(), repo_type="dataset", exist_ok=True, private=True)
    _repo_ready = True


def _read_json(path: str):
    repo = _repo()
    if not repo:
        return None
    try:
        local = hf_hub_download(repo, path, repo_type="dataset", token=_token())
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception:  # noqa: BLE001 - never let a read break a request
        return None
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: str) -> str:
    repo = _repo()
    if not repo:
        return ""
    try:
        local = hf_hub_download(repo, path, repo_type="dataset", token=_token())
    except (EntryNotFoundError, RepositoryNotFoundError):
        return ""
    except Exception:  # noqa: BLE001
        return ""
    with open(local, encoding="utf-8") as f:
        return f.read()


def _commit(files: dict[str, str], message: str) -> None:
    """Write multiple files (path -> text content) in a single dataset commit."""
    _ensure_repo()
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=c.encode("utf-8"))
           for p, c in files.items()]
    _api().create_commit(_repo(), repo_type="dataset", operations=ops,
                         commit_message=message)


def get_leaderboard() -> list[dict]:
    """Ranked channel summaries (highest reputation first)."""
    global _leaderboard_cache
    if _leaderboard_cache is not None:
        return _leaderboard_cache
    data = _read_json("leaderboard.json") or []
    _leaderboard_cache = data
    return data


def get_channel(channel_id: str) -> dict | None:
    return _read_json(f"channels/{channel_id}.json")


def record_run(result) -> None:
    """Persist one analyze run: merge call scores, recompute reputation, and
    write channel + video docs + leaderboard + a runs.jsonl line in one commit.

    No-op when the store is disabled or the run has no channel id (e.g. a single
    video that did not resolve to a channel).
    """
    global _leaderboard_cache
    if not is_enabled() or not result.channel_id:
        return

    with _LOCK:
        cid = result.channel_id
        channel = _read_json(f"channels/{cid}.json") or {"channel_id": cid, "scores": {}}
        scores: dict = channel.get("scores", {})
        prior_videos: list = channel.get("videos", [])

        # Keyed by video+ticker+horizon so re-analyzing a channel overwrites
        # rather than double-counts the same call.
        for v in result.videos:
            for c in v.calls:
                if c.score is not None:
                    scores[f"{v.video_id}:{c.ticker}:{c.horizon_days}"] = {
                        "score": c.score, "verdict": c.verdict}

        # Union this run's videos with previously-stored ones (this run wins on
        # overlap). Without a YOUTUBE_API_KEY a run only sees the ~15 most recent
        # uploads via RSS, so replacing wholesale would drop older videos that
        # earlier runs recorded; merging preserves the full history.
        new_videos = [asdict(v) for v in result.videos]
        new_ids = {v["video_id"] for v in new_videos}
        merged_videos = new_videos + [v for v in prior_videos
                                      if v.get("video_id") not in new_ids]

        rep = calculate_channel_score(list(scores.values()))
        channel.update({
            "channel_id": cid,
            "title": result.title,
            "model": result.model,
            "scores": scores,
            "reputation": rep["score"],
            "measured_calls": rep["measured_calls"],
            "win_rate": rep["win_rate"],
            "avg_weighted_score": rep["avg_weighted_score"],
            "last_updated": _now_iso(),
            "videos": merged_videos,
        })

        summary = {
            "channel_id": cid,
            "title": result.title,
            "reputation": rep["score"],
            "measured_calls": rep["measured_calls"],
            "win_rate": rep["win_rate"],
            "avg_weighted_score": rep["avg_weighted_score"],
            "last_updated": channel["last_updated"],
        }
        leaderboard = [e for e in (_read_json("leaderboard.json") or [])
                       if e.get("channel_id") != cid]
        leaderboard.append(summary)
        leaderboard.sort(key=lambda e: e.get("reputation", 0), reverse=True)

        run_line = json.dumps({
            "ts": channel["last_updated"],
            "channel_id": cid,
            "title": result.title,
            "videos_analyzed": len(result.videos),
            "measured_calls": rep["measured_calls"],
            "reputation": rep["score"],
        })
        runs = _read_text("runs.jsonl")
        runs = (runs + ("\n" if runs and not runs.endswith("\n") else "") + run_line + "\n")

        files = {
            f"channels/{cid}.json": json.dumps(channel, indent=2),
            "leaderboard.json": json.dumps(leaderboard, indent=2),
            "runs.jsonl": runs,
        }
        for v in result.videos:
            files[f"videos/{v.video_id}.json"] = json.dumps(asdict(v), indent=2)

        _commit(files, f"analyze {result.title} ({len(result.videos)} videos)")
        _leaderboard_cache = leaderboard
