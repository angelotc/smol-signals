"""Harvest transcript + existing-label pairs from the Smol Signals HF bucket.

The production bucket stores accepted model outputs but not raw transcripts, so
this script re-fetches transcripts for the stored videos and pairs them with the
stored normalized labels. It is resumable: rerunning skips video IDs already in
the output JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common import (  # noqa: E402
    TARGET_CHANNEL_TITLES,
    append_jsonl,
    canonical_assistant_output,
    read_jsonl,
)
from youtube_source import fetch_transcript  # noqa: E402


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _download_json(repo: str, path: str) -> Any:
    local = hf_hub_download(repo, path, repo_type="dataset", token=_token())
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def _target_channels(repo: str, titles: set[str]) -> list[dict[str, Any]]:
    leaderboard = _download_json(repo, "leaderboard.json")
    wanted = []
    for entry in leaderboard:
        if str(entry.get("title") or "").strip() in titles:
            wanted.append(_download_json(repo, f"channels/{entry['channel_id']}.json"))
    missing = titles - {str(c.get("title") or "").strip() for c in wanted}
    if missing:
        raise RuntimeError(f"Did not find target channels in leaderboard: {sorted(missing)}")
    return wanted


def _iter_video_jobs(
    channels: list[dict[str, Any]],
    *,
    limit_per_channel: int | None,
    only_with_calls: bool,
    include_errors: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for channel in channels:
        videos = channel.get("videos") or []
        if only_with_calls:
            videos = [v for v in videos if v.get("has_call") or v.get("calls")]
        if not include_errors:
            videos = [v for v in videos if not v.get("error")]
        if limit_per_channel is not None:
            videos = videos[:limit_per_channel]
        for video in videos:
            jobs.append((channel, video))
    return jobs


def _harvest_one(repo: str, channel: dict[str, Any], video: dict[str, Any]) -> dict[str, Any]:
    video_id = video["video_id"]
    transcript = fetch_transcript(video_id)
    assistant_json = canonical_assistant_output(video)
    return {
        "source_repo": repo,
        "label_source": "stored_gemma_output",
        "label_source_model": channel.get("model"),
        "channel_id": channel.get("channel_id"),
        "channel_title": channel.get("title"),
        "video_id": video_id,
        "video_title": video.get("title") or video_id,
        "published_at": video.get("published_at"),
        "url": video.get("url") or f"https://www.youtube.com/watch?v={video_id}",
        "stored_has_call": bool(video.get("has_call")),
        "assistant_json": assistant_json,
        "transcript": transcript,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.getenv("HF_BUCKET_NAME") or "ajc426/small-signals")
    parser.add_argument("--output", type=Path, default=Path("training/data/harvested.jsonl"))
    parser.add_argument("--titles", nargs="*", default=list(TARGET_CHANNEL_TITLES))
    parser.add_argument("--limit-per-channel", type=int, default=None)
    parser.add_argument("--only-with-calls", action="store_true")
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--sleep-after", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    if not _token():
        raise RuntimeError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN.")
    if not os.getenv("ZYTE_API_KEY"):
        raise RuntimeError("Set ZYTE_API_KEY to re-fetch YouTube transcripts.")

    # Fails early if the token cannot see the dataset.
    HfApi(token=_token()).repo_info(args.repo, repo_type="dataset")

    existing = {row["video_id"] for row in read_jsonl(args.output) if row.get("video_id")}
    channels = _target_channels(args.repo, {t.strip() for t in args.titles})
    jobs = [
        (c, v) for c, v in _iter_video_jobs(
            channels,
            limit_per_channel=args.limit_per_channel,
            only_with_calls=args.only_with_calls,
            include_errors=args.include_errors,
        )
        if v.get("video_id") not in existing
    ]

    print(f"Target channels: {', '.join(c['title'] for c in channels)}")
    print(f"Already harvested: {len(existing)}")
    print(f"Remaining jobs: {len(jobs)}")

    stats: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {pool.submit(_harvest_one, args.repo, c, v): (c, v) for c, v in jobs}
        for idx, future in enumerate(as_completed(futures), start=1):
            channel, video = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - record and keep harvesting
                stats["errors"] += 1
                print(f"[{idx}/{len(jobs)}] ERROR {channel.get('title')} {video.get('video_id')}: {exc}")
                continue
            append_jsonl(args.output, row)
            stats["ok"] += 1
            if row["assistant_json"]["signal"] == "NONE":
                stats["none"] += 1
            else:
                stats["calls"] += 1
            if args.print_every > 0 and (
                idx == 1 or idx == len(jobs) or idx % args.print_every == 0
            ):
                print(
                    f"[{idx}/{len(jobs)}] OK {row['channel_title']} {row['video_id']} "
                    f"signal={row['assistant_json']['signal']} stats={dict(stats)}"
                )
            if args.sleep_after:
                time.sleep(args.sleep_after)

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
