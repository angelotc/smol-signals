"""Build and optionally publish chat-format SFT JSONL for Gemma post-training."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common import is_grounded, make_messages, read_jsonl, write_jsonl  # noqa: E402


def _bucket(row: dict[str, Any]) -> str:
    label = row["assistant_json"]
    if label.get("signal") == "NONE":
        return "NONE"
    if label.get("rotations"):
        return "rotate"
    return str(label.get("signal") or "call")


def _split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_bucket(row)].append(row)

    train: list[dict] = []
    val: list[dict] = []
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        n_val = max(1, round(len(bucket_rows) * val_ratio)) if len(bucket_rows) > 1 else 0
        val.extend(bucket_rows[:n_val])
        train.extend(bucket_rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _to_example(row: dict[str, Any], max_transcript_chars: int) -> dict[str, Any]:
    return {
        "messages": make_messages(row, max_transcript_chars=max_transcript_chars),
        "metadata": {
            "channel_id": row.get("channel_id"),
            "channel_title": row.get("channel_title"),
            "video_id": row.get("video_id"),
            "video_title": row.get("video_title"),
            "published_at": row.get("published_at"),
            "url": row.get("url"),
            "label_source": row.get("label_source"),
            "label_source_model": row.get("label_source_model"),
            "signal": row.get("assistant_json", {}).get("signal"),
        },
    }


def _usable_rows(rows: list[dict[str, Any]], *, require_grounded_evidence: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        label = row.get("assistant_json") or {}
        if not row.get("transcript"):
            skipped["missing_transcript"] += 1
            continue
        if not isinstance(label, dict):
            skipped["bad_label"] += 1
            continue
        if require_grounded_evidence and not is_grounded(label, row["transcript"]):
            skipped["ungrounded_evidence"] += 1
            continue
        kept.append(row)
    return kept, skipped


def _write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    readme = f"""---
pretty_name: Smol Signals Gemma 12B SFT
task_categories:
- text-generation
language:
- en
tags:
- finance
- youtube
- extraction
- gemma
---

# Smol Signals Gemma 12B SFT

Chat-format supervised fine-tuning data for extracting explicit, forward-looking
stock-market calls from finance YouTube transcripts.

The inputs are real transcripts re-fetched for videos already analyzed by Smol
Signals. The assistant targets are reconstructed from the stored accepted model
outputs in the source dataset, not from market outcomes.

```json
{json.dumps(summary, indent=2)}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _push_to_hub(output_dir: Path, repo_id: str, private: bool) -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN to push the dataset.")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    ops = []
    for path in output_dir.iterdir():
        if path.is_file():
            ops.append(CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=str(path)))
    api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message="Build Smol Signals Gemma 12B SFT dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("training/data/harvested.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/sft_gemma12b"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-transcript-chars", type=int, default=24000)
    parser.add_argument("--require-grounded-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--extra-train-input",
        type=Path,
        action="append",
        default=[],
        help="Additional harvested-format JSONL files to append to train only.",
    )
    parser.add_argument("--extra-train-repeat", type=int, default=1)
    parser.add_argument("--push-repo", default=os.getenv("SFT_DATASET_REPO"))
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    kept, skipped = _usable_rows(
        read_jsonl(args.input),
        require_grounded_evidence=args.require_grounded_evidence,
    )

    if not kept:
        raise RuntimeError("No usable rows after filtering.")

    train_rows, val_rows = _split(kept, args.val_ratio, args.seed)
    extra_train_rows: list[dict[str, Any]] = []
    extra_skipped: Counter[str] = Counter()
    for extra_path in args.extra_train_input:
        extra_rows, skipped_extra = _usable_rows(
            read_jsonl(extra_path),
            require_grounded_evidence=args.require_grounded_evidence,
        )
        extra_train_rows.extend(extra_rows)
        extra_skipped.update(skipped_extra)
    if extra_train_rows and args.extra_train_repeat > 1:
        extra_train_rows = extra_train_rows * args.extra_train_repeat
    train_rows.extend(extra_train_rows)

    train = [_to_example(row, args.max_transcript_chars) for row in train_rows]
    val = [_to_example(row, args.max_transcript_chars) for row in val_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)

    summary = {
        "input": str(args.input),
        "train_rows": len(train),
        "val_rows": len(val),
        "skipped": dict(skipped),
        "signals": dict(Counter(row["assistant_json"]["signal"] for row in kept)),
        "channels": dict(Counter(row["channel_title"] for row in kept)),
        "extra_train_inputs": [str(path) for path in args.extra_train_input],
        "extra_train_rows": len(extra_train_rows),
        "extra_train_repeat": args.extra_train_repeat,
        "extra_train_skipped": dict(extra_skipped),
        "max_transcript_chars": args.max_transcript_chars,
        "require_grounded_evidence": args.require_grounded_evidence,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_readme(args.output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.push_repo:
        _push_to_hub(args.output_dir, args.push_repo, private=not args.public)
        print(f"Pushed dataset to hf://datasets/{args.push_repo}")


if __name__ == "__main__":
    main()
