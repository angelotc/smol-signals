"""Server-side run registry so analyses survive client disconnects.

A *run* is a background worker draining `analysis.iter_analysis`, independent of
any HTTP/SSE connection. Browser refreshes, second viewers, and the live activity
feed all attach to the same run through a per-run replay buffer + pub/sub, so the
work keeps going even when nobody is watching and a refreshed page re-attaches
mid-flight instead of restarting.

A Hugging Face Space is a single process, so this module-level state is shared
across every session. Caveat: it is in-memory — runs survive refreshes and
disconnects but NOT a Space restart/sleep (the process dies with the registry).
A multi-replica Space would fragment it (would need Redis or similar).

Concurrency is bounded by MAX_CONCURRENT_RUNS (default 2): extra runs sit in
`queued` until a slot frees, so a small Space isn't swamped (each running run
still uses ANALYZE_CONCURRENCY worker threads internally).
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field

import storage
from analysis import iter_analysis
from tickers import extract_video_id

# Finished runs linger this long so a late refresh can still fetch the result.
_DONE_TTL = 300.0
_TERMINAL = ("result", "error")


def _max_runs() -> int:
    try:
        return max(1, int(os.environ.get("MAX_CONCURRENT_RUNS", "2")))
    except ValueError:
        return 2


_lock = threading.Lock()                      # guards the _runs registry
_runs: dict[str, Run] = {}
_run_sema = threading.BoundedSemaphore(_max_runs())


@dataclass
class Run:
    run_id: str
    target: str
    status: str = "queued"                     # queued | running | done | error
    fraction: float = 0.0
    label: str = "Queued…"
    channel_id: str | None = None
    title: str = ""
    videos_done: int = 0
    videos_total: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Internal (not part of the public snapshot):
    result: dict | None = None                 # ChannelResult.to_dict() when done
    result_obj: object | None = None           # the ChannelResult, for persistence
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    replay: list[dict] = field(default_factory=list, repr=False)
    subscribers: set = field(default_factory=set, repr=False)

    def snapshot(self) -> dict:
        """JSON-safe public view for the live feed / status checks."""
        with self.lock:
            return {
                "run_id": self.run_id, "target": self.target, "status": self.status,
                "fraction": self.fraction, "label": self.label,
                "channel_id": self.channel_id, "title": self.title,
                "videos_done": self.videos_done, "videos_total": self.videos_total,
                "error": self.error, "started_at": self.started_at,
                "updated_at": self.updated_at,
            }


def _normalize(target: str) -> str:
    """Stable registry key. Single videos key on their id; channels on the text."""
    t = (target or "").strip()
    vid = extract_video_id(t)
    return f"vid:{vid}" if vid else f"chan:{t.lower()}"


def _publish(run: Run, event: dict) -> None:
    """Append an event to the replay buffer, fan it out, and sync snapshot fields."""
    with run.lock:
        run.replay.append(event)
        run.updated_at = time.time()
        if "fraction" in event:
            run.fraction = event["fraction"]
        if event.get("label"):
            run.label = event["label"]
        if event.get("channel_id"):
            run.channel_id = event["channel_id"]
        if event.get("title"):
            run.title = event["title"]
        if "videos_done" in event:
            run.videos_done = event["videos_done"]
        if "videos_total" in event:
            run.videos_total = event["videos_total"]
        for q in list(run.subscribers):
            q.put(event)


def _set(run: Run, **fields) -> None:
    with run.lock:
        for k, v in fields.items():
            setattr(run, k, v)
        run.updated_at = time.time()


def _worker(run: Run) -> None:
    """Drive one analysis to completion, publishing events as it goes.

    Runs in a daemon thread, detached from any client. Always publishes a
    terminal (`result` or `error`) event so subscribers never block forever.
    """
    terminal_published = False
    _publish(run, {"type": "status", "fraction": 0.0,
                   "label": "Queued — waiting for a free slot…"})
    _run_sema.acquire()
    try:
        _set(run, status="running")
        for kind, payload in iter_analysis(run.target):
            if kind == "result":
                _set(run, status="done", result=payload.to_dict(), result_obj=payload)
                _publish(run, {"type": "result", "result": run.result})
                terminal_published = True
                try:
                    storage.record_run(payload)
                except Exception as e:  # noqa: BLE001 - persistence must never crash a run
                    print(f"[runs] record_run failed: {e}")
            else:
                _publish(run, {"type": kind, **payload})
    except Exception as e:  # noqa: BLE001 - surface as a terminal error event
        _set(run, status="error", error=str(e))
        _publish(run, {"type": "error", "error": str(e)})
        terminal_published = True
    finally:
        _run_sema.release()
        if not terminal_published:
            # iter_analysis ended without a result (shouldn't happen) — unblock waiters.
            _set(run, status="error", error=run.error or "ended without a result")
            _publish(run, {"type": "error", "error": run.error or "ended without a result"})


def _prune() -> None:
    """Drop finished runs older than the TTL. Caller must hold _lock."""
    now = time.time()
    for key in [k for k, r in _runs.items()
                if r.status in ("done", "error") and now - r.updated_at > _DONE_TTL]:
        del _runs[key]


def start_run(target: str) -> str:
    """Start (or attach to) a run for `target`, returning its run_id.

    Dedupes only *active* runs: a second submit of a channel already queued or
    running attaches to the same worker. A finished run is replaced by a fresh
    one (so re-analyzing picks up new uploads / matures pending calls).
    """
    run_id = _normalize(target)
    with _lock:
        _prune()
        existing = _runs.get(run_id)
        if existing and existing.status in ("queued", "running"):
            return run_id
        run = Run(run_id=run_id, target=target.strip())
        _runs[run_id] = run
    threading.Thread(target=_worker, args=(run,), name=f"run:{run_id}",
                     daemon=True).start()
    return run_id


def get_run(run_id: str) -> Run | None:
    with _lock:
        return _runs.get(run_id)


def subscribe(run_id: str) -> tuple[list[dict], queue.Queue]:
    """Register a subscriber, returning (replay_snapshot, queue) atomically.

    The snapshot holds every event so far; the queue receives all subsequent
    events — no gap, no duplicate, because both happen under the run's lock.
    """
    run = get_run(run_id)
    q: queue.Queue = queue.Queue()
    if run is None:
        return [], q
    with run.lock:
        replay = list(run.replay)
        run.subscribers.add(q)
    return replay, q


def unsubscribe(run_id: str, q: queue.Queue) -> None:
    run = get_run(run_id)
    if run is None:
        return
    with run.lock:
        run.subscribers.discard(q)


def stream(run_id: str):
    """Yield a run's events (replayed + live) until a terminal event.

    Safe across disconnects: closing this generator only unsubscribes; the worker
    keeps running. A timeout re-checks run status so a vanished run can't hang us.
    """
    replay, q = subscribe(run_id)
    try:
        seen_terminal = False
        for event in replay:
            yield event
            if event.get("type") in _TERMINAL:
                seen_terminal = True
        while not seen_terminal:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                run = get_run(run_id)
                if run is None or run.status in ("done", "error"):
                    break
                continue
            yield event
            if event.get("type") in _TERMINAL:
                seen_terminal = True
    finally:
        unsubscribe(run_id, q)


def list_active() -> list[dict]:
    """Snapshots of queued/running runs, newest first — for the live feed."""
    with _lock:
        _prune()
        runs = [r for r in _runs.values() if r.status in ("queued", "running")]
    return sorted((r.snapshot() for r in runs),
                  key=lambda s: s["started_at"], reverse=True)
