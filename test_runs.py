"""Offline tests for sampling + the run registry. Run: python test_runs.py

No pytest in this repo, so this is a plain script: each check asserts and prints,
and a non-zero exit signals failure. Network is avoided by disabling the proxy
refresher and monkeypatching iter_analysis with a fake generator.
"""
from __future__ import annotations

import os
import threading
import time

# Must be set before importing youtube_source (via analysis): keeps the proxy
# pool refresher from spawning a network thread during tests.
os.environ["YOUTUBE_PROXY_REFRESH_SECONDS"] = "0"

import analysis  # noqa: E402
import runs  # noqa: E402
from youtube_source import Video  # noqa: E402


def _videos(n: int) -> list[Video]:
    # Newest-first, like the uploads playlist / RSS feed.
    return [Video(video_id=f"v{i:04d}", title=f"t{i}", published_at=None,
                  url=f"http://y/{i}") for i in range(n)]


def _ids(vs: list[Video]) -> list[str]:
    return [v.video_id for v in vs]


def wait_until(pred, timeout=5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


# --- Sampling ---------------------------------------------------------------

def test_sampling():
    os.environ["ANALYSIS_MAX_VIDEOS"] = "125"
    os.environ["ANALYSIS_LATEST_VIDEOS"] = "25"
    os.environ.pop("ANALYSIS_SAMPLE_SEED", None)

    # Under budget -> all videos returned unchanged.
    small = _videos(50)
    assert analysis._select_videos(small, {}, "UCx") == small, "50<=125 should return all"

    # Over budget -> exactly the cap.
    big = _videos(2896)
    sel = analysis._select_videos(big, {}, "UCabc")
    assert len(sel) == 125, f"expected 125, got {len(sel)}"

    # Latest 25 always included.
    latest_ids = set(_ids(big[:25]))
    assert latest_ids <= set(_ids(sel)), "latest 25 must all be present"

    # No duplicate ids.
    assert len(set(_ids(sel))) == len(sel), "no duplicate video ids"

    # Deterministic for the same channel id; differs by channel (via seed=cid).
    sel2 = analysis._select_videos(big, {}, "UCabc")
    assert _ids(sel) == _ids(sel2), "selection must be deterministic per channel"
    sel_other = analysis._select_videos(big, {}, "UCdifferent")
    assert _ids(sel) != _ids(sel_other), "different channels should sample differently"

    # Cached video with a pending call is kept even if not sampled.
    stored_map = {"v2000": {"video_id": "v2000",
                            "calls": [{"score": None, "signal": "buy"}]}}
    sel3 = analysis._select_videos(big, stored_map, "UCabc")
    assert "v2000" in set(_ids(sel3)), "pending cached video must be retained"

    print("ok: sampling")


# --- Fake analysis generator ------------------------------------------------

class FakeResult:
    def __init__(self, cid="UC1"):
        self.channel_id = cid

    def to_dict(self):
        return {"channel_id": self.channel_id, "title": "Fake"}


def _install_fake(*, gate: threading.Event | None = None, started: threading.Event | None = None,
                  calls: list | None = None):
    """Patch runs.iter_analysis with a controllable fake generator."""
    def fake(target):
        if calls is not None:
            calls.append(target)
        yield "status", {"fraction": 0.1, "label": "resolving",
                         "channel_id": "UC1", "title": "Fake"}
        if started is not None:
            started.set()
        if gate is not None:
            gate.wait(timeout=5)
        yield "video", {"video": {"video_id": "v1"}, "videos_done": 1,
                        "videos_total": 1, "fraction": 0.6}
        yield "result", FakeResult()
    runs.iter_analysis = fake


# --- Registry ---------------------------------------------------------------

def test_dedup():
    gate = threading.Event()
    started = threading.Event()
    calls: list = []
    _install_fake(gate=gate, started=started, calls=calls)

    rid1 = runs.start_run("@dedupchan")
    assert started.wait(timeout=5), "worker should start"
    rid2 = runs.start_run("@DedupChan")  # same key (case-insensitive), still active
    assert rid1 == rid2, "active run should dedupe"
    assert len(calls) == 1, f"only one worker should run, got {len(calls)}"

    gate.set()
    assert wait_until(lambda: runs.get_run(rid1).status == "done"), "run should finish"

    # After completion, a fresh submit starts a new worker.
    _install_fake(calls=calls)  # no gate -> runs to completion fast
    rid3 = runs.start_run("@dedupchan")
    assert wait_until(lambda: runs.get_run(rid3).status == "done")
    assert len(calls) == 2, "re-submit after done should start a new run"
    print("ok: dedup")


def test_replay_and_stream():
    _install_fake()  # fast
    rid = runs.start_run("@replaychan")
    assert wait_until(lambda: runs.get_run(rid).status == "done"), "run should finish"

    # Late subscriber gets the full replay including the terminal result.
    events = list(runs.stream(rid))
    types = [e["type"] for e in events]
    assert types[-1] == "result", f"stream must end with result, got {types}"
    assert "status" in types and "video" in types, f"replay missing events: {types}"
    print("ok: replay + stream")


def test_stream_heartbeat():
    gate = threading.Event()
    started = threading.Event()
    _install_fake(gate=gate, started=started)

    original_interval = runs._STREAM_HEARTBEAT_SEC
    runs._STREAM_HEARTBEAT_SEC = 0.05
    try:
        rid = runs.start_run("@heartbeatchan")
        assert started.wait(timeout=5), "worker should pause after first status"

        gen = runs.stream(rid)
        events = [next(gen), next(gen), next(gen)]
        assert events[-1]["type"] == "status", f"expected heartbeat status, got {events}"
        assert events[-1]["videos_total"] == runs.get_run(rid).videos_total

        gate.set()
        rest = list(gen)
        assert rest and rest[-1]["type"] == "result", "stream should still finish"
    finally:
        runs._STREAM_HEARTBEAT_SEC = original_interval
        gate.set()
    print("ok: stream heartbeat")


def test_list_active_and_prune():
    gate = threading.Event()
    started = threading.Event()
    _install_fake(gate=gate, started=started)
    rid = runs.start_run("@activechan")
    assert started.wait(timeout=5)
    active_ids = [r["run_id"] for r in runs.list_active()]
    assert rid in active_ids, "running run should appear in list_active"

    gate.set()
    assert wait_until(lambda: runs.get_run(rid).status == "done")
    assert rid not in [r["run_id"] for r in runs.list_active()], "done run not active"

    # Prune drops finished runs past the TTL.
    runs._DONE_TTL = 0.0
    runs.get_run(rid)  # still present
    with runs._lock:
        runs._prune()
    assert runs.get_run(rid) is None, "finished run should be pruned"
    print("ok: list_active + prune")


def test_error_is_terminal():
    def boom(target):
        raise ValueError("bad target")
        yield  # make it a generator
    runs.iter_analysis = boom
    rid = runs.start_run("@errchan")
    events = list(runs.stream(rid))
    assert events and events[-1]["type"] == "error", f"expected error terminal, got {events}"
    assert runs.get_run(rid).status == "error"
    print("ok: error terminal")


if __name__ == "__main__":
    test_sampling()
    test_dedup()
    test_replay_and_stream()
    test_stream_heartbeat()
    test_list_active_and_prune()
    test_error_is_terminal()
    print("\nALL PASSED")
