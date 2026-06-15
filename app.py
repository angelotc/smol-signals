"""Smol Signals — Gradio app + REST + custom frontend, one server on port 7860.

Built for the Build Small hackathon (<=32B model, Gradio, Spaces). The same
process serves:
  * the native Gradio Blocks UI at /gradio,
  * a queued structured JSON endpoint `/analyze_channel` (gr.api) for the custom
    frontend, which goes through Gradio's queue (no gateway timeout on the slow
    transcript+LLM loop),
  * plain JSON REST reads under /api (leaderboard / channel / health),
  * the custom Next.js frontend (static export in ./frontend/out) at /.

Serving a custom frontend from the Gradio server is the "Off-Brand" path while
staying a Gradio app on a Hugging Face Space.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

# Local dev convenience: load .env if present. On Spaces, secrets are real env
# vars and there is no .env, so this is a harmless no-op.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import runs
import storage
from signals import DEFAULT_MODEL, MODAL_GEMMA4_URL

APP_THEME = gr.themes.Soft()
_FRONTEND_DIR = Path(__file__).parent / "frontend" / "out"
_CACHE_HEADERS = {"Cache-Control": "public, max-age=60"}


def _server_host() -> str:
    return os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")


def _server_port() -> int:
    return int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))


def _blocks_kwargs() -> dict:
    kwargs = {"title": "Smol Signals"}
    if "theme" in inspect.signature(gr.Blocks).parameters:
        kwargs["theme"] = APP_THEME
    return kwargs


def analyze_channel_api(target: str) -> dict:
    """Structured JSON endpoint for the custom frontend (queued via gr.api).

    Starts (or attaches to) a server-side run and drains it to completion. The
    run's background worker owns persistence, so this just returns the result.
    """
    run_id = runs.start_run(target)
    for _ in runs.stream(run_id):
        pass
    run = runs.get_run(run_id)
    if run is None or run.result is None:
        raise gr.Error((run and run.error) or "Analysis failed.")
    return run.result


def analyze_channel_stream(target: str) -> dict:
    """Streaming endpoint: forwards a run's progress events, then its result.

    The `-> dict` annotation matters: gr.api uses it to bind one JSON output, so
    each yielded event reaches the client. Without it gr.api infers zero outputs
    and silently drops every yield.

    The run lives server-side (see runs.py), so the work survives a client
    disconnect and a refreshed page re-attaches to the same run instead of
    restarting it. Each yield is one JSON event the frontend renders live:
      * {"type": "status", "fraction", "label", ...}
      * {"type": "video",  "video", "videos_done", "videos_total", "fraction"}
      * {"type": "result", "result": <ChannelResult>}
      * {"type": "error",  "error": <message>}
    Consumed client-side via @gradio/client `submit()` over Gradio's queue/SSE.
    """
    run_id = runs.start_run(target)
    yield from runs.stream(run_id)


def _format_rows(result) -> list[list[str]]:
    rows: list[list[str]] = []
    for v in result.videos:
        if v.error:
            rows.append([v.title, "—", "error", v.error, v.url])
            continue
        if not v.has_call:
            rows.append([v.title, "NONE", "hold", v.summary or "no call", v.url])
            continue
        for c in v.calls:
            if c.signal == "rotate":
                rows.append([v.title, c.ticker, "rotate", c.summary, v.url])
                continue
            alpha = f"{c.alpha_pct:+.1f}%" if c.alpha_pct is not None else "—"
            score_txt = f"{c.score:+.1f}" if c.score is not None else c.verdict
            rows.append([v.title, c.ticker, c.signal,
                         f"{c.summary}  ·  30d α {alpha} · {c.verdict} ({score_txt})", v.url])
    return rows


def analyze(target: str, progress=gr.Progress()):
    """Gradio Blocks UI handler — markdown header + dataframe rows.

    Drives a server-side run (so it dedupes and survives disconnects) and feeds
    its events into the progress bar. The worker owns persistence.
    """
    run_id = runs.start_run(target)
    for ev in runs.stream(run_id):
        if ev["type"] == "error":
            return f"Error: {ev['error']}", []
        if ev["type"] == "result":
            break
        label = ev.get("label") or (
            f"Scored {ev.get('videos_done', '')}/{ev.get('videos_total', '')}")
        progress(ev.get("fraction", 0.0), desc=label)
    run = runs.get_run(run_id)
    result = run.result_obj if run else None
    if result is None:
        return "Analysis failed.", []
    header = (
        f"### {result.title}\n"
        f"**Reputation score:** {result.reputation:.0f}/100 (neutral = 50) · "
        f"model `{result.model}`\n\n"
        f"Measured calls: {result.measured_calls} · "
        f"Win rate: {result.win_rate * 100:.0f}% · "
        f"Avg weighted alpha: {result.avg_weighted_score:+.2f}\n\n"
        f"_Calls only score once 30 days have elapsed since publish; recent videos "
        f"show as `pending`._"
    )
    return header, _format_rows(result)


def _render_active_rows() -> list[list[str]]:
    """Rows for the global live-activity panel — everyone's in-flight runs."""
    rows: list[list[str]] = []
    for r in runs.list_active():
        name = r["title"] or r["target"]
        if r["videos_total"]:
            prog = f'{r["videos_done"]}/{r["videos_total"]} · {int(r["fraction"] * 100)}%'
        else:
            prog = r["label"]
        rows.append([name, r["status"], prog])
    return rows


with gr.Blocks(**_blocks_kwargs()) as demo:
    gr.Markdown(
        "# Smol Signals\n"
        "Scores finance YouTube calls against SPY after 30 days."
    )
    target = gr.Textbox(
        label="Channel or video",
        placeholder="@HandleName  ·  youtube.com/@channel  ·  channel URL  ·  a single video URL",
    )
    with gr.Row():
        run = gr.Button("Analyze", variant="primary")
        refresh = gr.Button("Refresh", variant="secondary")
    gr.Markdown(
        "_New uploads are analyzed; existing scores are reused._"
    )
    summary = gr.Markdown()
    table = gr.Dataframe(
        headers=["Video", "Ticker", "Call", "Summary / outcome", "URL"],
        wrap=True,
        label="Extracted calls",
    )
    run.click(analyze, inputs=[target], outputs=[summary, table])
    # Refresh re-runs the same incremental analysis on the current target;
    # already-scored videos are reused, so only new uploads do real work.
    refresh.click(analyze, inputs=[target], outputs=[summary, table])

    # Global live-activity feed: every visitor sees what's being processed on the
    # Space right now (runs are server-side, shared across sessions). The Timer
    # refreshes it for each connected client; the value callable seeds it on load.
    gr.Markdown("### Live activity\n_In-flight analyses on this Space._")
    activity = gr.Dataframe(
        headers=["Channel / target", "Status", "Progress"],
        value=_render_active_rows,
        wrap=True,
        label="Now processing",
    )
    gr.Timer(2.0).tick(_render_active_rows, outputs=[activity])

    # Structured JSON endpoints for the custom frontend (queued, SSE-backed).
    # analyze_channel: one-shot result. analyze_channel_stream: live progress.
    gr.api(analyze_channel_api, api_name="analyze_channel")
    gr.api(analyze_channel_stream, api_name="analyze_channel_stream")

    if (not MODAL_GEMMA4_URL
            and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))):
        gr.Markdown("> Set `MODAL_GEMMA4_URL` for fine-tuned Gemma 4 inference "
                    "or set `HF_TOKEN` for HF fallback inference. Set "
                    "`YOUTUBE_API_KEY` for full channel history; without it the app "
                    "uses the public RSS feed (most recent uploads only). "
                    "`ZYTE_API_KEY` is required for YouTube page/feed/transcript "
                    "access through Zyte.")

    model_backend_configured = (
        MODAL_GEMMA4_URL
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if model_backend_configured and not os.environ.get("ZYTE_API_KEY"):
        gr.Markdown("> Setup needed: set `ZYTE_API_KEY` for YouTube "
                    "page/feed/transcript access through Zyte.")


fastapi_app = FastAPI(title="Smol Signals")


@fastapi_app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model": DEFAULT_MODEL,
        "storage": "dataset" if storage.is_enabled() else "disabled",
    })


@fastapi_app.get("/api/leaderboard")
def leaderboard() -> JSONResponse:
    return JSONResponse(storage.get_leaderboard(), headers=_CACHE_HEADERS)


@fastapi_app.get("/api/channel/{channel_id}")
def channel(channel_id: str) -> JSONResponse:
    data = storage.get_channel(channel_id)
    if data is None:
        return JSONResponse({"error": "channel not found"}, status_code=404)
    return JSONResponse(data, headers=_CACHE_HEADERS)


@fastapi_app.get("/api/runs")
def active_runs() -> JSONResponse:
    """All in-flight runs (queued/running) for the live-activity feed."""
    return JSONResponse(runs.list_active(), headers={"Cache-Control": "no-store"})


@fastapi_app.get("/api/run/{run_id:path}")
def run_status(run_id: str) -> JSONResponse:
    """A single run's snapshot (+ result when done) for a quick status check."""
    run = runs.get_run(run_id)
    if run is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    snap = run.snapshot()
    snap["result"] = run.result
    return JSONResponse(snap, headers={"Cache-Control": "no-store"})


# Native Gradio UI + its queue/API live under /gradio.
app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/gradio",
    server_name=_server_host(),
    server_port=_server_port(),
    ssr_mode=False,
)

# Serve the built frontend at / (mounted LAST so /api/* and /gradio win).
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
else:
    @fastapi_app.get("/")
    def _placeholder() -> JSONResponse:
        return JSONResponse({
            "message": "Smol Signals is running. Native UI at /gradio. Build the "
                       "frontend (frontend/ -> next build) to serve the custom UI here.",
            "ui": "/gradio",
            "endpoints": ["/gradio (api: analyze_channel)", "/api/health",
                          "/api/leaderboard", "/api/channel/{id}", "/api/runs",
                          "/api/run/{id}"],
        })


def _serve() -> None:
    uvicorn.run(app, host=_server_host(), port=_server_port())


if __name__ == "__main__":
    _serve()
