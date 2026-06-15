---
title: Smol Signals
emoji: chart_with_upwards_trend
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
short_description: Score finance YouTubers' stock calls with a small model
---

# Smol Signals

Smol Signals turns finance YouTube videos into a track record. Give it a
creator channel, `@handle`, channel ID, channel URL, or a single video URL. The
app fetches transcripts, extracts explicit forward-looking market calls, scores
the calls against SPY after the relevant horizon, and rolls the results up into
a creator reputation score.

This repo is built as a Hugging Face Gradio Space with a custom static Next.js
frontend served by the same Python process.

## What It Does

1. Discovers recent videos for a YouTube channel or accepts a single video URL.
2. Fetches transcripts through Zyte-backed YouTube access.
3. Extracts explicit buy, sell, hold, and sector-rotation calls with a small
   model endpoint.
4. Scores matured calls against SPY with benchmark-relative, volatility-adjusted
   alpha.
5. Persists channel results and leaderboard data to a Hugging Face Dataset repo
   when storage is configured.

This is not financial advice. It measures historical on-camera calls against a
market benchmark.

## App Layout

One server process listens on port `7860` and serves:

| Path | Description |
| --- | --- |
| `/` | Static Next.js frontend exported to `frontend/out/` |
| `/gradio` | Native Gradio Blocks UI |
| `/analyze_channel` | Queued Gradio API endpoint used by the frontend |
| `/api/health` | Runtime health and model backend metadata |
| `/api/leaderboard` | Cached leaderboard JSON |
| `/api/channel/{channel_id}` | Cached channel result JSON |
| `/api/runs` | In-flight analysis runs |
| `/api/run/{run_id}` | Status and result for one run |

Key files:

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI, Gradio, API routes, and static frontend mount |
| `analysis.py` | End-to-end analysis orchestration |
| `signals.py` | Model prompting, response parsing, and signal normalization |
| `youtube_source.py` | YouTube discovery and transcript retrieval |
| `market.py` | Market data lookup and call scoring |
| `storage.py` | Hugging Face Dataset-backed persistence |
| `runs.py` | Server-side run lifecycle and progress streaming |
| `frontend/` | Static-exported Next.js frontend |
| `training/` and `serving/` | Model fine-tuning, validation, and Modal serving utilities |

## Configuration

Copy `.env.example` to `.env` for local development and fill in the values you
need. On Hugging Face Spaces, add the same values as Space secrets.

| Variable | Required | Description |
| --- | --- | --- |
| `HF_TOKEN` | Yes for storage or HF fallback | Hugging Face token for Dataset writes and fallback inference |
| `MODAL_GEMMA4_URL` | No | Preferred fine-tuned Gemma 4 endpoint, OpenAI-compatible at `/v1/chat/completions` |
| `MODAL_GEMMA4_MODEL` | No | Modal model name, defaults to `small-signals-gemma-4-12b:signals` |
| `MODAL_GEMMA4_TIMEOUT` | No | Modal request timeout in seconds, defaults to `1200` |
| `HF_MODEL` | No | Hugging Face fallback model when `MODAL_GEMMA4_URL` is not set |
| `HF_BUCKET_NAME` | No | Dataset repo such as `org/small-signals`; unset means stateless storage |
| `YOUTUBE_API_KEY` | No | Enables full channel history and name search; otherwise RSS fallback is used |
| `ZYTE_API_KEY` | Yes for YouTube analysis | Required for YouTube page, feed, and transcript access |
| `TRANSCRIPT_FETCH_RETRIES` | No | Retry count for transient transcript fetch failures, defaults to `2` |
| `TRANSCRIPT_FETCH_RETRY_BACKOFF` | No | Initial retry backoff in seconds, defaults to `1.5` |
| `YOUTUBE_TRANSCRIPT_LANG` | No | Preferred transcript language, defaults to `en` |
| `ANALYZE_CONCURRENCY` | No | Number of videos analyzed concurrently, defaults to `6` |

## Local Development

Install backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Build the custom frontend:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

Run the app:

```powershell
python app.py
```

Open `http://localhost:7860` for the custom frontend or
`http://localhost:7860/gradio` for the native Gradio UI.

If `frontend/out/` has not been built, the server still starts and returns a
small JSON placeholder at `/`; the Gradio UI remains available at `/gradio`.

## Frontend Notes

The Next.js app uses `output: "export"`, so `npm run build` emits static files
to `frontend/out/`. That directory is intentionally commit-able because
`app.py` serves it directly in the Hugging Face Space. Node is not required at
runtime.

## Deploy To Hugging Face Spaces

1. Build the frontend with `cd frontend && npm ci && npm run build`.
2. Commit the Python app, `requirements.txt`, `frontend/out/`, and supporting
   source files.
3. Create a Gradio Space and push this repo.
4. Add the required Space secrets from the configuration table.
5. If `HF_BUCKET_NAME` is set, make sure the token can create or write to that
   private Dataset repo.

## Scoring Model

- Calls only score after the target horizon has elapsed and market data exists.
- Recent calls remain pending until they mature.
- Per-call scores use benchmark-relative alpha against SPY.
- Channel reputation is centered at 50 and increases or decreases based on
  measured call quality and confidence from sample size.
- High-volatility tickers are adjusted so the score reflects excess alpha, not
  only amplified market beta.
