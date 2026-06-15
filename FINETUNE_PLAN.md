# LLM Usage Analysis & QLoRA Fine-Tune Plan

## Context

This repo (`simplefin/gradio-app`, "small-signals") makes exactly **one LLM call**:
`summarize_transcript_to_signal()` in `signals.py:132`. It sends a YouTube transcript
to `google/gemma-3-27b-it` via `huggingface_hub.InferenceClient.chat_completion`
(`temperature=0`, `max_tokens=1500`, `response_format={"type":"json_object"}`) and
expects a strict JSON object: `{signal, assets[], rotations[], summary}`. The prompt,
schema, and few-shot examples are `SYSTEM_PROMPT` + `SCHEMA_AND_RULES` (`signals.py:53-95`),
assembled by `_build_prompt()` (`signals.py:98`). Output is repaired by
`_safe_json_parse()` (`signals.py:112`) and validated/canonicalized by `_normalize()`
(`signals.py:154`): tickers via `canonical_ticker()`, evidence quotes must be ≥8 chars,
NONE→hold, dedupe, drop invalid tickers. Downstream, `market.py` grades each call vs SPY
and `storage.py` persists results to an HF Dataset repo (`HF_BUCKET_NAME`).

**Goal:** distill a strong teacher (OpenAI GPT‑5.5) into a small student
(**Qwen3‑8B**) using **QLoRA**, to get a cheaper/faster model that matches or beats
Gemma‑27B on extraction *fidelity*. The app already supports swapping the model via the
`HF_MODEL` env var (`signals.py:18`), so deployment is mostly a config switch.

### Two facts that shape the whole plan
1. **Transcripts are not stored.** `VideoResult` and `videos/*.json` keep the *output*
   (calls/summary) but not the *input transcript*. Training pairs must be re-harvested
   (or captured going forward) — the persisted data alone is insufficient.
2. **Market verdict ≠ extraction quality.** `market.py`'s verdict measures whether the
   *creator* was right, not whether the *model extracted faithfully*. The fine-tune target
   is extraction fidelity (correct ticker/signal, grounded evidence quote, correct NONE),
   so gold labels come from the **teacher model**, NOT from SPY scores. The market score is
   kept only as an orthogonal evaluation/triage signal, never as a training label.

## Approach (offline training pipeline in a new `training/` dir; app code barely changes)

All new work lives in a new `training/` folder so the runtime app stays clean. We
**reuse the app's exact prompt construction** so train-time and inference-time inputs match.

### 1. Harvest raw inputs — `training/harvest.py`
- Collect video IDs: read `runs.jsonl` / `videos/*.json` from the HF Dataset repo
  (`storage._read_text`, `storage._read_json`) for already-seen videos, and/or enumerate
  fresh channels with `youtube_source.resolve_channel()` + `list_channel_videos()`.
- For each video, fetch the transcript with `youtube_source.fetch_transcript(video_id)`
  (handles proxy rotation already) plus title/channel/published_at.
- Write `training/data/raw.jsonl`: one record per video with
  `{video_id, channel_title, video_title, published_at, transcript}`.
- Skip empty/blocked transcripts. Target a few thousand videos across diverse channels
  (finance, plus off-topic channels to supply hard NONE negatives).

### 2. Label with the teacher — `training/label_teacher.py`
- For each raw record, build the **same** user prompt via `signals._build_prompt(...)` and
  send `SYSTEM_PROMPT` + prompt to **OpenAI GPT‑5.5** with JSON mode (`response_format`).
- Run the teacher output through the app's own `_safe_json_parse()` + `_normalize()` so the
  gold target is already in canonical, schema-valid form (identical to what the app accepts).
- Re-serialize the normalized `ExtractedSignal` back to the canonical JSON shape
  (`signal/assets/rotations/summary` with `evidenceQuote`/`whyThisIsCurrentCall` keys) — this
  is the assistant target string.
- **Quality gates before keeping a pair:** valid JSON; every asset's `evidenceQuote` is an
  actual substring of the transcript (groundedness check — drop hallucinated quotes); ticker
  passes `canonical_ticker()`. Human spot-check a sample (~100-200) and a slice of NONE cases.
- Balance the set: keep a healthy fraction of NONE / off-topic examples so the student
  doesn't over-extract (the most common failure mode). Output `training/data/labeled.jsonl`.

### 3. Build the SFT dataset — `training/build_dataset.py`
- Emit chat-format JSONL: `messages = [{system: SYSTEM_PROMPT}, {user: _build_prompt(...)},
  {assistant: <canonical JSON>}]`. Reusing `_build_prompt` guarantees train==inference format.
- Stratified split (by `signal` and NONE-vs-call) into `train.jsonl` / `val.jsonl` (~90/10).
- Note token budget: transcripts are truncated to 40k chars (`tickers.truncate_for_prompt`),
  ≈10-13k tokens. Set `max_seq_len` to ~12-16k. Qwen3‑8B is natively 32k context, so this fits,
  but long sequences dominate VRAM — see step 4.

### 4. QLoRA training — `training/train_qlora.py`
- **Stack:** `transformers` + TRL `SFTTrainer` + PEFT LoRA + `bitsandbytes` 4-bit
  (nf4 + double quant, compute dtype bf16). Optionally Unsloth for ~2x speed / lower VRAM.
- **Base:** `Qwen/Qwen3-8B`. Use Qwen3 chat template with **thinking disabled**
  (`enable_thinking=False`) — this is deterministic JSON extraction, not reasoning; keeps
  outputs short and within `max_tokens`.
- **LoRA:** r=16-32, alpha=32, dropout=0.05, target all linear projections
  (`q,k,v,o,gate,up,down`). **Train on completion only** (mask the prompt tokens) so loss
  focuses on the JSON, not on memorizing transcripts.
- **Optim:** lr 1-2e-4, cosine, warmup ~3%, 1-3 epochs, grad checkpointing on, packing off
  (sequences are long and variable), effective batch via grad accumulation.
- **Hardware:** 8B QLoRA at ~16k seq len fits a single 24-40GB GPU (A100/4090/L4-class);
  if VRAM is tight, lower `max_seq_len` to 8k and tighten `truncate_for_prompt`'s budget for
  the training corpus (keep the runtime app at 40k or whatever you serve).

### 5. Evaluate — `training/eval.py`
- Run base Qwen3‑8B, the QLoRA student, and Gemma‑27B on the held-out `val.jsonl`.
- Metrics (reuse `_safe_json_parse`/`_normalize` to parse every model's output identically):
  JSON validity rate; signal exact-match; ticker-set precision/recall/F1; NONE
  precision/recall (over-extraction guard); evidence-quote-in-transcript rate (groundedness);
  rotation pair accuracy. Compare student vs Gemma; the bar is "≥ Gemma fidelity at a fraction
  of cost." Optionally cross-reference `market.py` scores as a sanity triage, not a metric.

### 6. Deploy — config-only change in the app
- Merge adapter or keep as adapter; push to HF Hub. Serve via an HF Inference Endpoint / TGI
  that supports the OpenAI-compatible `chat_completion` + `response_format` the app uses.
- Set `HF_MODEL=<your-org/qwen3-8b-signals>` (and point `InferenceClient` at the endpoint
  if self-hosted). **No code change needed** beyond env config — `signals.py:18,140` already
  read `HF_MODEL`/`model`. Verify the served model honors JSON mode; if not, the existing
  `_safe_json_parse` fallback already strips fences/prose.

### Optional 7. Preference tuning (DPO/ORPO) — later
- If SFT still over-extracts or mis-grounds, build preference pairs: chosen = teacher/corrected
  extraction, rejected = base-student error (hallucinated ticker, missed NONE). Train a short
  DPO pass on the same QLoRA adapter. Still extraction-fidelity based, never market outcome.

## New / touched files
- New: `training/harvest.py`, `training/label_teacher.py`, `training/build_dataset.py`,
  `training/train_qlora.py`, `training/eval.py`, `training/requirements.txt`
  (`transformers, trl, peft, bitsandbytes, accelerate, datasets, openai`).
- Reused (imported, not modified): `signals._build_prompt`, `SYSTEM_PROMPT`,
  `_safe_json_parse`, `_normalize`; `tickers.canonical_ticker`, `truncate_for_prompt`;
  `youtube_source.fetch_transcript/list_channel_videos/resolve_channel`; `storage._read_*`.
- App change at deploy time: **env only** (`HF_MODEL`), optionally endpoint URL in
  `signals.summarize_transcript_to_signal`'s `InferenceClient(...)`.
- Optional capture-going-forward: persist the transcript alongside each run (add a field in
  `analysis._analyze_video` / `storage.record_run`) so future training data needs no re-harvest.

## Verification
- **Data:** after step 3, eyeball `train.jsonl` — confirm every assistant target is valid JSON
  that round-trips through `_normalize()`, and that NONE examples are well represented.
- **Training:** watch val loss; run `training/eval.py` and confirm the QLoRA student meets/beats
  Gemma‑27B on signal exact-match, ticker F1, NONE recall, and groundedness.
- **End-to-end:** set `HF_MODEL` to the new model, run `python app.py`, analyze a known channel
  through the Gradio UI / `/analyze_channel`, and confirm `_normalize()` accepts the output and
  `market.py` scores render — i.e., the app behaves identically with the smaller, cheaper model.
