# Todo - Space channel timeout

- [ ] Confirm the deployed app is healthy but channel analysis can disappear or
      time out during long quiet periods.
- [ ] Add backend heartbeat/status events while a run is still active.
- [ ] Add enough failure logging to diagnose run crashes from Space logs.
- [ ] Tune interactive channel sample defaults through env/docs for Space
      reliability.
- [ ] Verify syntax, deploy the backend/config patch, and re-check Space health.

# Todo - Transcript retry policy

- [x] Confirm current transcript-fetch behavior and classify retryable vs.
      permanent transcript errors.
- [x] Add bounded retry logic for transient transcript/Zyte failures.
- [x] Verify syntax and review the focused diff.

## Review

- Existing behavior had no in-run transcript retry. It only timed requests out,
  surfaced per-video transcript errors, and retried errored videos on a later
  channel rerun.
- Added `TRANSCRIPT_FETCH_RETRIES` and `TRANSCRIPT_FETCH_RETRY_BACKOFF` around
  `fetch_transcript()`.
- Permanent transcript states such as unavailable/unplayable videos, disabled
  transcripts, missing requested-language transcripts, and invalid IDs are not
  retried.
- Transient request/blocking failures such as Zyte 429/5xx, timeouts, request
  blocking, and IP blocking are retried with exponential backoff.
- Verified `youtube_source.py` with `.venv\Scripts\python.exe -m py_compile`,
  module import, and a small retry-classification check.

# Todo - README and gitignore

- [x] Review the app entry points, dependency manifests, and deployment layout.
- [x] Create a README that explains local setup, configuration, runtime paths,
      and Space deployment.
- [x] Create a .gitignore that keeps secrets, caches, logs, and generated
      training artifacts out of git while allowing `frontend/out/` to be
      committed for Spaces.
- [x] Verify the final docs and working tree state.

## Review

- Replaced the README with an ASCII-only project guide covering the Gradio
  server, static Next.js frontend, environment variables, local run commands,
  Hugging Face Space deployment, and scoring behavior.
- Expanded `.gitignore` for local secrets, Python caches, virtual environments,
  logs, frontend build tooling, model/training outputs, and editor/OS files.
- Verified `*.log` ignores `gradio.stdout.log`, `.env.example` remains visible
  to git, and `frontend/out/index.html` is not ignored.

# Todo - Gemma 12B post-training dataset

## Modal Gemma 4 serving

- [x] Build a Modal SGLang server for `unsloth/gemma-4-12b` +
      `ajc426/small-signals-gemma-4-12b-sft-lora`.
- [x] Identify SGLang Gemma 4 incompatibility and stop the SGLang deployment.
- [x] Deploy a single-container Modal Transformers endpoint for the same
      fine-tuned Gemma 4 adapter.
- [x] Run the endpoint against current transcript/SFT examples and review
      JSON validity, grounded evidence, and call extraction quality.
- [x] Switch app inference from HF Inference to the Modal fine-tuned Gemma 4
      endpoint when `MODAL_GEMMA4_URL` is configured.
- [x] Verify the app extraction path with the new backend and document the
      deployed endpoint/config.

## SGLang Gemma 4 acceleration pass

- [x] Re-test Gemma 4 + LoRA on Modal's SGLang dev image with the Gemma 4
      VLM/runtime patches.
- [x] Confirm the current SGLang blocker is LoRA buffer loading for Gemma 4
      full-attention `o_proj` width mismatch, not stale Modal capacity.
- [x] Patch the SGLang LoRA buffer loader to tolerate non-uniform 2D adapter
      widths for the dev endpoint.
- [x] Run a one-row SGLang smoke eval against current transcripts.
- [x] Run the full 17-row SGLang validation eval.
- [x] Switch app inference to SGLang only if the smoke/full eval is stable;
      otherwise keep the verified Transformers endpoint as production.

## vLLM Gemma 4 serving spike

- [x] Research current vLLM Gemma 4, LoRA, structured-output, quantization, and
      Modal serving support.
- [x] Add a separate Modal vLLM dev server for `unsloth/gemma-4-12b` +
      `ajc426/small-signals-gemma-4-12b-sft-lora`.
- [x] Fix vLLM serving setup issues found during startup: use the current JSON
      object form for `--lora-modules`, pin `transformers==5.12.0` for
      `gemma4_unified`, force eager mode, and restrict LoRA targets to the
      adapter's projection modules.
- [ ] Smoke-test the vLLM dev server on one validation row.
- [ ] Run the full 17-row validation if the smoke test starts cleanly.
- [ ] Switch production only if vLLM matches or beats the Transformers endpoint
      on grounded JSON quality.

## Gemma 4 merged-model serving pass

- [x] Read the official Google Gemma Hugging Face QLoRA guide and confirm the
      recommended deployment path is to merge adapter weights into the base model
      with `merge_and_unload()` before using serving stacks like vLLM or TGI.
- [x] Add a Modal merge job for `unsloth/gemma-4-12b` +
      `ajc426/small-signals-gemma-4-12b-sft-lora`.
- [x] Push a merged HF model repo for serving without runtime LoRA.
- [x] Mirror the merged HF model into the official Build Small hackathon org
      for judge access and discoverability.
- [x] Smoke-test a merged-model server on one validation row.
- [x] Run the full 17-row validation if the smoke test is stable.
- [x] Switch production only if the merged-model server matches or beats the
      current Transformers endpoint on grounded JSON quality and latency.

Notes:

- First merge app run `ap-MHvvv806ylV068sY9zQa5v` built the CPU/high-memory
  Modal image but failed before model load because the Gemma 4 processor imports
  `torchvision.transforms.v2`; the merge image now includes the same processor
  dependencies used by the training image.
- Merge app run `ap-YlQlq6D7JDaj32a8dc8W5P` succeeded and pushed private model
  repo `ajc426/small-signals-gemma-4-12b-sft-merged` at
  `79f8df665fb2bab953eaaff3883af32cc90c1b55` with seven safetensor shards,
  tokenizer files, processor config, and `small_signals_merge.json`.
- Publish app run `ap-1vexsFenzTnQ6ipPMlJQG5` mirrored the merged checkpoint to
  public hackathon org repo
  `build-small-hackathon/small-signals-gemma-4-12b-sft-merged` at
  `e110b733a1975f66087a26ed289f9601fe683a9d`.
- The merged SGLang dev server now points at the public hackathon-org merged
  model repo, not the private `ajc426` mirror.
- First merged SGLang smoke attempt `ap-GZfqhmmsSByP7WB1cwSITr` failed before
  weight load because SGLang disallows `:` in `--served-model-name`; the merged
  dev server now uses `small-signals-gemma-4-12b-signals`.
- Merged SGLang smoke run `ap-Kv6EjDiv7wT0YLaGFcPgLV` loaded the public merged
  model on one L40S and passed one validation row: 1/1 valid JSON, 1/1
  grounded, 1/1 signal match. Cold path latency was 236.31 seconds.
- Full merged SGLang validation run `ap-o24YrbNAVEmOQsZvR4NJ3t` completed on
  one L40S against all 17 validation rows: 16/17 valid JSON, 15/17 grounded,
  12/17 signal matches, and 4 non-NONE ticker overlaps. First request latency
  was 238.81 seconds; warm requests averaged 4.84 seconds with 3.06 second
  median and 19.7 second max.
- Decision: do not switch production inference to merged SGLang. It is worse
  than the current Transformers endpoint, which previously scored 17/17 valid
  JSON, 17/17 grounded, 13/17 signal matches, and 4 non-NONE ticker overlaps
  on the same validation set.

## Gemma 4 12B follow-up

- [x] Verify `unsloth/gemma-4-12b` access and model metadata.
- [x] Adapt Modal trainer for Gemma 4 Unified (`AutoProcessor` + `AutoModelForMultimodalLM`).
- [x] Launch Modal training from `unsloth/gemma-4-12b`.
- [x] Verify and document the pushed Gemma 4 adapter repo.

## Gemma 3 12B completed run

- [x] Inspect persisted `ajc426/small-signals` data and identify saved videos for:
      Defiant Gatekeeper, Chicken Genius Singapore, Financial Education.
- [x] Build `training/harvest.py` to pull saved channel/video outputs from the HF Dataset
      repo and re-fetch transcripts for those videos.
- [x] Build `training/build_sft_dataset.py` to convert harvested records into chat-format
      SFT JSONL for the app's exact signal-extraction prompt/schema.
- [x] Build `training/modal_gemma12b_qlora.py` to run a Modal QLoRA/SFT job for
      Gemma 3 12B.
- [x] Run harvest + dataset build, then launch/verify the Modal training job.

## Review

- HF dataset inspection found exactly the three requested channels:
  Defiant Gatekeeper (84 saved videos), Chicken Genius Singapore (340), and
  Financial Education (2,896).
- Added a resumable transcript harvester, SFT JSONL builder, and Modal Gemma 12B
  QLoRA trainer under `training/`.
- Full transcript harvest finished with 209 valid harvested rows. The script skipped
  unavailable/unplayable transcript errors and kept going.
- Final grounded SFT dataset: 159 train rows, 17 validation rows, 33 rows skipped for
  ungrounded evidence quotes. Signal mix: 105 NONE, 45 buy, 20 hold, 6 sell.
- Pushed private HF Dataset repo: `ajc426/small-signals-gemma12b-sft`
  (`README.md`, `summary.json`, `train.jsonl`, `val.jsonl`).
- `google/gemma-3-12b-it` was gated for the current HF token, so the successful Modal
  run used accessible Gemma 3 12B base `unsloth/gemma-3-12b-it-bnb-4bit`.
- Modal run `ap-v156HrhwnwhAXnojlrMrHv`: 2 epochs, 40 optimizer steps, final
  train loss 0.3695, eval loss 0.3595.
- Pushed private HF model repo: `ajc426/small-signals-gemma-3-12b-sft-lora`
  (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files).
- First Gemma 4 Modal attempt `ap-2t6MKdaVhDhB6sc2O88VEm` loaded
  `unsloth/gemma-4-12b` and attached LoRA, then failed because the processor has
  no chat template. The trainer now renders Gemma 4 turn tokens explicitly.
- Successful Gemma 4 Modal run `ap-8AcaBE9TnAvqrJzoF9HG2g`: 2 epochs, 40
  optimizer steps, final train loss 0.5015, eval loss 0.4117.
- Pushed and verified private HF model repo:
  `ajc426/small-signals-gemma-4-12b-sft-lora` at
  `361450c0d82fddf6401a6c665d2de0e7cf08fc91` with
  `adapter_config.json`, `adapter_model.safetensors`, processor config, and
  tokenizer files.
- SGLang v0.5.10 can load the Gemma 4 base + LoRA after patches, but generation
  hits a non-uniform KV-cache shape failure for Gemma 4 full-attention layers, so
  it is not the production serving path yet.
- Deployed Modal Transformers endpoint:
  `https://angelotc--smol-signals-gemma4-transformers-serve.modal.run`, using
  `unsloth/gemma-4-12b` + `ajc426/small-signals-gemma-4-12b-sft-lora`,
  `max_containers=1`, a 2-minute idle scaledown window, and JSON-aware
  stopping.
- Full 17-row validation against the Transformers endpoint: 17/17 valid JSON,
  17/17 grounded, 13/17 signal matches, 4 non-NONE ticker overlaps. The model is
  conservative on some positive/hold calls, so the next quality pass should
  rebalance/oversample positive examples.
- App inference now prefers `MODAL_GEMMA4_URL` and reports
  `small-signals-gemma-4-12b:signals` from `/api/health`; HF Inference remains a
  fallback when the Modal URL is unset.
- Latest SGLang dev-image attempt starts the server and loads the adapter, but
  the first eval request fails while loading full-attention `o_proj` LoRA weights:
  SGLang allocates a `[16, 4096]` buffer for a `[16, 8192]` adapter tensor. The
  active app remains on the verified Transformers endpoint while this is patched.
- After patching the LoRA buffer loader, a one-row SGLang smoke eval succeeded:
  `200 OK`, 1/1 valid JSON, 1/1 grounded, and 1/1 signal match. Cold-start plus
  first request latency was about 143 seconds, so full-set latency and stability
  still need verification before switching the app.
- Full 17-row SGLang dev-image validation on L40S finished with no endpoint
  errors after lowering eval `max_tokens` to 512: 17/17 valid JSON, 15/17
  grounded, 13/17 signal matches, and 4 non-NONE ticker overlaps. Warm requests
  averaged 4.37 seconds with a 3.22 second median after a 136.05 second first
  request. This is not good enough to replace production because the current
  Transformers endpoint keeps 17/17 grounded on the same validation set.
- Decision: do not switch app inference to SGLang yet. The production path stays
  on `smol-signals-gemma4-transformers` while the SGLang LoRA shape workaround
  remains experimental.
- vLLM startup findings: `vllm==0.19.0` can parse the Gemma 4 config only after
  upgrading to `transformers==5.12.0`, but it falls back to the generic
  Transformers backend for `TransformersMultiModalForCausalLM`. With default
  compilation it fails in TorchDynamo; with `--enforce-eager` and explicit
  projection-only LoRA targets it still fails during engine profiling with
  `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x4096 and
  8192x3840)`. The one-row vLLM smoke never reached inference, so do not switch
  production to vLLM.
- Verified after the vLLM spike that the deployed production endpoint
  `https://angelotc--smol-signals-gemma4-transformers-serve.modal.run/health`
  is healthy and still reports backend `transformers` with model
  `small-signals-gemma-4-12b:signals`.

## Merged SGLang failure debug

- Re-scored the merged SGLang eval with app-style effective signals:
  `signal: NONE` plus non-empty `rotations` counts as effective `HOLD`.
  This makes row 5 a false alarm rather than a real model miss: merged SGLang
  extracted the correct `QQQ->XLV` rotation, but emitted top-level `NONE`.
- Fixed `serving/eval_sglang_endpoint.py` so `--structured-json-schema` uses
  SGLang's OpenAI-compatible `response_format: {type: json_schema, ...}` path
  instead of the ignored `structured_outputs` field. The schema now caps
  `assets` at 8 and `rotations` at 4 so constrained decoding can terminate.
- Row 2 targeted rerun with the corrected schema produced valid JSON instead
  of truncated JSON, but it still failed quality: predicted `BUY` without the
  target `QCOM` ticker, repeated TSLA/AAPL/NVDA assets, and used ungrounded
  evidence. Treat row 2 as an SGLang extraction-quality failure, not just a
  parser/max-token failure.
- Row 1 is a label-policy ambiguity: the teacher label marks `BUY TSLA` from
  "closed my GM short position to buy more Tesla stock"; merged SGLang,
  Transformers, and runtime-LoRA SGLang all returned `NONE`. This needs a
  deliberate policy decision and positive examples if disclosed current
  purchases should count as calls.
- Row 4 is a SGLang-specific false positive: merged/runtime-LoRA SGLang mapped
  generic "hold current investments / be careful making new investments" into
  `HOLD SPY`; Transformers correctly returned `NONE`. Add hard negatives for
  broad caution/portfolio-positioning language that should not force SPY.
- Row 8 is another label-policy ambiguity: the teacher label marks `BUY TSLA`
  from a long-term price-target statement, while all tested models returned
  `NONE`. Current prompt explicitly says not to output price target fields, so
  either relabel this as `NONE` or add examples saying long-horizon valuation
  targets imply a buy call.
- Projected merged-SGLang metrics after the row 2 schema fix and effective
  rotation scoring: 17/17 valid JSON, 15/17 grounded, 14/17 effective signal
  matches, 4 non-NONE ticker overlaps. Effective signal failures are rows
  1, 4, and 8; ungrounded valid rows are 2 and 4.
- Transformers under the same effective-signal scoring: 17/17 valid JSON,
  17/17 grounded, 14/17 effective signal matches, 4 non-NONE ticker overlaps.
  Its remaining effective signal failures are rows 1, 8, and 9.
- Current decision remains unchanged: do not switch production to merged
  SGLang until it beats the Transformers endpoint on groundedness and ticker
  overlap, not just valid JSON.

## Repaired merged SGLang deployed eval

- Deployed the repaired merged checkpoint as a Modal SGLang endpoint:
  `https://angelotc--smol-signals-gemma4-sglang-merged-repair-serve.modal.run`.
  The app routing and local `.env` were not changed to use it.
- First deployed eval attempt used the old colon model alias
  `small-signals-gemma-4-12b:signals`; SGLang treated that as a LoRA adapter
  request and returned 400 for every row. Re-ran with the actual served model
  name `small-signals-gemma-4-12b-signals`.
- Raw deployed SGLang eval on the 17-row validation set:
  17/17 valid JSON, 16/17 grounded, 14/17 effective signal matches, and
  5 non-NONE ticker overlaps. Output:
  `training/output/sglang_gemma4_merged_repair_deployed_eval.json`.
- App-normalized deployed SGLang eval:
  17/17 valid JSON, 17/17 grounded, 15/17 effective signal matches, and
  5 non-NONE ticker overlaps. Output:
  `training/output/sglang_gemma4_merged_repair_deployed_eval_normalized.json`.
- Remaining app-normalized misses are row 1
  (`Why isn’t General Motors Bankcrupt`, target `BUY TSLA`, predicted `NONE`)
  and row 8 (`Being successful in the stock market`, target `BUY TSLA`,
  predicted `NONE`). Row 4's raw false positive `HOLD SPY` is removed by the
  app quote-grounding normalizer.
