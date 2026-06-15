# Lessons

- When the user corrects process or asks for a revised answer, read the local
  `AGENTS.md` first and apply its workflow before answering.
- When the user changes a backend from optional fallback to mandatory
  replacement, remove the old code paths and docs instead of layering the new
  backend onto legacy behavior.
- For Zyte-backed Python requests, prefer the extract API over proxy mode when
  the goal is hosted reliability; proxy mode requires CA bundle setup and can
  surface `CERTIFICATE_VERIFY_FAILED` in transcript fetching.
- For Zyte extract HTTP requests, do not forward `requests.Session` default
  headers. Use `customHttpRequestHeaders` as a list of `{name, value}` objects,
  and avoid browser-only `requestHeaders` for raw `httpResponseBody` calls.
- For Modal GPU web-server smoke tests, set `max_containers=1` before sending
  retrying clients. A read-timeout retry can trigger another cold-started GPU
  container while the first one is still loading.
- When saying a backend was removed, distinguish between removed from production
  routing and deleted from the repo. If the user asks to remove something, either
  delete/archive the experimental files too or explicitly say they remain as
  non-production experiments.
- Before resuming a persisted backend goal, reconcile it with the latest
  explicit direction from the user and the latest serving decision. Do not
  launch GPU experiments for an older SGLang objective after the thread has
  moved to a vLLM/Transformers decision without calling out the mismatch first.
