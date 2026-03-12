# Core-Model Instruction Plan (Performance + Reliability)

## Context Update
Previous infrastructure/dashboard plan is considered completed.
This document replaces the old plan and defines the next optimization wave for the proxy and routing stack.

## Hard Constraints
1. Preserve compatibility for existing client endpoints:
- /api/chat
- /api/generate
- /api/embed
- /api/embeddings
- /api/models

2. Keep all runtime behavior environment-driven.

3. Do not start or restore local Qwen3.5-4B on this server.
- Local service vllm-embed-gpu1 is removed.
- Remote models on other servers must remain untouched unless explicitly requested.

## Priority Roadmap

### 1) Shared Upstream HTTP Client [DONE]
Goal:
Reduce latency and socket churn by reusing one AsyncClient per worker instead of creating clients per request.

Implementation:
- Add app-lifecycle managed shared client in upstream services.
- Reuse the same client in request forwarding and model status checks.

Acceptance criteria:
- No behavior change in API responses.
- Lower connection overhead under parallel load.

Status:
- Completed on 2026-03-12.
- Shared AsyncClient lifecycle added and reused by upstream forwarding + model status probes.

### 2) Deduplicate Model Probes in /api/models [DONE]
Goal:
Prevent duplicate entries and redundant health checks when aliases point to the same model+base_url.

Implementation:
- Deduplicate route checks by key: model_vllm + base_url + type.
- Keep strict alias resolution, but probe each unique target once.

Acceptance criteria:
- /api/models contains no duplicate rows for identical upstream target.
- Poller performs fewer redundant requests.

Status:
- Completed on 2026-03-12.
- Probes deduplicated by normalized model_vllm + base_url + type.
- Availability cache still resolves by aliases.

### 3) Configurable Connection Pooling Limits [DONE]
Goal:
Improve throughput and stability under concurrent traffic.

Implementation:
- Add env vars for pool tuning:
  - UPSTREAM_MAX_CONNECTIONS
  - UPSTREAM_MAX_KEEPALIVE_CONNECTIONS
  - UPSTREAM_KEEPALIVE_EXPIRY_SECONDS
- Wire these values into shared HTTP client creation.

Acceptance criteria:
- Defaults are safe and backward compatible.
- Values are exposed in .env and docker-compose passthrough.

Status:
- Completed on 2026-03-12.
- Added UPSTREAM_MAX_CONNECTIONS, UPSTREAM_MAX_KEEPALIVE_CONNECTIONS, UPSTREAM_KEEPALIVE_EXPIRY_SECONDS.
- Wired into shared AsyncClient limits in upstream service.

### 4) Centralized Bounded Retries (5xx/Transient Only) [DONE]
Goal:
Increase resilience without causing retry storms.

Implementation:
- Retry only for transient upstream conditions:
  - HTTP 5xx
  - connection reset / timeouts
- Add bounded retries with small jitter and cap.
- Keep non-retryable 4xx behavior unchanged.

Acceptance criteria:
- No infinite retries.
- Better success rate for intermittent failures.

Status:
- Completed on 2026-03-12.
- Retry policy centralized in upstream service.
- Retries apply only to transient request errors and HTTP 5xx.

### 5) Basic Prometheus Metrics [DONE]
Goal:
Make performance and error trends observable.

Implementation:
- Expose metrics endpoint and counters/histograms for:
  - endpoint latency
  - upstream latency by route
  - upstream errors by status code
  - model availability state

Acceptance criteria:
- Metrics endpoint available and scrapeable.
- Key dashboards can be built without code changes.

Status:
- Completed on 2026-03-12.
- Added /metrics endpoint and core request/upstream/model gauges.

### 6) Adaptive Status Polling [DONE]
Goal:
Reduce unnecessary status traffic while reacting quickly to outages.

Implementation:
- Use longer poll interval on stable state.
- Temporarily shorten interval after errors.

Acceptance criteria:
- Lower background probe load in stable periods.
- Faster detection during incidents.

Status:
- Completed on 2026-03-12.
- Poller now uses short interval on refresh errors or unavailable models.

### 7) Lightweight Perf Smoke Script [DONE]
Goal:
Track regression risk with quick repeatable checks.

Implementation:
- Add simple script with profiles:
  - warm single request
  - short burst concurrency
  - mixed chat/embed calls
- Print latency summary and error counts.

Acceptance criteria:
- Script runs from repo root with one command.
- Produces comparable before/after numbers.

Status:
- Completed on 2026-03-12.
- Added scripts/perf_smoke.py with warm, burst, and mixed profiles.

## Recommended Execution Order
1. Shared HTTP client.
2. Probe deduplication.
3. Pooling env settings.
4. Retry policy centralization.
5. Adaptive polling.
6. Metrics.
7. Perf smoke script.

## Validation Checklist Per Step
1. python3 -m py_compile proxy/app.py
2. python3 -m py_compile proxy/services/*.py
3. sudo docker compose up -d --build ollama-proxy
4. curl -sS http://127.0.0.1:11434/api/models
5. Smoke test one chat route and one embedding route.

## Definition of Done
1. Compatibility preserved for all public endpoints.
2. No local Qwen3.5-4B usage on this server.
3. Lower average upstream overhead in smoke tests.
4. Better observability and cleaner operational triage.
