# Model Registry Migration Roadmap

Purpose
- Move model configuration from hardcoded env/code routing to a managed registry with safe rollout and progress tracking.

Execution priority note
- Current priority is local Docker/Compose flow.
- k3s rollout is intentionally deferred due to unavailable servers.

How status is tracked
- [ ] Not started
- [~] In progress
- [x] Completed
- [!] Blocked

## Scope
- Chat and embeddings routes (vLLM + llama backends)
- Ollama-compatible endpoints (/api/models, /api/tags, /api/chat, /api/chat-ui)
- Runtime controls per model (context window, max tokens, reasoning toggle, defaults)

## Target data model (what to store)

### A. Model registry (source of truth)
- [ ] model_key (stable internal key)
- [ ] public_name (API-visible name)
- [ ] backend_model_id (upstream model id)
- [ ] backend_type (vllm | llama_cpp | ollama | other)
- [ ] base_url
- [ ] model_type (chat | embeddings)
- [ ] is_enabled
- [ ] priority_or_weight

### B. Runtime config (per model)
- [ ] max_context_tokens
- [ ] default_max_tokens
- [ ] max_tokens_cap
- [ ] min_context_headroom
- [ ] reasoning_default_enabled
- [ ] stream_supported
- [ ] timeout_seconds (optional)
- [ ] retry_policy (optional)

### C. Metadata (for UI and /api/tags)
- [ ] family
- [ ] families[]
- [ ] parameter_size
- [ ] quantization_level
- [ ] format
- [ ] digest
- [ ] size
- [ ] modified_at

### D. Operational/audit fields
- [ ] created_at
- [ ] updated_at
- [ ] updated_by
- [ ] notes

## Implementation phases

### Phase 0: Design freeze and contracts
- [~] Agree final schema fields and naming
- [x] Decide DB choice (PostgreSQL/SQLite for first step)
- [x] Define ownership/edit process (who changes model records)
- [x] Define fallback behavior if DB is unavailable
- Exit criteria:
  - One approved schema contract doc
  - One approved rollout plan

## Phase 0 decisions (draft)

### DB choice (phase 1)
- Primary choice: PostgreSQL for production.
- First implementation path: PostgreSQL driver in proxy runtime.
- Runtime strategy: PostgreSQL + env fallback on connectivity/config issues.

### Ownership and change process
- Source of truth owner: backend platform team.
- Change channels:
  - Short term: seed/migration scripts in repo + PR review.
  - Mid term: internal admin API/script with audit logging.
- Required reviewers for production model changes:
  - one backend owner
  - one ops owner (for endpoint/base_url/runtime limits)

### Fallback behavior if DB is unavailable
- Request path must not hard-fail on transient DB errors.
- Read flow order:
  1. In-memory registry cache (fresh)
  2. Last known good snapshot (stale but valid)
  3. Env-based routing fallback
- On fallback activation:
  - emit warning logs
  - increment metric counter
  - preserve API compatibility for /api/models, /api/tags, /api/chat, /api/chat-ui

## Schema contract (draft v1)

### Table: model_registry
- id: uuid, PK, required
- model_key: text, unique, required
- public_name: text, required
- backend_model_id: text, required
- backend_type: enum(vllm, llama_cpp, ollama, other), required
- model_type: enum(chat, embeddings), required
- base_url: text, required
- is_enabled: bool, required, default true
- priority: int, required, default 100
- created_at: timestamptz, required, default now
- updated_at: timestamptz, required, default now
- updated_by: text, optional
- notes: text, optional

### Table: model_runtime_config
- id: uuid, PK, required
- model_id: uuid, FK -> model_registry.id, unique, required
- max_context_tokens: int, required
- default_max_tokens: int, required
- max_tokens_cap: int, required
- min_context_headroom: int, required, default 256
- stream_supported: bool, required, default false
- reasoning_supported: bool, required, default false
- reasoning_default_enabled: bool, required, default false
- timeout_seconds: numeric, optional
- retry_attempts: int, optional
- retry_base_delay_seconds: numeric, optional
- retry_jitter_seconds: numeric, optional

### Table: model_metadata
- id: uuid, PK, required
- model_id: uuid, FK -> model_registry.id, unique, required
- family: text, optional
- families: jsonb/text[], optional
- parameter_size: text, optional
- quantization_level: text, optional
- format: text, optional
- digest: text, optional
- size_bytes: bigint, optional, default 0
- modified_at: timestamptz, optional

### Table: model_aliases
- id: uuid, PK, required
- model_id: uuid, FK -> model_registry.id, required
- alias: text, required
- alias_type: enum(public, legacy, internal), required, default public
- is_active: bool, required, default true
- unique constraint: (alias, is_active=true)

### Validation rules (must enforce)
- If model_type=embeddings then chat endpoints cannot resolve this model.
- If model_type=chat then embeddings endpoints cannot resolve this model.
- base_url must be normalized (rstrip '/').
- priority must be non-negative.
- max_context_tokens >= default_max_tokens.
- max_tokens_cap >= default_max_tokens.

### Compatibility mapping
- /api/models: map from model_registry + runtime status cache.
- /api/tags: map from model_registry + metadata (chat + embeddings as currently required).
- /api/chat and /api/chat-ui: resolve by aliases first, then public_name, then backend_model_id.
- Reasoning toggle: apply only where reasoning_supported=true (llama routes in current behavior).

### Phase 1: Read-only registry integration
- [x] Add storage layer and migrations
- [x] Seed current models from .env into registry
- [x] Add read path for /api/models and /api/tags from registry
- [x] Keep routing still env-driven (no behavior change)
- [x] Add feature flag: MODEL_REGISTRY_ENABLED (read-only)
- Exit criteria:
  - /api/models and /api/tags output matches current behavior
  - Can disable registry and fallback to current behavior

### Phase 2: Routing by registry (safe switch)
- [x] Resolve chat/embed targets using registry records
- [x] Preserve alias matching and strict endpoint compatibility
- [x] Preserve llama-only reasoning toggle behavior
- [x] Preserve stream behavior and done/error guarantees
- [x] Keep env fallback for emergency rollback
- Exit criteria:
  - All smoke tests pass on both vLLM and llama models
  - Rollback toggle tested

### Phase 3: Runtime controls from registry
- [~] Use per-model max_context_tokens/default_max_tokens/caps
- [x] Use per-model capabilities (stream/reasoning support)
- [x] Enforce endpoint capability checks from registry
- Exit criteria:
  - No hardcoded per-model context values left in runtime path
  - One model can be tuned without redeploy

### Phase 4: Operations and governance
- [ ] Add admin update workflow (script or internal endpoint)
- [ ] Add audit logging for config changes
- [ ] Add monitoring for registry read errors and stale cache
- [ ] Add backup/restore docs
- Exit criteria:
  - Change process documented and tested
  - Alerting in place

## Risks and mitigations
- [ ] DB unavailable during request path
  - Mitigation: in-memory cache + last known good snapshot + env fallback
- [ ] Misconfiguration breaks routing
  - Mitigation: validation rules + dry-run checker + staged rollout
- [ ] Drift between env and registry during migration
  - Mitigation: one-way sync script + cutover checklist

## Test plan
- [ ] Unit tests for route resolution by capabilities/model type
- [ ] Contract tests for /api/models and /api/tags shape
- [ ] Smoke tests for /api/chat and /api/chat-ui (stream)
- [ ] Smoke tests for /api/embed and /api/embeddings
- [ ] Failure-path tests (upstream unavailable, model disabled)

## Rollout checklist
- [ ] Deploy with registry disabled
- [ ] Run sync and validate parity
- [ ] Enable registry for read-only endpoints
- [ ] Enable registry-based routing for small subset
- [ ] Full cutover
- [ ] Remove deprecated env-only paths (final cleanup)

## Initial seed plan (.env -> registry)

### Seed targets from current environment
- Chat routes:
  - QWEN chat: QWEN_CHAT_MODEL / QWEN_CHAT_BASE_URL / PUBLIC_QWEN_CHAT_MODEL
  - MINISTRAL chat: MINISTRAL_CHAT_MODEL / MINISTRAL_CHAT_BASE_URL / PUBLIC_MINISTRAL_CHAT_MODEL
  - CPU Q4 chat: CPU_CHAT_Q4_MODEL / CPU_CHAT_Q4_BASE_URL / PUBLIC_CPU_CHAT_Q4_MODEL
  - CPU Q6 chat: CPU_CHAT_Q6_MODEL / CPU_CHAT_Q6_BASE_URL / PUBLIC_CPU_CHAT_Q6_MODEL
- Embedding routes:
  - QWEN embed default: QWEN_EMBED_MODEL / QWEN_EMBED_BASE_URL / PUBLIC_QWEN_EMBED_MODEL
  - QWEN embed 8B: QWEN_EMBED_8B_MODEL / QWEN_EMBED_8B_BASE_URL / PUBLIC_QWEN_EMBED_8B_MODEL
  - QWEN embed 4B: QWEN_EMBED_4B_MODEL / QWEN_EMBED_4B_BASE_URL / PUBLIC_QWEN_EMBED_4B_MODEL

### Mapping rules: model_registry
- model_key:
  - normalize(public_name) when present
  - else normalize(backend_model_id)
- public_name: PUBLIC_* env value (fallback to backend_model_id)
- backend_model_id: *_MODEL env value
- backend_type:
  - vllm for qwen/ministral routes on /v1-compatible servers
  - llama_cpp for CPU Q4/Q6 routes
- model_type:
  - chat for QWEN_CHAT, MINISTRAL_CHAT, CPU_CHAT_Q4, CPU_CHAT_Q6
  - embeddings for QWEN_EMBED*
- base_url: *_BASE_URL env value normalized with rstrip('/').
- is_enabled: true for all seeded records
- priority:
  - 100 default
  - optional override: CPU Q4=110, CPU Q6=120 if we want explicit fallback order

### Mapping rules: model_runtime_config
- max_context_tokens:
  - QWEN_CHAT_MAX_CONTEXT_TOKENS (QWEN chat)
  - MINISTRAL_CHAT_MAX_CONTEXT_TOKENS (Ministral chat)
  - CPU_CHAT_Q4_MAX_CONTEXT_TOKENS (CPU Q4)
  - CPU_CHAT_Q6_MAX_CONTEXT_TOKENS (CPU Q6)
  - QWEN_EMBED_MAX_CONTEXT_TOKENS / QWEN_EMBED_8B_MAX_CONTEXT_TOKENS / QWEN_EMBED_4B_MAX_CONTEXT_TOKENS (embeddings)
- default_max_tokens: VLLM_DEFAULT_MAX_TOKENS
- max_tokens_cap: VLLM_MAX_TOKENS_CAP
- min_context_headroom: VLLM_MIN_CONTEXT_HEADROOM
- stream_supported:
  - true for chat routes
  - false for embeddings
- reasoning_supported:
  - true only for CPU Q4/Q6 llama routes
  - false for vLLM routes and embeddings
- reasoning_default_enabled:
  - false when DISABLE_THINKING=1
  - true when DISABLE_THINKING=0
- timeout/retry defaults:
  - timeout_seconds <- UPSTREAM_TIMEOUT_SECONDS
  - retry_attempts <- UPSTREAM_RETRY_ATTEMPTS
  - retry_base_delay_seconds <- UPSTREAM_RETRY_BASE_DELAY_SECONDS
  - retry_jitter_seconds <- UPSTREAM_RETRY_JITTER_SECONDS

### Mapping rules: model_metadata
- family/families:
  - qwen for QWEN*
  - mistral for MINISTRAL*
- parameter_size:
  - parse from model string (e.g. 9B, 14B, 8B, 4B)
- quantization_level:
  - Q4_K_M for CPU_CHAT_Q4_MODEL
  - Q6_K for CPU_CHAT_Q6_MODEL
  - empty for non-quantized entries
- format:
  - gguf for CPU Q4/Q6
  - unknown for current vLLM routes unless explicit format added later
- digest/size_bytes/modified_at:
  - optional in phase 1 (can be null/default values)

### Mapping rules: model_aliases
- Add alias rows for both public and backend names.
- alias_type=public for PUBLIC_* names.
- alias_type=internal for backend_model_id values.
- Keep aliases active by default.

### Seed execution steps
1. Parse .env into route candidates.
2. Skip invalid candidates (missing model/base_url pair).
3. Upsert model_registry by model_key.
4. Upsert runtime_config and metadata by model_id.
5. Upsert aliases and deactivate removed aliases.
6. Produce seed report (inserted/updated/skipped) for CI logs.

### Seed parity checks (must pass)
- /api/models parity:
  - same model names and types as current env-driven behavior
  - same max_context_tokens values per model
- /api/tags parity:
  - same set of models currently exposed (chat + embeddings)
- Routing parity:
  - chat aliases resolve to same backend as before
  - embedding aliases resolve to same backend as before

## Progress log
- 2026-03-23
  - [x] Created roadmap document and initial phased checklist.
  - [~] Phase 0 started: schema and rollout discussion underway.
  - [x] Drafted schema contract v1 (tables, fields, defaults, validations).
  - [x] Chosen DB strategy for phase 1 (PostgreSQL primary, SQLite local/dev path).
  - [x] Defined ownership and DB-unavailable fallback strategy.
  - [x] Added detailed .env-to-registry seed mapping and parity checks.
  - [x] Implemented read-only registry service with SQLite schema and env-sync seed upsert.
  - [x] Added feature flags (MODEL_REGISTRY_ENABLED, MODEL_REGISTRY_DB_PATH) in settings/.env/compose/k8s.
  - [x] Wired /api/models and /api/tags data source through registry-enabled status checks with env fallback.
  - [x] Smoke-tested both modes: registry disabled and enabled.
  - [x] Switched model registry implementation to PostgreSQL (psycopg) and DSN-based configuration.
  - [x] Added local PostgreSQL service in docker-compose and wired proxy dependency.
  - [x] Added sync parity report logging (inserted/updated/unchanged/removed).
  - [x] Fixed concurrent startup schema race with PostgreSQL advisory lock.
  - [x] Added Prometheus metrics for registry DB health, sync counters, and fallback reasons.
  - [x] Marked k3s rollout as deferred; local/docker execution is primary.
  - [x] Added MODEL_REGISTRY_ROUTING_ENABLED feature flag and env/compose wiring.
  - [x] Implemented registry-fed resolver path for chat/embeddings with fallback to legacy env routing.
  - [x] Verified routing parity ON vs OFF for chat/embed + mismatch validation.
  - [x] Verified stream terminal events and llama reasoning flag behavior with routing ON.
  - [x] Added runtime limit fields to registry sync (default_max_tokens/max_tokens_cap/min_context_headroom).
  - [x] Wired chat/generate max_tokens resolver to per-target runtime limits from registry-fed status cache.
  - [x] Added stream/reasoning capability fields to registry-fed targets and enforced them in /api/chat and /api/chat-ui.
  - [x] Added targeted capability smoke checks to scripts/perf_smoke.py and validated expected 400 fail-fast responses.
  - [x] Hardened smoke checks with startup wait and extended retries for transient post-restart resets.
  - [x] Added preflight + guards-only smoke mode to separate proxy regressions from upstream availability issues.
  - [x] Added --exit-on-infra-errors policy to full smoke for CI-friendly WARN mode and strict default mode.
  - [x] Enabled MODEL_REGISTRY_ROUTING_ENABLED by default for local Docker (.env + compose fallback).
  - [x] Added MODEL_REGISTRY_SYNC_FROM_ENV toggle and switched local mode to DB-first (sync disabled by default).
  - [x] Added one-shot seeding workflow for registry via scripts/seed_model_registry.py.
  - [x] Simplified API routing to registry-first path by default; env resolver remains fallback-only.
  - [~] Stage/k3s default enablement deferred until cluster access is restored.

## Current focus (next actions)
1. Enable routing by default in stage/k3s once server access is restored.
2. Return to k3s manifests only when server access is restored.
