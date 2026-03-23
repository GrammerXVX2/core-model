# Model Registry Migration Roadmap

Purpose
- Move model configuration from hardcoded env/code routing to a managed registry with safe rollout and progress tracking.

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
- [ ] Decide DB choice (PostgreSQL/SQLite for first step)
- [ ] Define ownership/edit process (who changes model records)
- [ ] Define fallback behavior if DB is unavailable
- Exit criteria:
  - One approved schema contract doc
  - One approved rollout plan

### Phase 1: Read-only registry integration
- [ ] Add storage layer and migrations
- [ ] Seed current models from .env into registry
- [ ] Add read path for /api/models and /api/tags from registry
- [ ] Keep routing still env-driven (no behavior change)
- [ ] Add feature flag: MODEL_REGISTRY_ENABLED (read-only)
- Exit criteria:
  - /api/models and /api/tags output matches current behavior
  - Can disable registry and fallback to current behavior

### Phase 2: Routing by registry (safe switch)
- [ ] Resolve chat/embed targets using registry records
- [ ] Preserve alias matching and strict endpoint compatibility
- [ ] Preserve llama-only reasoning toggle behavior
- [ ] Preserve stream behavior and done/error guarantees
- [ ] Keep env fallback for emergency rollback
- Exit criteria:
  - All smoke tests pass on both vLLM and llama models
  - Rollback toggle tested

### Phase 3: Runtime controls from registry
- [ ] Use per-model max_context_tokens/default_max_tokens/caps
- [ ] Use per-model capabilities (stream/reasoning support)
- [ ] Enforce endpoint capability checks from registry
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

## Progress log
- 2026-03-23
  - [x] Created roadmap document and initial phased checklist.
  - [~] Phase 0 started: schema and rollout discussion underway.

## Current focus (next actions)
1. Finalize schema contract (fields, required/optional, defaults).
2. Choose DB engine for first rollout step.
3. Prepare migration strategy from current .env model config to initial seed.
