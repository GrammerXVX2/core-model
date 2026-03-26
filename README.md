vLLM + Ollama-compat proxy

Overview
- vLLM serves OpenAI-compatible API on port 8000.
- Proxy exposes Ollama-compatible endpoints on port 11434.

Requirements
- Docker with NVIDIA runtime
- 2x RTX 5090
- vLLM nightly image (required for Qwen3.5 support before vLLM 0.17.0)

Setup
1) Get a Hugging Face token:
   - https://huggingface.co/settings/tokens
   - Create a "Read" token.
2) Copy .env.example to .env and set HF_TOKEN if required.

Run
- docker compose pull
- docker compose up

Endpoints
- POST http://localhost:11434/api/chat
- POST http://localhost:11434/api/chat-ui
- POST http://localhost:11434/api/generate
- POST http://localhost:11434/api/embed

PostgreSQL (pgAdmin)
- Host: `127.0.0.1`
- Port: `5432` (or `MODEL_REGISTRY_POSTGRES_PORT` from `.env`)
- Database: `model_registry`
- Username: `postgres`
- Password: `postgres`
- If pgAdmin runs in Docker on the same compose network, use host `postgres` and port `5432`.

Model Registry Source of Truth
- Runtime model routing/limits are DB-only and read from PostgreSQL.
- If registry is empty or unavailable, API returns explicit errors; env model definitions are not used in runtime path.
- New models should be added only in DB (do not add model routes/aliases to `.env`).
- Optional bulk seed from JSON checks file:
  - `python3 scripts/seed_model_registry.py --checks-file ./checks.json`
- Quick register/update single model (no env edits required):
   - `python3 scripts/register_model.py --public-model Qwen3.5-122B-A10B-FP8 --backend-model qwen3 --base-url http://10.77.163.200:8000/v1 --type chat --max-context 131072 --default-max 131072 --max-cap 131072 --headroom 256 --stream-supported --no-reasoning-supported`
- API register/update endpoint:
   - `POST /api/models/register`
   - Example body:
     `{"public_model":"Qwen3.5-122B-A10B-FP8","vllm_model":"qwen3","model_type":"chat","base_url":"http://10.77.163.200:8000/v1","max_context_tokens":131072,"default_max_tokens":131072,"max_tokens_cap":131072,"min_context_headroom":256,"stream_supported":true,"reasoning_supported":false,"aliases":["Qwen3.5-122B-A10B-FP8","qwen3"]}`
- Full CRUD (ID-based):
   - `POST /api/models` create model (no `id` in body; generated automatically)
   - `GET /api/models/{id}` get model by id
   - `PUT /api/models/{id}` update model by id
   - `DELETE /api/models/{id}` disable model by id
   - `id` is available in `GET /api/models` response.

Smoke checks
- Guards-only validation (recommended when upstream chat backend is unstable):
   - `python3 scripts/perf_smoke.py --base-url http://127.0.0.1:11434 --guards-only --startup-wait-seconds 3`
- Full lightweight smoke with capability checks:
   - `python3 scripts/perf_smoke.py --base-url http://127.0.0.1:11434 --burst-requests 2 --mixed-requests 2 --concurrency 2 --capability-checks --startup-wait-seconds 3`
- CI-friendly full smoke (treat infra failures as warnings, still keep capability checks strict):
   - `python3 scripts/perf_smoke.py --base-url http://127.0.0.1:11434 --burst-requests 2 --mixed-requests 2 --concurrency 2 --capability-checks --startup-wait-seconds 3 --no-exit-on-infra-errors`
- Optional strict preflight (fails when required models are unavailable in `/api/models`):
   - add `--preflight-strict`

Notes
- `/api/chat` returns non-stream response only and always with reasoning disabled.
- `/api/chat-ui` supports stream toggle (`stream=true/false`) and reasoning toggle (`reasoning/thinking/enable_thinking`).
- `/api/generate` streaming is not implemented yet.
- `/api/generate` returns non-stream response only and always with reasoning disabled.
- `/api/embeddings` is removed; use `/api/embed`.
- `TOKEN_CAP_DYNAMIC_MODE=1` enables dynamic output cap per request: `hard_cap = max_context_tokens - estimated_input_tokens - headroom`.
- In dynamic mode, static `max_tokens_cap` does not clamp responses; limit is derived from each model context budget.
- `TOKEN_BUDGET_STRICT_MODE=1` enables fail-fast 400 errors when requested output does not fit model token budget.
- Set `TOKEN_BUDGET_STRICT_MODE=0` to restore auto-clamp behavior (silent max_tokens reduction).
- Adjust --max-num-seqs and --gpu-memory-utilization in docker-compose.yml based on load.
- vLLM is pinned to `vllm/vllm-openai:nightly` for Qwen3.5 compatibility.
