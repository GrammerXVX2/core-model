# Copilot Instructions for `core-model`

## Project Context
- This repository hosts an Ollama-compatible FastAPI proxy for multiple OpenAI-compatible model backends.
- Runtime is Docker Compose based.
- Main entrypoint is `proxy/app.py`.

## Primary Goals
- Keep proxy behavior stable for existing clients (`/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/models`).
- Prefer additive changes over breaking changes.
- Keep model routing explicit and env-driven.

## Configuration Rules
- All runtime settings must come from environment variables (`.env` + `docker-compose.yml` passthrough).
- For every new env variable used in code:
  1. Add a sensible default in code.
  2. Add it to `.env`.
  3. Pass it through in `docker-compose.yml` for `ollama-proxy`.
- Do not hardcode server IPs in Python code.

## API and Compatibility Rules
- Preserve current request compatibility behavior (JSON, form-urlencoded raw JSON, prompt/text fallbacks).
- Keep embeddings response compatibility fields: `embedding` and `embeddings`.
- Keep strict model matching by configured aliases and model IDs.
- If upstream model is unavailable, return explicit HTTP errors (avoid raw traceback leaks).

## Availability and Routing Rules
- `/api/models` must reflect real availability and include route metadata.
- Do not duplicate model entries in `/api/models`.
- Additional model routes should be implemented in one place and reused consistently.

## Reliability Rules
- Network timeouts should fail fast and be configurable.
- Prefer bounded retries only for transient upstream 5xx cases.
- Avoid introducing blocking operations in request handlers.

## Code Organization Guidelines
- Prefer small, focused helper functions over large inlined blocks.
- Keep logging useful but concise; avoid logging secrets/tokens.
- Use ASCII-only edits unless file already requires Unicode.

## Validation Checklist (after changes)
Run from repo root:

```bash
python3 -m py_compile proxy/app.py
sudo docker compose up -d --build ollama-proxy
curl -sS http://127.0.0.1:11434/api/models
```

For routing changes, also smoke-test at least one chat model and one embedding model.

## Non-Goals
- Do not migrate to Kubernetes in routine fixes.
- Do not rewrite large sections unless required for a concrete bug/feature.
- Do not remove working backward-compat behavior without explicit request.
