import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response

from api.chat import router as chat_router
from api.embeddings import router as embeddings_router
from api.models import router as models_router
from services.metrics import export_metrics, now_seconds, observe_request_latency
from services.model_registry import (
    shutdown_model_registry as _shutdown_model_registry,
    startup_model_registry as _startup_model_registry,
)
from services.status_cache import (
    shutdown_status_poller as _status_shutdown_poller,
    startup_status_poller as _status_startup_poller,
)
from services.upstream import (
    shutdown_http_client as _shutdown_http_client,
    startup_http_client as _startup_http_client,
)

app = FastAPI(
    title="Ollama-Compatible Proxy for vLLM",
    description=(
        "Proxy layer that exposes Ollama-style endpoints and forwards requests "
        "to vLLM/TEI OpenAI-compatible APIs. "
        "Model routing is DB-first via model_registry_checks; use /api/models CRUD "
        "to add, update, disable, and tune models without redeploy."
    ),
    version="1.0.0",
    openapi_tags=[
        {
            "name": "chat",
            "description": "Chat completion endpoints. Supports Ollama-style messages and optional images for VL models.",
        },
        {
            "name": "generate",
            "description": "Prompt-style text generation endpoint (/api/generate).",
        },
        {
            "name": "embeddings",
            "description": "Embedding endpoints with OpenAI- and TEI-compatible upstream fallback.",
        },
        {
            "name": "models",
            "description": "Model registry CRUD and runtime availability snapshot. Includes modality and vision support markers.",
        },
    ],
)

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(embeddings_router)


@app.middleware("http")
async def _request_metrics_middleware(request: Request, call_next):
    started_at = now_seconds()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        observe_request_latency(request.url.path, request.method, status_code, started_at)


@app.get("/metrics", tags=["models"], summary="Prometheus Metrics")
async def metrics() -> Response:
    payload, content_type = export_metrics()
    return Response(content=payload, media_type=content_type)


@app.on_event("startup")
async def _startup_model_poller() -> None:
    await _startup_http_client()
    await _startup_model_registry()
    await _status_startup_poller()


@app.on_event("shutdown")
async def _shutdown_model_poller() -> None:
    await _status_shutdown_poller()
    await _shutdown_model_registry()
    await _shutdown_http_client()
