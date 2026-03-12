import logging

from fastapi import FastAPI

from api.chat import router as chat_router
from api.embeddings import router as embeddings_router
from api.models import router as models_router
from services.status_cache import (
    shutdown_status_poller as _status_shutdown_poller,
    startup_status_poller as _status_startup_poller,
)

app = FastAPI(
    title="Ollama-Compatible Proxy for vLLM",
    description=(
        "Proxy layer that exposes Ollama-style endpoints and forwards requests "
        "to vLLM OpenAI-compatible APIs."
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "chat", "description": "Chat-style text generation."},
        {"name": "generate", "description": "Prompt-style text generation."},
        {"name": "embeddings", "description": "Text embedding endpoints."},
        {"name": "models", "description": "Model registry and availability status."},
    ],
)

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(embeddings_router)


@app.on_event("startup")
async def _startup_model_poller() -> None:
    await _status_startup_poller()


@app.on_event("shutdown")
async def _shutdown_model_poller() -> None:
    await _status_shutdown_poller()
