import json
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from constants import (
    EMBEDDING_PATH_CANDIDATES,
    ERR_NO_EMBEDDING_MODELS,
    LOG_EMBED_ADAPTED,
    LOG_EMBED_INCOMING,
    LOG_EMBED_VLLM_ERROR,
    LOG_EMBED_VLLM_RESPONSE,
)

from api.common import ns, safe_preview
from schemas import EmbedResponseModel
from settings import EMBED_DEBUG_LOG
from services.request_parser import read_request_body_as_dict as _read_request_body_as_dict
from services.status_cache import ensure_model_available as _ensure_model_available
from services.status_cache import resolve_target_from_status_cache as _resolve_target_from_status_cache
from services.upstream import post_json_to as _post_json_to

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["embeddings"])


async def _post_embeddings_with_fallback(base_url: str, model_id: str, input_data: Any) -> Dict[str, Any]:
    attempts = [
        (EMBEDDING_PATH_CANDIDATES[0], {"model": model_id, "input": input_data}),
        (EMBEDDING_PATH_CANDIDATES[2], {"model": model_id, "input": input_data}),
        (EMBEDDING_PATH_CANDIDATES[1], {"inputs": input_data}),
        (EMBEDDING_PATH_CANDIDATES[3], {"inputs": input_data}),
    ]

    last_exc: HTTPException | None = None
    for path, payload in attempts:
        try:
            data = await _post_json_to(base_url, path, payload)
            if isinstance(data, dict):
                return data
        except HTTPException as exc:
            last_exc = exc
            if exc.status_code in (404, 405, 422):
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise HTTPException(status_code=502, detail="embedding upstream is unavailable")


def _extract_embeddings(data: Dict[str, Any]) -> List[List[float]]:
    # OpenAI-style response.
    if isinstance(data.get("data"), list):
        return [item.get("embedding", []) for item in data.get("data", []) if isinstance(item, dict)]

    # TEI native response can return `embeddings` or a single vector list.
    if isinstance(data.get("embeddings"), list):
        emb = data.get("embeddings")
        if emb and isinstance(emb[0], list):
            return emb
    if isinstance(data.get("embedding"), list):
        emb = data.get("embedding")
        if emb and isinstance(emb[0], (int, float)):
            return [emb]

    # Some TEI deployments return raw vector (single input).
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [data]
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data

    return []

EMBED_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "example": {
                    "model": os.getenv("OPENAPI_EMBED_EXAMPLE_MODEL", "qwen-embed-4b-tei"),
                    "input": json.loads(
                        os.getenv(
                            "OPENAPI_EMBED_EXAMPLE_INPUT_JSON",
                            '["Как настроить OAuth2 авторизацию в FastAPI?", "Как сбросить пароль через форму восстановления аккаунта.", "Пошаговая настройка OAuth2 в FastAPI: /token, bearer scheme, проверка access token."]',
                        )
                    ),
                },
            }
        },
    }
}


@router.post(
    "/api/embed",
    tags=["embeddings"],
    summary="Embed Text",
    response_model=EmbedResponseModel,
    openapi_extra=EMBED_OPENAPI_EXTRA,
)
async def api_embed(request: Request) -> Dict[str, Any]:
    body_data = await _read_request_body_as_dict(request)
    if EMBED_DEBUG_LOG:
        logger.info(LOG_EMBED_INCOMING, safe_preview(body_data))

    requested_model = body_data.get("model")
    target = await _resolve_target_from_status_cache(requested_model, expected_type="embeddings")
    if target is None:
        raise HTTPException(status_code=503, detail=ERR_NO_EMBEDDING_MODELS)
    await _ensure_model_available(target)
    model = requested_model or target["public_model"]
    input_data = body_data.get("input")

    if input_data is None:
        input_data = body_data.get("prompt") or body_data.get("text")

    if input_data is None and isinstance(body_data.get("message"), dict):
        msg = body_data.get("message") or {}
        input_data = msg.get("content")

    if input_data is None and isinstance(body_data.get("messages"), list):
        merged = []
        for msg in body_data.get("messages") or []:
            if isinstance(msg, dict) and msg.get("content") is not None:
                merged.append(str(msg.get("content")))
        if merged:
            input_data = "\n".join(merged)

    if input_data is None:
        raise HTTPException(status_code=400, detail="input is required")

    start_ns = ns()
    if EMBED_DEBUG_LOG:
        logger.info(
            LOG_EMBED_ADAPTED,
            model,
            target["vllm_model"],
            target["base_url"],
            type(input_data).__name__,
            safe_preview(input_data),
        )

    try:
        data = await _post_embeddings_with_fallback(target["base_url"], target["vllm_model"], input_data)
    except HTTPException as exc:
        if EMBED_DEBUG_LOG:
            logger.error(
                LOG_EMBED_VLLM_ERROR,
                exc.status_code,
                safe_preview(exc.detail),
            )
        raise

    embeddings = _extract_embeddings(data)
    usage = data.get("usage") or {}

    if EMBED_DEBUG_LOG:
        logger.info(
            LOG_EMBED_VLLM_RESPONSE,
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
            usage.get("prompt_tokens", 0),
        )

    return {
        "model": model,
        "embedding": embeddings[0] if embeddings else [],
        "embeddings": embeddings,
        "total_duration": max(0, ns() - start_ns),
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }


@router.post(
    "/api/dev/embeddings/info",
    tags=["embeddings"],
    summary="Dev: Embedding Model Vector Size",
)
async def api_dev_embeddings_info(request: Request) -> Dict[str, Any]:
    body_data = await _read_request_body_as_dict(request)
    requested_model = body_data.get("model")
    target = await _resolve_target_from_status_cache(requested_model, expected_type="embeddings")
    if target is None:
        raise HTTPException(status_code=503, detail=ERR_NO_EMBEDDING_MODELS)
    await _ensure_model_available(target)

    probe_input = body_data.get("input")
    if probe_input is None:
        probe_input = "test"

    data = await _post_embeddings_with_fallback(target["base_url"], target["vllm_model"], probe_input)

    vectors = _extract_embeddings(data)
    vector_size = len(vectors[0]) if vectors else 0

    return {
        "model": requested_model or target["public_model"],
        "model_vllm": target["vllm_model"],
        "base_url": target["base_url"],
        "vector_size": vector_size,
        "vectors_count": len(vectors),
    }
