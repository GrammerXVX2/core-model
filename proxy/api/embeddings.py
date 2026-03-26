import json
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from api.common import ns, safe_preview
from schemas import EmbedResponseModel
from settings import EMBED_DEBUG_LOG
from services.request_parser import read_request_body_as_dict as _read_request_body_as_dict
from services.status_cache import ensure_model_available as _ensure_model_available
from services.status_cache import resolve_target_from_status_cache as _resolve_target_from_status_cache
from services.upstream import post_json_to as _post_json_to

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["embeddings"])

EMBED_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "example": {
                    "model": os.getenv("OPENAPI_EMBED_EXAMPLE_MODEL", "Qwen3-Embedding-8B"),
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
        logger.info("embed.incoming body=%s", safe_preview(body_data))

    requested_model = body_data.get("model")
    target = await _resolve_target_from_status_cache(requested_model, expected_type="embeddings")
    if target is None:
        raise HTTPException(status_code=503, detail="no embedding models registered in status cache")
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
    payload = {
        "model": target["vllm_model"],
        "input": input_data,
    }

    if EMBED_DEBUG_LOG:
        logger.info(
            "embed.adapted model=%s route_model=%s base_url=%s input_type=%s input_preview=%s",
            model,
            target["vllm_model"],
            target["base_url"],
            type(input_data).__name__,
            safe_preview(input_data),
        )

    try:
        data = await _post_json_to(target["base_url"], "/embeddings", payload)
    except HTTPException as exc:
        if EMBED_DEBUG_LOG:
            logger.error(
                "embed.vllm_error status=%s detail=%s",
                exc.status_code,
                safe_preview(exc.detail),
            )
        raise

    embeddings = [item.get("embedding", []) for item in data.get("data", [])]
    usage = data.get("usage") or {}

    if EMBED_DEBUG_LOG:
        logger.info(
            "embed.vllm_response vectors=%s first_dim=%s prompt_tokens=%s",
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
        raise HTTPException(status_code=503, detail="no embedding models registered in status cache")
    await _ensure_model_available(target)

    probe_input = body_data.get("input")
    if probe_input is None:
        probe_input = "test"

    payload = {
        "model": target["vllm_model"],
        "input": probe_input,
    }
    data = await _post_json_to(target["base_url"], "/embeddings", payload)

    vectors = [item.get("embedding", []) for item in data.get("data", [])]
    vector_size = len(vectors[0]) if vectors else 0

    return {
        "model": requested_model or target["public_model"],
        "model_vllm": target["vllm_model"],
        "base_url": target["base_url"],
        "vector_size": vector_size,
        "vectors_count": len(vectors),
    }
