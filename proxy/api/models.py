import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from constants import MODEL_STATUS_AVAILABLE

from schemas import (
    ModelRegistryCrudPayload,
    ModelRegistryCrudResponse,
    ModelRegistryItem,
    ModelRegistryUpsertRequest,
    ModelRegistryUpsertResponse,
    ModelStatusItem,
)
from services.model_registry import (
    create_registry_check as _create_registry_check,
    disable_registry_check_by_id as _disable_registry_check_by_id,
    get_registry_check_by_id as _get_registry_check_by_id,
    update_registry_check_by_id as _update_registry_check_by_id,
    upsert_registry_check as _upsert_registry_check,
)
from services.status_cache import (
    get_models_snapshot as _get_models_snapshot,
    refresh_model_status_cache as _refresh_model_status_cache,
)

router = APIRouter(tags=["models"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_quantization_level(model_name: str) -> str:
    upper = model_name.upper()
    if "Q4_K_M" in upper:
        return "Q4_K_M"
    if "Q6_K" in upper:
        return "Q6_K"
    if "Q8" in upper:
        return "Q8"
    return ""


def _guess_family(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen" in lower:
        return "qwen"
    if "ministral" in lower or "mistral" in lower:
        return "mistral"
    if "llama" in lower:
        return "llama"
    return "unknown"


def _guess_parameter_size(model_name: str) -> str:
    lower = model_name.lower().replace("_", "-")
    for token in lower.split("-"):
        if token.endswith("b") and token[:-1].isdigit():
            return token.upper()
    if "14b" in lower:
        return "14B"
    if "9b" in lower:
        return "9B"
    if "8b" in lower:
        return "8B"
    if "7b" in lower:
        return "7B"
    return ""


def _to_ollama_tag_item(item: Dict[str, Any]) -> Dict[str, Any]:
    public_model = str(item.get("model") or "").strip()
    backend_model = str(item.get("model_vllm") or public_model).strip()
    model_type = str(item.get("type") or "")
    model_name = public_model or backend_model
    family = _guess_family(backend_model or model_name)
    quantization = _extract_quantization_level(backend_model or model_name)
    modality = str(item.get("modality") or "llm")
    vision_supported = bool(item.get("vision_supported", False))
    is_gguf = bool(quantization)
    if modality == "vl":
        fmt = "vl"
    else:
        fmt = "gguf" if is_gguf else "vllm"
    digest = _stable_digest(f"{model_name}|{backend_model}|{model_type}")

    return {
        "name": model_name,
        "model": model_name,
        "modified_at": _now_iso(),
        "size": 0,
        "digest": digest,
        "details": {
            "parent_model": backend_model,
            "format": fmt,
            "family": family,
            "families": [family] if family != "unknown" else [],
            "parameter_size": _guess_parameter_size(backend_model or model_name),
            "quantization_level": quantization if is_gguf else "",
            "modality": modality,
            "vision_supported": vision_supported,
        },
    }


@router.get("/", tags=["chat"], summary="Proxy Status")
async def root_status() -> Dict[str, str]:
    return {
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@router.get(
    "/api/models",
    tags=["models"],
    summary="List Models and Availability",
)
async def api_models() -> Dict[str, List[ModelStatusItem]]:
    snapshot = await _get_models_snapshot()
    return {"models": snapshot}


@router.get(
    "/api/tags",
    tags=["models"],
    summary="List Models (Ollama Tags Format)",
)
async def api_tags() -> Dict[str, List[Dict[str, Any]]]:
    snapshot = await _get_models_snapshot()

    # Include currently available chat and embedding routes in Ollama-like tags.
    available = [
        item
        for item in snapshot
        if str(item.get("status", "")).lower() == MODEL_STATUS_AVAILABLE
        and str(item.get("type", "")) in {"chat", "embeddings"}
    ]

    deduped: Dict[str, Dict[str, Any]] = {}
    for item in available:
        tag_item = _to_ollama_tag_item(item)
        deduped[tag_item["name"]] = tag_item

    return {"models": list(deduped.values())}


def _validate_registry_payload(payload: ModelRegistryCrudPayload) -> None:
    if payload.default_max_tokens > payload.max_context_tokens:
        raise HTTPException(status_code=400, detail="default_max_tokens cannot exceed max_context_tokens")
    if payload.max_tokens_cap < payload.default_max_tokens:
        raise HTTPException(status_code=400, detail="max_tokens_cap cannot be smaller than default_max_tokens")


def _payload_to_check(payload: ModelRegistryCrudPayload) -> Dict[str, Any]:
    aliases = {payload.public_model.strip(), payload.vllm_model.strip()}
    aliases.update(a.strip() for a in (payload.aliases or []) if str(a).strip())

    return {
        "public_model": payload.public_model.strip(),
        "vllm_model": payload.vllm_model.strip(),
        "type": payload.model_type,
        "base_url": payload.base_url.strip().rstrip("/"),
        "max_context_tokens": int(payload.max_context_tokens),
        "default_max_tokens": int(payload.default_max_tokens),
        "max_tokens_cap": int(payload.max_tokens_cap),
        "min_context_headroom": int(payload.min_context_headroom),
        "stream_supported": bool(payload.stream_supported),
        "reasoning_supported": bool(payload.reasoning_supported),
        "aliases": aliases,
    }


def _row_to_crud_response(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": {
            "id": int(row.get("id") or 0),
            "public_model": str(row.get("public_model") or ""),
            "vllm_model": str(row.get("vllm_model") or ""),
            "model_type": str(row.get("model_type") or row.get("type") or "chat"),
            "base_url": str(row.get("base_url") or "").rstrip("/"),
            "max_context_tokens": int(row.get("max_context_tokens") or 0),
            "default_max_tokens": int(row.get("default_max_tokens") or 0),
            "max_tokens_cap": int(row.get("max_tokens_cap") or 0),
            "min_context_headroom": int(row.get("min_context_headroom") or 0),
            "stream_supported": bool(row.get("stream_supported") or False),
            "reasoning_supported": bool(row.get("reasoning_supported") or False),
            "aliases": list(row.get("aliases") or []),
            "is_enabled": bool(row.get("is_enabled") if row.get("is_enabled") is not None else True),
        },
    }


@router.post(
    "/api/models",
    tags=["models"],
    summary="Create Model Route",
    response_model=ModelRegistryCrudResponse,
)
async def api_create_model(payload: ModelRegistryCrudPayload) -> Dict[str, Any]:
    _validate_registry_payload(payload)
    try:
        row = await _create_registry_check(_payload_to_check(payload))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"registry create failed: {str(exc)}")

    asyncio.create_task(_refresh_model_status_cache())
    return _row_to_crud_response(row)


@router.get(
    "/api/models/{model_id}",
    tags=["models"],
    summary="Get Model Route By ID",
    response_model=ModelRegistryCrudResponse,
)
async def api_get_model(model_id: int) -> Dict[str, Any]:
    row = await _get_registry_check_by_id(int(model_id))
    if not row:
        raise HTTPException(status_code=404, detail=f"model not found: id={model_id}")
    return _row_to_crud_response(row)


@router.put(
    "/api/models/{model_id}",
    tags=["models"],
    summary="Update Model Route By ID",
    response_model=ModelRegistryCrudResponse,
)
async def api_update_model(model_id: int, payload: ModelRegistryCrudPayload) -> Dict[str, Any]:
    _validate_registry_payload(payload)
    try:
        row = await _update_registry_check_by_id(int(model_id), _payload_to_check(payload))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"registry update failed: {str(exc)}")
    if not row:
        raise HTTPException(status_code=404, detail=f"model not found: id={model_id}")

    asyncio.create_task(_refresh_model_status_cache())
    return _row_to_crud_response(row)


@router.delete(
    "/api/models/{model_id}",
    tags=["models"],
    summary="Disable Model Route By ID",
    response_model=ModelRegistryCrudResponse,
)
async def api_delete_model(model_id: int) -> Dict[str, Any]:
    try:
        row = await _disable_registry_check_by_id(int(model_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"registry delete failed: {str(exc)}")
    if not row:
        raise HTTPException(status_code=404, detail=f"model not found: id={model_id}")

    asyncio.create_task(_refresh_model_status_cache())
    return _row_to_crud_response(row)


@router.post(
    "/api/models/register",
    tags=["models"],
    summary="Register or Update Model Route",
    response_model=ModelRegistryUpsertResponse,
)
async def api_register_model(payload: ModelRegistryUpsertRequest) -> Dict[str, Any]:
    if payload.default_max_tokens > payload.max_context_tokens:
        raise HTTPException(status_code=400, detail="default_max_tokens cannot exceed max_context_tokens")
    if payload.max_tokens_cap < payload.default_max_tokens:
        raise HTTPException(status_code=400, detail="max_tokens_cap cannot be smaller than default_max_tokens")

    aliases = {payload.public_model.strip(), payload.vllm_model.strip()}
    aliases.update(a.strip() for a in (payload.aliases or []) if str(a).strip())

    check = {
        "public_model": payload.public_model.strip(),
        "vllm_model": payload.vllm_model.strip(),
        "type": payload.model_type,
        "base_url": payload.base_url.strip().rstrip("/"),
        "max_context_tokens": int(payload.max_context_tokens),
        "default_max_tokens": int(payload.default_max_tokens),
        "max_tokens_cap": int(payload.max_tokens_cap),
        "min_context_headroom": int(payload.min_context_headroom),
        "stream_supported": bool(payload.stream_supported),
        "reasoning_supported": bool(payload.reasoning_supported),
        "aliases": aliases,
    }

    try:
        row = await _upsert_registry_check(check)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"registry upsert failed: {str(exc)}")

    asyncio.create_task(_refresh_model_status_cache())

    return {
        "status": "ok",
        "model": {
            "id": int(row.get("id") or 0),
            "model": row["public_model"],
            "model_vllm": row["vllm_model"],
            "type": row["type"],
            "base_url": row["base_url"],
            "max_context_tokens": row["max_context_tokens"],
            "status": "registered",
            "detail": "model saved to registry; status refresh started",
        },
        "aliases": row["aliases"],
    }
