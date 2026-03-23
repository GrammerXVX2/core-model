import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter

from schemas import ModelStatusItem
from services.status_cache import get_models_snapshot as _get_models_snapshot

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
    fmt = "gguf" if quantization else "unknown"
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
            "quantization_level": quantization,
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
        if str(item.get("status", "")).lower() == "доступен"
        and str(item.get("type", "")) in {"chat", "embeddings"}
    ]

    deduped: Dict[str, Dict[str, Any]] = {}
    for item in available:
        tag_item = _to_ollama_tag_item(item)
        deduped[tag_item["name"]] = tag_item

    return {"models": list(deduped.values())}
