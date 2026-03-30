import asyncio
import logging
from typing import Any, Dict, List, Union

from fastapi import HTTPException
from constants import (
    EMBEDDING_PATH_CANDIDATES,
    ERR_MODEL_DOES_NOT_SUPPORT_ENDPOINT_FMT,
    ERR_MODEL_UNAVAILABLE_FMT,
    ERR_NO_MODELS_REGISTERED,
    ERR_UNSUPPORTED_MODEL_FOR_ENDPOINT_FMT,
    LOG_MODELS_POLLER_ERROR,
    MODEL_STATUS_AVAILABLE,
    MODEL_STATUS_POLLING_DETAIL,
    MODEL_STATUS_UNAVAILABLE,
    MODEL_STATUS_WARMING,
)

from settings import (
    DEFAULT_MAX_TOKENS,
    MAX_TOKENS_CAP,
    MIN_CONTEXT_HEADROOM,
    MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS,
    MODEL_STATUS_POLL_INTERVAL_SECONDS,
)
from services.metrics import set_model_availability
from services.model_registry import get_registry_checks as _get_registry_checks
from services.upstream import get_http_client as _get_http_client


logger = logging.getLogger("uvicorn.error")

MODEL_STATUS_CACHE: Dict[str, Dict[str, Any]] = {}
MODEL_STATUS_LIST_CACHE: List[Dict[str, Any]] = []
MODEL_STATUS_CACHE_LOCK = asyncio.Lock()
MODEL_STATUS_POLLER_TASK: asyncio.Task | None = None


def _normalize_model_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _endpoint_name(expected_type: str) -> str:
    if expected_type == "chat":
        return "chat"
    if expected_type == "embeddings":
        return "embeddings"
    return expected_type or "model"


def _is_vision_capable(check: Dict[str, Any]) -> bool:
    aliases = {_normalize_model_name(str(a or "")) for a in (check.get("aliases") or set())}
    if "vl" in aliases or "vision" in aliases or "multimodal" in aliases:
        return True
    public_model = _normalize_model_name(str(check.get("public_model") or ""))
    backend_model = _normalize_model_name(str(check.get("vllm_model") or ""))
    # Backward-compatible fallback: legacy 122B route is considered vision-capable.
    return "122b" in public_model or "122b" in backend_model


def _warm_item_from_check(check: Dict[str, Any]) -> Dict[str, Any]:
    vision_supported = _is_vision_capable(check)
    return {
        "id": int(check.get("id") or 0),
        "model": str(check.get("public_model") or ""),
        "model_vllm": str(check.get("vllm_model") or ""),
        "type": str(check.get("type") or ""),
        "modality": "vl" if vision_supported else "llm",
        "vision_supported": vision_supported,
        "base_url": str(check.get("base_url") or "").rstrip("/"),
        "max_context_tokens": int(check.get("max_context_tokens") or 0),
        "default_max_tokens": int(check.get("default_max_tokens") or DEFAULT_MAX_TOKENS),
        "max_tokens_cap": int(check.get("max_tokens_cap") or MAX_TOKENS_CAP),
        "min_context_headroom": int(check.get("min_context_headroom") or MIN_CONTEXT_HEADROOM),
        "stream_supported": bool(check.get("stream_supported", False)),
        "reasoning_supported": bool(check.get("reasoning_supported", False)),
        "status": MODEL_STATUS_WARMING,
        "detail": MODEL_STATUS_POLLING_DETAIL,
    }


async def _probe_model_status(
    model_id: int,
    public_model: str,
    vllm_model: str,
    model_type: str,
    base_url: str,
    max_context_tokens: int,
    default_max_tokens: int,
    max_tokens_cap: int,
    min_context_headroom: int,
    stream_supported: bool,
    reasoning_supported: bool,
    vision_supported: bool,
) -> Dict[str, Union[str, int]]:
    async def _probe_via_models_endpoint() -> tuple[bool, str]:
        url = f"{base_url.rstrip('/')}/models"
        try:
            client = await _get_http_client()
            resp = await client.get(url)
        except Exception as exc:
            return False, f"connection error: {str(exc)}"

        if resp.status_code >= 400:
            return False, f"http {resp.status_code}"

        try:
            data = resp.json()
        except Exception:
            return False, "invalid json from /models"

        models = data.get("data") if isinstance(data, dict) else []
        ids = [str(item.get("id", "")) for item in models if isinstance(item, dict)]
        target = _normalize_model_name(vllm_model)
        found = any(_normalize_model_name(mid) == target for mid in ids)
        if found:
            return True, ""
        preview = ", ".join(ids[:5])
        return False, f"model id not found in /models; seen: {preview}"

    async def _probe_via_json_post(paths: List[str], payloads: List[Dict[str, Any]], kind: str) -> tuple[bool, str]:
        last_detail = ""
        client = await _get_http_client()
        for path in paths:
            for payload in payloads:
                url = f"{base_url.rstrip('/')}{path}"
                try:
                    resp = await client.post(url, json=payload)
                except Exception as exc:
                    last_detail = f"{kind} probe connection error on {path}: {str(exc)}"
                    continue

                if resp.status_code in (401, 403, 422):
                    # Endpoint exists; payload/auth mismatch is enough for liveness probe.
                    return True, ""

                if resp.status_code >= 400:
                    last_detail = f"{kind} probe http {resp.status_code} on {path}"
                    continue

                try:
                    data = resp.json()
                except Exception:
                    last_detail = f"{kind} probe invalid json on {path}"
                    continue

                if isinstance(data, dict):
                    if kind == "embeddings" and (
                        isinstance(data.get("data"), list)
                        or isinstance(data.get("embeddings"), list)
                        or isinstance(data.get("embedding"), list)
                    ):
                        return True, ""
                elif kind == "embeddings" and isinstance(data, list):
                    return True, ""

                last_detail = f"unexpected {kind} response shape on {path}"

        return False, last_detail or f"{kind} probe failed"

    is_available = False
    detail = ""
    normalized_type = str(model_type or "").lower()
    if normalized_type == "chat":
        is_available, detail = await _probe_via_models_endpoint()
    elif normalized_type == "embeddings":
        is_available, detail = await _probe_via_models_endpoint()
        if not is_available:
            is_available, detail = await _probe_via_json_post(
                paths=list(EMBEDDING_PATH_CANDIDATES),
                payloads=[
                    {"model": vllm_model, "input": "healthcheck"},
                    {"inputs": "healthcheck"},
                ],
                kind="embeddings",
            )
    else:
        detail = f"unsupported model type: {model_type}"

    if is_available:
        return {
            "id": int(model_id),
            "model": public_model,
            "model_vllm": vllm_model,
            "type": model_type,
            "modality": "vl" if vision_supported else "llm",
            "vision_supported": vision_supported,
            "base_url": base_url,
            "max_context_tokens": max_context_tokens,
            "default_max_tokens": default_max_tokens,
            "max_tokens_cap": max_tokens_cap,
            "min_context_headroom": min_context_headroom,
            "stream_supported": stream_supported,
            "reasoning_supported": reasoning_supported,
            "status": MODEL_STATUS_AVAILABLE,
            "detail": "",
        }

    return {
        "id": int(model_id),
        "model": public_model,
        "model_vllm": vllm_model,
        "type": model_type,
        "modality": "vl" if vision_supported else "llm",
        "vision_supported": vision_supported,
        "base_url": base_url,
        "max_context_tokens": max_context_tokens,
        "default_max_tokens": default_max_tokens,
        "max_tokens_cap": max_tokens_cap,
        "min_context_headroom": min_context_headroom,
        "stream_supported": stream_supported,
        "reasoning_supported": reasoning_supported,
        "status": MODEL_STATUS_UNAVAILABLE,
        "detail": detail,
    }


async def _build_model_checks() -> List[Dict[str, Any]]:
    checks = await _get_registry_checks()
    if checks:
        return checks
    return []


async def refresh_model_status_cache() -> None:
    checks = await _build_model_checks()
    if not checks:
        async with MODEL_STATUS_CACHE_LOCK:
            MODEL_STATUS_CACHE.clear()
            MODEL_STATUS_LIST_CACHE.clear()
        return
    results = await asyncio.gather(
        *[
            _probe_model_status(
                int(check.get("id") or 0),
                str(check["public_model"]),
                str(check["vllm_model"]),
                str(check["type"]),
                str(check["base_url"]),
                int(check["max_context_tokens"]),
                int(check.get("default_max_tokens") or DEFAULT_MAX_TOKENS),
                int(check.get("max_tokens_cap") or MAX_TOKENS_CAP),
                int(check.get("min_context_headroom") or MIN_CONTEXT_HEADROOM),
                bool(check.get("stream_supported", False)),
                bool(check.get("reasoning_supported", False)),
                bool(_is_vision_capable(check)),
            )
            for check in checks
        ]
    )

    cache: Dict[str, Dict[str, Any]] = {}
    for item, check in zip(results, checks):
        set_model_availability(
            str(item.get("model", "")),
            str(item.get("type", "")),
            str(item.get("base_url", "")),
            str(item.get("status", "")),
        )
        model = str(item.get("model", ""))
        model_vllm = str(item.get("model_vllm", ""))
        if model:
            cache[model] = item
        if model_vllm:
            cache[model_vllm] = item
        aliases = check.get("aliases") or set()
        for alias in aliases:
            alias_name = str(alias or "").strip()
            if alias_name:
                cache[alias_name] = item

    async with MODEL_STATUS_CACHE_LOCK:
        MODEL_STATUS_CACHE.clear()
        MODEL_STATUS_CACHE.update(cache)
        MODEL_STATUS_LIST_CACHE.clear()
        MODEL_STATUS_LIST_CACHE.extend(results)


async def _model_status_poller() -> None:
    while True:
        sleep_for = MODEL_STATUS_POLL_INTERVAL_SECONDS
        try:
            await refresh_model_status_cache()
            async with MODEL_STATUS_CACHE_LOCK:
                has_unavailable = any(item.get("status") != MODEL_STATUS_AVAILABLE for item in MODEL_STATUS_LIST_CACHE)
            if has_unavailable:
                sleep_for = MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS
        except Exception as exc:
            logger.warning(LOG_MODELS_POLLER_ERROR, str(exc))
            sleep_for = MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS
        await asyncio.sleep(max(1, sleep_for))


async def ensure_model_available(target: Dict[str, str]) -> None:
    async with MODEL_STATUS_CACHE_LOCK:
        status = MODEL_STATUS_CACHE.get(target.get("public_model", "")) or MODEL_STATUS_CACHE.get(target.get("vllm_model", ""))

    if not status:
        await refresh_model_status_cache()
        async with MODEL_STATUS_CACHE_LOCK:
            status = MODEL_STATUS_CACHE.get(target.get("public_model", "")) or MODEL_STATUS_CACHE.get(target.get("vllm_model", ""))

    if status and status.get("status") != MODEL_STATUS_AVAILABLE:
        detail = status.get("detail", "")
        raise HTTPException(
            status_code=503,
            detail=ERR_MODEL_UNAVAILABLE_FMT.format(
                model=status.get("model"),
                base_url=status.get("base_url"),
                detail=detail,
            ).strip(),
        )


async def get_models_snapshot() -> List[Dict[str, Any]]:
    async with MODEL_STATUS_CACHE_LOCK:
        cached = [dict(item) for item in MODEL_STATUS_LIST_CACHE]
    if cached:
        return cached

    checks = await _build_model_checks()
    if not checks:
        return []
    return [_warm_item_from_check(check) for check in checks]


async def resolve_target_from_status_cache(requested_model: str | None, expected_type: str) -> Dict[str, Any] | None:
    requested = (requested_model or "").strip()
    if not requested:
        return None

    async with MODEL_STATUS_CACHE_LOCK:
        status = MODEL_STATUS_CACHE.get(requested)

    if status:
        actual_type = str(status.get("type", ""))
        public_model = str(status.get("model", requested))
        if actual_type != expected_type:
            endpoint = _endpoint_name(expected_type)
            raise HTTPException(
                status_code=400,
                detail=ERR_MODEL_DOES_NOT_SUPPORT_ENDPOINT_FMT.format(endpoint=endpoint, model=public_model),
            )

        return {
            "public_model": public_model,
            "vllm_model": str(status.get("model_vllm", public_model)),
            "base_url": str(status.get("base_url", "")).rstrip("/"),
            "type": actual_type,
            "modality": "vl" if bool(status.get("vision_supported", False)) else "llm",
            "vision_supported": bool(status.get("vision_supported", False)),
            "max_context_tokens": int(status.get("max_context_tokens") or 0),
            "default_max_tokens": int(status.get("default_max_tokens") or DEFAULT_MAX_TOKENS),
            "max_tokens_cap": int(status.get("max_tokens_cap") or MAX_TOKENS_CAP),
            "min_context_headroom": int(status.get("min_context_headroom") or MIN_CONTEXT_HEADROOM),
            "stream_supported": bool(status.get("stream_supported", False)),
            "reasoning_supported": bool(status.get("reasoning_supported", False)),
        }

    checks = await _build_model_checks()
    for check in checks:
        aliases = {str(check.get("public_model") or ""), str(check.get("vllm_model") or "")}
        aliases.update(str(a or "").strip() for a in (check.get("aliases") or set()))
        if requested not in aliases:
            continue

        actual_type = str(check.get("type") or "")
        public_model = str(check.get("public_model") or requested)
        if actual_type != expected_type:
            endpoint = _endpoint_name(expected_type)
            raise HTTPException(
                status_code=400,
                detail=ERR_MODEL_DOES_NOT_SUPPORT_ENDPOINT_FMT.format(endpoint=endpoint, model=public_model),
            )

        return {
            "public_model": public_model,
            "vllm_model": str(check.get("vllm_model") or public_model),
            "base_url": str(check.get("base_url") or "").rstrip("/"),
            "type": actual_type,
            "modality": "vl" if _is_vision_capable(check) else "llm",
            "vision_supported": _is_vision_capable(check),
            "max_context_tokens": int(check.get("max_context_tokens") or 0),
            "default_max_tokens": int(check.get("default_max_tokens") or DEFAULT_MAX_TOKENS),
            "max_tokens_cap": int(check.get("max_tokens_cap") or MAX_TOKENS_CAP),
            "min_context_headroom": int(check.get("min_context_headroom") or MIN_CONTEXT_HEADROOM),
            "stream_supported": bool(check.get("stream_supported", False)),
            "reasoning_supported": bool(check.get("reasoning_supported", False)),
        }

    async with MODEL_STATUS_CACHE_LOCK:
        all_items = [dict(item) for item in MODEL_STATUS_LIST_CACHE]
    if not all_items:
        if not checks:
            raise HTTPException(status_code=503, detail=ERR_NO_MODELS_REGISTERED)
        allowed = [str(item.get("public_model", "")) for item in checks if str(item.get("type", "")) == expected_type]
        endpoint = _endpoint_name(expected_type)
        raise HTTPException(
            status_code=400,
            detail=ERR_UNSUPPORTED_MODEL_FOR_ENDPOINT_FMT.format(endpoint=endpoint, allowed=", ".join(allowed)),
        )

    allowed = [str(item.get("model", "")) for item in all_items if str(item.get("type", "")) == expected_type]
    endpoint = _endpoint_name(expected_type)
    raise HTTPException(
        status_code=400,
        detail=ERR_UNSUPPORTED_MODEL_FOR_ENDPOINT_FMT.format(endpoint=endpoint, allowed=", ".join(allowed)),
    )


async def startup_status_poller() -> None:
    global MODEL_STATUS_POLLER_TASK
    MODEL_STATUS_POLLER_TASK = asyncio.create_task(_model_status_poller())


async def shutdown_status_poller() -> None:
    global MODEL_STATUS_POLLER_TASK
    if MODEL_STATUS_POLLER_TASK is None:
        return
    MODEL_STATUS_POLLER_TASK.cancel()
    try:
        await MODEL_STATUS_POLLER_TASK
    except asyncio.CancelledError:
        pass
    MODEL_STATUS_POLLER_TASK = None
