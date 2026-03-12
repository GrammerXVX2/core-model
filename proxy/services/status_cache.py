import asyncio
import logging
from typing import Any, Dict, List, Union

from fastapi import HTTPException

from settings import (
    CPU_CHAT_Q4_MAX_CONTEXT_TOKENS,
    CPU_CHAT_Q4_MODEL,
    CPU_CHAT_Q6_MAX_CONTEXT_TOKENS,
    MINISTRAL_CHAT_MAX_CONTEXT_TOKENS,
    MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS,
    MODEL_STATUS_POLL_INTERVAL_SECONDS,
    PUBLIC_MINISTRAL_CHAT_MODEL,
    PUBLIC_QWEN_CHAT_MODEL,
    PUBLIC_QWEN_EMBED_MODEL,
    QWEN_CHAT_MAX_CONTEXT_TOKENS,
    QWEN_CHAT_MODEL,
    QWEN_EMBED_4B_MAX_CONTEXT_TOKENS,
    QWEN_EMBED_4B_MODEL,
    QWEN_EMBED_8B_MAX_CONTEXT_TOKENS,
    QWEN_EMBED_8B_MODEL,
    QWEN_EMBED_MAX_CONTEXT_TOKENS,
    QWEN_EMBED_MODEL,
    MINISTRAL_CHAT_MODEL,
    MINISTRAL_CHAT_BASE_URL,
    QWEN_CHAT_BASE_URL,
    QWEN_EMBED_BASE_URL,
)
from services.metrics import set_model_availability
from services.routing import _additional_chat_routes, _embed_routes, _normalize_model_name
from services.upstream import get_http_client as _get_http_client


logger = logging.getLogger("uvicorn.error")

MODEL_STATUS_CACHE: Dict[str, Dict[str, Any]] = {}
MODEL_STATUS_LIST_CACHE: List[Dict[str, Any]] = []
MODEL_STATUS_CACHE_LOCK = asyncio.Lock()
MODEL_STATUS_POLLER_TASK: asyncio.Task | None = None


async def _probe_model_status(
    public_model: str,
    vllm_model: str,
    model_type: str,
    base_url: str,
    max_context_tokens: int,
) -> Dict[str, Union[str, int]]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        client = await _get_http_client()
        resp = await client.get(url)
    except Exception as exc:
        return {
            "model": public_model,
            "model_vllm": vllm_model,
            "type": model_type,
            "base_url": base_url,
            "max_context_tokens": max_context_tokens,
            "status": "недоступен",
            "detail": f"connection error: {str(exc)}",
        }

    if resp.status_code >= 400:
        return {
            "model": public_model,
            "model_vllm": vllm_model,
            "type": model_type,
            "base_url": base_url,
            "max_context_tokens": max_context_tokens,
            "status": "недоступен",
            "detail": f"http {resp.status_code}",
        }

    try:
        data = resp.json()
    except Exception:
        return {
            "model": public_model,
            "model_vllm": vllm_model,
            "type": model_type,
            "base_url": base_url,
            "max_context_tokens": max_context_tokens,
            "status": "недоступен",
            "detail": "invalid json from /models",
        }

    models = data.get("data") if isinstance(data, dict) else []
    ids = [str(item.get("id", "")) for item in models if isinstance(item, dict)]
    target = _normalize_model_name(vllm_model)
    found = any(_normalize_model_name(mid) == target for mid in ids)
    if found:
        return {
            "model": public_model,
            "model_vllm": vllm_model,
            "type": model_type,
            "base_url": base_url,
            "max_context_tokens": max_context_tokens,
            "status": "доступен",
            "detail": "",
        }

    preview = ", ".join(ids[:5])
    return {
        "model": public_model,
        "model_vllm": vllm_model,
        "type": model_type,
        "base_url": base_url,
        "max_context_tokens": max_context_tokens,
        "status": "недоступен",
        "detail": f"model id not found in /models; seen: {preview}",
    }


def _build_model_checks() -> List[Dict[str, Any]]:
    checks = [
        {
            "public_model": PUBLIC_QWEN_CHAT_MODEL,
            "vllm_model": QWEN_CHAT_MODEL,
            "type": "chat",
            "base_url": QWEN_CHAT_BASE_URL,
            "max_context_tokens": QWEN_CHAT_MAX_CONTEXT_TOKENS,
            "aliases": {PUBLIC_QWEN_CHAT_MODEL, QWEN_CHAT_MODEL},
        },
        {
            "public_model": PUBLIC_QWEN_EMBED_MODEL,
            "vllm_model": QWEN_EMBED_MODEL,
            "type": "embeddings",
            "base_url": QWEN_EMBED_BASE_URL,
            "max_context_tokens": QWEN_EMBED_MAX_CONTEXT_TOKENS,
            "aliases": {PUBLIC_QWEN_EMBED_MODEL, QWEN_EMBED_MODEL},
        },
        {
            "public_model": PUBLIC_MINISTRAL_CHAT_MODEL,
            "vllm_model": MINISTRAL_CHAT_MODEL,
            "type": "chat",
            "base_url": MINISTRAL_CHAT_BASE_URL,
            "max_context_tokens": MINISTRAL_CHAT_MAX_CONTEXT_TOKENS,
            "aliases": {PUBLIC_MINISTRAL_CHAT_MODEL, MINISTRAL_CHAT_MODEL},
        },
    ]
    for route in _embed_routes()[1:]:
        if route["vllm_model"] == QWEN_EMBED_8B_MODEL:
            max_ctx = QWEN_EMBED_8B_MAX_CONTEXT_TOKENS
        elif route["vllm_model"] == QWEN_EMBED_4B_MODEL:
            max_ctx = QWEN_EMBED_4B_MAX_CONTEXT_TOKENS
        else:
            max_ctx = QWEN_EMBED_MAX_CONTEXT_TOKENS

        checks.append(
            {
                "public_model": route["public_model"],
                "vllm_model": route["vllm_model"],
                "type": "embeddings",
                "base_url": route["base_url"],
                "max_context_tokens": max_ctx,
                "aliases": {route["public_model"], route["vllm_model"]},
            }
        )
    for route in _additional_chat_routes():
        max_ctx = CPU_CHAT_Q4_MAX_CONTEXT_TOKENS if route["vllm_model"] == CPU_CHAT_Q4_MODEL else CPU_CHAT_Q6_MAX_CONTEXT_TOKENS
        checks.append(
            {
                "public_model": route["public_model"],
                "vllm_model": route["vllm_model"],
                "type": "chat",
                "base_url": route["base_url"],
                "max_context_tokens": max_ctx,
                "aliases": {route["public_model"], route["vllm_model"]},
            }
        )

    deduped: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for check in checks:
        key = (
            _normalize_model_name(str(check["vllm_model"])),
            str(check["base_url"]).rstrip("/"),
            str(check["type"]),
        )
        if key not in deduped:
            deduped[key] = check
            continue
        deduped[key]["aliases"].update(check["aliases"])

    return list(deduped.values())


async def refresh_model_status_cache() -> None:
    checks = _build_model_checks()
    results = await asyncio.gather(
        *[
            _probe_model_status(
                str(check["public_model"]),
                str(check["vllm_model"]),
                str(check["type"]),
                str(check["base_url"]),
                int(check["max_context_tokens"]),
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
                has_unavailable = any(item.get("status") != "доступен" for item in MODEL_STATUS_LIST_CACHE)
            if has_unavailable:
                sleep_for = MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS
        except Exception as exc:
            logger.warning("models.poller.error=%s", str(exc))
            sleep_for = MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS
        await asyncio.sleep(max(1, sleep_for))


async def ensure_model_available(target: Dict[str, str]) -> None:
    async with MODEL_STATUS_CACHE_LOCK:
        status = MODEL_STATUS_CACHE.get(target.get("public_model", "")) or MODEL_STATUS_CACHE.get(target.get("vllm_model", ""))

    if not status:
        await refresh_model_status_cache()
        async with MODEL_STATUS_CACHE_LOCK:
            status = MODEL_STATUS_CACHE.get(target.get("public_model", "")) or MODEL_STATUS_CACHE.get(target.get("vllm_model", ""))

    if status and status.get("status") != "доступен":
        detail = status.get("detail", "")
        raise HTTPException(
            status_code=503,
            detail=(
                f"model unavailable: {status.get('model')} at {status.get('base_url')}. "
                f"detail: {detail}"
            ).strip(),
        )


async def get_models_snapshot() -> List[Dict[str, Any]]:
    if not MODEL_STATUS_LIST_CACHE:
        await refresh_model_status_cache()
    async with MODEL_STATUS_CACHE_LOCK:
        return [dict(item) for item in MODEL_STATUS_LIST_CACHE]


async def startup_status_poller() -> None:
    global MODEL_STATUS_POLLER_TASK
    await refresh_model_status_cache()
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
