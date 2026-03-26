from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from settings import (
    CPU_CHAT_Q4_BASE_URL,
    CPU_CHAT_Q4_MODEL,
    CPU_CHAT_Q6_BASE_URL,
    CPU_CHAT_Q6_MODEL,
    QWEN_122B_CHAT_BASE_URL,
    QWEN_122B_CHAT_MODEL,
    MINISTRAL_CHAT_BASE_URL,
    MINISTRAL_CHAT_MODEL,
    PUBLIC_QWEN_122B_CHAT_MODEL,
    PUBLIC_CPU_CHAT_Q4_MODEL,
    PUBLIC_CPU_CHAT_Q6_MODEL,
    PUBLIC_MINISTRAL_CHAT_MODEL,
    PUBLIC_QWEN_CHAT_MODEL,
    PUBLIC_QWEN_EMBED_4B_MODEL,
    PUBLIC_QWEN_EMBED_8B_MODEL,
    PUBLIC_QWEN_EMBED_MODEL,
    QWEN_CHAT_BASE_URL,
    QWEN_CHAT_MODEL,
    QWEN_EMBED_4B_BASE_URL,
    QWEN_EMBED_4B_MODEL,
    QWEN_EMBED_8B_BASE_URL,
    QWEN_EMBED_8B_MODEL,
    QWEN_EMBED_BASE_URL,
    QWEN_EMBED_MODEL,
    VLLM_MODEL,
)


def _normalize_model_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _is_qwen_chat_alias(name: str) -> bool:
    n = _normalize_model_name(name)
    return "qwen" in n and "35" in n and "9b" in n


def _is_ministral_alias(name: str) -> bool:
    n = _normalize_model_name(name)
    return "ministral" in n and "14b" in n


def _resolve_default_chat_route() -> Dict[str, str]:
    if _is_ministral_alias(VLLM_MODEL):
        return {
            "public_model": MINISTRAL_CHAT_MODEL,
            "vllm_model": MINISTRAL_CHAT_MODEL,
            "base_url": MINISTRAL_CHAT_BASE_URL,
            "type": "chat",
        }
    return {
        "public_model": QWEN_CHAT_MODEL,
        "vllm_model": QWEN_CHAT_MODEL,
        "base_url": QWEN_CHAT_BASE_URL,
        "type": "chat",
    }


def _additional_chat_routes() -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    if QWEN_122B_CHAT_MODEL and QWEN_122B_CHAT_BASE_URL and PUBLIC_QWEN_122B_CHAT_MODEL:
        routes.append(
            {
                "public_model": PUBLIC_QWEN_122B_CHAT_MODEL,
                "vllm_model": QWEN_122B_CHAT_MODEL,
                "base_url": QWEN_122B_CHAT_BASE_URL,
                "type": "chat",
            }
        )
    if CPU_CHAT_Q4_MODEL and CPU_CHAT_Q4_BASE_URL and PUBLIC_CPU_CHAT_Q4_MODEL:
        routes.append(
            {
                "public_model": PUBLIC_CPU_CHAT_Q4_MODEL,
                "vllm_model": CPU_CHAT_Q4_MODEL,
                "base_url": CPU_CHAT_Q4_BASE_URL,
                "type": "chat",
            }
        )
    if CPU_CHAT_Q6_MODEL and CPU_CHAT_Q6_BASE_URL and PUBLIC_CPU_CHAT_Q6_MODEL:
        routes.append(
            {
                "public_model": PUBLIC_CPU_CHAT_Q6_MODEL,
                "vllm_model": CPU_CHAT_Q6_MODEL,
                "base_url": CPU_CHAT_Q6_BASE_URL,
                "type": "chat",
            }
        )
    return routes


def _embed_routes() -> List[Dict[str, str]]:
    routes: List[Dict[str, str]] = [
        {
            "public_model": PUBLIC_QWEN_EMBED_MODEL,
            "vllm_model": QWEN_EMBED_MODEL,
            "base_url": QWEN_EMBED_BASE_URL,
            "type": "embeddings",
        }
    ]
    if QWEN_EMBED_8B_MODEL and QWEN_EMBED_8B_BASE_URL and PUBLIC_QWEN_EMBED_8B_MODEL:
        routes.append(
            {
                "public_model": PUBLIC_QWEN_EMBED_8B_MODEL,
                "vllm_model": QWEN_EMBED_8B_MODEL,
                "base_url": QWEN_EMBED_8B_BASE_URL,
                "type": "embeddings",
            }
        )
    if QWEN_EMBED_4B_MODEL and QWEN_EMBED_4B_BASE_URL and PUBLIC_QWEN_EMBED_4B_MODEL:
        routes.append(
            {
                "public_model": PUBLIC_QWEN_EMBED_4B_MODEL,
                "vllm_model": QWEN_EMBED_4B_MODEL,
                "base_url": QWEN_EMBED_4B_BASE_URL,
                "type": "embeddings",
            }
        )
    return routes


def _resolve_chat_target(requested_model: Optional[str]) -> Dict[str, str]:
    requested = (requested_model or "").strip()
    if not requested:
        return _resolve_default_chat_route()

    for route in [
        {
            "public_model": PUBLIC_QWEN_CHAT_MODEL,
            "vllm_model": QWEN_CHAT_MODEL,
            "base_url": QWEN_CHAT_BASE_URL,
            "type": "chat",
        },
        {
            "public_model": PUBLIC_MINISTRAL_CHAT_MODEL,
            "vllm_model": MINISTRAL_CHAT_MODEL,
            "base_url": MINISTRAL_CHAT_BASE_URL,
            "type": "chat",
        },
        *_additional_chat_routes(),
    ]:
        if requested == route["vllm_model"] or requested == route["public_model"]:
            return route

    for route in _embed_routes():
        if requested == route["public_model"] or requested == route["vllm_model"]:
            raise HTTPException(status_code=400, detail=f"model does not support chat endpoint: {route['public_model']}")

    allowed = [PUBLIC_QWEN_CHAT_MODEL, PUBLIC_MINISTRAL_CHAT_MODEL]
    for route in _additional_chat_routes():
        allowed.append(route["public_model"])
    raise HTTPException(status_code=400, detail=f"unsupported model for chat; allowed: {', '.join(allowed)}")


def _resolve_embed_target(requested_model: Optional[str]) -> Dict[str, str]:
    requested = (requested_model or "").strip()
    embed_routes = _embed_routes()
    default_route = embed_routes[0]

    if not requested:
        return default_route

    for route in embed_routes:
        if requested == route["public_model"] or requested == route["vllm_model"]:
            return route

    known_chat = [
        PUBLIC_QWEN_CHAT_MODEL,
        QWEN_CHAT_MODEL,
        PUBLIC_MINISTRAL_CHAT_MODEL,
        MINISTRAL_CHAT_MODEL,
    ]
    for route in _additional_chat_routes():
        known_chat.extend([route["public_model"], route["vllm_model"]])

    if requested in known_chat:
        raise HTTPException(status_code=400, detail=f"model does not support embeddings endpoint: {requested}")

    allowed = [route["public_model"] for route in embed_routes]
    raise HTTPException(
        status_code=400,
        detail=f"unsupported model for embeddings; allowed: {', '.join(allowed)}",
    )
