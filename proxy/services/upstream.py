import asyncio
import random
from typing import Any, Dict

import httpx
from fastapi import HTTPException

from services.metrics import inc_upstream_error, now_seconds, observe_upstream_latency
from settings import (
    UPSTREAM_HTTP_LIMITS,
    UPSTREAM_HTTP_TIMEOUT,
    UPSTREAM_RETRY_ATTEMPTS,
    UPSTREAM_RETRY_BASE_DELAY_SECONDS,
    UPSTREAM_RETRY_JITTER_SECONDS,
    VLLM_BASE_URLS,
)


_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def _new_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=UPSTREAM_HTTP_TIMEOUT, limits=UPSTREAM_HTTP_LIMITS)


async def startup_http_client() -> None:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _new_http_client()


async def shutdown_http_client() -> None:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        return
    await _SHARED_HTTP_CLIENT.aclose()
    _SHARED_HTTP_CLIENT = None


async def get_http_client() -> httpx.AsyncClient:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _new_http_client()
    return _SHARED_HTTP_CLIENT


def _retry_delay(attempt_index: int) -> float:
    base = max(0.0, UPSTREAM_RETRY_BASE_DELAY_SECONDS)
    jitter = max(0.0, UPSTREAM_RETRY_JITTER_SECONDS)
    return base * (2 ** attempt_index) + random.uniform(0.0, jitter)


def _is_retryable_request_error(exc: httpx.RequestError) -> bool:
    retryable_types = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    )
    return isinstance(exc, retryable_types)


def _is_retryable_status(status_code: int) -> bool:
    return status_code >= 500


async def _request_json_with_retries(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    attempts = max(1, UPSTREAM_RETRY_ATTEMPTS)
    client = await get_http_client()
    route = url

    for attempt in range(attempts):
        started_at = now_seconds()
        try:
            resp = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            if attempt + 1 < attempts and _is_retryable_request_error(exc):
                observe_upstream_latency(route, 502, started_at)
                await asyncio.sleep(_retry_delay(attempt))
                continue
            observe_upstream_latency(route, 502, started_at)
            inc_upstream_error(route, 502)
            raise HTTPException(status_code=502, detail=f"upstream connection error: {url}: {str(exc) or exc.__class__.__name__}")

        if resp.status_code >= 400:
            observe_upstream_latency(route, resp.status_code, started_at)
            if attempt + 1 < attempts and _is_retryable_status(resp.status_code):
                await asyncio.sleep(_retry_delay(attempt))
                continue
            inc_upstream_error(route, resp.status_code)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        observe_upstream_latency(route, resp.status_code, started_at)
        return resp.json()

    inc_upstream_error(route, 502)
    raise HTTPException(status_code=502, detail=f"upstream retry exhausted: {url}")


async def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = VLLM_BASE_URLS[0]
    url = f"{base_url}{path}"
    return await _request_json_with_retries(url, payload)


async def post_json_to(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    return await _request_json_with_retries(url, payload)
