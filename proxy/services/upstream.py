from typing import Any, Dict

import httpx
from fastapi import HTTPException

from settings import UPSTREAM_HTTP_TIMEOUT, VLLM_BASE_URLS


async def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = VLLM_BASE_URLS[0]
    url = f"{base_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"upstream connection error: {url}: {str(exc) or exc.__class__.__name__}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def post_json_to(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"upstream connection error: {url}: {str(exc) or exc.__class__.__name__}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()
