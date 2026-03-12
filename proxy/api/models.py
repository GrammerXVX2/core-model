from typing import Dict, List

from fastapi import APIRouter

from schemas import ModelStatusItem
from services.status_cache import get_models_snapshot as _get_models_snapshot

router = APIRouter(tags=["models"])


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
