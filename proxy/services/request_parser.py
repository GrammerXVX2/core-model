import json
import logging
from typing import Any, Dict
from urllib.parse import parse_qs

from fastapi import Request

from settings import LOG_TEXT_PREVIEW_CHARS, REQUEST_DEBUG_LOG

logger = logging.getLogger("uvicorn.error")


def _truncate_text(value: str, max_chars: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated {len(value) - max_chars} chars>"


def _safe_preview(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], str):
            return [_truncate_text(v) for v in value[:5]]
        return f"<list len={len(value)}>"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in list(value.items())[:20]:
            if "token" in k.lower() or "authorization" in k.lower() or "password" in k.lower():
                out[k] = "***"
            else:
                out[k] = _safe_preview(v)
        return out
    return value


async def read_request_body_as_dict(request: Request) -> Dict[str, Any]:
    """Tolerant parser for backend payloads to avoid 422 on shape/content-type drift."""
    content_type = (request.headers.get("content-type") or "").lower()
    content_length = request.headers.get("content-length", "")

    if REQUEST_DEBUG_LOG:
        logger.info(
            "req.parse.start path=%s content_type=%s content_length=%s",
            request.url.path,
            content_type,
            content_length,
        )

    if "application/json" in content_type:
        try:
            parsed = await request.json()
        except Exception as exc:
            if REQUEST_DEBUG_LOG:
                logger.warning("req.parse.json_error path=%s error=%s", request.url.path, str(exc))
            parsed = {}
    elif "application/x-www-form-urlencoded" in content_type:
        try:
            raw_bytes = await request.body()
            raw = raw_bytes.decode("utf-8", errors="ignore")
            raw_stripped = raw.strip()

            if raw_stripped.startswith("{") or raw_stripped.startswith("["):
                try:
                    parsed = json.loads(raw_stripped)
                except Exception:
                    parsed = {}
            else:
                parsed = {}

            if not parsed:
                form_qs = parse_qs(raw, keep_blank_values=True)
                parsed = {
                    k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                    for k, v in form_qs.items()
                }

            for container_key in ("body", "payload", "data", "request", "json"):
                container_val = parsed.get(container_key)
                if isinstance(container_val, str):
                    candidate = container_val.strip()
                    if candidate.startswith("{") or candidate.startswith("["):
                        try:
                            nested = json.loads(candidate)
                            if isinstance(nested, dict):
                                parsed = nested
                            elif isinstance(nested, list):
                                parsed = {"input": nested}
                            else:
                                parsed = {"input": nested, "prompt": str(nested)}
                            break
                        except Exception:
                            pass

            if (
                isinstance(parsed, dict)
                and len(parsed) == 1
            ):
                only_key, only_val = next(iter(parsed.items()))
                if isinstance(only_key, str) and only_key.lstrip().startswith(("{", "[")):
                    reconstructed = only_key
                    if isinstance(only_val, str):
                        reconstructed = f"{only_key}={only_val}"
                    try:
                        recovered = json.loads(reconstructed)
                        if isinstance(recovered, dict):
                            parsed = recovered
                        elif isinstance(recovered, list):
                            parsed = {"input": recovered}
                        else:
                            parsed = {"input": recovered, "prompt": str(recovered)}
                    except Exception:
                        pass
        except Exception as exc:
            if REQUEST_DEBUG_LOG:
                logger.warning("req.parse.form_urlencoded_error path=%s error=%s", request.url.path, str(exc))
            parsed = {}
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            parsed = dict(form)
        except Exception as exc:
            if REQUEST_DEBUG_LOG:
                logger.warning("req.parse.multipart_error path=%s error=%s", request.url.path, str(exc))
            parsed = {}
    else:
        raw_bytes = await request.body()
        raw = raw_bytes.decode("utf-8", errors="ignore").strip()
        if REQUEST_DEBUG_LOG:
            logger.info("req.parse.raw_preview path=%s raw=%s", request.url.path, _safe_preview(raw))
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"prompt": raw, "input": raw}

    if isinstance(parsed, dict):
        if REQUEST_DEBUG_LOG and parsed:
            logger.info("req.parse.keys path=%s keys=%s", request.url.path, list(parsed.keys())[:20])
        if REQUEST_DEBUG_LOG and not parsed:
            logger.warning("req.parse.empty_dict path=%s", request.url.path)
        return parsed
    if isinstance(parsed, str):
        return {"prompt": parsed, "input": parsed}
    if isinstance(parsed, list):
        return {"input": parsed}
    return {}
