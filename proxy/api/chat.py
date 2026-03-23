import json
import logging
import os
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from schemas import OllamaTextResponseModel
from settings import (
    CHAT_DEBUG_LOG,
    CHAT_EMPTY_FALLBACK_USER_TEXT,
    CPU_CHAT_Q4_MODEL,
    CPU_CHAT_Q6_MODEL,
    DISABLE_THINKING,
    PUBLIC_CPU_CHAT_Q4_MODEL,
    PUBLIC_CPU_CHAT_Q6_MODEL,
)
from services.request_parser import read_request_body_as_dict as _read_request_body_as_dict
from services.routing import _resolve_chat_target
from services.status_cache import ensure_model_available as _ensure_model_available
from services.upstream import post_json_to as _post_json_to

from api.common import (
    estimate_chat_input_tokens,
    estimate_input_tokens_from_text,
    extract_chat_text,
    extract_finish_reason,
    inject_system_language_prompt,
    language_instruction,
    ns,
    now_iso,
    ollama_response,
    resolve_max_tokens,
    safe_preview,
    sse_event,
    strip_reasoning_prefix,
)

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["chat", "generate"])


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _extract_reasoning_flag(body_data: Dict[str, Any]) -> Any:
    for key in ("reasoning", "thinking", "enable_thinking"):
        if key in body_data:
            parsed = _coerce_bool(body_data.get(key))
            if parsed is not None:
                return parsed

    options = body_data.get("options")
    if isinstance(options, dict):
        for key in ("reasoning", "thinking", "enable_thinking"):
            if key in options:
                parsed = _coerce_bool(options.get(key))
                if parsed is not None:
                    return parsed

    return None


def _is_llama_route(target: Dict[str, str]) -> bool:
    public_model = target.get("public_model", "")
    backend_model = target.get("vllm_model", "")
    return public_model in {PUBLIC_CPU_CHAT_Q4_MODEL, PUBLIC_CPU_CHAT_Q6_MODEL} or backend_model in {
        CPU_CHAT_Q4_MODEL,
        CPU_CHAT_Q6_MODEL,
    }

CHAT_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "examples": {
                    "qwen_chat": {
                        "summary": "Qwen chat route",
                        "value": {
                            "model": os.getenv("OPENAPI_CHAT_EXAMPLE_MODEL", "Qwen3.5-9B"),
                            "temperature": 0,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": os.getenv(
                                        "OPENAPI_CHAT_EXAMPLE_CONTENT",
                                        "Выбери лучший документ по запросу и ответь JSON.",
                                    ),
                                }
                            ],
                        },
                    },
                    "ministral_chat": {
                        "summary": "Ministral chat route",
                        "value": {
                            "model": os.getenv(
                                "OPENAPI_MINISTRAL_CHAT_EXAMPLE_MODEL",
                                os.getenv("PUBLIC_MINISTRAL_CHAT_MODEL", "Ministral3-14B"),
                            ),
                            "temperature": 0.2,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Кратко объясни разницу между договором и офертой.",
                                }
                            ],
                        },
                    },
                },
            }
        },
    }
}

GENERATE_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "examples": {
                    "qwen_generate": {
                        "summary": "Qwen generate route",
                        "value": {
                            "model": os.getenv("OPENAPI_GENERATE_EXAMPLE_MODEL", "Qwen3.5-9B"),
                            "prompt": os.getenv(
                                "OPENAPI_GENERATE_EXAMPLE_PROMPT",
                                "Кратко объясни, как работает OAuth2 в FastAPI.",
                            ),
                            "temperature": 0,
                        },
                    },
                    "ministral_generate": {
                        "summary": "Ministral generate route",
                        "value": {
                            "model": os.getenv(
                                "OPENAPI_MINISTRAL_GENERATE_EXAMPLE_MODEL",
                                os.getenv("PUBLIC_MINISTRAL_CHAT_MODEL", "Ministral3-14B"),
                            ),
                            "prompt": "Сформулируй краткое определение понятия сервитута.",
                            "temperature": 0.2,
                        },
                    },
                },
            }
        },
    }
}


@router.post(
    "/api/chat",
    tags=["chat"],
    summary="Chat Completion",
    response_model=OllamaTextResponseModel,
    openapi_extra=CHAT_OPENAPI_EXTRA,
)
async def api_chat(request: Request) -> Any:
    body_data = await _read_request_body_as_dict(request)
    if CHAT_DEBUG_LOG:
        logger.info("chat.incoming body=%s", safe_preview(body_data))

    requested_model = body_data.get("model")
    target = _resolve_chat_target(requested_model)
    await _ensure_model_available(target)
    model = requested_model or target["public_model"]
    messages: List[Dict[str, Any]] = body_data.get("messages", [])

    if not messages:
        fallback_text = (
            body_data.get("prompt")
            or body_data.get("input")
            or body_data.get("text")
            or body_data.get("query")
        )
        if fallback_text is not None:
            messages = [{"role": "user", "content": str(fallback_text)}]
        elif isinstance(body_data.get("message"), dict):
            msg = body_data.get("message") or {}
            msg_content = msg.get("content")
            if msg_content is not None:
                messages = [{"role": msg.get("role", "user"), "content": str(msg_content)}]

    if not messages:
        messages = [{"role": "user", "content": CHAT_EMPTY_FALLBACK_USER_TEXT}]
    messages = inject_system_language_prompt(messages)
    stream = bool(body_data.get("stream", False))

    start_ns = ns()
    estimated_input_tokens = estimate_chat_input_tokens(messages)
    resolved_max_tokens = resolve_max_tokens(body_data, estimated_input_tokens=estimated_input_tokens)
    payload = {
        "model": target["vllm_model"],
        "messages": messages,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolved_max_tokens,
    }
    reasoning_flag = _extract_reasoning_flag(body_data)
    if _is_llama_route(target):
        if reasoning_flag is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning_flag}
        elif DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    if stream:
        stream_payload = {**payload, "stream": True}
        return StreamingResponse(
            _stream_chat_ollama_events(
                target,
                stream_payload,
                model,
                start_ns,
                include_reasoning=reasoning_flag is True,
            ),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    if CHAT_DEBUG_LOG:
        roles = [str(m.get("role", "")) for m in messages[:10]]
        logger.info(
            "chat.adapted model=%s route_model=%s base_url=%s messages=%s roles=%s est_input_tokens=%s max_tokens=%s",
            model,
            target["vllm_model"],
            target["base_url"],
            len(messages),
            roles,
            estimated_input_tokens,
            resolved_max_tokens,
        )

    data = await _post_json_to(target["base_url"], "/chat/completions", payload)

    if CHAT_DEBUG_LOG:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        logger.info(
            "chat.vllm_response finish_reason=%s content_preview=%s reasoning_preview=%s",
            choice.get("finish_reason"),
            safe_preview(message.get("content")),
            safe_preview(message.get("reasoning")),
        )

    content = strip_reasoning_prefix(extract_chat_text(data))
    done_reason = extract_finish_reason(data)
    return ollama_response(model, content, start_ns, done_reason=done_reason)


async def _stream_chat_ollama_events(
    target: Dict[str, str],
    payload: Dict[str, Any],
    public_model: str,
    start_ns: int,
    include_reasoning: bool = False,
):
    url = f"{target['base_url'].rstrip('/')}/chat/completions"
    final_reason = "stop"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    detail = await resp.aread()
                    message = detail.decode("utf-8", errors="ignore")
                    raise HTTPException(status_code=resp.status_code, detail=message)

                async for raw_line in resp.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()

                    if line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        final_reason = finish_reason

                    token = delta.get("content")
                    if token is None and include_reasoning:
                        token = delta.get("reasoning_content")
                    if token is None:
                        continue

                    token_text = str(token)
                    if not token_text:
                        continue

                    yield json.dumps(
                        {
                            "model": public_model,
                            "created_at": now_iso(),
                            "message": {"role": "assistant", "content": token_text},
                            "done": False,
                        },
                        ensure_ascii=False,
                    ) + "\n"
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"upstream stream error: {str(exc) or exc.__class__.__name__}")

    end_ns = ns()
    total_ns = max(0, end_ns - start_ns)
    yield json.dumps(
        {
            "model": public_model,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": final_reason,
            "total_duration": total_ns,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        },
        ensure_ascii=False,
    ) + "\n"


async def _stream_chat_ui_events(
    target: Dict[str, str],
    payload: Dict[str, Any],
    public_model: str,
    include_reasoning: bool = False,
):
    url = f"{target['base_url'].rstrip('/')}/chat/completions"
    total_text = ""
    terminal_emitted = False

    yield sse_event({"type": "start", "model": public_model})

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    detail = await resp.aread()
                    terminal_emitted = True
                    yield sse_event(
                        {
                            "type": "error",
                            "status": resp.status_code,
                            "detail": detail.decode("utf-8", errors="ignore"),
                        }
                    )
                    yield "data: [DONE]\n\n"
                    return

                async for raw_line in resp.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()

                    if line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}

                    content_token = delta.get("content")
                    reasoning_token = delta.get("reasoning_content") if include_reasoning else None

                    if reasoning_token is not None:
                        token_text = str(reasoning_token)
                        if token_text:
                            yield sse_event({"type": "token", "model": public_model, "token": token_text, "is_reasoning": True})
                        continue

                    if content_token is None:
                        continue
                    token_text = str(content_token)
                    if not token_text:
                        continue
                    total_text += token_text
                    yield sse_event({"type": "token", "model": public_model, "token": token_text})

        terminal_emitted = True
        yield sse_event({"type": "done", "model": public_model, "text": total_text})
        yield "data: [DONE]\n\n"
    except Exception as exc:
        if not terminal_emitted:
            yield sse_event({"type": "error", "status": 500, "detail": str(exc)})
            yield "data: [DONE]\n\n"


@router.post(
    "/api/chat-ui",
    tags=["chat"],
    summary="Chat Completion Stream (UI)",
)
@router.post(
    "/api/chat/ui",
    tags=["chat"],
    summary="Chat Completion Stream (UI Alias)",
)
async def api_chat_ui(request: Request) -> StreamingResponse:
    body_data = await _read_request_body_as_dict(request)
    requested_model = body_data.get("model")
    target = _resolve_chat_target(requested_model)
    await _ensure_model_available(target)
    model = requested_model or target["public_model"]

    messages: List[Dict[str, Any]] = body_data.get("messages", [])
    if not messages:
        fallback_text = (
            body_data.get("prompt")
            or body_data.get("input")
            or body_data.get("text")
            or body_data.get("query")
        )
        if fallback_text is not None:
            messages = [{"role": "user", "content": str(fallback_text)}]

    if not messages:
        messages = [{"role": "user", "content": CHAT_EMPTY_FALLBACK_USER_TEXT}]

    messages = inject_system_language_prompt(messages)
    estimated_input_tokens = estimate_chat_input_tokens(messages)
    resolved_max_tokens = resolve_max_tokens(body_data, estimated_input_tokens=estimated_input_tokens)

    payload = {
        "model": target["vllm_model"],
        "messages": messages,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolved_max_tokens,
        "stream": True,
    }
    reasoning_flag = _extract_reasoning_flag(body_data)
    if _is_llama_route(target):
        if reasoning_flag is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning_flag}
        elif DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    if CHAT_DEBUG_LOG:
        logger.info(
            "chat.ui.adapted model=%s route_model=%s base_url=%s messages=%s est_input_tokens=%s max_tokens=%s",
            model,
            target["vllm_model"],
            target["base_url"],
            len(messages),
            estimated_input_tokens,
            resolved_max_tokens,
        )

    return StreamingResponse(
        _stream_chat_ui_events(target, payload, model, include_reasoning=reasoning_flag is True),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/api/generate",
    tags=["generate"],
    summary="Prompt Completion",
    response_model=OllamaTextResponseModel,
    openapi_extra=GENERATE_OPENAPI_EXTRA,
)
async def api_generate(request: Request) -> Dict[str, Any]:
    body_data = await _read_request_body_as_dict(request)
    requested_model = body_data.get("model")
    target = _resolve_chat_target(requested_model)
    await _ensure_model_available(target)
    model = requested_model or target["public_model"]
    prompt: str = str(body_data.get("prompt", ""))
    prompt = f"{language_instruction()}\n\n{prompt}".strip()
    stream = bool(body_data.get("stream", False))

    if stream:
        raise HTTPException(status_code=501, detail="stream not implemented")

    start_ns = ns()
    estimated_input_tokens = estimate_input_tokens_from_text(prompt) + 16
    payload = {
        "model": target["vllm_model"],
        "prompt": prompt,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolve_max_tokens(body_data, estimated_input_tokens=estimated_input_tokens),
    }
    data = await _post_json_to(target["base_url"], "/completions", payload)
    content = strip_reasoning_prefix(data["choices"][0]["text"])
    done_reason = extract_finish_reason(data)
    return ollama_response(model, content, start_ns, done_reason=done_reason)
