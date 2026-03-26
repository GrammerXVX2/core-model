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
    DISABLE_THINKING,
    TOKEN_BUDGET_STRICT_MODE,
)
from services.request_parser import read_request_body_as_dict as _read_request_body_as_dict
from services.status_cache import ensure_model_available as _ensure_model_available
from services.status_cache import resolve_target_from_status_cache as _resolve_target_from_status_cache
from services.upstream import post_json_to as _post_json_to

from api.common import (
    analyze_max_tokens_budget,
    estimate_chat_input_tokens,
    estimate_input_tokens_from_text,
    extract_chat_text,
    extract_finish_reason,
    inject_system_language_prompt,
    language_instruction,
    ns,
    now_iso,
    ollama_response,
    safe_preview,
    sse_event,
    strip_reasoning_prefix,
)

logger = logging.getLogger("uvicorn.error")
router = APIRouter()


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


def _stream_supported(target: Dict[str, Any]) -> bool:
    if "stream_supported" in target:
        return bool(target.get("stream_supported"))
    # Backward-compatible default for legacy env chat routes.
    return True


def _reasoning_supported(target: Dict[str, Any]) -> bool:
    if "reasoning_supported" in target:
        return bool(target.get("reasoning_supported"))
    return False


def _maybe_raise_strict_token_budget_error(model: str, budget: Dict[str, Any]) -> None:
    if not TOKEN_BUDGET_STRICT_MODE:
        return

    requested_output_tokens = budget.get("requested_output_tokens")
    available_output_tokens = int(budget.get("available_output_tokens") or 0)
    hard_cap = int(budget.get("hard_cap") or 1)

    overflow_requested = requested_output_tokens is not None and int(requested_output_tokens) > hard_cap
    no_budget_left = available_output_tokens < 1

    if not overflow_requested and not no_budget_left:
        return

    detail = {
        "error": "token_budget_exceeded",
        "model": model,
        "message": "Requested output exceeds available token budget for this model.",
        "estimated_input_tokens": int(budget.get("estimated_input_tokens") or 0),
        "model_max_context_tokens": int(budget.get("max_context_tokens") or 0),
        "min_context_headroom": int(budget.get("min_context_headroom") or 0),
        "available_output_tokens": available_output_tokens,
        "requested_output_tokens": requested_output_tokens,
        "requested_source": budget.get("requested_source"),
        "max_tokens_cap": int(budget.get("max_tokens_cap") or 0),
        "hard_cap": hard_cap,
        "resolved_max_tokens": int(budget.get("resolved_max_tokens") or 1),
        "hint": "Reduce input size, lower requested max_tokens, or raise model-specific context/cap limits.",
    }
    raise HTTPException(status_code=400, detail=detail)

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
                                "Ministral3-14B",
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
                                "Ministral3-14B",
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


CHAT_UI_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "examples": {
                    "ui_stream": {
                        "summary": "UI stream mode",
                        "value": {
                            "model": os.getenv("OPENAPI_CHAT_UI_EXAMPLE_MODEL", "Qwen3.5-9B"),
                            "stream": True,
                            "reasoning": False,
                            "temperature": 0,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Answer in one short sentence.",
                                }
                            ],
                        },
                    },
                    "ui_non_stream": {
                        "summary": "UI non-stream mode",
                        "value": {
                            "model": os.getenv("OPENAPI_CHAT_UI_EXAMPLE_MODEL", "Qwen3.5-9B"),
                            "stream": False,
                            "reasoning": True,
                            "max_tokens": 128,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Summarize token budgeting in two bullets.",
                                }
                            ],
                        },
                    },
                }
            }
        },
    },
    "responses": {
        "200": {
            "description": "When stream=true returns text/event-stream (SSE). When stream=false returns Ollama-style JSON.",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "data: {\"type\": \"start\", \"model\": \"Qwen3.5-9B\"}\\n\\ndata: [DONE]\\n\\n",
                    }
                },
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/OllamaTextResponseModel"}
                },
            },
        }
    },
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
    target = await _resolve_target_from_status_cache(requested_model, expected_type="chat")
    if target is None:
        raise HTTPException(status_code=503, detail="no chat models registered in status cache")
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
    if stream:
        raise HTTPException(status_code=501, detail="stream is not supported on /api/chat; use /api/chat-ui")

    start_ns = ns()
    estimated_input_tokens = estimate_chat_input_tokens(messages)
    token_budget = analyze_max_tokens_budget(
        body_data,
        estimated_input_tokens=estimated_input_tokens,
        max_context_tokens=target.get("max_context_tokens"),
        max_tokens_cap=target.get("max_tokens_cap"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )
    _maybe_raise_strict_token_budget_error(model, token_budget)
    resolved_max_tokens = int(token_budget["resolved_max_tokens"])
    payload = {
        "model": target["vllm_model"],
        "messages": messages,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolved_max_tokens,
    }
    # /api/chat always runs with reasoning disabled; UI toggle lives only in /api/chat-ui.
    payload["chat_template_kwargs"] = {"enable_thinking": False}

    await _ensure_model_available(target)

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
    openapi_extra=CHAT_UI_OPENAPI_EXTRA,
)
@router.post(
    "/api/chat/ui",
    tags=["chat"],
    summary="Chat Completion Stream (UI Alias)",
    openapi_extra=CHAT_UI_OPENAPI_EXTRA,
)
async def api_chat_ui(request: Request) -> Any:
    body_data = await _read_request_body_as_dict(request)
    requested_model = body_data.get("model")
    target = await _resolve_target_from_status_cache(requested_model, expected_type="chat")
    if target is None:
        raise HTTPException(status_code=503, detail="no chat models registered in status cache")
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
    token_budget = analyze_max_tokens_budget(
        body_data,
        estimated_input_tokens=estimated_input_tokens,
        max_context_tokens=target.get("max_context_tokens"),
        max_tokens_cap=target.get("max_tokens_cap"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )
    _maybe_raise_strict_token_budget_error(model, token_budget)
    resolved_max_tokens = int(token_budget["resolved_max_tokens"])

    payload = {
        "model": target["vllm_model"],
        "messages": messages,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolved_max_tokens,
    }
    stream = bool(body_data.get("stream", True))
    payload["stream"] = stream
    reasoning_flag = _extract_reasoning_flag(body_data)
    if reasoning_flag is not None and not _reasoning_supported(target):
        raise HTTPException(status_code=400, detail=f"model does not support reasoning toggle: {model}")
    if _reasoning_supported(target):
        if reasoning_flag is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning_flag}
        elif DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    if stream and not _stream_supported(target):
        raise HTTPException(status_code=400, detail=f"model does not support stream mode: {model}")

    await _ensure_model_available(target)

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

    if not stream:
        start_ns = ns()
        data = await _post_json_to(target["base_url"], "/chat/completions", payload)
        content = strip_reasoning_prefix(extract_chat_text(data))
        done_reason = extract_finish_reason(data)
        return ollama_response(model, content, start_ns, done_reason=done_reason)

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
    target = await _resolve_target_from_status_cache(requested_model, expected_type="chat")
    if target is None:
        raise HTTPException(status_code=503, detail="no chat models registered in status cache")
    await _ensure_model_available(target)
    model = requested_model or target["public_model"]
    prompt: str = str(body_data.get("prompt", ""))
    prompt = f"{language_instruction()}\n\n{prompt}".strip()
    stream = bool(body_data.get("stream", False))

    if stream:
        raise HTTPException(status_code=501, detail="stream not implemented")

    start_ns = ns()
    estimated_input_tokens = estimate_input_tokens_from_text(prompt) + 16
    token_budget = analyze_max_tokens_budget(
        body_data,
        estimated_input_tokens=estimated_input_tokens,
        max_context_tokens=target.get("max_context_tokens"),
        max_tokens_cap=target.get("max_tokens_cap"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )
    _maybe_raise_strict_token_budget_error(model, token_budget)
    payload = {
        "model": target["vllm_model"],
        "prompt": prompt,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        # /api/generate is always no-reasoning for deterministic output contract.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = await _post_json_to(target["base_url"], "/completions", payload)
    content = strip_reasoning_prefix(data["choices"][0]["text"])
    done_reason = extract_finish_reason(data)
    return ollama_response(model, content, start_ns, done_reason=done_reason)
