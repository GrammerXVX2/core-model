import os
import json
import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="Ollama-Compatible Proxy for vLLM",
    description=(
        "Proxy layer that exposes Ollama-style endpoints and forwards requests "
        "to vLLM OpenAI-compatible APIs."
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "chat", "description": "Chat-style text generation."},
        {"name": "generate", "description": "Prompt-style text generation."},
        {"name": "embeddings", "description": "Text embedding endpoints."},
        {"name": "models", "description": "Model registry and availability status."},
    ],
)


class ChatOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    num_predict: Optional[int] = Field(default=None, description="Ollama-style max output tokens.")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(description="Message role.", examples=["user"])
    content: Any = Field(description="Message text/content payload.", examples=["Кратко объясни, что такое RAG."])


class ChatRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = Field(default=None, description="Model name requested by client.")
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Conversation messages.",
        examples=[[{"role": "user", "content": "Сделай краткое резюме текста"}]],
    )
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=None, description="Requested max output tokens.")
    stream: bool = Field(default=False, description="Streaming is not supported.")
    options: Optional[ChatOptions] = Field(default=None, description="Ollama options.")


class GenerateRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = Field(default=None, description="Model name requested by client.")
    prompt: Optional[str] = Field(default=None, description="Input prompt.", examples=["Объясни разницу между REST и gRPC."])
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=None, description="Requested max output tokens.")
    stream: bool = Field(default=False, description="Streaming is not supported.")
    options: Optional[ChatOptions] = Field(default=None, description="Ollama options.")


class EmbedRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = Field(default=None, description="Ignored by proxy; embed model is fixed in config.")
    input: Optional[Any] = Field(
        default=None,
        description="Input text or list of texts for embedding.",
        examples=["Ошибка 502 при оплате картой"],
    )


class OllamaTextResponseModel(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class EmbedResponseModel(BaseModel):
    model: str
    embedding: List[float]
    embeddings: List[List[float]]
    total_duration: int
    load_duration: int
    prompt_eval_count: int


class ModelStatusItem(BaseModel):
    model: str
    model_vllm: str
    type: str
    base_url: str
    max_context_tokens: int
    status: str
    detail: str = ""


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
                            "model": "Ministral3-14B",
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
                            "model": "Ministral3-14B",
                            "prompt": "Сформулируй краткое определение понятия сервитута.",
                            "temperature": 0.2,
                        },
                    },
                },
            }
        },
    }
}

EMBED_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "example": {
                    "model": os.getenv("OPENAPI_EMBED_EXAMPLE_MODEL", "Qwen3.5-4B"),
                    "input": json.loads(
                        os.getenv(
                            "OPENAPI_EMBED_EXAMPLE_INPUT_JSON",
                            '["Как настроить OAuth2 авторизацию в FastAPI?", "Как сбросить пароль через форму восстановления аккаунта.", "Пошаговая настройка OAuth2 в FastAPI: /token, bearer scheme, проверка access token."]',
                        )
                    )
                },
            }
        },
    }
}

VLLM_BASE_URLS = [
    url.strip().rstrip("/")
    for url in os.getenv("VLLM_BASE_URLS", os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).split(",")
    if url.strip()
]
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", VLLM_BASE_URLS[0])
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "Qwen/Qwen3.5-4B")

# Model routing config.
DEFAULT_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", VLLM_BASE_URLS[0])
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "Qwen/Qwen3.5-9B")
QWEN_CHAT_BASE_URL = os.getenv("QWEN_CHAT_BASE_URL", DEFAULT_CHAT_BASE_URL)
MINISTRAL_CHAT_MODEL = os.getenv("MINISTRAL_CHAT_MODEL", "Ministral-3-14B")
MINISTRAL_CHAT_BASE_URL = os.getenv("MINISTRAL_CHAT_BASE_URL", DEFAULT_CHAT_BASE_URL)
QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", VLLM_EMBED_MODEL)
QWEN_EMBED_BASE_URL = os.getenv("QWEN_EMBED_BASE_URL", VLLM_EMBED_BASE_URL)
QWEN_CHAT_MAX_CONTEXT_TOKENS = int(os.getenv("QWEN_CHAT_MAX_CONTEXT_TOKENS", os.getenv("VLLM_MAX_CONTEXT_TOKENS", "10240")))
MINISTRAL_CHAT_MAX_CONTEXT_TOKENS = int(os.getenv("MINISTRAL_CHAT_MAX_CONTEXT_TOKENS", "8192"))
QWEN_EMBED_MAX_CONTEXT_TOKENS = int(os.getenv("QWEN_EMBED_MAX_CONTEXT_TOKENS", "8192"))

DEFAULT_RESPONSE_LANGUAGE = os.getenv("DEFAULT_RESPONSE_LANGUAGE", "ru")
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1") == "1"
DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_DEFAULT_MAX_TOKENS", "256"))
MAX_TOKENS_CAP = int(os.getenv("VLLM_MAX_TOKENS_CAP", "1024"))
MAX_CONTEXT_TOKENS = int(os.getenv("VLLM_MAX_CONTEXT_TOKENS", "4096"))
MIN_CONTEXT_HEADROOM = int(os.getenv("VLLM_MIN_CONTEXT_HEADROOM", "256"))
CHAT_DEBUG_LOG = os.getenv("CHAT_DEBUG_LOG", "1") == "1"
EMBED_DEBUG_LOG = os.getenv("EMBED_DEBUG_LOG", "1") == "1"
LOG_TEXT_PREVIEW_CHARS = int(os.getenv("LOG_TEXT_PREVIEW_CHARS", "500"))
REQUEST_DEBUG_LOG = os.getenv("REQUEST_DEBUG_LOG", "1") == "1"
EMBED_FORCE_MODEL = os.getenv("EMBED_FORCE_MODEL", "1") == "1"
CHAT_EMPTY_FALLBACK_USER_TEXT = os.getenv("CHAT_EMPTY_FALLBACK_USER_TEXT", "")
LANGUAGE_INSTRUCTION_RU = os.getenv(
    "LANGUAGE_INSTRUCTION_RU",
    "Отвечай только на русском языке.",
)
LANGUAGE_INSTRUCTION_RU_FINAL_ONLY = os.getenv(
    "LANGUAGE_INSTRUCTION_RU_FINAL_ONLY",
    "Не показывай ход рассуждений, верни только финальный ответ.",
)
LANGUAGE_INSTRUCTION_DEFAULT = os.getenv(
    "LANGUAGE_INSTRUCTION_DEFAULT",
    "Answer in the configured default language. Provide only the final answer.",
)
REASONING_PREFIX_MARKERS = [
    m.strip()
    for m in os.getenv(
        "REASONING_PREFIX_MARKERS",
        "Thinking Process:|Reasoning:|Ход рассуждений:",
    ).split("|")
    if m.strip()
]

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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


def _language_instruction() -> str:
    lang = DEFAULT_RESPONSE_LANGUAGE.lower()
    if lang in {"ru", "russian", "рус", "русский"}:
        base = LANGUAGE_INSTRUCTION_RU
        if DISABLE_THINKING:
            return f"{base} {LANGUAGE_INSTRUCTION_RU_FINAL_ONLY}".strip()
        return base
    return LANGUAGE_INSTRUCTION_DEFAULT


def _inject_system_language_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    instruction = _language_instruction()
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        if instruction in existing:
            return messages
        patched = messages.copy()
        patched[0] = {"role": "system", "content": f"{existing}\n{instruction}".strip()}
        return patched
    return [{"role": "system", "content": instruction}, *messages]


def _estimate_input_tokens_from_text(text: str) -> int:
    # Rough heuristic: 1 token ~= 4 chars for mixed RU/EN text.
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_chat_input_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += _estimate_input_tokens_from_text(str(msg.get("content", "")))
        total += 8
    return total + 16


def _resolve_max_tokens(body: Dict[str, Any], estimated_input_tokens: int = 0) -> int:
    # Ollama clients often send options.num_predict instead of max_tokens.
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    requested = body.get("max_tokens")
    if requested is None:
        requested = options.get("num_predict")

    # Keep room for prompt/system tokens and scheduler overhead.
    dynamic_cap = max(1, MAX_CONTEXT_TOKENS - max(0, estimated_input_tokens) - MIN_CONTEXT_HEADROOM)
    hard_cap = max(1, min(MAX_TOKENS_CAP, dynamic_cap))

    if requested is None:
        return min(DEFAULT_MAX_TOKENS, hard_cap)

    try:
        value = int(requested)
    except (TypeError, ValueError):
        return min(DEFAULT_MAX_TOKENS, hard_cap)

    if value < 1:
        return min(DEFAULT_MAX_TOKENS, hard_cap)
    return min(value, hard_cap)


def _extract_chat_text(data: Dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    content = message.get("content")
    if isinstance(content, str) and content:
        return content

    # Qwen reasoning models can return text under `reasoning` with null `content`.
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    return ""


def _strip_reasoning_prefix(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for marker in REASONING_PREFIX_MARKERS:
        if cleaned.startswith(marker):
            # Keep only the tail after a blank line if present.
            parts = cleaned.split("\n\n", 1)
            cleaned = parts[1] if len(parts) == 2 else cleaned
    return cleaned.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ns() -> int:
    return time.perf_counter_ns()


def _normalize_model_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _is_qwen_chat_alias(name: str) -> bool:
    n = _normalize_model_name(name)
    return "qwen" in n and "35" in n and "9b" in n


def _is_qwen_embed_alias(name: str) -> bool:
    n = _normalize_model_name(name)
    return "qwen" in n and ("4b" in n or "embedding" in n)


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


def _resolve_chat_target(requested_model: Optional[str]) -> Dict[str, str]:
    requested = (requested_model or "").strip()
    if not requested:
        return _resolve_default_chat_route()

    if _is_qwen_chat_alias(requested) or _normalize_model_name(requested) == _normalize_model_name(QWEN_CHAT_MODEL):
        return {
            "public_model": "Qwen3.5-9B",
            "vllm_model": QWEN_CHAT_MODEL,
            "base_url": QWEN_CHAT_BASE_URL,
            "type": "chat",
        }

    if _is_ministral_alias(requested) or _normalize_model_name(requested) == _normalize_model_name(MINISTRAL_CHAT_MODEL):
        return {
            "public_model": "Ministral3-14B",
            "vllm_model": MINISTRAL_CHAT_MODEL,
            "base_url": MINISTRAL_CHAT_BASE_URL,
            "type": "chat",
        }

    if _is_qwen_embed_alias(requested):
        raise HTTPException(status_code=400, detail="model does not support chat endpoint: Qwen3.5-4B")

    raise HTTPException(
        status_code=400,
        detail="unsupported model for chat; allowed: Qwen3.5-9B, Ministral3-14B",
    )


def _resolve_embed_target(requested_model: Optional[str]) -> Dict[str, str]:
    requested = (requested_model or "").strip()

    if EMBED_FORCE_MODEL:
        return {
            "public_model": "Qwen3.5-4B",
            "vllm_model": QWEN_EMBED_MODEL,
            "base_url": QWEN_EMBED_BASE_URL,
            "type": "embeddings",
        }

    if not requested:
        return {
            "public_model": "Qwen3.5-4B",
            "vllm_model": QWEN_EMBED_MODEL,
            "base_url": QWEN_EMBED_BASE_URL,
            "type": "embeddings",
        }

    if _is_qwen_embed_alias(requested) or _normalize_model_name(requested) == _normalize_model_name(QWEN_EMBED_MODEL):
        return {
            "public_model": "Qwen3.5-4B",
            "vllm_model": QWEN_EMBED_MODEL,
            "base_url": QWEN_EMBED_BASE_URL,
            "type": "embeddings",
        }

    if _is_qwen_chat_alias(requested) or _is_ministral_alias(requested):
        raise HTTPException(status_code=400, detail=f"model does not support embeddings endpoint: {requested}")

    raise HTTPException(
        status_code=400,
        detail="unsupported model for embeddings; allowed: Qwen3.5-4B",
    )


async def _probe_model_status(
    public_model: str,
    vllm_model: str,
    model_type: str,
    base_url: str,
    max_context_tokens: int,
) -> Dict[str, Union[str, int]]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
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


def _ollama_response(model: str, content: str, start_ns: int, load_ns: int = 0) -> Dict[str, Any]:
    end_ns = _ns()
    total_ns = max(0, end_ns - start_ns)
    return {
        "model": model,
        "created_at": _now_iso(),
        "response": content,
        "done": True,
        "done_reason": "stop",
        "total_duration": total_ns,
        "load_duration": load_ns,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


async def _read_request_body_as_dict(request: Request) -> Dict[str, Any]:
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

    # Try JSON first for common clients.
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

            # Some clients send raw JSON but mark it as form-urlencoded.
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

            # Many clients wrap JSON in one urlencoded field.
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

            # Heuristic recovery for malformed form parse when raw JSON contained '='.
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
            # Fallback for plain-text clients: treat body as prompt/input text.
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


async def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = VLLM_BASE_URLS[0]
    url = f"{base_url}{path}"
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def _post_json_to(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/", tags=["chat"], summary="Proxy Status")
async def root_status() -> Dict[str, str]:
    return {
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get(
    "/api/models",
    tags=["models"],
    summary="List Models and Availability",
)
async def api_models() -> Dict[str, List[ModelStatusItem]]:
    checks = [
        _probe_model_status(
            "Qwen3.5-9B",
            QWEN_CHAT_MODEL,
            "chat",
            QWEN_CHAT_BASE_URL,
            QWEN_CHAT_MAX_CONTEXT_TOKENS,
        ),
        _probe_model_status(
            "Qwen3.5-4B",
            QWEN_EMBED_MODEL,
            "embeddings",
            QWEN_EMBED_BASE_URL,
            QWEN_EMBED_MAX_CONTEXT_TOKENS,
        ),
        _probe_model_status(
            "Ministral3-14B",
            MINISTRAL_CHAT_MODEL,
            "chat",
            MINISTRAL_CHAT_BASE_URL,
            MINISTRAL_CHAT_MAX_CONTEXT_TOKENS,
        ),
    ]
    results = await asyncio.gather(*checks)
    return {"models": results}


@app.post(
    "/api/chat",
    tags=["chat"],
    summary="Chat Completion",
    response_model=OllamaTextResponseModel,
    openapi_extra=CHAT_OPENAPI_EXTRA,
)
async def api_chat(request: Request) -> Dict[str, Any]:
    body_data = await _read_request_body_as_dict(request)
    if CHAT_DEBUG_LOG:
        logger.info("chat.incoming body=%s", _safe_preview(body_data))

    requested_model = body_data.get("model")
    target = _resolve_chat_target(requested_model)
    model = requested_model or target["public_model"]
    messages: List[Dict[str, Any]] = body_data.get("messages", [])
    # Compatibility: some clients send prompt-like payload to /api/chat.
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
        # Last-resort fallback to prevent hard 400 on empty payloads.
        messages = [{"role": "user", "content": CHAT_EMPTY_FALLBACK_USER_TEXT}]
    messages = _inject_system_language_prompt(messages)
    stream = bool(body_data.get("stream", False))

    if stream:
        raise HTTPException(status_code=501, detail="stream not implemented")

    start_ns = _ns()
    estimated_input_tokens = _estimate_chat_input_tokens(messages)
    resolved_max_tokens = _resolve_max_tokens(body_data, estimated_input_tokens=estimated_input_tokens)
    payload = {
        "model": target["vllm_model"],
        "messages": messages,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolved_max_tokens,
    }
    if DISABLE_THINKING and "qwen" in _normalize_model_name(target["vllm_model"]):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

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

    try:
        data = await _post_json_to(target["base_url"], "/chat/completions", payload)
    except HTTPException as exc:
        if CHAT_DEBUG_LOG:
            logger.error(
                "chat.vllm_error status=%s detail=%s",
                exc.status_code,
                _safe_preview(exc.detail),
            )
        raise

    if CHAT_DEBUG_LOG:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        logger.info(
            "chat.vllm_response finish_reason=%s content_preview=%s reasoning_preview=%s",
            choice.get("finish_reason"),
            _safe_preview(message.get("content")),
            _safe_preview(message.get("reasoning")),
        )

    content = _strip_reasoning_prefix(_extract_chat_text(data))
    return _ollama_response(model, content, start_ns)


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_chat_ui_events(target: Dict[str, str], payload: Dict[str, Any], public_model: str):
    url = f"{target['base_url'].rstrip('/')}/chat/completions"
    total_text = ""

    yield _sse_event({"type": "start", "model": public_model})

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    detail = await resp.aread()
                    yield _sse_event(
                        {
                            "type": "error",
                            "status": resp.status_code,
                            "detail": detail.decode("utf-8", errors="ignore"),
                        }
                    )
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
                    token = delta.get("content")
                    if token is None:
                        continue

                    token_text = str(token)
                    if not token_text:
                        continue
                    total_text += token_text
                    yield _sse_event({"type": "token", "model": public_model, "token": token_text})

        yield _sse_event({"type": "done", "model": public_model, "text": total_text})
    except Exception as exc:
        yield _sse_event({"type": "error", "status": 500, "detail": str(exc)})


@app.post(
    "/api/chat-ui",
    tags=["chat"],
    summary="Chat Completion Stream (UI)",
)
@app.post(
    "/api/chat/ui",
    tags=["chat"],
    summary="Chat Completion Stream (UI Alias)",
)
async def api_chat_ui(request: Request) -> StreamingResponse:
    body_data = await _read_request_body_as_dict(request)
    requested_model = body_data.get("model")
    target = _resolve_chat_target(requested_model)
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

    messages = _inject_system_language_prompt(messages)
    estimated_input_tokens = _estimate_chat_input_tokens(messages)
    resolved_max_tokens = _resolve_max_tokens(body_data, estimated_input_tokens=estimated_input_tokens)

    payload = {
        "model": target["vllm_model"],
        "messages": messages,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": resolved_max_tokens,
        "stream": True,
    }
    if DISABLE_THINKING and "qwen" in _normalize_model_name(target["vllm_model"]):
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
        _stream_chat_ui_events(target, payload, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post(
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
    model = requested_model or target["public_model"]
    prompt: str = str(body_data.get("prompt", ""))
    prompt = f"{_language_instruction()}\n\n{prompt}".strip()
    stream = bool(body_data.get("stream", False))

    if stream:
        raise HTTPException(status_code=501, detail="stream not implemented")

    start_ns = _ns()
    estimated_input_tokens = _estimate_input_tokens_from_text(prompt) + 16
    payload = {
        "model": target["vllm_model"],
        "prompt": prompt,
        "temperature": body_data.get("temperature", 0.7),
        "max_tokens": _resolve_max_tokens(body_data, estimated_input_tokens=estimated_input_tokens),
    }
    data = await _post_json_to(target["base_url"], "/completions", payload)
    content = _strip_reasoning_prefix(data["choices"][0]["text"])
    return _ollama_response(model, content, start_ns)


@app.post(
    "/api/embed",
    tags=["embeddings"],
    summary="Embed Text (Alias)",
    response_model=EmbedResponseModel,
    openapi_extra=EMBED_OPENAPI_EXTRA,
)
@app.post(
    "/api/embeddings",
    tags=["embeddings"],
    summary="Embed Text",
    response_model=EmbedResponseModel,
    openapi_extra=EMBED_OPENAPI_EXTRA,
)
async def api_embed(request: Request) -> Dict[str, Any]:
    body_data = await _read_request_body_as_dict(request)
    if EMBED_DEBUG_LOG:
        logger.info("embed.incoming body=%s", _safe_preview(body_data))

    requested_model = body_data.get("model")
    target = _resolve_embed_target(requested_model)
    model = requested_model or target["public_model"]
    input_data = body_data.get("input")

    # Compatibility: some clients send text in prompt-like fields.
    if input_data is None:
        input_data = body_data.get("prompt") or body_data.get("text")

    if input_data is None and isinstance(body_data.get("message"), dict):
        msg = body_data.get("message") or {}
        input_data = msg.get("content")

    if input_data is None and isinstance(body_data.get("messages"), list):
        merged = []
        for msg in body_data.get("messages") or []:
            if isinstance(msg, dict) and msg.get("content") is not None:
                merged.append(str(msg.get("content")))
        if merged:
            input_data = "\n".join(merged)

    if input_data is None:
        raise HTTPException(status_code=400, detail="input is required")

    start_ns = _ns()
    payload = {
        "model": target["vllm_model"],
        "input": input_data,
    }

    if EMBED_DEBUG_LOG:
        logger.info(
            "embed.adapted model=%s route_model=%s base_url=%s input_type=%s input_preview=%s",
            model,
            target["vllm_model"],
            target["base_url"],
            type(input_data).__name__,
            _safe_preview(input_data),
        )

    try:
        data = await _post_json_to(target["base_url"], "/embeddings", payload)
    except HTTPException as exc:
        if EMBED_DEBUG_LOG:
            logger.error(
                "embed.vllm_error status=%s detail=%s",
                exc.status_code,
                _safe_preview(exc.detail),
            )
        raise

    embeddings = [item.get("embedding", []) for item in data.get("data", [])]
    usage = data.get("usage") or {}

    if EMBED_DEBUG_LOG:
        logger.info(
            "embed.vllm_response vectors=%s first_dim=%s prompt_tokens=%s",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
            usage.get("prompt_tokens", 0),
        )

    return {
        "model": model,
        "embedding": embeddings[0] if embeddings else [],
        "embeddings": embeddings,
        "total_duration": max(0, _ns() - start_ns),
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }
