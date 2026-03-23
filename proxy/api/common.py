import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from settings import (
    DEFAULT_RESPONSE_LANGUAGE,
    LANGUAGE_INSTRUCTION_DEFAULT,
    LANGUAGE_INSTRUCTION_RU,
    LOG_TEXT_PREVIEW_CHARS,
    MAX_CONTEXT_TOKENS,
    MAX_TOKENS_CAP,
    MIN_CONTEXT_HEADROOM,
    DEFAULT_MAX_TOKENS,
    REASONING_PREFIX_MARKERS,
)


def truncate_text(value: str, max_chars: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated {len(value) - max_chars} chars>"


def safe_preview(value: Any) -> Any:
    if isinstance(value, str):
        return truncate_text(value)
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], str):
            return [truncate_text(v) for v in value[:5]]
        return f"<list len={len(value)}>"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in list(value.items())[:20]:
            if "token" in k.lower() or "authorization" in k.lower() or "password" in k.lower():
                out[k] = "***"
            else:
                out[k] = safe_preview(v)
        return out
    return value


def language_instruction() -> str:
    lang = DEFAULT_RESPONSE_LANGUAGE.lower()
    if lang in {"ru", "russian", "рус", "русский"}:
        return LANGUAGE_INSTRUCTION_RU
    return LANGUAGE_INSTRUCTION_DEFAULT


def inject_system_language_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    instruction = language_instruction()
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        if instruction in existing:
            return messages
        patched = messages.copy()
        patched[0] = {"role": "system", "content": f"{existing}\n{instruction}".strip()}
        return patched
    return [{"role": "system", "content": instruction}, *messages]


def estimate_input_tokens_from_text(text: str) -> int:
    # Rough heuristic: 1 token ~= 4 chars for mixed RU/EN text.
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_chat_input_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_input_tokens_from_text(str(msg.get("content", "")))
        total += 8
    return total + 16


def resolve_max_tokens(body: Dict[str, Any], estimated_input_tokens: int = 0) -> int:
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    requested = body.get("max_tokens")
    if requested is None:
        requested = options.get("num_predict")

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


def extract_chat_text(data: Dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    content = message.get("content")
    if isinstance(content, str) and content:
        return content

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    return ""


def extract_finish_reason(data: Dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    reason = choice.get("finish_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "stop"


def strip_reasoning_prefix(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for marker in REASONING_PREFIX_MARKERS:
        if cleaned.startswith(marker):
            parts = cleaned.split("\n\n", 1)
            cleaned = parts[1] if len(parts) == 2 else cleaned
    return cleaned.strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ns() -> int:
    return time.perf_counter_ns()


def ollama_response(model: str, content: str, start_ns: int, load_ns: int = 0, done_reason: str = "stop") -> Dict[str, Any]:
    end_ns = ns()
    total_ns = max(0, end_ns - start_ns)
    return {
        "model": model,
        "created_at": now_iso(),
        "response": content,
        "done": True,
        "done_reason": done_reason,
        "total_duration": total_ns,
        "load_duration": load_ns,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


def sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
