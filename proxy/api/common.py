import json
import re
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
    TOKEN_CAP_DYNAMIC_MODE,
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
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text") is not None:
                        parts.append(str(item.get("text")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        if isinstance(content, dict):
            text = content.get("text")
            return str(text) if text is not None else ""
        return ""

    total = 0
    for msg in messages:
        total += estimate_input_tokens_from_text(_content_to_text(msg.get("content", "")))
        total += 8
    return total + 16


def analyze_max_tokens_budget(
    body: Dict[str, Any],
    estimated_input_tokens: int = 0,
    max_context_tokens: int | None = None,
    max_tokens_cap: int | None = None,
    min_context_headroom: int | None = None,
    default_max_tokens: int | None = None,
) -> Dict[str, Any]:
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    requested_raw = body.get("max_tokens")
    requested_source = "max_tokens"
    if requested_raw is None:
        requested_raw = options.get("num_predict")
        requested_source = "options.num_predict"

    requested_value = None
    requested_valid = False
    if requested_raw is not None:
        try:
            requested_value = int(requested_raw)
            requested_valid = requested_value >= 1
        except (TypeError, ValueError):
            requested_value = None
            requested_valid = False

    resolved_context = MAX_CONTEXT_TOKENS if max_context_tokens is None else max(1, int(max_context_tokens))
    resolved_cap = MAX_TOKENS_CAP if max_tokens_cap is None else max(1, int(max_tokens_cap))
    resolved_headroom = MIN_CONTEXT_HEADROOM if min_context_headroom is None else max(0, int(min_context_headroom))
    resolved_default = DEFAULT_MAX_TOKENS if default_max_tokens is None else max(1, int(default_max_tokens))

    available_output_tokens = resolved_context - max(0, estimated_input_tokens) - resolved_headroom
    dynamic_cap = max(1, available_output_tokens)
    if TOKEN_CAP_DYNAMIC_MODE:
        hard_cap = dynamic_cap
        effective_cap = dynamic_cap
        cap_mode = "dynamic"
    else:
        hard_cap = max(1, min(resolved_cap, dynamic_cap))
        effective_cap = resolved_cap
        cap_mode = "static"

    if not requested_valid:
        resolved = min(resolved_default, hard_cap)
    else:
        resolved = min(int(requested_value), hard_cap)

    return {
        "estimated_input_tokens": max(0, int(estimated_input_tokens)),
        "max_context_tokens": resolved_context,
        "max_tokens_cap": effective_cap,
        "min_context_headroom": resolved_headroom,
        "default_max_tokens": resolved_default,
        "available_output_tokens": max(0, int(available_output_tokens)),
        "requested_output_tokens": int(requested_value) if requested_valid else None,
        "requested_source": requested_source if requested_raw is not None else None,
        "had_requested_output_tokens": requested_raw is not None,
        "resolved_max_tokens": int(resolved),
        "hard_cap": int(hard_cap),
        "cap_mode": cap_mode,
    }


def resolve_max_tokens(
    body: Dict[str, Any],
    estimated_input_tokens: int = 0,
    max_context_tokens: int | None = None,
    max_tokens_cap: int | None = None,
    min_context_headroom: int | None = None,
    default_max_tokens: int | None = None,
) -> int:
    budget = analyze_max_tokens_budget(
        body,
        estimated_input_tokens=estimated_input_tokens,
        max_context_tokens=max_context_tokens,
        max_tokens_cap=max_tokens_cap,
        min_context_headroom=min_context_headroom,
        default_max_tokens=default_max_tokens,
    )
    return int(budget["resolved_max_tokens"])


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


def strip_reasoning_artifacts(text: str) -> str:
    if not text:
        return text
    # Remove explicit reasoning blocks some models emit in completion mode.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return strip_reasoning_prefix(cleaned)


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
