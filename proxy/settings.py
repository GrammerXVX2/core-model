import os

import httpx

VLLM_BASE_URLS = [
    url.strip().rstrip("/")
    for url in os.getenv("VLLM_BASE_URLS", os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).split(",")
    if url.strip()
]

# Model routing is DB-only (model_registry_checks).
# Model add/update/delete must be done via DB or /api/models CRUD endpoints.

DEFAULT_RESPONSE_LANGUAGE = os.getenv("DEFAULT_RESPONSE_LANGUAGE", "ru")
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1") == "1"
DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_DEFAULT_MAX_TOKENS", "32768"))
MAX_TOKENS_CAP = int(os.getenv("VLLM_MAX_TOKENS_CAP", "16384"))
MAX_CONTEXT_TOKENS = int(os.getenv("VLLM_MAX_CONTEXT_TOKENS", "4096"))
MIN_CONTEXT_HEADROOM = int(os.getenv("VLLM_MIN_CONTEXT_HEADROOM", "256"))
TOKEN_CAP_DYNAMIC_MODE = os.getenv("TOKEN_CAP_DYNAMIC_MODE", "0") == "1"
TOKEN_BUDGET_STRICT_MODE = os.getenv("TOKEN_BUDGET_STRICT_MODE", "0") == "1"
CHAT_DEBUG_LOG = os.getenv("CHAT_DEBUG_LOG", "1") == "1"
EMBED_DEBUG_LOG = os.getenv("EMBED_DEBUG_LOG", "1") == "1"
LOG_TEXT_PREVIEW_CHARS = int(os.getenv("LOG_TEXT_PREVIEW_CHARS", "500"))
REQUEST_DEBUG_LOG = os.getenv("REQUEST_DEBUG_LOG", "1") == "1"
CHAT_EMPTY_FALLBACK_USER_TEXT = os.getenv("CHAT_EMPTY_FALLBACK_USER_TEXT", "")
LANGUAGE_INSTRUCTION_RU = os.getenv(
    "LANGUAGE_INSTRUCTION_RU",
    "Отвечай только на русском языке.",
)
LANGUAGE_INSTRUCTION_DEFAULT = os.getenv(
    "LANGUAGE_INSTRUCTION_DEFAULT",
    "Answer in the configured default language.",
)
REASONING_PREFIX_MARKERS = [
    m.strip()
    for m in os.getenv(
        "REASONING_PREFIX_MARKERS",
        "Thinking Process:|Reasoning:|Ход рассуждений:",
    ).split("|")
    if m.strip()
]

UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "10"))
UPSTREAM_MAX_CONNECTIONS = int(os.getenv("UPSTREAM_MAX_CONNECTIONS", "200"))
UPSTREAM_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("UPSTREAM_MAX_KEEPALIVE_CONNECTIONS", "50"))
UPSTREAM_KEEPALIVE_EXPIRY_SECONDS = float(os.getenv("UPSTREAM_KEEPALIVE_EXPIRY_SECONDS", "30"))
UPSTREAM_RETRY_ATTEMPTS = int(os.getenv("UPSTREAM_RETRY_ATTEMPTS", "2"))
UPSTREAM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("UPSTREAM_RETRY_BASE_DELAY_SECONDS", "0.2"))
UPSTREAM_RETRY_JITTER_SECONDS = float(os.getenv("UPSTREAM_RETRY_JITTER_SECONDS", "0.1"))
MODEL_STATUS_POLL_INTERVAL_SECONDS = int(os.getenv("MODEL_STATUS_POLL_INTERVAL_SECONDS", "60"))
MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS = int(os.getenv("MODEL_STATUS_POLL_INTERVAL_ERROR_SECONDS", "15"))
MODEL_REGISTRY_ENABLED = os.getenv("MODEL_REGISTRY_ENABLED", "0") == "1"
MODEL_REGISTRY_ROUTING_ENABLED = os.getenv("MODEL_REGISTRY_ROUTING_ENABLED", "0") == "1"
MODEL_REGISTRY_SYNC_FROM_ENV = os.getenv("MODEL_REGISTRY_SYNC_FROM_ENV", "1") == "1"
MODEL_REGISTRY_DB_DSN = os.getenv("MODEL_REGISTRY_DB_DSN", "")
UPSTREAM_HTTP_TIMEOUT = httpx.Timeout(timeout=UPSTREAM_TIMEOUT_SECONDS)
UPSTREAM_HTTP_LIMITS = httpx.Limits(
    max_connections=UPSTREAM_MAX_CONNECTIONS,
    max_keepalive_connections=UPSTREAM_MAX_KEEPALIVE_CONNECTIONS,
    keepalive_expiry=UPSTREAM_KEEPALIVE_EXPIRY_SECONDS,
)
