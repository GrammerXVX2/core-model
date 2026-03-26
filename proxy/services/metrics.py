import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_LATENCY_SECONDS = Histogram(
    "proxy_http_request_duration_seconds",
    "HTTP request latency for proxy endpoints.",
    ["endpoint", "method", "status"],
)

UPSTREAM_LATENCY_SECONDS = Histogram(
    "proxy_upstream_request_duration_seconds",
    "Upstream request latency by route and status.",
    ["route", "status"],
)

UPSTREAM_ERRORS_TOTAL = Counter(
    "proxy_upstream_errors_total",
    "Count of upstream errors by route and status.",
    ["route", "status"],
)

MODEL_AVAILABILITY = Gauge(
    "proxy_model_availability",
    "Model availability flag (1 available, 0 unavailable).",
    ["model", "type", "base_url"],
)

MODEL_REGISTRY_DB_UP = Gauge(
    "proxy_model_registry_db_up",
    "Model registry DB connectivity flag (1 connected, 0 fallback).",
)

MODEL_REGISTRY_SYNC_COUNTS = Gauge(
    "proxy_model_registry_sync_counts",
    "Last model registry sync counters by kind.",
    ["kind"],
)

MODEL_REGISTRY_FALLBACK_TOTAL = Counter(
    "proxy_model_registry_fallback_total",
    "Model registry fallback count by reason.",
    ["reason"],
)


def now_seconds() -> float:
    return time.perf_counter()


def observe_request_latency(endpoint: str, method: str, status: int, started_at: float) -> None:
    REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, method=method, status=str(status)).observe(max(0.0, time.perf_counter() - started_at))


def observe_upstream_latency(route: str, status: int, started_at: float) -> None:
    UPSTREAM_LATENCY_SECONDS.labels(route=route, status=str(status)).observe(max(0.0, time.perf_counter() - started_at))


def inc_upstream_error(route: str, status: int) -> None:
    UPSTREAM_ERRORS_TOTAL.labels(route=route, status=str(status)).inc()


def set_model_availability(model: str, model_type: str, base_url: str, status: str) -> None:
    MODEL_AVAILABILITY.labels(model=model, type=model_type, base_url=base_url).set(1 if status == "доступен" else 0)


def set_model_registry_db_up(is_up: bool) -> None:
    MODEL_REGISTRY_DB_UP.set(1 if is_up else 0)


def set_model_registry_sync_counts(
    env_total: int,
    db_total_before: int,
    inserted: int,
    updated: int,
    unchanged: int,
    removed: int,
) -> None:
    MODEL_REGISTRY_SYNC_COUNTS.labels(kind="env_total").set(max(0, env_total))
    MODEL_REGISTRY_SYNC_COUNTS.labels(kind="db_total_before").set(max(0, db_total_before))
    MODEL_REGISTRY_SYNC_COUNTS.labels(kind="inserted").set(max(0, inserted))
    MODEL_REGISTRY_SYNC_COUNTS.labels(kind="updated").set(max(0, updated))
    MODEL_REGISTRY_SYNC_COUNTS.labels(kind="unchanged").set(max(0, unchanged))
    MODEL_REGISTRY_SYNC_COUNTS.labels(kind="removed").set(max(0, removed))


def inc_model_registry_fallback(reason: str) -> None:
    MODEL_REGISTRY_FALLBACK_TOTAL.labels(reason=reason or "unknown").inc()


def export_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
