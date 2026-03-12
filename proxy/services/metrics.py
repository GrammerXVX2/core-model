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


def export_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
