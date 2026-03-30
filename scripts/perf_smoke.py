#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple


def _post_json(url: str, payload: Dict) -> Tuple[int, float, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            latency = time.perf_counter() - started
            return int(resp.status), latency, ""
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - started
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(exc)
        return int(exc.code), latency, detail
    except Exception as exc:
        latency = time.perf_counter() - started
        return 0, latency, str(exc)


def _post_json_with_retry(url: str, payload: Dict, attempts: int = 3, backoff_seconds: float = 0.4) -> Tuple[int, float, str]:
    last: Tuple[int, float, str] = (0, 0.0, "no attempts")
    for i in range(max(1, attempts)):
        last = _post_json(url, payload)
        if last[0] != 0:
            return last
        if i < attempts - 1:
            time.sleep(backoff_seconds * (i + 1))
    return last


def _get_json(url: str) -> Tuple[int, Dict[str, Any], str]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(body) if body else {}
            return int(resp.status), data, ""
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(exc)
        return int(exc.code), {}, detail
    except Exception as exc:
        return 0, {}, str(exc)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int((len(sorted_vals) - 1) * p)
    return sorted_vals[index]


def _print_summary(title: str, results: List[Tuple[int, float, str]]) -> None:
    latencies = [lat for _, lat, _ in results]
    errors = [r for r in results if r[0] != 200]
    avg = statistics.mean(latencies) if latencies else 0.0
    p95 = _percentile(latencies, 0.95)
    max_lat = max(latencies) if latencies else 0.0
    print(f"[{title}]")
    print(f"total={len(results)} errors={len(errors)} avg_ms={avg * 1000:.2f} p95_ms={p95 * 1000:.2f} max_ms={max_lat * 1000:.2f}")
    if errors:
        samples = errors[:3]
        for code, _, err in samples:
            print(f"error_sample status={code} detail={err[:180]}")
    print()


def _is_infra_error(code: int, detail: str) -> bool:
    if code == 0:
        return True
    detail_lower = (detail or "").lower()
    if code in {502, 503, 504}:
        markers = [
            "model unavailable",
            "connection error",
            "name resolution",
            "timeout",
            "temporarily unavailable",
            "refused",
            "broken pipe",
            "connection reset",
        ]
        return any(marker in detail_lower for marker in markers)
    return False


def _count_error_types(results: List[Tuple[int, float, str]]) -> Tuple[int, int]:
    infra = 0
    app = 0
    for code, _latency, detail in results:
        if code == 200:
            continue
        if _is_infra_error(code, detail):
            infra += 1
        else:
            app += 1
    return infra, app


def _print_check_result(name: str, expected: int, actual: int, detail: str) -> bool:
    ok = expected == actual
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: expected={expected} actual={actual}")
    if detail:
        print(f"  detail={detail[:240]}")
    return ok


def _is_available_status(raw: str) -> bool:
    normalized = (raw or "").strip().lower()
    return normalized in {"доступен", "available", "ok", "up", "healthy"}


def _run_preflight(base_url: str, required_models: List[str], strict: bool) -> bool:
    url = f"{base_url.rstrip('/')}/api/models"
    code, data, detail = _get_json(url)
    print("[preflight]")
    if code != 200:
        print(f"[FAIL] models_endpoint: expected=200 actual={code}")
        if detail:
            print(f"  detail={detail[:240]}")
        print()
        return False

    snapshot = data.get("models") or []
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in snapshot:
        name = str(row.get("model") or "").strip()
        if name:
            by_name[name] = row

    ok = True
    for model_name in required_models:
        row = by_name.get(model_name)
        if row is None:
            print(f"[FAIL] model_present {model_name}: missing in /api/models")
            ok = False
            continue
        status = str(row.get("status") or "")
        available = _is_available_status(status)
        level = "PASS" if available else ("FAIL" if strict else "WARN")
        print(f"[{level}] model_status {model_name}: {status or 'unknown'}")
        if strict and not available:
            ok = False

    print()
    return ok


def _chat_payload(model: str) -> Dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Кратко опиши назначение этого smoke-теста."}],
        "max_tokens": 24,
        "temperature": 0,
    }


def _embed_payload(model: str) -> Dict:
    return {
        "model": model,
        "input": ["smoke test one", "smoke test two"],
    }


def _run_burst(url: str, payload: Dict, total: int, concurrency: int) -> List[Tuple[int, float, str]]:
    results: List[Tuple[int, float, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_post_json, url, payload) for _ in range(total)]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_capability_checks(
    base_url: str,
    non_llama_chat_model: str,
    embed_model: str,
    attempts: int = 6,
    backoff_seconds: float = 0.6,
) -> bool:
    user_msg = [{"role": "user", "content": "ping"}]
    cases = [
        (
            "chat_reasoning_rejected_for_non_llama",
            f"{base_url.rstrip('/')}/api/chat",
            {"model": non_llama_chat_model, "messages": user_msg, "reasoning": True},
            400,
        ),
        (
            "chat_ui_reasoning_rejected_for_non_llama",
            f"{base_url.rstrip('/')}/api/chat-ui",
            {"model": non_llama_chat_model, "messages": user_msg, "reasoning": True},
            400,
        ),
        (
            "chat_ui_embed_model_rejected",
            f"{base_url.rstrip('/')}/api/chat-ui",
            {"model": embed_model, "messages": user_msg, "stream": True},
            400,
        ),
    ]

    print("[capability_checks]")
    all_ok = True
    for name, url, payload, expected in cases:
        code, _latency, detail = _post_json_with_retry(
            url,
            payload,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
        )
        all_ok = _print_check_result(name, expected, code, detail) and all_ok
    print()
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight perf smoke for proxy chat/embed endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Proxy base URL")
    parser.add_argument("--chat-model", default="Qwen3.5-122B-A10B-FP8", help="Public chat model")
    parser.add_argument("--embed-model", default="qwen-embed-4b-tei", help="Public embeddings model")
    parser.add_argument("--burst-requests", type=int, default=20, help="Requests in burst profile")
    parser.add_argument("--mixed-requests", type=int, default=20, help="Requests in mixed profile")
    parser.add_argument("--concurrency", type=int, default=5, help="Worker count for concurrent profiles")
    parser.add_argument(
        "--non-llama-chat-model",
        default="Qwen3.5-122B-A10B-FP8",
        help="Chat model expected to reject reasoning toggle (used in capability checks)",
    )
    parser.add_argument(
        "--capability-checks",
        action="store_true",
        help="Run fail-fast capability checks (expected HTTP statuses)",
    )
    parser.add_argument(
        "--guards-only",
        action="store_true",
        help="Run preflight/capability checks only; skip perf burst profiles",
    )
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run /api/models preflight before smoke profiles",
    )
    parser.add_argument(
        "--preflight-strict",
        action="store_true",
        help="Fail if required models are present but unavailable in preflight",
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=2.0,
        help="Sleep before running checks to avoid post-restart transient resets",
    )
    parser.add_argument(
        "--exit-on-infra-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, full smoke exits non-zero on infrastructure errors (502/503/504/connection failures)",
    )
    args = parser.parse_args()

    if args.startup_wait_seconds > 0:
        time.sleep(args.startup_wait_seconds)

    if args.guards_only:
        args.capability_checks = True

    chat_url = f"{args.base_url.rstrip('/')}/api/chat"
    embed_url = f"{args.base_url.rstrip('/')}/api/embeddings"

    if args.preflight:
        required = [args.chat_model, args.embed_model]
        preflight_ok = _run_preflight(args.base_url, required, strict=args.preflight_strict)
        if not preflight_ok:
            raise SystemExit(1)

    if not args.guards_only:
        warm_results = [
            _post_json(chat_url, _chat_payload(args.chat_model)),
            _post_json(embed_url, _embed_payload(args.embed_model)),
        ]
        _print_summary("warm_single_request", warm_results)

        burst_results = _run_burst(chat_url, _chat_payload(args.chat_model), args.burst_requests, args.concurrency)
        _print_summary("short_burst_concurrency", burst_results)

        mixed_results: List[Tuple[int, float, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = []
            for i in range(args.mixed_requests):
                if i % 2 == 0:
                    futures.append(pool.submit(_post_json, chat_url, _chat_payload(args.chat_model)))
                else:
                    futures.append(pool.submit(_post_json, embed_url, _embed_payload(args.embed_model)))
            for future in as_completed(futures):
                mixed_results.append(future.result())
        _print_summary("mixed_chat_embed", mixed_results)

        combined = [*warm_results, *burst_results, *mixed_results]
        infra_errors, app_errors = _count_error_types(combined)
        if app_errors:
            print(f"[FAIL] non_infra_errors={app_errors} infra_errors={infra_errors}")
            raise SystemExit(1)
        if infra_errors:
            level = "FAIL" if args.exit_on_infra_errors else "WARN"
            print(f"[{level}] infra_errors={infra_errors} non_infra_errors=0")
            if args.exit_on_infra_errors:
                raise SystemExit(1)
        else:
            print("[PASS] full_smoke_errors=0")

    if args.capability_checks:
        ok = _run_capability_checks(args.base_url, args.non_llama_chat_model, args.embed_model)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
