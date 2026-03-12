#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple


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
        return int(exc.code), latency, str(exc)
    except Exception as exc:
        latency = time.perf_counter() - started
        return 0, latency, str(exc)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight perf smoke for proxy chat/embed endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Proxy base URL")
    parser.add_argument("--chat-model", default="Qwen3.5-9B", help="Public chat model")
    parser.add_argument("--embed-model", default="Qwen3-Embedding-8B", help="Public embeddings model")
    parser.add_argument("--burst-requests", type=int, default=20, help="Requests in burst profile")
    parser.add_argument("--mixed-requests", type=int, default=20, help="Requests in mixed profile")
    parser.add_argument("--concurrency", type=int, default=5, help="Worker count for concurrent profiles")
    args = parser.parse_args()

    chat_url = f"{args.base_url.rstrip('/')}/api/chat"
    embed_url = f"{args.base_url.rstrip('/')}/api/embeddings"

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


if __name__ == "__main__":
    main()
