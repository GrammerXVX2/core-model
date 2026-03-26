#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from typing import List


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _model_key(public_model: str, model_type: str, base_url: str) -> str:
    return f"{_normalize(public_model)}:{model_type}:{_normalize(base_url)}"


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _bool_literal(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _run(cmd: List[str]) -> None:
    completed = subprocess.run(cmd, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick upsert into model_registry_checks via docker compose + psql.")
    parser.add_argument("--public-model", required=True, help="Public model alias exposed by proxy")
    parser.add_argument("--backend-model", required=True, help="Upstream model id used in /chat/completions")
    parser.add_argument("--base-url", required=True, help="Upstream base URL, e.g. http://10.77.163.200:8000/v1")
    parser.add_argument("--type", choices=["chat", "embeddings"], default="chat", help="Model endpoint type")
    parser.add_argument("--max-context", type=int, required=True, help="Model context window")
    parser.add_argument("--default-max", type=int, default=None, help="Default output max_tokens")
    parser.add_argument("--max-cap", type=int, default=None, help="Static cap (ignored when TOKEN_CAP_DYNAMIC_MODE=1)")
    parser.add_argument("--headroom", type=int, default=256, help="Context headroom reserve")
    parser.add_argument("--stream-supported", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reasoning-supported", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aliases", default="", help="Comma-separated aliases; public/backend names are auto-added")
    parser.add_argument("--docker-service", default="postgres", help="Compose service name with psql")
    parser.add_argument("--db-user", default=os.getenv("MODEL_REGISTRY_POSTGRES_USER", "postgres"))
    parser.add_argument("--db-name", default=os.getenv("MODEL_REGISTRY_POSTGRES_DB", "model_registry"))
    parser.add_argument("--dry-run", action="store_true", help="Print SQL and exit")
    args = parser.parse_args()

    default_max = args.default_max if args.default_max is not None else args.max_context
    max_cap = args.max_cap if args.max_cap is not None else args.max_context
    stream_supported = args.stream_supported
    if stream_supported is None:
        stream_supported = args.type == "chat"

    aliases = {args.public_model.strip(), args.backend_model.strip()}
    if args.aliases.strip():
        aliases.update(a.strip() for a in args.aliases.split(",") if a.strip())
    aliases_json = json.dumps(sorted(aliases), ensure_ascii=False)

    model_key = _model_key(args.public_model, args.type, args.base_url.rstrip("/"))

    sql = f"""
INSERT INTO model_registry_checks (
    model_key,
    public_model,
    vllm_model,
    model_type,
    base_url,
    max_context_tokens,
    default_max_tokens,
    max_tokens_cap,
    min_context_headroom,
    stream_supported,
    reasoning_supported,
    aliases_json,
    is_enabled,
    updated_at
) VALUES (
    '{_sql_escape(model_key)}',
    '{_sql_escape(args.public_model)}',
    '{_sql_escape(args.backend_model)}',
    '{_sql_escape(args.type)}',
    '{_sql_escape(args.base_url.rstrip('/'))}',
    {int(args.max_context)},
    {int(default_max)},
    {int(max_cap)},
    {int(args.headroom)},
    {_bool_literal(bool(stream_supported))},
    {_bool_literal(bool(args.reasoning_supported))},
    '{_sql_escape(aliases_json)}'::jsonb,
    TRUE,
    NOW()
)
ON CONFLICT (vllm_model, base_url, model_type) DO UPDATE SET
    model_key = EXCLUDED.model_key,
    public_model = EXCLUDED.public_model,
    max_context_tokens = EXCLUDED.max_context_tokens,
    default_max_tokens = EXCLUDED.default_max_tokens,
    max_tokens_cap = EXCLUDED.max_tokens_cap,
    min_context_headroom = EXCLUDED.min_context_headroom,
    stream_supported = EXCLUDED.stream_supported,
    reasoning_supported = EXCLUDED.reasoning_supported,
    aliases_json = EXCLUDED.aliases_json,
    is_enabled = TRUE,
    updated_at = NOW();

SELECT public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text
FROM model_registry_checks
WHERE vllm_model = '{_sql_escape(args.backend_model)}' AND base_url = '{_sql_escape(args.base_url.rstrip('/'))}' AND model_type = '{_sql_escape(args.type)}';
""".strip()

    if args.dry_run:
        print(sql)
        return

    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        args.docker_service,
        "psql",
        "-U",
        args.db_user,
        "-d",
        args.db_name,
        "-c",
        sql,
    ]
    print("Running upsert via:", " ".join(cmd[:-1]), file=sys.stderr)
    _run(cmd)


if __name__ == "__main__":
    main()
