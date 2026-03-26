import json
import logging
from typing import Any, Dict, List

from settings import MODEL_REGISTRY_DB_DSN, MODEL_REGISTRY_ENABLED, MODEL_REGISTRY_SYNC_FROM_ENV
from services.metrics import (
    inc_model_registry_fallback,
    set_model_registry_db_up,
    set_model_registry_sync_counts,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:
    psycopg = None
    dict_row = None

logger = logging.getLogger("uvicorn.error")
_WARNED_SYNC_FROM_ENV = False


def _normalize_model_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _open_conn() -> Any:
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    if not MODEL_REGISTRY_DB_DSN:
        raise RuntimeError("MODEL_REGISTRY_DB_DSN is empty")
    return psycopg.connect(MODEL_REGISTRY_DB_DSN, row_factory=dict_row)


def _init_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(hashtext('model_registry_checks_schema_v1'))")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry_checks (
                id BIGSERIAL PRIMARY KEY,
                model_key TEXT NOT NULL UNIQUE,
                public_model TEXT NOT NULL,
                vllm_model TEXT NOT NULL,
                model_type TEXT NOT NULL,
                base_url TEXT NOT NULL,
                max_context_tokens INTEGER NOT NULL,
                default_max_tokens INTEGER NOT NULL DEFAULT 2048,
                max_tokens_cap INTEGER NOT NULL DEFAULT 16384,
                min_context_headroom INTEGER NOT NULL DEFAULT 256,
                stream_supported BOOLEAN NOT NULL DEFAULT FALSE,
                reasoning_supported BOOLEAN NOT NULL DEFAULT FALSE,
                aliases_json JSONB NOT NULL,
                is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute("ALTER TABLE model_registry_checks ADD COLUMN IF NOT EXISTS default_max_tokens INTEGER NOT NULL DEFAULT 2048")
        cur.execute("ALTER TABLE model_registry_checks ADD COLUMN IF NOT EXISTS max_tokens_cap INTEGER NOT NULL DEFAULT 16384")
        cur.execute("ALTER TABLE model_registry_checks ADD COLUMN IF NOT EXISTS min_context_headroom INTEGER NOT NULL DEFAULT 256")
        cur.execute("ALTER TABLE model_registry_checks ADD COLUMN IF NOT EXISTS stream_supported BOOLEAN NOT NULL DEFAULT FALSE")
        cur.execute("ALTER TABLE model_registry_checks ADD COLUMN IF NOT EXISTS reasoning_supported BOOLEAN NOT NULL DEFAULT FALSE")
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_unique_target
            ON model_registry_checks(vllm_model, base_url, model_type)
            """
        )
        cur.execute("SELECT pg_advisory_unlock(hashtext('model_registry_checks_schema_v1'))")
    conn.commit()


def _model_key(check: Dict[str, Any]) -> str:
    model = _normalize_model_name(str(check.get("public_model") or check.get("vllm_model") or "model"))
    model_type = str(check.get("type") or "chat")
    base_url = str(check.get("base_url") or "").rstrip("/")
    return f"{model}:{model_type}:{_normalize_model_name(base_url)}"


def _sync_payload(check: Dict[str, Any]) -> Dict[str, Any]:
    aliases = sorted({str(a).strip() for a in (check.get("aliases") or set()) if str(a).strip()})
    return {
        "model_key": _model_key(check),
        "public_model": str(check.get("public_model") or ""),
        "vllm_model": str(check.get("vllm_model") or ""),
        "model_type": str(check.get("type") or "chat"),
        "base_url": str(check.get("base_url") or "").rstrip("/"),
        "max_context_tokens": int(check.get("max_context_tokens") or 0),
        "default_max_tokens": int(check.get("default_max_tokens") or 0),
        "max_tokens_cap": int(check.get("max_tokens_cap") or 0),
        "min_context_headroom": int(check.get("min_context_headroom") or 0),
        "stream_supported": bool(check.get("stream_supported", False)),
        "reasoning_supported": bool(check.get("reasoning_supported", False)),
        "aliases_json": json.dumps(aliases, ensure_ascii=False),
    }


def _payload_changed(existing: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    return any(
        [
            str(existing.get("public_model") or "") != payload["public_model"],
            str(existing.get("vllm_model") or "") != payload["vllm_model"],
            str(existing.get("model_type") or "") != payload["model_type"],
            str(existing.get("base_url") or "").rstrip("/") != payload["base_url"],
            int(existing.get("max_context_tokens") or 0) != payload["max_context_tokens"],
            int(existing.get("default_max_tokens") or 0) != payload["default_max_tokens"],
            int(existing.get("max_tokens_cap") or 0) != payload["max_tokens_cap"],
            int(existing.get("min_context_headroom") or 0) != payload["min_context_headroom"],
            bool(existing.get("stream_supported") or False) != payload["stream_supported"],
            bool(existing.get("reasoning_supported") or False) != payload["reasoning_supported"],
            str(existing.get("aliases_json") or "") != payload["aliases_json"],
        ]
    )


def _load_existing_rows(conn: Any) -> Dict[str, Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_key, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json
            FROM model_registry_checks
            """
        )
        rows = cur.fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[str(row.get("model_key") or "")] = row
    return out


def _upsert_from_env(conn: Any, env_checks: List[Dict[str, Any]]) -> Dict[str, int]:
    existing = _load_existing_rows(conn)
    inserted = 0
    updated = 0
    unchanged = 0
    seen_keys: set[str] = set()

    with conn.cursor() as cur:
        for check in env_checks:
            payload = _sync_payload(check)
            model_key = payload["model_key"]
            seen_keys.add(model_key)

            prev = existing.get(model_key)
            if prev is None:
                inserted += 1
            elif _payload_changed(prev, payload):
                updated += 1
            else:
                unchanged += 1

            cur.execute(
                """
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
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, TRUE, NOW())
                ON CONFLICT(model_key) DO UPDATE SET
                    public_model=EXCLUDED.public_model,
                    vllm_model=EXCLUDED.vllm_model,
                    model_type=EXCLUDED.model_type,
                    base_url=EXCLUDED.base_url,
                    max_context_tokens=EXCLUDED.max_context_tokens,
                    default_max_tokens=EXCLUDED.default_max_tokens,
                    max_tokens_cap=EXCLUDED.max_tokens_cap,
                    min_context_headroom=EXCLUDED.min_context_headroom,
                    stream_supported=EXCLUDED.stream_supported,
                    reasoning_supported=EXCLUDED.reasoning_supported,
                    aliases_json=EXCLUDED.aliases_json,
                    updated_at=NOW()
                """,
                (
                    payload["model_key"],
                    payload["public_model"],
                    payload["vllm_model"],
                    payload["model_type"],
                    payload["base_url"],
                    payload["max_context_tokens"],
                    payload["default_max_tokens"],
                    payload["max_tokens_cap"],
                    payload["min_context_headroom"],
                    payload["stream_supported"],
                    payload["reasoning_supported"],
                    payload["aliases_json"],
                ),
            )
    conn.commit()
    removed = len({k for k in existing.keys() if k not in seen_keys})
    return {
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "removed": removed,
        "env_total": len(env_checks),
        "db_total_before": len(existing),
    }


def _read_checks(conn: Any) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json
            FROM model_registry_checks
            WHERE is_enabled = TRUE
            ORDER BY model_type, public_model
            """
        )
        rows = cur.fetchall()

    checks: List[Dict[str, Any]] = []
    for row in rows:
        aliases_raw = str(row.get("aliases_json") or "[]")
        try:
            aliases = set(json.loads(aliases_raw))
        except Exception:
            aliases = {str(row.get("public_model") or ""), str(row.get("vllm_model") or "")}

        checks.append(
            {
                "id": int(row.get("id") or 0),
                "public_model": str(row.get("public_model") or ""),
                "vllm_model": str(row.get("vllm_model") or ""),
                "type": str(row.get("model_type") or "chat"),
                "base_url": str(row.get("base_url") or "").rstrip("/"),
                "max_context_tokens": int(row.get("max_context_tokens") or 0),
                "default_max_tokens": int(row.get("default_max_tokens") or 0),
                "max_tokens_cap": int(row.get("max_tokens_cap") or 0),
                "min_context_headroom": int(row.get("min_context_headroom") or 0),
                "stream_supported": bool(row.get("stream_supported") or False),
                "reasoning_supported": bool(row.get("reasoning_supported") or False),
                "aliases": aliases,
            }
        )
    return checks


def _row_to_registry_item(row: Dict[str, Any]) -> Dict[str, Any]:
    aliases_raw = str(row.get("aliases_json") or "[]")
    try:
        aliases = sorted({str(a).strip() for a in json.loads(aliases_raw) if str(a).strip()})
    except Exception:
        aliases = sorted({str(row.get("public_model") or ""), str(row.get("vllm_model") or "")})

    return {
        "id": int(row.get("id") or 0),
        "public_model": str(row.get("public_model") or ""),
        "vllm_model": str(row.get("vllm_model") or ""),
        "model_type": str(row.get("model_type") or "chat"),
        "base_url": str(row.get("base_url") or "").rstrip("/"),
        "max_context_tokens": int(row.get("max_context_tokens") or 0),
        "default_max_tokens": int(row.get("default_max_tokens") or 0),
        "max_tokens_cap": int(row.get("max_tokens_cap") or 0),
        "min_context_headroom": int(row.get("min_context_headroom") or 0),
        "stream_supported": bool(row.get("stream_supported") or False),
        "reasoning_supported": bool(row.get("reasoning_supported") or False),
        "aliases": aliases,
        "is_enabled": bool(row.get("is_enabled") if row.get("is_enabled") is not None else True),
    }


def _insert_registry_check(conn: Any, check: Dict[str, Any]) -> Dict[str, Any]:
    payload = _sync_payload(check)
    with conn.cursor() as cur:
        cur.execute(
            """
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
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, TRUE, NOW())
            RETURNING id, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json, is_enabled
            """,
            (
                payload["model_key"],
                payload["public_model"],
                payload["vllm_model"],
                payload["model_type"],
                payload["base_url"],
                payload["max_context_tokens"],
                payload["default_max_tokens"],
                payload["max_tokens_cap"],
                payload["min_context_headroom"],
                payload["stream_supported"],
                payload["reasoning_supported"],
                payload["aliases_json"],
            ),
        )
        row = cur.fetchone() or {}
    conn.commit()
    return _row_to_registry_item(row)


def _get_registry_check_by_id(conn: Any, model_id: int) -> Dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json, is_enabled
            FROM model_registry_checks
            WHERE id = %s
            """,
            (int(model_id),),
        )
        row = cur.fetchone()
    if not row:
        return None
    return _row_to_registry_item(row)


def _update_registry_check_by_id(conn: Any, model_id: int, check: Dict[str, Any]) -> Dict[str, Any] | None:
    payload = _sync_payload(check)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE model_registry_checks
            SET model_key = %s,
                public_model = %s,
                vllm_model = %s,
                model_type = %s,
                base_url = %s,
                max_context_tokens = %s,
                default_max_tokens = %s,
                max_tokens_cap = %s,
                min_context_headroom = %s,
                stream_supported = %s,
                reasoning_supported = %s,
                aliases_json = %s::jsonb,
                is_enabled = TRUE,
                updated_at = NOW()
            WHERE id = %s
            RETURNING id, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json, is_enabled
            """,
            (
                payload["model_key"],
                payload["public_model"],
                payload["vllm_model"],
                payload["model_type"],
                payload["base_url"],
                payload["max_context_tokens"],
                payload["default_max_tokens"],
                payload["max_tokens_cap"],
                payload["min_context_headroom"],
                payload["stream_supported"],
                payload["reasoning_supported"],
                payload["aliases_json"],
                int(model_id),
            ),
        )
        row = cur.fetchone()
    if not row:
        conn.rollback()
        return None
    conn.commit()
    return _row_to_registry_item(row)


def _disable_registry_check_by_id(conn: Any, model_id: int) -> Dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE model_registry_checks
            SET is_enabled = FALSE,
                updated_at = NOW()
            WHERE id = %s
            RETURNING id, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json, is_enabled
            """,
            (int(model_id),),
        )
        row = cur.fetchone()
    if not row:
        conn.rollback()
        return None
    conn.commit()
    return _row_to_registry_item(row)


async def get_registry_checks() -> List[Dict[str, Any]]:
    if not MODEL_REGISTRY_ENABLED:
        set_model_registry_db_up(False)
        return []

    if psycopg is None:
        set_model_registry_db_up(False)
        inc_model_registry_fallback("psycopg_not_installed")
        logger.warning("model_registry.fallback reason=psycopg_not_installed")
        return []

    if not MODEL_REGISTRY_DB_DSN:
        set_model_registry_db_up(False)
        inc_model_registry_fallback("empty_dsn")
        logger.warning("model_registry.fallback reason=empty_dsn")
        return []

    try:
        with _open_conn() as conn:
            global _WARNED_SYNC_FROM_ENV
            _init_schema(conn)
            set_model_registry_db_up(True)
            if MODEL_REGISTRY_SYNC_FROM_ENV and not _WARNED_SYNC_FROM_ENV:
                logger.warning("model_registry.sync_from_env_enabled but runtime is DB-only; skipping env sync")
                _WARNED_SYNC_FROM_ENV = True
            checks = _read_checks(conn)
            set_model_registry_sync_counts(
                env_total=0,
                db_total_before=len(checks),
                inserted=0,
                updated=0,
                unchanged=0,
                removed=0,
            )
            if checks:
                return checks
            inc_model_registry_fallback("db_empty")
            logger.warning("model_registry.fallback reason=db_empty")
    except Exception as exc:
        set_model_registry_db_up(False)
        inc_model_registry_fallback("db_error")
        logger.warning("model_registry.fallback reason=%s", str(exc))

    return []


async def startup_model_registry() -> None:
    if not MODEL_REGISTRY_ENABLED:
        set_model_registry_db_up(False)
        return
    if psycopg is None:
        set_model_registry_db_up(False)
        inc_model_registry_fallback("psycopg_not_installed")
        logger.warning("model_registry.startup.skip reason=psycopg_not_installed")
        return
    if not MODEL_REGISTRY_DB_DSN:
        set_model_registry_db_up(False)
        inc_model_registry_fallback("empty_dsn")
        logger.warning("model_registry.startup.skip reason=empty_dsn")
        return
    try:
        with _open_conn() as conn:
            _init_schema(conn)
            set_model_registry_db_up(True)
    except Exception as exc:
        set_model_registry_db_up(False)
        inc_model_registry_fallback("startup_error")
        logger.warning("model_registry.startup.error=%s", str(exc))


async def shutdown_model_registry() -> None:
    return


def sync_registry_from_env_checks(env_checks: List[Dict[str, Any]]) -> Dict[str, int]:
    with _open_conn() as conn:
        _init_schema(conn)
        report = _upsert_from_env(conn, env_checks)
    return {
        "env_total": int(report.get("env_total", 0)),
        "db_total_before": int(report.get("db_total_before", 0)),
        "inserted": int(report.get("inserted", 0)),
        "updated": int(report.get("updated", 0)),
        "unchanged": int(report.get("unchanged", 0)),
        "removed": int(report.get("removed", 0)),
    }


async def upsert_registry_check(check: Dict[str, Any]) -> Dict[str, Any]:
    if not MODEL_REGISTRY_ENABLED:
        raise RuntimeError("model registry is disabled")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    if not MODEL_REGISTRY_DB_DSN:
        raise RuntimeError("MODEL_REGISTRY_DB_DSN is empty")

    payload = _sync_payload(check)
    with _open_conn() as conn:
        _init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
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
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, TRUE, NOW())
                ON CONFLICT (vllm_model, base_url, model_type) DO UPDATE SET
                    model_key=EXCLUDED.model_key,
                    public_model=EXCLUDED.public_model,
                    max_context_tokens=EXCLUDED.max_context_tokens,
                    default_max_tokens=EXCLUDED.default_max_tokens,
                    max_tokens_cap=EXCLUDED.max_tokens_cap,
                    min_context_headroom=EXCLUDED.min_context_headroom,
                    stream_supported=EXCLUDED.stream_supported,
                    reasoning_supported=EXCLUDED.reasoning_supported,
                    aliases_json=EXCLUDED.aliases_json,
                    is_enabled=TRUE,
                    updated_at=NOW()
                RETURNING id, public_model, vllm_model, model_type, base_url, max_context_tokens, default_max_tokens, max_tokens_cap, min_context_headroom, stream_supported, reasoning_supported, aliases_json::text AS aliases_json, is_enabled
                """,
                (
                    payload["model_key"],
                    payload["public_model"],
                    payload["vllm_model"],
                    payload["model_type"],
                    payload["base_url"],
                    payload["max_context_tokens"],
                    payload["default_max_tokens"],
                    payload["max_tokens_cap"],
                    payload["min_context_headroom"],
                    payload["stream_supported"],
                    payload["reasoning_supported"],
                    payload["aliases_json"],
                ),
            )
            row = cur.fetchone() or {}
        conn.commit()

    aliases_raw = str(row.get("aliases_json") or "[]")
    try:
        aliases = sorted({str(a).strip() for a in json.loads(aliases_raw) if str(a).strip()})
    except Exception:
        aliases = sorted({payload["public_model"], payload["vllm_model"]})

    return {
        "id": int(row.get("id") or 0),
        "public_model": str(row.get("public_model") or payload["public_model"]),
        "vllm_model": str(row.get("vllm_model") or payload["vllm_model"]),
        "type": str(row.get("model_type") or payload["model_type"]),
        "base_url": str(row.get("base_url") or payload["base_url"]),
        "max_context_tokens": int(row.get("max_context_tokens") or payload["max_context_tokens"]),
        "default_max_tokens": int(row.get("default_max_tokens") or payload["default_max_tokens"]),
        "max_tokens_cap": int(row.get("max_tokens_cap") or payload["max_tokens_cap"]),
        "min_context_headroom": int(row.get("min_context_headroom") or payload["min_context_headroom"]),
        "stream_supported": bool(row.get("stream_supported") or False),
        "reasoning_supported": bool(row.get("reasoning_supported") or False),
        "aliases": aliases,
        "is_enabled": bool(row.get("is_enabled") if row.get("is_enabled") is not None else True),
    }


async def create_registry_check(check: Dict[str, Any]) -> Dict[str, Any]:
    if not MODEL_REGISTRY_ENABLED:
        raise RuntimeError("model registry is disabled")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    if not MODEL_REGISTRY_DB_DSN:
        raise RuntimeError("MODEL_REGISTRY_DB_DSN is empty")

    with _open_conn() as conn:
        _init_schema(conn)
        return _insert_registry_check(conn, check)


async def get_registry_check_by_id(model_id: int) -> Dict[str, Any] | None:
    if not MODEL_REGISTRY_ENABLED or psycopg is None or not MODEL_REGISTRY_DB_DSN:
        return None
    with _open_conn() as conn:
        _init_schema(conn)
        return _get_registry_check_by_id(conn, int(model_id))


async def update_registry_check_by_id(model_id: int, check: Dict[str, Any]) -> Dict[str, Any] | None:
    if not MODEL_REGISTRY_ENABLED:
        raise RuntimeError("model registry is disabled")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    if not MODEL_REGISTRY_DB_DSN:
        raise RuntimeError("MODEL_REGISTRY_DB_DSN is empty")

    with _open_conn() as conn:
        _init_schema(conn)
        return _update_registry_check_by_id(conn, int(model_id), check)


async def disable_registry_check_by_id(model_id: int) -> Dict[str, Any] | None:
    if not MODEL_REGISTRY_ENABLED:
        raise RuntimeError("model registry is disabled")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    if not MODEL_REGISTRY_DB_DSN:
        raise RuntimeError("MODEL_REGISTRY_DB_DSN is empty")

    with _open_conn() as conn:
        _init_schema(conn)
        return _disable_registry_check_by_id(conn, int(model_id))
