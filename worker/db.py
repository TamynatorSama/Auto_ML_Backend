"""
db.py
-----
Connecting to Postgres, and applying worker/sql/*.sql in order.

Migrations are plain SQL files, applied once each, in name order, and recorded
in schema_migrations; one transaction covers them all. LangGraph's checkpoint
tables are created by PostgresSaver.setup(), which setup() below also calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

SQL_DIR = Path(__file__).parent / "sql"
FILES_KEPT_DAYS = 7               # a finished job's run directory and checkpoints (A4)
MIGRATION_LOCK = 72001            # advisory lock: two processes starting at once migrate one at a time
# what PostgresSaver needs from its connections; our own code uses the same
CONNECT = {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (see docs/PHASE2.md §1)")
    return url


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or database_url(), **CONNECT)


def pool(url: str | None = None, size: int = 8) -> ConnectionPool:
    # a pool, not one connection: the pipeline emits events from many threads at once
    return ConnectionPool(url or database_url(), min_size=1, max_size=size, kwargs=CONNECT, open=True)


def migrate(conn: psycopg.Connection) -> list:
    """Apply the SQL files not applied yet; returns their names."""
    applied = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}
        for path in sorted(SQL_DIR.glob("*.sql")):
            if path.name not in done:
                # prepare=False: a prepared statement can't hold several commands
                conn.execute(path.read_text(encoding="utf-8"), prepare=False)
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
                applied.append(path.name)
    return applied


def checkpointer(pool: ConnectionPool):
    """LangGraph's Postgres checkpointer, allowed to load the pipeline's own classes."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from main import checkpoint_types

    return PostgresSaver(pool, serde=JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types()))


def setup(pool: ConnectionPool) -> list:
    """Our migrations, then LangGraph's checkpoint tables."""
    with pool.connection() as conn:
        applied = migrate(conn)
    checkpointer(pool).setup()
    return applied


def jsonb(value) -> Jsonb:
    # jsonb has no NaN or Infinity: they become null
    return Jsonb(json.loads(json.dumps(value, default=str), parse_constant=lambda _: None))
