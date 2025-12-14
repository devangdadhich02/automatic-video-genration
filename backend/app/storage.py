from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> str:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    return os.path.join(settings.DATA_DIR, "content.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


CONTENT_COLUMNS = [
    "content_id",
    "use_case",
    "status",
    "version",
    "prompt_version",
    "created_at",
    "updated_at",
    # RAW INPUT (OR-combination allowed)
    "raw_text",
    "web_url",
    "youtube_url",
    "channel_link",
    # CLEANED
    "merged_text",
    "cleaned_text",
    # CLASSIFIED
    "domain",
    "topic",
    "sub_topic",
    "story_type",
    "channel",
    "channel_dna_json",
    # CONFIG / MODEL
    "config_json",
    "ai_provider",
    "ai_model",
    "total_chars_target",
    # GENERATION
    "steps_json",
    "generated_steps_json",
    "final_output",
    "assets_json",
    # ERROR
    "error_log",
]


def init_db() -> None:
    """Initialize local persistence.

    This DB is the authoritative store when Google Sheets is not configured.
    """

    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contents (
              content_id TEXT PRIMARY KEY,
              use_case TEXT NOT NULL,
              status TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1,
              prompt_version TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              raw_text TEXT,
              web_url TEXT,
              youtube_url TEXT,
              channel_link TEXT,
              merged_text TEXT,
              cleaned_text TEXT,
              domain TEXT,
              topic TEXT,
              sub_topic TEXT,
              story_type TEXT,
              channel TEXT,
              channel_dna_json TEXT,
              config_json TEXT,
              ai_provider TEXT,
              ai_model TEXT,
              total_chars_target INTEGER,
              steps_json TEXT,
              generated_steps_json TEXT,
              final_output TEXT,
              assets_json TEXT,
              error_log TEXT
            )
            """
        )

        # Training ingredients (kept separate from contents, but still unified/role-based)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS story_types (
              type_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              description TEXT,
              structure_json TEXT,
              rules_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS step_prompts (
              prompt_id TEXT PRIMARY KEY,
              type_id TEXT NOT NULL,
              step_index INTEGER NOT NULL,
              step_name TEXT,
              objective TEXT,
              prompt_text TEXT NOT NULL,
              example_ref TEXT,
              ratio REAL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(type_id, step_index)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS examples (
              example_id TEXT PRIMARY KEY,
              type_id TEXT,
              channel TEXT,
              title TEXT,
              source_url TEXT,
              raw_text TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )

        # Channel DNA store (training ingredient). Kept separate from `contents` rows.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_dna (
              channel TEXT NOT NULL,
              story_type TEXT,
              version INTEGER NOT NULL DEFAULT 1,
              dna_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (channel, story_type)
            )
            """
        )

        conn.commit()


def create_content(*, use_case: str, status: str, fields: Dict[str, Any]) -> str:
    import uuid

    init_db()
    content_id = uuid.uuid4().hex
    now = _utc_now_iso()

    row: Dict[str, Any] = {
        "content_id": content_id,
        "use_case": use_case,
        "status": status,
        "version": int(fields.get("version") or 1),
        "prompt_version": fields.get("prompt_version"),
        "created_at": now,
        "updated_at": now,
    }

    for k in CONTENT_COLUMNS:
        if k in row:
            continue
        if k in fields:
            v = fields[k]
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v

    cols = [c for c in CONTENT_COLUMNS if c in row]
    vals = [row[c] for c in cols]
    placeholders = ",".join(["?"] * len(cols))

    with _connect() as conn:
        conn.execute(
            f"INSERT INTO contents ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        conn.commit()

    return content_id


def _row_to_dict(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    # decode json-ish fields
    for jf in ["channel_dna_json", "config_json", "steps_json", "generated_steps_json", "assets_json"]:
        if d.get(jf):
            try:
                d[jf] = json.loads(d[jf])
            except Exception:
                pass
    return d


def get_content(content_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM contents WHERE content_id = ?", (content_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def list_contents(
    *,
    use_case: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    init_db()
    q = "SELECT * FROM contents"
    args: List[Any] = []
    where: List[str] = []

    if use_case:
        where.append("use_case = ?")
        args.append(use_case)
    if status:
        where.append("status = ?")
        args.append(status)

    if where:
        q += " WHERE " + " AND ".join(where)

    q += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    args.extend([int(limit), int(offset)])

    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
        return [_row_to_dict(r) for r in rows]


def update_content(content_id: str, fields: Dict[str, Any]) -> None:
    init_db()
    if not fields:
        return

    fields = dict(fields)
    fields["updated_at"] = _utc_now_iso()

    sets: List[str] = []
    args: List[Any] = []

    for k, v in fields.items():
        if k not in CONTENT_COLUMNS:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        sets.append(f"{k} = ?")
        args.append(v)

    if not sets:
        return

    args.append(content_id)

    with _connect() as conn:
        conn.execute(f"UPDATE contents SET {', '.join(sets)} WHERE content_id = ?", args)
        conn.commit()


def upsert_story_type(*, type_id: str, name: str, description: str | None, structure: Any, rules: Any) -> None:
    init_db()
    now = _utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO story_types (type_id, name, description, structure_json, rules_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(type_id) DO UPDATE SET
              name=excluded.name,
              description=excluded.description,
              structure_json=excluded.structure_json,
              rules_json=excluded.rules_json,
              updated_at=excluded.updated_at
            """,
            (
                type_id,
                name,
                description,
                json.dumps(structure or {}, ensure_ascii=False),
                json.dumps(rules or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()


def list_story_types() -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM story_types ORDER BY updated_at DESC").fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            for jf in ["structure_json", "rules_json"]:
                if d.get(jf):
                    try:
                        d[jf] = json.loads(d[jf])
                    except Exception:
                        pass
            out.append(d)
        return out


def upsert_step_prompt(
    *,
    prompt_id: str,
    type_id: str,
    step_index: int,
    step_name: str | None,
    objective: str | None,
    prompt_text: str,
    example_ref: str | None,
    ratio: float | None,
) -> None:
    init_db()
    now = _utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO step_prompts (prompt_id, type_id, step_index, step_name, objective, prompt_text, example_ref, ratio, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(type_id, step_index) DO UPDATE SET
              prompt_id=excluded.prompt_id,
              step_name=excluded.step_name,
              objective=excluded.objective,
              prompt_text=excluded.prompt_text,
              example_ref=excluded.example_ref,
              ratio=excluded.ratio,
              updated_at=excluded.updated_at
            """,
            (
                prompt_id,
                type_id,
                int(step_index),
                step_name,
                objective,
                prompt_text,
                example_ref,
                float(ratio) if ratio is not None else None,
                now,
                now,
            ),
        )
        conn.commit()


def list_step_prompts(type_id: str) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM step_prompts WHERE type_id = ? ORDER BY step_index ASC",
            (type_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def upsert_example(
    *,
    example_id: str,
    type_id: str | None,
    channel: str | None,
    title: str | None,
    source_url: str | None,
    raw_text: str,
) -> None:
    init_db()
    now = _utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO examples (example_id, type_id, channel, title, source_url, raw_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(example_id) DO UPDATE SET
              type_id=excluded.type_id,
              channel=excluded.channel,
              title=excluded.title,
              source_url=excluded.source_url,
              raw_text=excluded.raw_text,
              updated_at=excluded.updated_at
            """,
            (example_id, type_id, channel, title, source_url, raw_text, now, now),
        )
        conn.commit()


def list_examples(*, type_id: str | None = None, channel: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    q = "SELECT * FROM examples"
    args: List[Any] = []
    where: List[str] = []
    if type_id:
        where.append("type_id = ?")
        args.append(type_id)
    if channel:
        where.append("channel = ?")
        args.append(channel)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY updated_at DESC LIMIT ?"
    args.append(int(limit))

    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]


def upsert_channel_dna(*, channel: str, story_type: str | None, version: int, dna: Dict[str, Any]) -> None:
    init_db()
    now = _utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO channel_dna (channel, story_type, version, dna_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel, story_type) DO UPDATE SET
              version=excluded.version,
              dna_json=excluded.dna_json,
              updated_at=excluded.updated_at
            """,
            (
                channel,
                story_type,
                int(version),
                json.dumps(dna or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()


def get_channel_dna(channel: str, story_type: str | None = None) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM channel_dna WHERE channel = ? AND story_type IS ?",
            (channel, story_type),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("dna_json"):
            try:
                d["dna_json"] = json.loads(d["dna_json"])
            except Exception:
                pass
        return d


def list_channel_dna(*, channel: str | None = None, story_type: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    q = "SELECT * FROM channel_dna"
    args: List[Any] = []
    where: List[str] = []
    if channel:
        where.append("channel = ?")
        args.append(channel)
    if story_type is not None:
        where.append("story_type IS ?")
        args.append(story_type)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY updated_at DESC LIMIT ?"
    args.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("dna_json"):
                try:
                    d["dna_json"] = json.loads(d["dna_json"])
                except Exception:
                    pass
            out.append(d)
        return out
