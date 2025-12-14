from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .config import settings


# Single-main-table sheet name (MANDATORY requirement)
CONTENT_SHEET_NAME = "CONTENT_MAIN"
TRAINING_SHEET_NAME = "TRAINING_DB"


CONTENT_SHEET_COLUMNS: List[str] = [
    "content_id",
    "use_case",
    "status",
    "version",
    "prompt_version",
    "updated_at",
    "created_at",
    # RAW (OR)
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


def sheets_enabled() -> bool:
    return bool(settings.GOOGLE_SHEETS_ID and settings.GOOGLE_SERVICE_ACCOUNT_JSON)


def _load_service_account_info() -> Dict[str, Any]:
    src = settings.GOOGLE_SERVICE_ACCOUNT_JSON or ""
    src = src.strip()
    if not src:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON missing")

    # file path
    if os.path.exists(src):
        with open(src, "r", encoding="utf-8") as f:
            return json.load(f)

    # raw json
    return json.loads(src)


def _client():
    """Lazy import to keep backend working without Sheets deps."""
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Google Sheets is not available. Install deps: gspread google-auth. "
            f"Original error: {e}"
        )

    info = _load_service_account_info()
    creds = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def _open_spreadsheet():
    if not sheets_enabled():
        raise RuntimeError("Google Sheets is not configured")
    gc = _client()
    return gc.open_by_key(settings.GOOGLE_SHEETS_ID)


def _ensure_worksheet(spreadsheet, name: str, headers: List[str]):
    try:
        ws = spreadsheet.worksheet(name)
    except Exception:
        ws = spreadsheet.add_worksheet(title=name, rows=2000, cols=max(26, len(headers) + 5))

    # Ensure header row
    values = ws.get_all_values()
    if not values:
        ws.append_row(headers)
        return ws

    current = values[0]
    if current[: len(headers)] != headers:
        # Rewrite header row to match expected schema; keep existing data rows.
        ws.update("A1", [headers])
    return ws


def ensure_sheets_schema() -> None:
    """Create (if missing) the required role-based worksheets.

    Mandatory: keep CONTENT in a single main table.
    """

    if not sheets_enabled():
        return

    ss = _open_spreadsheet()
    _ensure_worksheet(ss, CONTENT_SHEET_NAME, CONTENT_SHEET_COLUMNS)
    _ensure_worksheet(
        ss,
        TRAINING_SHEET_NAME,
        ["kind", "id", "json", "updated_at", "created_at"],
    )


def _normalize_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def upsert_content_row(content: Dict[str, Any]) -> None:
    if not sheets_enabled():
        return

    ensure_sheets_schema()
    ss = _open_spreadsheet()
    ws = ss.worksheet(CONTENT_SHEET_NAME)

    # Find existing row by content_id in column A
    content_id = str(content.get("content_id") or "").strip()
    if not content_id:
        return

    try:
        cell = ws.find(content_id)
        row_index = cell.row
    except Exception:
        row_index = None

    row = [_normalize_cell(content.get(c)) for c in CONTENT_SHEET_COLUMNS]

    if row_index and row_index > 1:
        ws.update(f"A{row_index}", [row])
    else:
        ws.append_row(row)


def upsert_training_record(kind: str, rec_id: str, payload: Dict[str, Any], created_at: str | None = None) -> None:
    """Store training ingredients (story types, step prompts, examples metadata) in TRAINING_DB sheet."""

    if not sheets_enabled():
        return

    ensure_sheets_schema()
    ss = _open_spreadsheet()
    ws = ss.worksheet(TRAINING_SHEET_NAME)

    now = payload.get("updated_at") or payload.get("updatedAt")
    if not now:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

    created = created_at or payload.get("created_at") or payload.get("createdAt") or now

    # Search for existing by kind+id (simple scan; acceptable for small tables)
    rows = ws.get_all_values()
    target_row = None
    for idx, r in enumerate(rows[1:], start=2):
        if len(r) >= 2 and r[0] == kind and r[1] == rec_id:
            target_row = idx
            break

    row = [kind, rec_id, json.dumps(payload, ensure_ascii=False), str(now), str(created)]
    if target_row:
        ws.update(f"A{target_row}", [row])
    else:
        ws.append_row(row)
