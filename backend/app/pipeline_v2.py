from __future__ import annotations

import asyncio
import io
import json
import math
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from .cleaning import CleanInputs, clean_and_merge
from .exporter import export_script
from .llm_router import chat, safe_provider_and_model
from .rag import rag_service
from .script_jobs import create_job, get_job, update_job
from .sheets_store import upsert_content_row, upsert_training_record
from .storage import (
    create_content,
    get_channel_dna,
    get_content,
    list_contents,
    list_channel_dna,
    list_step_prompts,
    list_story_types,
    update_content,
    upsert_channel_dna,
    upsert_example,
    upsert_step_prompt,
    upsert_story_type,
)

router = APIRouter(prefix="/v2", tags=["pipeline-v2"])


# -----------------
# Models
# -----------------


class RawInputRequest(BaseModel):
    """RAW input (OR logic)."""

    use_case: str = Field("TRAINING", description="TRAINING or GENERATION")
    content_id: Optional[str] = Field(None, description="Optional: reuse an existing content_id")

    raw_text: Optional[str] = None
    web_url: Optional[str] = None
    youtube_url: Optional[str] = None
    channel_link: Optional[str] = None

    # Optional metadata columns (stored as columns, not tabs)
    domain: Optional[str] = None
    topic: Optional[str] = None
    sub_topic: Optional[str] = None
    channel: Optional[str] = None


class CleanResponse(BaseModel):
    job_id: str
    content_id: str


class ClassifyRequest(BaseModel):
    manual_story_type: Optional[str] = None
    manual_domain: Optional[str] = None
    manual_topic: Optional[str] = None
    manual_sub_topic: Optional[str] = None


class GenerateRequest(BaseModel):
    # Either manual story_type or use already classified
    story_type: Optional[str] = None

    # Model selection (from sheet dropdown)
    ai_provider: Optional[str] = None  # GPT/Claude/Gemini/OpenRouter -> normalized
    ai_model: Optional[str] = None

    prompt_version: Optional[str] = None

    total_chars_target: int = 30000

    # Optional override: step count
    steps: Optional[int] = Field(None, ge=5, le=20)

    # Optional config_json merge (retry/timeout/asset prompt options etc.)
    config: Dict[str, Any] | None = None


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    formats: List[str] = Field(default_factory=lambda: ["docx", "pdf"])
    filename_prefix: Optional[str] = None


class StoryTypeUpsertRequest(BaseModel):
    type_id: str
    name: str
    description: Optional[str] = None
    structure: Dict[str, Any] = Field(default_factory=dict)
    rules: Dict[str, Any] = Field(default_factory=dict)


class StepPromptUpsertRequest(BaseModel):
    prompt_id: str
    type_id: str
    step_index: int
    step_name: Optional[str] = None
    objective: Optional[str] = None
    prompt_text: str
    example_ref: Optional[str] = None
    ratio: Optional[float] = None


class ExampleUpsertRequest(BaseModel):
    example_id: str
    type_id: Optional[str] = None
    channel: Optional[str] = None
    title: Optional[str] = None
    source_url: Optional[str] = None
    raw_text: str


class ChannelDNAUpsertRequest(BaseModel):
    channel: str
    story_type: Optional[str] = None
    version: int = 1
    dna: Dict[str, Any] = Field(default_factory=dict)


# -----------------
# Helpers
# -----------------


def _normalize_use_case(u: str) -> str:
    u = (u or "").strip().upper()
    if u not in {"TRAINING", "GENERATION"}:
        return "TRAINING"
    return u


def _sync_sheet(content_id: str) -> None:
    c = get_content(content_id)
    if not c:
        return
    try:
        upsert_content_row(c)
    except Exception:
        # never fail core pipeline due to sheets
        pass


def _mark_error(content_id: str, err: str) -> None:
    update_content(content_id, {"status": "ERROR", "error_log": err})
    _sync_sheet(content_id)


def _basic_story_type_guess(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["breaking", "today", "yesterday", "latest", "headline"]):
        return "News-type"
    if any(k in t for k in ["how to", "step-by-step", "steps", "guide", "tutorial"]):
        return "Problem-solving"
    return "Narrative"


def _extract_channel_dna(cleaned_text: str) -> Dict[str, Any]:
    """Lightweight channel DNA extraction (heuristic, fast, no external calls)."""

    t = cleaned_text or ""
    # Hook: first 2 sentences
    sents = re.split(r"(?<=[.!?])\s+", t.strip())
    hook = " ".join([s.strip() for s in sents[:2] if s.strip()])

    # Tone heuristic
    tone = "neutral"
    if re.search(r"\b(shocking|insane|unbelievable|secret|never told)\b", t, re.I):
        tone = "high-energy"
    if re.search(r"\bdata|chart|numbers|evidence\b", t, re.I):
        tone = "analytical"

    return {
        "hook_pattern": hook,
        "tone": tone,
        "style": "spoken voiceover",
        "notes": "Heuristic extraction; replace with LLM-based analysis if desired.",
    }


def _choose_story_type(cleaned_text: str) -> str:
    """Pick a story_type id from trained types when possible."""
    trained = list_story_types()
    if not trained:
        return _basic_story_type_guess(cleaned_text)

    t = (cleaned_text or "").lower()
    best_id = trained[0].get("type_id") or trained[0].get("name") or "Narrative"
    best_score = -1.0

    for st in trained:
        type_id = (st.get("type_id") or st.get("name") or "").strip() or "Narrative"
        blob = " ".join(
            [
                str(st.get("name") or ""),
                str(st.get("description") or ""),
                json.dumps(st.get("structure_json") or st.get("structure") or {}, ensure_ascii=False),
                json.dumps(st.get("rules_json") or st.get("rules") or {}, ensure_ascii=False),
            ]
        ).lower()

        # Simple relevance scoring: keyword overlap + bonus for explicit matches
        score = 0.0
        for w in set(re.findall(r"[a-z]{4,}", blob)):
            if w in t:
                score += 1.0
        if type_id.lower() in t:
            score += 5.0
        if score > best_score:
            best_score = score
            best_id = type_id

    return best_id


def _estimate_tokens(text: str) -> int:
    s = (text or "").strip()
    if not s:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        return max(1, int(len(s) / 4))


def _get_effective_config(content: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    base = content.get("config_json") if isinstance(content.get("config_json"), dict) else {}
    cfg = dict(base or {})
    if override:
        # shallow merge is enough for our needs (clients store flat config keys)
        cfg.update(override)
    return cfg


def _mark_key_terms_heuristic(text: str) -> str:
    """Mark likely visual keywords using *term*.

    Heuristic-only (safe + offline):
    - numbers/currency/percent
    - simple proper-noun sequences
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Avoid double-marking
    def wrap(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("*") and s.endswith("*"):
            return s
        return f"*{s}*"

    # Numbers / currency / percentages
    t = re.sub(r"(?<!\*)\b(?:\$|₹|€|£)?\d[\d,]*(?:\.\d+)?%?\b(?!\*)", wrap, t)
    # Proper nouns (very conservative)
    t = re.sub(r"(?<!\*)\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b(?!\*)", wrap, t)
    return t


def _collapse_spaces(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """Best-effort PDF text extraction (offline).

    If the PDF is scanned images (no embedded text), this returns empty or near-empty text.
    OCR is intentionally not added here to keep dependencies light and avoid runtime complexity.
    """
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(data))  # type: ignore[name-defined]
        parts: List[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                parts.append(t)
        return _collapse_spaces("\n\n".join(parts))
    except Exception:
        return ""


def _token_limit_for_provider(provider: str, cfg: Dict[str, Any]) -> int:
    # Client spec: GPT step-based generation should consider ~2,500 token output cap.
    v = cfg.get("max_tokens_per_step")
    if isinstance(v, int) and v > 200:
        return v
    if provider in {"openai"}:
        return 2500
    return 4000


def _sleep_backoff(i: int) -> None:
    time.sleep(min(8.0, 0.6 * (2 ** i)))


def _chat_with_retries(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int,
) -> str:
    last: Exception | None = None
    # Model fallback (OpenAI only) to avoid common invalid model ids (e.g. "gpt-5.2-mini").
    model_candidates = [m for m in [model, "gpt-4o-mini"] if m and isinstance(m, str)]
    seen = set()
    model_candidates = [m for m in model_candidates if not (m in seen or seen.add(m))]

    for m in model_candidates:
        for attempt in range(max(0, int(retries)) + 1):
            try:
                return chat(provider=provider, model=m, messages=messages, temperature=temperature, max_tokens=max_tokens).text
            except Exception as e:
                last = e
                msg = str(e).lower()
                # If model doesn't exist, try next candidate (OpenAI only)
                if provider == "openai" and ("model" in msg and ("not found" in msg or "does not exist" in msg or "model_not_found" in msg)):
                    break
                if attempt >= retries:
                    break
                _sleep_backoff(attempt)
    raise RuntimeError(str(last) if last else "llm_call_failed")


def _build_outline_for_steps(cleaned: str, story_type: str, steps: List[Dict[str, Any]], provider: str, model: str, cfg: Dict[str, Any]) -> List[str]:
    """Generate a step-aligned outline (best-effort)."""
    # If LLM is not available, fall back to objectives.
    retries = int(cfg.get("retry_count") or 1)
    max_tokens = min(900, _token_limit_for_provider(provider, cfg))

    labels = [f"{i+1}. {s.get('step_name') or f'Step {i+1}'} — {s.get('objective') or ''}".strip() for i, s in enumerate(steps)]
    fallback = [s.get("objective") or s.get("step_name") or f"Step {i+1}" for i, s in enumerate(steps)]

    system = "You are an expert YouTube script strategist. Output plain text only."
    user = (
        f"STORY_TYPE: {story_type}\n\n"
        f"SOURCE (cleaned, truncated):\n{cleaned[:4000]}\n\n"
        "Create a 1-line outline bullet for EACH step below, matching order and count.\n"
        "Do not add extra steps. Do not number beyond the provided steps.\n\n"
        "STEPS:\n" + "\n".join(labels) + "\n\n"
        "Output format:\n"
        "One bullet per line, exactly the same number of lines as steps.\n"
    )
    try:
        out = _chat_with_retries(
            provider=provider,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            max_tokens=max_tokens,
            retries=retries,
        )
        lines = [re.sub(r"^\s*[-*\d\.)]+\s*", "", ln.strip()) for ln in (out or "").splitlines() if ln.strip()]
        if len(lines) >= len(steps):
            return lines[: len(steps)]
    except Exception:
        pass
    return fallback


def _step_plan_from_prompts(type_id: str) -> List[Dict[str, Any]]:
    prompts = list_step_prompts(type_id)
    steps = []
    for p in prompts:
        steps.append(
            {
                "step_index": int(p.get("step_index") or 0),
                "step_name": p.get("step_name") or f"Step {int(p.get('step_index') or 0) + 1}",
                "objective": p.get("objective") or "",
                "prompt_text": p.get("prompt_text") or "",
                "example_ref": p.get("example_ref"),
                "ratio": float(p.get("ratio")) if p.get("ratio") is not None else None,
            }
        )
    return steps


def _default_steps(story_type: str, count: int) -> List[Dict[str, Any]]:
    # Minimal defaults (covers 5-20)
    base = [
        ("Hook", "Hook the viewer and preview the promise."),
        ("Context", "Set up background and stakes."),
        ("Development", "Build the key ideas with examples."),
        ("Conflict", "Introduce the main tension/problem and deepen it."),
        ("Resolution", "Resolve the tension with clear takeaways."),
        ("Recap", "Summarize and reinforce key points."),
        ("Outro", "Close with CTA and smooth landing."),
    ]

    if count <= len(base):
        chosen = base[:count]
    else:
        chosen = base + [(f"Deep Dive {i}", "Continue the development with new angles and examples.") for i in range(1, count - len(base) + 1)]

    steps: List[Dict[str, Any]] = []
    for i, (name, obj) in enumerate(chosen):
        steps.append(
            {
                "step_index": i,
                "step_name": name,
                "objective": obj,
                "prompt_text": f"Write the {name} section in a {story_type} style.",
                "example_ref": None,
                "ratio": None,
            }
        )
    return steps


def _allocate_lengths(total_chars: int, steps: List[Dict[str, Any]]) -> List[int]:
    ratios = []
    for s in steps:
        r = s.get("ratio")
        ratios.append(float(r) if r is not None else 0.0)

    if sum(ratios) <= 0:
        ratios = [1.0 for _ in steps]

    total = float(sum(ratios))
    raw = [max(200, int(round(total_chars * (r / total)))) for r in ratios]

    # fix rounding drift
    drift = total_chars - sum(raw)
    i = 0
    while drift != 0 and raw:
        j = i % len(raw)
        raw[j] += 1 if drift > 0 else -1
        drift += -1 if drift > 0 else 1
        i += 1

    return raw


def _sentence_asset_prompts(final_output: str) -> List[Dict[str, str]]:
    # Sentence split
    text = (final_output or "").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    out: List[Dict[str, str]] = []
    for i, s in enumerate([x.strip() for x in sents if x.strip()]):
        # Light prompt template
        out.append(
            {
                "index": str(i),
                "sentence": s,
                "image_prompt": (
                    "Cinematic b-roll, realistic, sharp focus, 16:9. "
                    "Visualize the following narration sentence as a single scene: "
                    + s
                ),
            }
        )
    return out


# -----------------
# Content APIs
# -----------------


@router.get("/contents")
def api_list_contents(use_case: Optional[str] = None, status: Optional[str] = None, limit: int = 50, offset: int = 0):
    return {"items": list_contents(use_case=use_case, status=status, limit=limit, offset=offset)}


@router.get("/contents/{content_id}")
def api_get_content(content_id: str):
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")
    return c


@router.post("/content/raw")
def api_create_or_update_raw(req: RawInputRequest):
    use_case = _normalize_use_case(req.use_case)

    fields: Dict[str, Any] = {
        "raw_text": req.raw_text,
        "web_url": req.web_url,
        "youtube_url": req.youtube_url,
        "channel_link": req.channel_link,
        "domain": req.domain,
        "topic": req.topic,
        "sub_topic": req.sub_topic,
        "channel": req.channel,
    }

    if req.content_id:
        existing = get_content(req.content_id)
        if not existing:
            raise HTTPException(status_code=404, detail="content_id_not_found")
        update_content(req.content_id, {"use_case": use_case, "status": "RAW", **fields})
        _sync_sheet(req.content_id)
        return {"content_id": req.content_id, "status": "RAW"}

    content_id = create_content(use_case=use_case, status="RAW", fields=fields)
    _sync_sheet(content_id)
    return {"content_id": content_id, "status": "RAW"}


@router.post("/content/{content_id}/upload/pdf")
async def api_upload_pdf(content_id: str, file: UploadFile = File(...)):
    """Upload a PDF as an input source (optional).

    Extracted text is stored into the same `contents.raw_text` column to preserve:
    - single main table
    - OR-combination input logic
    """
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")

    if not file or not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="pdf_required")

    data = await file.read()
    extracted = _extract_text_from_pdf_bytes(data or b"")
    extracted_chars = len(extracted or "")

    raw_text_existing = (c.get("raw_text") or "").strip()
    if extracted and extracted.strip():
        if raw_text_existing:
            new_raw = raw_text_existing + "\n\n" + extracted.strip()
        else:
            new_raw = extracted.strip()
        update_content(content_id, {"raw_text": new_raw, "status": "RAW"})

    # Store lineage-safe asset record (derived from content_id only)
    assets = c.get("assets_json") if isinstance(c.get("assets_json"), list) else []
    assets = list(assets or [])
    assets.append(
        {
            "parent_content_id": content_id,
            "kind": "source_pdf",
            "filename": file.filename,
            "bytes": len(data or b""),
            "extracted_chars": extracted_chars,
        }
    )
    update_content(content_id, {"assets_json": assets})
    _sync_sheet(content_id)

    warning = None
    if extracted_chars < 50:
        warning = "pdf_text_extraction_low_or_empty (scanned PDF may need OCR)"

    return {"status": "ok", "content_id": content_id, "extracted_chars": extracted_chars, "warning": warning}


@router.post("/content/{content_id}/clean/async", response_model=CleanResponse)
async def api_clean_async(content_id: str):
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")

    job = create_job()
    update_job(job.id, status="running", progress={"phase": "cleaning"}, result={"content_id": content_id})

    async def _run():
        try:
            update_content(content_id, {"status": "RAW"})
            _sync_sheet(content_id)

            merged, cleaned, sources = await clean_and_merge(
                CleanInputs(
                    raw_text=c.get("raw_text"),
                    web_url=c.get("web_url"),
                    youtube_url=c.get("youtube_url"),
                    channel_link=c.get("channel_link"),
                )
            )

            update_content(
                content_id,
                {
                    "merged_text": merged,
                    "cleaned_text": cleaned,
                    "status": "CLEANED",
                    "error_log": None,
                },
            )
            _sync_sheet(content_id)

            update_job(job.id, status="completed", result={"content_id": content_id, "sources_used": sources})
        except Exception as e:
            _mark_error(content_id, str(e))
            update_job(job.id, status="failed", error=str(e), result={"content_id": content_id})

    asyncio.create_task(_run())
    return {"job_id": job.id, "content_id": content_id}


@router.post("/content/{content_id}/classify/async")
async def api_classify_async(content_id: str, req: ClassifyRequest):
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")

    job = create_job()
    update_job(job.id, status="running", progress={"phase": "classifying"}, result={"content_id": content_id})

    async def _run():
        try:
            cleaned = (c.get("cleaned_text") or c.get("merged_text") or c.get("raw_text") or "").strip()
            if not cleaned:
                raise RuntimeError("no_text_to_classify")

            story_type = (req.manual_story_type or c.get("story_type") or "").strip() or _choose_story_type(cleaned)

            update_content(
                content_id,
                {
                    "story_type": story_type,
                    "domain": (req.manual_domain or c.get("domain")),
                    "topic": (req.manual_topic or c.get("topic")),
                    "sub_topic": (req.manual_sub_topic or c.get("sub_topic")),
                    "status": "CLASSIFIED",
                    "error_log": None,
                },
            )

            # Channel DNA extraction + version increment (TRAINING phase only)
            if (c.get("use_case") or "").upper() == "TRAINING":
                dna = _extract_channel_dna(cleaned)
                version = int(c.get("version") or 1) + 1
                update_content(content_id, {"channel_dna_json": dna, "version": version})

                # Persist channel DNA into training store (keyed by channel + story_type).
                ch = (c.get("channel") or "").strip()
                if ch:
                    upsert_channel_dna(channel=ch, story_type=story_type, version=version, dna=dna)
                    try:
                        upsert_training_record(
                            "channel_dna",
                            f"{ch}:{story_type}",
                            {"channel": ch, "story_type": story_type, "version": version, "dna": dna},
                        )
                    except Exception:
                        pass

            _sync_sheet(content_id)
            update_job(job.id, status="completed", result={"content_id": content_id, "story_type": story_type})
        except Exception as e:
            _mark_error(content_id, str(e))
            update_job(job.id, status="failed", error=str(e), result={"content_id": content_id})

    asyncio.create_task(_run())
    return {"job_id": job.id, "content_id": content_id}


@router.get("/jobs/{job_id}")
def api_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": getattr(job, "progress", None),
        "result": job.result,
        "error": job.error,
    }


# -----------------
# Training DB APIs
# -----------------


@router.get("/training/story-types")
def api_list_story_types():
    return {"items": list_story_types()}


@router.post("/training/story-types")
def api_upsert_story_type(req: StoryTypeUpsertRequest):
    upsert_story_type(
        type_id=req.type_id,
        name=req.name,
        description=req.description,
        structure=req.structure,
        rules=req.rules,
    )

    try:
        upsert_training_record(
            "story_type",
            req.type_id,
            {"type_id": req.type_id, "name": req.name, "description": req.description, "structure": req.structure, "rules": req.rules},
        )
    except Exception:
        pass

    return {"status": "ok"}


@router.get("/training/step-prompts/{type_id}")
def api_list_step_prompts(type_id: str):
    return {"items": list_step_prompts(type_id)}


@router.post("/training/step-prompts")
def api_upsert_step_prompt(req: StepPromptUpsertRequest):
    upsert_step_prompt(
        prompt_id=req.prompt_id,
        type_id=req.type_id,
        step_index=req.step_index,
        step_name=req.step_name,
        objective=req.objective,
        prompt_text=req.prompt_text,
        example_ref=req.example_ref,
        ratio=req.ratio,
    )

    try:
        upsert_training_record(
            "step_prompt",
            req.prompt_id,
            req.model_dump(),
        )
    except Exception:
        pass

    return {"status": "ok"}


@router.get("/training/examples")
def api_list_examples(type_id: Optional[str] = None, channel: Optional[str] = None, limit: int = 50):
    from .storage import list_examples

    return {"items": list_examples(type_id=type_id, channel=channel, limit=limit)}


@router.post("/training/examples")
def api_upsert_example(req: ExampleUpsertRequest):
    """Upload example text (optionally extracted elsewhere). Also ingests to vector DB for RAG."""

    upsert_example(
        example_id=req.example_id,
        type_id=req.type_id,
        channel=req.channel,
        title=req.title,
        source_url=req.source_url,
        raw_text=req.raw_text,
    )

    # Ingest into vector DB with metadata filters
    meta = {
        "kind": "example",
        "example_id": req.example_id,
        "story_type": req.type_id or "",
        "channel": req.channel or "",
        "source_url": req.source_url or "",
    }
    try:
        rag_service.ingest_text(req.raw_text, metadata=meta)
    except Exception:
        # never fail upload if embeddings not configured
        pass

    try:
        upsert_training_record(
            "example",
            req.example_id,
            req.model_dump(),
        )
    except Exception:
        pass

    return {"status": "ok", "example_id": req.example_id}


@router.get("/training/channel-dna")
def api_list_channel_dna(channel: Optional[str] = None, story_type: Optional[str] = None, limit: int = 100):
    return {"items": list_channel_dna(channel=channel, story_type=story_type, limit=limit)}


@router.post("/training/channel-dna")
def api_upsert_channel_dna(req: ChannelDNAUpsertRequest):
    upsert_channel_dna(channel=req.channel, story_type=req.story_type, version=req.version, dna=req.dna)
    try:
        upsert_training_record(
            "channel_dna",
            f"{req.channel}:{req.story_type or ''}",
            req.model_dump(),
        )
    except Exception:
        pass
    return {"status": "ok"}


# -----------------
# Generation APIs
# -----------------


@router.post("/generation/{content_id}/generate/async")
async def api_generate_async(content_id: str, req: GenerateRequest):
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")

    job = create_job()
    update_job(job.id, status="running", progress={"phase": "starting"}, result={"content_id": content_id})

    async def _run():
        try:
            # Re-read in case the UI just uploaded a PDF or edited raw fields.
            c2 = get_content(content_id) or {}
            # Ensure CLEANED exists
            cleaned = (c2.get("cleaned_text") or "").strip()
            if not cleaned:
                merged = (c2.get("merged_text") or "").strip()
                if merged:
                    cleaned = merged
                else:
                    # Auto-clean (no-error UX): if any RAW inputs exist, run cleaning now.
                    raw_text = (c2.get("raw_text") or "").strip()
                    web_url = (c2.get("web_url") or "").strip()
                    youtube_url = (c2.get("youtube_url") or "").strip()
                    channel_link = (c2.get("channel_link") or "").strip()
                    if raw_text or web_url or youtube_url or channel_link:
                        update_job(job.id, progress={"phase": "auto_cleaning"})
                        merged2, cleaned2, _sources = await clean_and_merge(
                            CleanInputs(
                                raw_text=raw_text or None,
                                web_url=web_url or None,
                                youtube_url=youtube_url or None,
                                channel_link=channel_link or None,
                            )
                        )
                        update_content(
                            content_id,
                            {
                                "merged_text": merged2,
                                "cleaned_text": cleaned2,
                                "status": "CLEANED",
                                "error_log": None,
                            },
                        )
                        _sync_sheet(content_id)
                        cleaned = (cleaned2 or "").strip() or (merged2 or "").strip()
                    if not cleaned:
                        raise RuntimeError("content_not_cleaned")

            story_type = (req.story_type or c2.get("story_type") or "").strip() or _basic_story_type_guess(cleaned)

            provider, model = safe_provider_and_model(req.ai_provider or c2.get("ai_provider"), req.ai_model or c2.get("ai_model"))
            cfg = _get_effective_config(c2, req.config)
            retry_count = int(cfg.get("retry_count") or 1)
            max_tokens_per_step = _token_limit_for_provider(provider, cfg)

            update_content(
                content_id,
                {
                    "use_case": "GENERATION",
                    "ai_provider": provider,
                    "ai_model": model,
                    "prompt_version": req.prompt_version or c.get("prompt_version"),
                    "total_chars_target": int(req.total_chars_target),
                    "story_type": story_type,
                    "status": "CLASSIFIED",
                    "error_log": None,
                    "config_json": cfg,
                },
            )
            _sync_sheet(content_id)

            # Build step plan
            steps = _step_plan_from_prompts(story_type)
            desired_steps = int(req.steps) if req.steps else None

            if steps:
                # If user overrides, pad/truncate
                if desired_steps and desired_steps != len(steps):
                    if desired_steps < len(steps):
                        steps = steps[:desired_steps]
                    else:
                        extra = _default_steps(story_type, desired_steps)[len(steps) :]
                        steps.extend(extra)
            else:
                steps = _default_steps(story_type, desired_steps or 8)

            # Dynamic adjustment for output limits (keep per-step chunks small)
            max_chars_per_step = int(cfg.get("max_chars_per_step") or (6000 if provider in {"openai", "openrouter", "farai"} else 9000))
            if req.total_chars_target > max_chars_per_step * len(steps):
                target_steps = min(20, max(len(steps), int(math.ceil(req.total_chars_target / max_chars_per_step))))
                steps = _default_steps(story_type, target_steps)

            lengths = _allocate_lengths(int(req.total_chars_target), steps)
            for i, s in enumerate(steps):
                s["target_chars"] = lengths[i]
                s["target_tokens"] = max(200, int(min(max_tokens_per_step, max(300, lengths[i] / 4))))

            # Generate an outline aligned to the step plan (best-effort)
            update_job(job.id, progress={"phase": "outline"})
            outline_lines = _build_outline_for_steps(cleaned, story_type, steps, provider, model, cfg)
            for i in range(len(steps)):
                steps[i]["outline_line"] = outline_lines[i] if i < len(outline_lines) else ""

            update_content(content_id, {"steps_json": steps, "status": "CLASSIFIED"})
            _sync_sheet(content_id)

            # Generate step outputs
            generated_steps: List[Dict[str, Any]] = []
            prev = ""

            # Prefer trained Channel DNA (TRAINING_DB) over per-content heuristic
            trained_dna = None
            if c2.get("channel"):
                trained_dna = get_channel_dna(str(c2.get("channel")), story_type)
            channel_dna = {}
            if trained_dna and isinstance(trained_dna.get("dna_json"), dict):
                channel_dna = trained_dna.get("dna_json") or {}
            elif isinstance(c2.get("channel_dna_json"), dict):
                channel_dna = c2.get("channel_dna_json") or {}
            dna_text = ""
            if isinstance(channel_dna, dict) and channel_dna:
                dna_text = "\n".join([f"{k}: {v}" for k, v in channel_dna.items() if v])

            for idx, step in enumerate(steps):
                update_job(job.id, progress={"phase": "generating", "step": idx + 1, "total_steps": len(steps)})

                query = f"{step.get('step_name')}: {step.get('objective')}\n{cleaned[:1500]}"
                where = {"kind": "example"}
                if story_type:
                    where["story_type"] = story_type
                if c2.get("channel"):
                    where["channel"] = c2.get("channel")
                if step.get("example_ref"):
                    where["example_id"] = step.get("example_ref")

                snippets: List[str] = []
                try:
                    docs = rag_service.retrieve(query, k=5, where=where)
                    snippets = [d.page_content for d in docs if getattr(d, "page_content", None)]
                except Exception:
                    snippets = []

                system = (
                    "You are an expert long-form scriptwriter. Output plain text only. "
                    "Maintain consistent tone across steps."
                )

                user = (
                    f"STORY_TYPE: {story_type}\n"
                    f"STEP: {step.get('step_name')}\n"
                    f"OBJECTIVE: {step.get('objective')}\n"
                    f"OUTLINE_LINE: {step.get('outline_line') or ''}\n"
                    f"TARGET_CHARS: {step.get('target_chars')}\n\n"
                    f"CHANNEL_DNA (if any):\n{dna_text if dna_text else '[none]'}\n\n"
                    f"CLEANED_SOURCE_TEXT (for facts/context):\n{cleaned[:6000]}\n\n"
                    f"EXAMPLE_SNIPPETS (RAG):\n{('\n\n'.join(snippets)) if snippets else '[none]'}\n\n"
                    f"PREVIOUS_STEP_CONTEXT (continue, don\'t repeat):\n{prev[-2500:] if prev else '[none]'}\n\n"
                    f"STEP_PROMPT:\n{step.get('prompt_text') or ''}\n\n"
                    "Write this step now.\n"
                    "Rules:\n"
                    "- Keep it engaging and spoken.\n"
                    "- Avoid duplicated sentences.\n"
                    "- Use smooth transitions.\n"
                )

                token_budget = min(max_tokens_per_step, max(400, int(step.get("target_tokens") or 800)))
                try:
                    resp = await asyncio.to_thread(
                        _chat_with_retries,
                        provider=provider,
                        model=model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=0.8,
                        max_tokens=token_budget,
                        retries=retry_count,
                    )
                    text = (resp or "").strip()
                except Exception as e:
                    # No-error guarantee: if LLM provider is not configured (missing SDK/key),
                    # emit a deterministic fallback so the pipeline can still complete.
                    text = (
                        "[FALLBACK_STEP]\n"
                        f"Reason: {str(e)}\n\n"
                        f"{step.get('step_name')}\n"
                        f"Objective: {step.get('objective')}\n\n"
                        "Draft (from cleaned source text):\n"
                        + (cleaned[: int(step.get("target_chars") or 1200)] if cleaned else "[no source text]")
                    ).strip()
                prev = (prev + "\n\n" + text).strip()
                generated_steps.append({"step_index": idx, "step_name": step.get("step_name"), "text": text})

                update_content(content_id, {"generated_steps_json": generated_steps})
                _sync_sheet(content_id)

            # Merge + light tone normalization (optional)
            merged_text = "\n\n".join([s.get("text") or "" for s in generated_steps]).strip()

            update_content(content_id, {"status": "GENERATED"})
            _sync_sheet(content_id)

            # If LLM available, do a final polish in one pass (best-effort)
            polished = merged_text
            try:
                update_job(job.id, progress={"phase": "polishing"})
                ptxt = await asyncio.to_thread(
                    _chat_with_retries,
                    provider=provider,
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert editor. Output plain text only."},
                        {
                            "role": "user",
                            "content": (
                                "Unify tone and remove repetition. Keep factual integrity. Output plain text only.\n\n"
                                f"TEXT:\n{merged_text}\n"
                            ),
                        },
                    ],
                    temperature=0.4,
                    max_tokens=min(max_tokens_per_step, max(800, int(req.total_chars_target / 6))),
                    retries=retry_count,
                )
                if ptxt and len(ptxt) > 200:
                    polished = ptxt.strip()
            except Exception:
                polished = merged_text

            # Key term marking stage (for downstream visual asset generation)
            if cfg.get("mark_key_terms", True):
                polished = _mark_key_terms_heuristic(polished)

            assets = _sentence_asset_prompts(polished)
            # Ensure lineage: assets must reference parent_content_id only
            for a in assets:
                a["parent_content_id"] = content_id

            update_content(
                content_id,
                {
                    "final_output": polished,
                    "assets_json": assets,
                    "status": "OUTPUT",
                    "error_log": None,
                },
            )
            _sync_sheet(content_id)

            update_job(job.id, status="completed", progress={"phase": "done"}, result={"content_id": content_id})
        except Exception as e:
            _mark_error(content_id, str(e))
            update_job(job.id, status="failed", error=str(e), result={"content_id": content_id})

    asyncio.create_task(_run())
    return {"job_id": job.id, "content_id": content_id}


@router.post("/contents/{content_id}/config")
def api_update_config(content_id: str, req: ConfigUpdateRequest):
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")
    cfg = _get_effective_config(c, req.config)
    update_content(content_id, {"config_json": cfg})
    _sync_sheet(content_id)
    return {"status": "ok", "content_id": content_id, "config_json": cfg}


@router.post("/contents/{content_id}/export")
def api_export(content_id: str, req: ExportRequest):
    c = get_content(content_id)
    if not c:
        raise HTTPException(status_code=404, detail="content_not_found")
    text = (c.get("final_output") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="no_final_output")

    fmts = []
    for f in req.formats or []:
        ff = (f or "").strip().lower()
        if ff in {"docx", "pdf", "txt"}:
            fmts.append(ff)
    if not fmts:
        fmts = ["docx", "pdf"]

    result = export_script(
        content_id=content_id,
        title=str(c.get("topic") or c.get("story_type") or "Script"),
        text=text,
        formats=fmts,  # type: ignore[arg-type]
        filename_prefix=req.filename_prefix,
    )

    # Store export artifacts in assets_json (derivative of content_id only)
    assets = c.get("assets_json") if isinstance(c.get("assets_json"), list) else []
    assets = list(assets or [])
    assets.append({"parent_content_id": content_id, "kind": "exports", "result": result})
    update_content(content_id, {"assets_json": assets})
    _sync_sheet(content_id)
    return result
