from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .cleaning import CleanInputs, clean_and_merge
from .llm_router import chat, safe_provider_and_model
from .rag import rag_service
from .script_jobs import create_job, get_job, update_job
from .sheets_store import upsert_content_row, upsert_training_record
from .storage import (
    create_content,
    get_content,
    list_contents,
    list_step_prompts,
    list_story_types,
    update_content,
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

            story_type = (req.manual_story_type or c.get("story_type") or "").strip() or _basic_story_type_guess(cleaned)

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

            # Channel DNA extraction + version increment (TRAINING only, but safe)
            dna = _extract_channel_dna(cleaned)
            version = int(c.get("version") or 1) + 1
            update_content(content_id, {"channel_dna_json": dna, "version": version})

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
            # Ensure CLEANED exists
            cleaned = (c.get("cleaned_text") or "").strip()
            if not cleaned:
                merged = (c.get("merged_text") or "").strip()
                if merged:
                    cleaned = merged
                else:
                    raise RuntimeError("content_not_cleaned")

            story_type = (req.story_type or c.get("story_type") or "").strip() or _basic_story_type_guess(cleaned)

            provider, model = safe_provider_and_model(req.ai_provider or c.get("ai_provider"), req.ai_model or c.get("ai_model"))

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
            max_chars_per_step = 6000 if provider in {"openai", "openrouter"} else 9000
            if req.total_chars_target > max_chars_per_step * len(steps):
                target_steps = min(20, max(len(steps), int(math.ceil(req.total_chars_target / max_chars_per_step))))
                steps = _default_steps(story_type, target_steps)

            lengths = _allocate_lengths(int(req.total_chars_target), steps)
            for i, s in enumerate(steps):
                s["target_chars"] = lengths[i]
                s["target_tokens"] = max(200, int(lengths[i] / 4))

            update_content(content_id, {"steps_json": steps, "status": "GENERATED"})
            _sync_sheet(content_id)

            # Generate step outputs
            generated_steps: List[Dict[str, Any]] = []
            prev = ""

            channel_dna = c.get("channel_dna_json") or {}
            dna_text = ""
            if isinstance(channel_dna, dict) and channel_dna:
                dna_text = "\n".join([f"{k}: {v}" for k, v in channel_dna.items() if v])

            for idx, step in enumerate(steps):
                update_job(job.id, progress={"phase": "generating", "step": idx + 1, "total_steps": len(steps)})

                query = f"{step.get('step_name')}: {step.get('objective')}\n{cleaned[:1500]}"
                where = {"kind": "example"}
                if story_type:
                    where["story_type"] = story_type
                if c.get("channel"):
                    where["channel"] = c.get("channel")
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

                token_budget = min(2500, max(400, int(step.get("target_tokens") or 800)))
                try:
                    resp = await asyncio.to_thread(
                        chat,
                        provider=provider,
                        model=model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=0.8,
                        max_tokens=token_budget,
                    )
                    text = resp.text.strip()
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

            # If LLM available, do a final polish in one pass (best-effort)
            polished = merged_text
            try:
                update_job(job.id, progress={"phase": "polishing"})
                p = await asyncio.to_thread(
                    chat,
                    provider=provider,
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert editor. Output plain text only."},
                        {
                            "role": "user",
                            "content": (
                                f"Unify tone and remove repetition. Keep factual integrity.\n\nTEXT:\n{merged_text}\n"
                            ),
                        },
                    ],
                    temperature=0.4,
                    max_tokens=min(2500, max(800, int(req.total_chars_target / 6))),
                )
                if p.text and len(p.text) > 200:
                    polished = p.text.strip()
            except Exception:
                polished = merged_text

            assets = _sentence_asset_prompts(polished)

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
