from typing import Any, Dict, List, Optional
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .rag import rag_service
from .script_service import ScriptGenerationOptions, script_service
from .script_jobs import create_job, get_job, update_job
from .video_pipeline import video_pipeline
from .sns import sns_service
from .storage import init_db
from .sheets_store import ensure_sheets_schema
from .pipeline_v2 import router as pipeline_v2_router

app = FastAPI(title="Video Automation Backend", version="0.1.0")

# Allow the static frontend (and tools like n8n) to call this API during dev.
# For production, restrict origins to your deployed dashboard domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_v2_router)

@app.on_event("startup")
def _startup() -> None:
    # Always ensure local DB exists (no-error baseline).
    init_db()
    # Optional Google Sheets schema (only if configured).
    try:
        ensure_sheets_schema()
    except Exception:
        # Never block startup if Sheets is misconfigured.
        pass


class IngestRequest(BaseModel):
    source: str
    text: str
    metadata: Dict[str, Any] | None = None


class GenerateScriptRequest(BaseModel):
    outline: str
    query: Optional[str] = None
    target_minutes: int = 60
    language: str = "English"
    mode: str = "fast"  # fast | sequential


class GenerateVideoRequest(BaseModel):
    script: str
    narration_audio_path: Optional[str] = None


class SNSPostRequest(BaseModel):
    title: str
    text_body: str
    long_video_url: Optional[str] = None
    short_video_url: Optional[str] = None
    tags: List[str] | None = None
    webhook_url: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest/script")
def ingest_script(req: IngestRequest) -> Dict[str, Any]:
    """Called from Google Apps Script / n8n after scraping reviews, SNS, etc."""
    try:
        count = rag_service.ingest_text(
            req.text,
            metadata={"source": req.source, **(req.metadata or {})},
        )
        return {"chunks_added": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/script")
def generate_script(req: GenerateScriptRequest) -> Dict[str, Any]:
    """Generate a long-form script using RAG + OpenAI."""
    try:
        # Try to retrieve research from RAG (even if empty, that's okay)
        docs = []
        try:
            docs = rag_service.retrieve(req.query or req.outline, k=12)
            research_snippets = [d.page_content for d in docs]
        except Exception:
            # If RAG fails, continue with empty research
            research_snippets = []

        opts = ScriptGenerationOptions(
            outline=req.outline,
            research_snippets=research_snippets,
            target_minutes=req.target_minutes,
            language=req.language,
        )
        # script_service.generate_script() handles OpenAI errors internally
        # and returns a fallback script if needed - it should NOT raise exceptions
        script_text = script_service.generate_script(opts)

        # Return script plus light metadata so n8n / MCP can inspect sources.
        return {
            "script": script_text,
            "research_count": len(research_snippets),
            "sources": [getattr(d, "metadata", {}) for d in docs] if docs else [],
        }
    except Exception as e:
        # Only catch unexpected errors that script_service couldn't handle
        error_msg = str(e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {error_msg}")


@app.post("/generate/script/async")
async def generate_script_async(req: GenerateScriptRequest) -> Dict[str, Any]:
    """Async script generation: returns job_id immediately, frontend polls status."""
    job = create_job()
    update_job(job.id, status="running", started_at=asyncio.get_event_loop().time(), progress={"phase": "starting"})

    async def _run():
        try:
            docs = []
            try:
                docs = rag_service.retrieve(req.query or req.outline, k=12)
                research_snippets = [d.page_content for d in docs]
            except Exception:
                research_snippets = []

            opts = ScriptGenerationOptions(
                outline=req.outline,
                research_snippets=research_snippets,
                target_minutes=req.target_minutes,
                language=req.language,
            )

            # FAST MODE: generate sections in parallel to reduce wall-clock time.
            # SEQUENTIAL MODE: generate part-by-part (slower but slightly more coherent).
            start = asyncio.get_event_loop().time()
            deadline = start + float(getattr(settings, "SCRIPT_JOB_MAX_SECONDS", 900))

            title, sections = script_service.parse_outline(req.outline)
            if not sections:
                sections = ["Main Topic"]
            total_sections = len(sections)

            # Time allocation
            total_minutes = max(5, int(req.target_minutes))
            intro_minutes = min(4, max(2, total_minutes // 15))
            outro_minutes = min(4, max(2, total_minutes // 15))
            body_minutes = max(1, total_minutes - intro_minutes - outro_minutes)
            per_section_minutes = max(3, body_minutes // max(1, total_sections))

            update_job(
                job.id,
                progress={"phase": "intro", "section_index": 0, "total_sections": total_sections},
                result={"script": "", "research_count": len(research_snippets), "sources": [getattr(d, "metadata", {}) for d in docs] if docs else [], "partial": True},
            )

            intro_text = await asyncio.to_thread(
                script_service.generate_intro,
                opts,
                title or "Video",
                intro_minutes,
            )

            # Generate sections
            section_texts: Dict[int, str] = {}
            completed = 0

            async def run_one(i: int, section_title: str, next_title: Optional[str]):
                return i, await asyncio.to_thread(
                    script_service.generate_section,
                    opts,
                    title or "Video",
                    section_title,
                    next_title,
                    per_section_minutes,
                )

            update_job(
                job.id,
                progress={"phase": "sections", "section_index": 0, "total_sections": total_sections},
                result={"script": intro_text, "research_count": len(research_snippets), "sources": [getattr(d, "metadata", {}) for d in docs] if docs else [], "partial": True},
            )

            tasks = []
            if req.mode == "sequential":
                for i, s in enumerate(sections):
                    if asyncio.get_event_loop().time() > deadline:
                        raise RuntimeError("job_time_cap_reached")
                    nxt = sections[i + 1] if i + 1 < total_sections else None
                    idx, text = await run_one(i, s, nxt)
                    section_texts[idx] = text
                    completed += 1
                    update_job(
                        job.id,
                        progress={"phase": "sections", "section_index": completed, "total_sections": total_sections},
                        result={"script": intro_text + "\n\n" + "\n\n".join(section_texts.get(k, "") for k in range(completed)), "research_count": len(research_snippets), "sources": [getattr(d, "metadata", {}) for d in docs] if docs else [], "partial": True},
                    )
            else:
                for i, s in enumerate(sections):
                    nxt = sections[i + 1] if i + 1 < total_sections else None
                    tasks.append(asyncio.create_task(run_one(i, s, nxt)))

                for coro in asyncio.as_completed(tasks):
                    if asyncio.get_event_loop().time() > deadline:
                        raise RuntimeError("job_time_cap_reached")
                    idx, text = await coro
                    section_texts[idx] = text
                    completed = len(section_texts)
                    # Provide partial script in correct order (fill missing with nothing)
                    ordered = [section_texts.get(i, "") for i in range(total_sections)]
                    partial_script = "\n\n".join([intro_text] + [t for t in ordered if t])
                    update_job(
                        job.id,
                        progress={"phase": "sections", "section_index": completed, "total_sections": total_sections},
                        result={"script": partial_script, "research_count": len(research_snippets), "sources": [getattr(d, "metadata", {}) for d in docs] if docs else [], "partial": True},
                    )

            # Outro
            update_job(job.id, progress={"phase": "outro", "section_index": total_sections, "total_sections": total_sections})
            outro_text = await asyncio.to_thread(
                script_service.generate_outro,
                opts,
                title or "Video",
                outro_minutes,
                "\n\n".join([intro_text] + [section_texts[i] for i in range(total_sections)]),
            )

            full_script = "\n\n".join([intro_text] + [section_texts[i] for i in range(total_sections)] + [outro_text]).strip()

            update_job(
                job.id,
                status="completed",
                finished_at=asyncio.get_event_loop().time(),
                result={
                    "script": full_script,
                    "research_count": len(research_snippets),
                    "sources": [getattr(d, "metadata", {}) for d in docs] if docs else [],
                    "partial": False,
                },
                error=None,
                progress={"phase": "done", "section_index": total_sections, "total_sections": total_sections},
            )
        except Exception as e:
            update_job(
                job.id,
                status="failed",
                finished_at=asyncio.get_event_loop().time(),
                error=str(e),
            )

    asyncio.create_task(_run())
    return {"job_id": job.id, "status": job.status}


@app.get("/generate/script/status/{job_id}")
def generate_script_status(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "progress": getattr(job, "progress", None),
    }


@app.post("/generate/video")
def generate_video(req: GenerateVideoRequest) -> Dict[str, Any]:
    """Create long + short video variants from a script."""
    try:
        result = video_pipeline.generate_long_and_short(
            script=req.script, narration_audio=req.narration_audio_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/video/async")
async def generate_video_async(req: GenerateVideoRequest) -> Dict[str, Any]:
    """Async video generation: returns job_id immediately, poll status."""
    job = create_job()
    update_job(job.id, status="running", started_at=asyncio.get_event_loop().time())

    async def _run():
        try:
            result = await asyncio.to_thread(
                video_pipeline.generate_long_and_short,
                req.script,
                req.narration_audio_path,
            )
            update_job(job.id, status="completed", finished_at=asyncio.get_event_loop().time(), result=result)
        except Exception as e:
            update_job(job.id, status="failed", finished_at=asyncio.get_event_loop().time(), error=str(e))

    asyncio.create_task(_run())
    return {"job_id": job.id, "status": job.status}


@app.get("/generate/video/status/{job_id}")
def generate_video_status(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {"job_id": job.id, "status": job.status, "result": job.result, "error": job.error}


@app.post("/sns/post")
async def sns_post(req: SNSPostRequest) -> Dict[str, Any]:
    """Build a clean SNS payload and forward to PowerAutomate / n8n webhook."""
    payload = sns_service.build_payload(
        title=req.title,
        text_body=req.text_body,
        long_video_url=req.long_video_url,
        short_video_url=req.short_video_url,
        tags=req.tags,
    )
    try:
        resp = await sns_service.post_to_webhook(req.webhook_url, payload)
        return {"status": "forwarded", "webhook_response": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FullPipelineRequest(BaseModel):
    """End-to-end pipeline input used by n8n / UI / MCP."""

    outline: str
    narration_audio_path: Optional[str] = None
    webhook_url: Optional[str] = None
    tags: List[str] | None = None
    target_minutes: int = 60
    language: str = "English"


@app.post("/pipeline/full")
async def full_pipeline(req: FullPipelineRequest) -> Dict[str, Any]:
    """End-to-end: outline -> script (RAG + OpenAI) -> videos -> optional SNS webhook."""
    script_resp = generate_script(
        GenerateScriptRequest(
            outline=req.outline,
            target_minutes=req.target_minutes,
            language=req.language,
        )
    )
    videos = generate_video(
        GenerateVideoRequest(
            script=script_resp["script"],
            narration_audio_path=req.narration_audio_path,
        )
    )

    result: Dict[str, Any] = {"script": script_resp["script"], "videos": videos}

    if req.webhook_url:
        payload = sns_service.build_payload(
            title=req.outline[:120],
            text_body=script_resp["script"][:1500],
            long_video_url=videos.get("long"),
            short_video_url=videos.get("short"),
            tags=req.tags,
        )
        try:
            resp = await sns_service.post_to_webhook(req.webhook_url, payload)
            result["sns_result"] = resp
        except Exception as e:
            result["sns_error"] = str(e)

    return result


@app.post("/pipeline/full/async")
async def full_pipeline_async(req: FullPipelineRequest) -> Dict[str, Any]:
    """Async full pipeline: outline -> script -> videos -> optional SNS webhook."""
    job = create_job()
    update_job(job.id, status="running", started_at=asyncio.get_event_loop().time())

    async def _run():
        try:
            script_resp = generate_script(
                GenerateScriptRequest(
                    outline=req.outline,
                    target_minutes=req.target_minutes,
                    language=req.language,
                )
            )
            videos = await asyncio.to_thread(
                video_pipeline.generate_long_and_short,
                script_resp["script"],
                req.narration_audio_path,
            )
            result: Dict[str, Any] = {"script": script_resp["script"], "videos": videos}

            if req.webhook_url:
                payload = sns_service.build_payload(
                    title=req.outline[:120],
                    text_body=script_resp["script"][:1500],
                    long_video_url=videos.get("long"),
                    short_video_url=videos.get("short"),
                    tags=req.tags,
                )
                try:
                    resp = await sns_service.post_to_webhook(req.webhook_url, payload)
                    result["sns_result"] = resp
                except Exception as e:
                    result["sns_error"] = str(e)

            update_job(job.id, status="completed", finished_at=asyncio.get_event_loop().time(), result=result)
        except Exception as e:
            update_job(job.id, status="failed", finished_at=asyncio.get_event_loop().time(), error=str(e))

    asyncio.create_task(_run())
    return {"job_id": job.id, "status": job.status}


@app.get("/pipeline/full/status/{job_id}")
def full_pipeline_status(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {"job_id": job.id, "status": job.status, "result": job.result, "error": job.error}


