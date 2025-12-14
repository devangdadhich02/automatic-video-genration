from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ScriptJob:
    id: str
    status: str  # queued | running | completed | failed
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


# Simple in-memory store (good for local/dev). For production, move to Redis/Supabase.
_jobs: Dict[str, ScriptJob] = {}


def create_job() -> ScriptJob:
    job_id = uuid.uuid4().hex
    job = ScriptJob(id=job_id, status="queued", created_at=time.time())
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[ScriptJob]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    for k, v in kwargs.items():
        setattr(job, k, v)


