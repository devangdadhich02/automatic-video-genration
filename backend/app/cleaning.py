from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import httpx


@dataclass
class CleanInputs:
    raw_text: str | None = None
    web_url: str | None = None
    youtube_url: str | None = None
    channel_link: str | None = None


def _collapse_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_sentences(text: str) -> List[str]:
    # Conservative splitter (works decently for English and many latin scripts)
    t = _collapse_spaces(text)
    if not t:
        return []

    # protect common abbreviations
    t = re.sub(r"\b(e\.g|i\.e|mr|mrs|dr|vs)\.", lambda m: m.group(0).replace(".", "<DOT>"), t, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+", t)
    out: List[str] = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if p:
            out.append(p)
    return out


def dedupe_sentences(text: str) -> str:
    """Remove repeated/duplicated sentences while preserving order."""

    sents = _split_sentences(text)
    seen = set()
    out = []
    for s in sents:
        key = re.sub(r"\W+", "", s).lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return " ".join(out).strip()


def organize_by_paragraph(text: str, *, max_chars_per_paragraph: int = 900) -> str:
    """Organize text into readable paragraphs.

    - Collapses whitespace
    - Splits into sentences
    - Re-wraps into paragraphs by character budget
    """

    sents = _split_sentences(text)
    if not sents:
        return ""

    paras: List[str] = []
    buf: List[str] = []
    size = 0

    for s in sents:
        if size + len(s) + 1 > max_chars_per_paragraph and buf:
            paras.append(" ".join(buf).strip())
            buf = [s]
            size = len(s)
        else:
            buf.append(s)
            size += len(s) + 1

    if buf:
        paras.append(" ".join(buf).strip())

    return "\n\n".join([p for p in paras if p]).strip()


async def extract_main_text_from_url(url: str, *, timeout: float = 30.0) -> str:
    """Fetch URL and extract main article body.

    Most efficient method: `trafilatura` if installed.
    Fallback: BeautifulSoup text extraction.
    """

    if not url:
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text

    html = html or ""

    # Prefer trafilatura
    try:
        import trafilatura  # type: ignore

        downloaded = trafilatura.extract(html, include_comments=False, include_tables=False)
        if downloaded:
            return _collapse_spaces(downloaded)
    except Exception:
        pass

    # Fallback: BeautifulSoup
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        # remove junk
        for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "nav"]):
            tag.decompose()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return _collapse_spaces(text)
    except Exception:
        # Worst-case: strip tags crudely
        text = re.sub(r"<[^>]+>", " ", html)
        return _collapse_spaces(text)


def _youtube_video_id(url_or_id: str) -> Optional[str]:
    u = (url_or_id or "").strip()
    if not u:
        return None

    # If it's already a bare ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", u):
        return u

    # Common URL patterns
    m = re.search(r"v=([A-Za-z0-9_-]{11})", u)
    if m:
        return m.group(1)

    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", u)
    if m:
        return m.group(1)

    m = re.search(r"/shorts/([A-Za-z0-9_-]{11})", u)
    if m:
        return m.group(1)

    return None


async def extract_youtube_transcript(url_or_id: str, *, prefer_langs: Sequence[str] = ("en", "en-US", "en-GB")) -> str:
    """Extract transcript text from a YouTube video.

    Uses `youtube-transcript-api` when available.
    """

    vid = _youtube_video_id(url_or_id)
    if not vid:
        return ""

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

        # Try preferred languages first; fallback to any.
        try:
            items = YouTubeTranscriptApi.get_transcript(vid, languages=list(prefer_langs))
        except Exception:
            items = YouTubeTranscriptApi.get_transcript(vid)

        lines = [it.get("text", "").strip() for it in items if it.get("text")]
        return _collapse_spaces("\n".join(lines))
    except Exception:
        return ""


async def clean_and_merge(inputs: CleanInputs) -> Tuple[str, str, List[str]]:
    """Return (merged_text, cleaned_text, sources_used)."""

    sources: List[str] = []
    pieces: List[str] = []

    if inputs.raw_text and inputs.raw_text.strip():
        pieces.append(inputs.raw_text.strip())
        sources.append("raw_text")

    if inputs.web_url and inputs.web_url.strip():
        try:
            web_text = await extract_main_text_from_url(inputs.web_url.strip())
        except Exception:
            web_text = ""
        if web_text:
            pieces.append(web_text)
            sources.append("web_url")

    if inputs.youtube_url and inputs.youtube_url.strip():
        yt_text = await extract_youtube_transcript(inputs.youtube_url.strip())
        if yt_text:
            pieces.append(yt_text)
            sources.append("youtube_url")

    # Channel link bulk extraction is optional; keep it non-blocking.
    # If yt-dlp is installed, you can extend this to pull last N videos.
    if inputs.channel_link and inputs.channel_link.strip():
        sources.append("channel_link")

    merged = "\n\n".join([p for p in pieces if p and p.strip()]).strip()
    merged = _collapse_spaces(merged)

    # Cleaned pipeline: dedupe -> organize paragraphs
    deduped = dedupe_sentences(merged)
    cleaned = organize_by_paragraph(deduped)

    return merged, cleaned, sources
