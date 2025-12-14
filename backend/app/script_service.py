from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Optional

try:
    from openai import OpenAI  # type: ignore
    from openai import APIError, RateLimitError, APIConnectionError  # type: ignore

    _OPENAI_OK = True
except Exception:
    OpenAI = None  # type: ignore
    APIError = RateLimitError = APIConnectionError = Exception  # type: ignore
    _OPENAI_OK = False

from .config import settings


@dataclass
class ScriptGenerationOptions:
    """Inputs for script generation."""

    outline: str
    research_snippets: Sequence[str]
    target_minutes: int = 60
    language: str = "English"


class ScriptService:
    """OpenAI-first script generation.

    Key design goals:
    - Any language supported (handled by the model).
    - Long scripts supported via section-by-section generation (avoids output limits).
    - If OpenAI fails (quota/billing), return a minimal fallback message instead of fake content.
    """

    def __init__(self) -> None:
        if _OPENAI_OK:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT_SECONDS)  # type: ignore[misc]
        else:
            self._client = None
        self.model = settings.OPENAI_MODEL

    def generate_script(self, opts: ScriptGenerationOptions) -> str:
        """Synchronous generation (best for small target_minutes).

        For long scripts, prefer the async job endpoint which generates parts progressively.
        """
        state = self.init_state(opts)
        try:
            # Generate until completion (synchronous)
            while state["phase"] != "done":
                state = self.generate_next(state, opts)
            return "\n\n".join(state["parts"]).strip()
        except Exception as e:
            return self._fallback(opts, reason=str(e))

    def parse_outline(self, outline: str) -> Tuple[str, List[str]]:
        return self._parse_outline((outline or "").strip())

    def generate_intro(self, opts: ScriptGenerationOptions, title: str, minutes: int) -> str:
        return self._generate_part(
            language=opts.language,
            title=title,
            part_name="INTRO",
            minutes=minutes,
            outline=(opts.outline or "").strip(),
            research_text="\n\n".join([s for s in opts.research_snippets if s]).strip(),
            section_title=None,
            previous_context="",
        )

    def generate_section(
        self,
        opts: ScriptGenerationOptions,
        title: str,
        section_title: str,
        next_section_title: Optional[str],
        minutes: int,
    ) -> str:
        # Provide next section to enforce good bridging.
        bridge = f" Next section after this: {next_section_title}." if next_section_title else ""
        return self._generate_part(
            language=opts.language,
            title=title,
            part_name=f"SECTION: {section_title}",
            minutes=minutes,
            outline=(opts.outline or "").strip() + bridge,
            research_text="\n\n".join([s for s in opts.research_snippets if s]).strip(),
            section_title=section_title,
            previous_context="",
        )

    def generate_outro(self, opts: ScriptGenerationOptions, title: str, minutes: int, prior_text: str) -> str:
        return self._generate_part(
            language=opts.language,
            title=title,
            part_name="OUTRO",
            minutes=minutes,
            outline=(opts.outline or "").strip(),
            research_text="\n\n".join([s for s in opts.research_snippets if s]).strip(),
            section_title=None,
            previous_context=prior_text[-4000:] if prior_text else "",
        )

    def init_state(self, opts: ScriptGenerationOptions) -> dict:
        outline = (opts.outline or "").strip()
        research_text = "\n\n".join([s for s in opts.research_snippets if s]).strip()
        title, sections = self._parse_outline(outline)
        if not sections:
            sections = [outline] if outline else ["Main Topic"]

        total_minutes = max(5, int(opts.target_minutes))
        intro_minutes = min(4, max(2, total_minutes // 15))
        outro_minutes = min(4, max(2, total_minutes // 15))
        body_minutes = max(1, total_minutes - intro_minutes - outro_minutes)
        per_section_minutes = max(3, body_minutes // max(1, len(sections)))

        return {
            "title": title or "Video",
            "outline": outline,
            "research_text": research_text,
            "sections": sections,
            "intro_minutes": intro_minutes,
            "outro_minutes": outro_minutes,
            "per_section_minutes": per_section_minutes,
            "phase": "intro",  # intro | section | outro | done
            "index": 0,  # section index (0-based)
            "context": "",
            "parts": [],
        }

    def generate_next(self, state: dict, opts: ScriptGenerationOptions) -> dict:
        """Generate the next part and update state."""
        if not settings.OPENAI_API_KEY:
            state["phase"] = "done"
            state["parts"].append(self._fallback(opts, reason="OPENAI_API_KEY missing"))
            return state

        outline = state["outline"]
        research_text = state["research_text"]
        title = state["title"]
        ctx = state["context"]

        if state["phase"] == "intro":
            text = self._generate_part(
                language=opts.language,
                title=title,
                part_name="INTRO",
                minutes=state["intro_minutes"],
                outline=outline,
                research_text=research_text,
                section_title=None,
                previous_context="",
            )
            state["parts"].append(text)
            state["context"] = (ctx + "\n\n" + text)[-4000:]
            state["phase"] = "section"
            state["index"] = 0
            return state

        if state["phase"] == "section":
            sections = state["sections"]
            idx = int(state["index"])
            if idx >= len(sections):
                state["phase"] = "outro"
                return state

            sec = sections[idx]
            text = self._generate_part(
                language=opts.language,
                title=title,
                part_name=f"SECTION {idx + 1}",
                minutes=state["per_section_minutes"],
                outline=outline,
                research_text=research_text,
                section_title=sec,
                previous_context=ctx,
            )
            state["parts"].append(text)
            state["context"] = (ctx + "\n\n" + text)[-4000:]
            state["index"] = idx + 1
            return state

        if state["phase"] == "outro":
            text = self._generate_part(
                language=opts.language,
                title=title,
                part_name="OUTRO",
                minutes=state["outro_minutes"],
                outline=outline,
                research_text=research_text,
                section_title=None,
                previous_context=ctx,
            )
            state["parts"].append(text)
            state["context"] = (ctx + "\n\n" + text)[-4000:]
            state["phase"] = "done"
            return state

        state["phase"] = "done"
        return state

    def _generate_part(
        self,
        *,
        language: str,
        title: str,
        part_name: str,
        minutes: int,
        outline: str,
        research_text: str,
        section_title: str | None,
        previous_context: str,
    ) -> str:
        # 150 words/min is a good baseline for spoken voiceover
        target_words = max(300, minutes * 150)

        # System prompt: stable “writer persona” + output constraints
        system_prompt = (
            "You are a world-class YouTube long-form scriptwriter.\n"
            "Write natural, spoken voiceover text with strong pacing and engagement.\n"
            "IMPORTANT:\n"
            "- Output MUST be in the requested language.\n"
            "- Output MUST be plain text (no markdown, no bullet lists explaining what you did).\n"
            "- Avoid repetition; keep it dynamic and human.\n"
            "- Use smooth transitions, storytelling, and periodic micro-recaps.\n"
            "- When useful, add [VISUAL: ...] cues for charts, screenshots, b-roll, or screen recordings.\n"
        )

        # User prompt: part-specific instruction
        sec_line = f"SECTION TOPIC: {section_title}\n" if section_title else ""
        prev_line = (
            f"CONTEXT FROM PREVIOUS PART (keep continuity, do not repeat verbatim):\n{previous_context}\n\n"
            if previous_context
            else ""
        )

        user_prompt = (
            f"LANGUAGE: {language}\n"
            f"VIDEO TITLE: {title}\n"
            f"PART: {part_name}\n"
            f"TARGET LENGTH: ~{minutes} minutes (~{target_words} words)\n\n"
            f"{sec_line}"
            f"FULL OUTLINE:\n{outline}\n\n"
            f"RESEARCH SNIPPETS (optional, use when relevant):\n{research_text if research_text else '[none]'}\n\n"
            f"{prev_line}"
            "Write this part now.\n"
            "Rules:\n"
            "- Make this part self-contained but consistent with the overall outline.\n"
            "- If this is INTRO: hook hard in the first 30 seconds, preview what viewers will learn.\n"
            "- If this is a SECTION: explain deeply, add examples, and end with a bridge to the next topic.\n"
            "- If this is OUTRO: recap key takeaways and end with a strong call-to-action.\n"
            "- NEVER ask the user for missing context. If something is missing, make a reasonable assumption and continue.\n"
            "- Stay on-topic (business/entrepreneurship content if the outline is business). Do not drift into unrelated metaphors.\n"
        )

        # Keep calls reasonably fast; long scripts are built by stitching parts.
        # If the model truncates, we only continue a couple of times to avoid very long waits.
        TOKENS_PER_CALL = 1200
        CONTINUE_LIMIT = 2

        # Initial generation
        text, finish_reason = self._chat(system_prompt, user_prompt, max_tokens=TOKENS_PER_CALL, temperature=0.85)
        chunks = [text]

        # Continue if truncated
        safety_loops = 0
        while finish_reason == "length" and safety_loops < CONTINUE_LIMIT:
            safety_loops += 1
            # Provide the last generated text as context so the model can continue seamlessly.
            cont_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "\n".join(chunks).strip()},
                {
                    "role": "user",
                    "content": (
                        f"LANGUAGE: {language}\n"
                        f"Continue the SAME PART ({part_name}) from the EXACT point you stopped.\n"
                        "Do not restart. Do not summarize. Keep the same tone and structure.\n"
                        "Output plain text only.\n"
                        "NEVER ask for missing context.\n"
                    ),
                },
            ]
            more, finish_reason = self._chat_messages(cont_messages, max_tokens=TOKENS_PER_CALL, temperature=0.85)
            chunks.append(more)

        return "\n".join(c.strip() for c in chunks if c and c.strip()).strip()

    def _chat(self, system_prompt: str, user_prompt: str, *, max_tokens: int, temperature: float) -> Tuple[str, str]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self._chat_messages(messages, max_tokens=max_tokens, temperature=temperature)

    def _chat_messages(self, messages, *, max_tokens: int, temperature: float) -> Tuple[str, str]:

        # Model fallback strategy:
        # - If the configured model doesn't exist / no access, try a couple of known-fast, widely available models.
        # NOTE: `gpt-5.2-mini` is NOT a valid model id (there is `gpt-5-mini`).
        model_candidates = [self.model, "gpt-5.2-mini", "gpt-4o-mini"]
        seen = set()
        model_candidates = [m for m in model_candidates if m and not (m in seen or seen.add(m))]

        last_exc: Exception | None = None

        for model in model_candidates:
            try:
                resp = self._create_completion_with_adaptive_params(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    token_budget=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                finish_reason = getattr(resp.choices[0], "finish_reason", "") or ""
                return content.strip(), finish_reason
            except Exception as e:
                last_exc = e
                msg = str(e)
                # If model is invalid / not accessible, try next candidate
                if "model" in msg.lower() and ("does not exist" in msg.lower() or "model_not_found" in msg.lower()):
                    continue
                # Otherwise, surface the error (no point trying other models)
                break

        # If we couldn't satisfy any model, raise the last error
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenAI request failed with no exception details.")

    def _create_completion_with_adaptive_params(self, *, model: str, messages, temperature: float, token_budget: int):
        """Create a chat completion while adapting to model-specific parameter support."""
        if not self._client:
            raise RuntimeError("openai_sdk_missing_or_not_configured")
        # Try with max_tokens first
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=token_budget,
            )
        except Exception as e:
            msg = str(e)

            # If model doesn't support temperature values (only default), retry without temperature
            if "Unsupported value" in msg and "temperature" in msg:
                try:
                    return self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=token_budget,
                    )
                except Exception as e2:
                    msg2 = str(e2)
                    # If it also doesn't support max_tokens, fall through to next handler
                    msg = msg2
                    e = e2

            # If model requires max_completion_tokens, retry using that
            if "Unsupported parameter" in msg and "max_tokens" in msg and "max_completion_tokens" in msg:
                try:
                    return self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=token_budget,
                    )
                except Exception as e3:
                    msg3 = str(e3)
                    # If temperature also unsupported here, retry without temperature
                    if "Unsupported value" in msg3 and "temperature" in msg3:
                        return self._client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_completion_tokens=token_budget,
                        )
                    raise

            raise

    def _parse_outline(self, outline: str) -> Tuple[str, List[str]]:
        """Extract a title and a list of section titles from common outline formats."""
        if not outline:
            return "", []

        # Title: try "X-hour YouTube video: TITLE. Sections: ..."
        title = ""
        m = re.search(r"(?:\d+\s*-?\s*hour\s+YouTube\s+video:\s*)?(.+?)(?:\.\s*Sections?:|$)", outline, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            title = re.sub(r"^\d+\s*-?\s*hour\s+YouTube\s+video:\s*", "", title, flags=re.IGNORECASE).strip()

        # Sections: handle "1) ... 2) ..." patterns
        sections = re.findall(r"\d+\)\s*([^0-9]+?)(?=\s*\d+\)|$)", outline)
        sections = [s.strip().strip(",.") for s in sections if s.strip()]
        return title, sections

    def _fallback(self, opts: ScriptGenerationOptions, reason: str) -> str:
        # Minimal fallback: don’t pretend to generate in every language without a model.
        # This can happen due to billing/quota OR unsupported params for a chosen model.
        return (
            "[FALLBACK]\n"
            "OpenAI generation is currently unavailable (quota/billing/rate-limit or key missing).\n"
            f"Reason: {reason}\n\n"
            "To get a real dynamic script in the requested language and length, please fix OpenAI billing/quota OR switch to a compatible model/config.\n\n"
            "Request:\n"
            f"- language: {opts.language}\n"
            f"- target_minutes: {opts.target_minutes}\n\n"
            "Outline:\n"
            f"{opts.outline}\n"
        )

    # (no longer needed) time-cap notes are handled by async job progress


script_service = ScriptService()


