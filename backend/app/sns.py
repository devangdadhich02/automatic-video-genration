from typing import Any, Dict

import httpx


class SNSService:
    """Lightweight SNS + webhook integration layer.

    Idea:
    - This backend produces a clean JSON payload (title, text, video URLs, tags).
    - PowerAutomate / n8n / Apps Script receive it via webhook and fan out
      to LinkedIn, YouTube, X, Instagram, email, etc.
    """

    def build_payload(
        self,
        title: str,
        text_body: str,
        long_video_url: str | None = None,
        short_video_url: str | None = None,
        tags: list[str] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "title": title,
            "text": text_body,
            "long_video_url": long_video_url,
            "short_video_url": short_video_url,
            "tags": tags or [],
        }
        if extra:
            payload.update(extra)
        return payload

    async def post_to_webhook(self, webhook_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generic poster – use this for PowerAutomate / n8n HTTP trigger URLs."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(webhook_url, json=payload)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return {"status_code": r.status_code, "text": r.text}


sns_service = SNSService()


