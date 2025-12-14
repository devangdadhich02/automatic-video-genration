import os
from pathlib import Path
from typing import List

import httpx

from .config import settings


class BrollService:
    """Fetch B-roll from Pixabay / Pexels and optionally upload to cloud.

    For simplicity we only download locally; you can extend `upload_to_cloud`
    to push to S3, Supabase storage, etc.
    """

    def __init__(self):
        self.assets_dir = Path(settings.ASSETS_DIR)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url: str, dest: Path) -> Path:
        with httpx.stream("GET", url) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest

    def fetch_broll(self, query: str, max_items: int = 5) -> List[str]:
        paths: List[str] = []

        if self._is_valid_key(settings.PIXABAY_API_KEY):
            paths.extend(self._fetch_from_pixabay(query, max_items))
        if self._is_valid_key(settings.PEXELS_API_KEY) and len(paths) < max_items:
            remaining = max_items - len(paths)
            paths.extend(self._fetch_from_pexels(query, remaining))

        return paths

    def _is_valid_key(self, key: str | None) -> bool:
        """Treat empty and placeholder values as missing."""
        if not key:
            return False
        lowered = key.strip().lower()
        placeholders = {
            "your-pixabay-api-key",
            "your-pexels-api-key",
            "pixabay_api_key",
            "pexels_api_key",
            "changeme",
        }
        return lowered not in placeholders and "your-" not in lowered

    def _fetch_from_pixabay(self, query: str, max_items: int) -> List[str]:
        url = "https://pixabay.com/api/"
        params = {
            "key": settings.PIXABAY_API_KEY,
            "q": query,
            "image_type": "photo",
            "per_page": max_items,
        }
        try:
            r = httpx.get(url, params=params, timeout=30)
            r.raise_for_status()
        except Exception:
            # If API key is invalid or rate-limited, just return no results.
            return []

        data = r.json()
        hits = data.get("hits", [])
        paths: List[str] = []
        for hit in hits[:max_items]:
            img_url = hit.get("largeImageURL")
            if not img_url:
                continue
            filename = os.path.basename(img_url.split("?")[0])
            dest = self.assets_dir / filename
            self._download_file(img_url, dest)
            paths.append(str(dest))
        return paths

    def _fetch_from_pexels(self, query: str, max_items: int) -> List[str]:
        if max_items <= 0:
            return []
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": settings.PEXELS_API_KEY}
        params = {"query": query, "per_page": max_items}
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception:
            return []

        data = r.json()
        photos = data.get("photos", [])
        paths: List[str] = []
        for photo in photos[:max_items]:
            src = photo.get("src", {})
            img_url = src.get("large") or src.get("original")
            if not img_url:
                continue
            filename = os.path.basename(img_url.split("?")[0])
            dest = self.assets_dir / filename
            self._download_file(img_url, dest)
            paths.append(str(dest))
        return paths

    def upload_to_cloud(self, path: str) -> str:
        """Placeholder for cloud upload.

        Implement this with your preferred provider (Supabase storage, S3, etc.)
        and return a public URL.
        """
        if not settings.CLOUD_BUCKET_URL:
            return path
        # Example pattern (pseudo):
        # return f"{settings.CLOUD_BUCKET_URL}/{os.path.basename(path)}"
        return path


broll_service = BrollService()


