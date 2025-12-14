import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import settings
from .broll import BrollService


@dataclass
class Scene:
    text: str
    image_paths: List[str]
    duration: float


class VideoPipeline:
    """High-level script → video pipeline.

    This is intentionally opinionated but simple:
    - Split script into scenes (by paragraph / delimiter)
    - For each scene, fetch B-roll images
    - Build a slideshow-style video
    - Optionally attach pre-generated narration audio using FFmpeg
    """

    def __init__(self):
        self.assets_dir = Path(settings.ASSETS_DIR)
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.broll = BrollService()

    def script_to_scenes(self, script: str, seconds_per_scene: int = 10) -> List[Scene]:
        paragraphs = [p.strip() for p in script.split("\n") if p.strip()]
        scenes: List[Scene] = []
        for p in paragraphs:
            # Simple heuristic: first few words become the search keyword
            keyword = " ".join(p.split()[:5])
            image_paths = self.broll.fetch_broll(keyword, max_items=3)
            # If no B-roll available (missing keys, rate-limits), create a local placeholder image
            if not image_paths:
                image_paths = [self._create_placeholder_image(p)]
            scenes.append(Scene(text=p, image_paths=image_paths, duration=seconds_per_scene))
        return scenes

    def _create_placeholder_image(self, text: str) -> str:
        """Create a simple placeholder image so pipeline works without Pixabay/Pexels keys."""
        from PIL import Image, ImageDraw, ImageFont

        w, h = 1280, 720
        img = Image.new("RGB", (w, h), color=(10, 20, 40))
        draw = ImageDraw.Draw(img)

        # Basic wrap
        clean = " ".join(text.split())
        if len(clean) > 220:
            clean = clean[:220] + "..."

        # Try a default font, fall back if not found
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            font = ImageFont.load_default()

        margin = 80
        lines = []
        line = ""
        for word in clean.split(" "):
            test = (line + " " + word).strip()
            if draw.textlength(test, font=font) <= (w - 2 * margin):
                line = test
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        y = h // 2 - (len(lines) * 50) // 2
        for ln in lines[:8]:
            draw.text((margin, y), ln, fill=(230, 235, 245), font=font)
            y += 55

        # Save
        name = f"placeholder_{uuid.uuid4().hex}.png"
        path = str(self.assets_dir / name)
        img.save(path, "PNG")
        return path

    def build_video_from_scenes(
        self,
        scenes: List[Scene],
        audio_path: str | None = None,
        output_name: str | None = None,
    ) -> str:
        # Lazy import so that backend can run even if moviepy is misconfigured;
        # errors surface only when video generation is requested.
        #
        # MoviePy v2 removed `moviepy.editor` import path, so support both v1 and v2.
        try:
            from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips  # type: ignore
        except ModuleNotFoundError:
            try:
                # MoviePy v2 style
                from moviepy import AudioFileClip, ImageClip, concatenate_videoclips  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "MoviePy import failed. Please install/repair MoviePy in the backend virtualenv.\n"
                    "Try: pip install -U moviepy\n"
                    "If it still fails, pin MoviePy v1: pip install 'moviepy==1.0.3'"
                ) from exc

        if not output_name:
            output_name = f"video_{uuid.uuid4().hex}.mp4"

        clips: List[ImageClip] = []
        for scene in scenes:
            if not scene.image_paths:
                continue
            for img in scene.image_paths:
                clip = ImageClip(img)
                # MoviePy v1: set_duration(); MoviePy v2: with_duration()
                if hasattr(clip, "set_duration"):
                    clip = clip.set_duration(scene.duration)
                elif hasattr(clip, "with_duration"):
                    clip = clip.with_duration(scene.duration)
                else:
                    # Fallback: set attribute if API changes
                    clip.duration = scene.duration  # type: ignore[attr-defined]
                clips.append(clip)

        if not clips:
            raise ValueError("No clips generated from scenes – B-roll might be missing.")

        video_clip = concatenate_videoclips(clips, method="compose")

        # Temporary file without audio
        temp_path = str(self.output_dir / f"temp_{output_name}")
        final_path = str(self.output_dir / output_name)

        video_clip.write_videofile(temp_path, fps=24)
        video_clip.close()

        if audio_path:
            # Merge video and audio using FFmpeg
            cmd = [
                settings.FFMPEG_PATH,
                "-y",
                "-i",
                temp_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                final_path,
            ]
            subprocess.run(cmd, check=True)
            os.remove(temp_path)
        else:
            final_path = temp_path

        return final_path

    def generate_long_and_short(
        self,
        script: str,
        narration_audio: str | None = None,
        long_seconds_per_scene: int = 30,
        short_seconds_per_scene: int = 5,
    ) -> dict:
        """Generate both long-form (e.g. ~1hr) and short-form versions.

        To reach ~1hr, adjust `long_seconds_per_scene` and script length.
        """
        long_scenes = self.script_to_scenes(script, seconds_per_scene=long_seconds_per_scene)
        short_scenes = self.script_to_scenes(script, seconds_per_scene=short_seconds_per_scene)

        long_path = self.build_video_from_scenes(long_scenes, narration_audio, "long_form.mp4")
        short_path = self.build_video_from_scenes(short_scenes, narration_audio, "short_form.mp4")
        return {"long": long_path, "short": short_path}


video_pipeline = VideoPipeline()


