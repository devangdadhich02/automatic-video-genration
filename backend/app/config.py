import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for the backend services."""

    # General
    ENV: str = os.getenv("ENV", "dev")
    BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

    # LLM / embeddings
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    # OpenAI-compatible provider (OpenRouter etc.)
    OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")
    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
    # Optional: Far.ai (OpenAI-compatible). If you use this provider, configure both.
    FARAI_API_KEY: str | None = os.getenv("FARAI_API_KEY")
    FARAI_BASE_URL: str = os.getenv("FARAI_BASE_URL", "https://api.far.ai/v1")
    # Recommended default: fast + high quality.
    # Note: "gpt-5.2-mini" / "gpt-5.1-mini" are not valid model ids in your models list.
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
    # Total background job time cap for long script generation (seconds).
    # The async job will keep generating parts until complete or until this limit.
    SCRIPT_JOB_MAX_SECONDS: float = float(os.getenv("SCRIPT_JOB_MAX_SECONDS", "900"))

    # Vector DB
    VECTOR_DB_PROVIDER: str = os.getenv(
        "VECTOR_DB_PROVIDER", "chroma"
    )  # chroma | pinecone | weaviate
    VECTOR_DB_DIR: str = os.path.join(DATA_DIR, "chroma_db")

    # Pinecone (optional)
    PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV: str | None = os.getenv("PINECONE_ENV")
    PINECONE_INDEX: str | None = os.getenv("PINECONE_INDEX")

    # B-roll sources
    PIXABAY_API_KEY: str | None = os.getenv("PIXABAY_API_KEY")
    PEXELS_API_KEY: str | None = os.getenv("PEXELS_API_KEY")

    # Cloud storage (generic – plug in S3/GCS/Supabase storage etc.)
    CLOUD_BUCKET_URL: str | None = os.getenv("CLOUD_BUCKET_URL")

    # FFmpeg
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")

    # Supabase (used mainly by the web UI directly, but kept here for jobs if needed)
    SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: str | None = os.getenv("SUPABASE_ANON_KEY")

    # Google Sheets (optional control panel + storage sync)
    # If not configured, the system uses local SQLite (`backend/data/content.db`) as the source of truth.
    GOOGLE_SHEETS_ID: str | None = os.getenv("GOOGLE_SHEETS_ID")
    # Service account json file path (recommended) OR raw JSON string.
    GOOGLE_SERVICE_ACCOUNT_JSON: str | None = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    # YouTube bulk transcript extraction (channel_link)
    YOUTUBE_CHANNEL_MAX_VIDEOS: int = int(os.getenv("YOUTUBE_CHANNEL_MAX_VIDEOS", "5"))
    YOUTUBE_TRANSCRIPT_LANGS: str = os.getenv("YOUTUBE_TRANSCRIPT_LANGS", "en,en-US,en-GB")

    # Exports
    EXPORTS_DIR: str = os.path.join(OUTPUT_DIR, "exports")


settings = Settings()


