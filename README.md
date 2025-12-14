## Video Automation System (n8n + RAG + FFmpeg + Supabase)

This project implements an end‑to‑end **1‑hour video automation system** based on your
client’s requirements:

- **n8n automation setup + flow replication**
- **RAG + Vector DB (Chroma by default; Pinecone/Weaviate pluggable)**
- **Script → image/video generation pipeline with FFmpeg**
- **Pixabay/Pexels API B‑roll fetch + local/cloud upload hook**
- **Long + short auto video rendering**
- **SNS auto‑posting via PowerAutomate (webhook)**
- **Basic Supabase UI/UX for customers + payments entry points**
- **Google Apps Script example instead of n8n for scraping / SNS analysis**

The repo is designed to be runnable on a single Windows machine, while offloading
SNS posting to PowerAutomate and optionally scraping/data collection to Google Apps Script.

---

### 1. Project structure

- `backend/`
  - `app/`
    - `config.py` – environment + API key config
    - `rag.py` – RAG service (Chroma by default)
    - `broll.py` – B‑roll downloader (Pixabay/Pexels)
    - `video_pipeline.py` – script → long/short video assembly (FFmpeg + MoviePy)
    - `sns.py` – webhook payload builder for PowerAutomate / n8n
    - `main.py` – FastAPI app with all endpoints
  - `requirements.txt` – Python deps
- `automation/`
  - `n8n/video_automation_flow.json` – example n8n flow (trigger → backend → PowerAutomate)
  - `google_apps_script/scraper.gs` – Apps Script to push scraped text to backend
  - `power_automate/README.md` – how to configure a Flow for SNS fan‑out
- `web/static/`
  - `index.html` – simple dashboard
  - `styles.css` – basic dark UI
  - `app.js` – Supabase auth + customer CRUD + pipeline trigger
- `ARCHITECTURE.md` – deeper architecture + sequence overview

---

### 2. Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and available in `PATH` (or configure `FFMPEG_PATH` in env)
- **Node/n8n (optional)** – if you want to import the flow
- **Supabase project** for:
  - Auth (magic links)
  - Tables: at least `customers` and optionally `payments`
- **API keys (optional but recommended)**:
  - `OPENAI_API_KEY` – for better script generation
  - `PIXABAY_API_KEY`, `PEXELS_API_KEY` – B‑roll fetching

Create a `.env` file in `backend/`:

```bash
ENV=dev
OPENAI_API_KEY=sk-...
PIXABAY_API_KEY=...
PEXELS_API_KEY=...
VECTOR_DB_PROVIDER=chroma
FFMPEG_PATH=ffmpeg
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY
```

---

### 3. Backend setup (FastAPI + RAG + FFmpeg)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Key endpoints:

- `GET /health` – health check
- `POST /ingest/script` – ingest scraped text into RAG
- `POST /generate/script` – outline + RAG → draft script
- `POST /generate/video` – script → `{ long, short }` video files
- `POST /sns/post` – build payload and forward to PowerAutomate webhook
- `POST /pipeline/full` – outline → script → videos → optional SNS webhook (for n8n/UI)

Outputs are written under `backend/outputs/`.

---

### 4. Web UI (Supabase + basic dashboard)

The UI is a **static HTML/JS dashboard** that:

- Authenticates users with Supabase magic links.
- Lists and creates entries in a `customers` table.
- Triggers the full pipeline by calling `POST /pipeline/full`.

Serve the static files any way you like, for local dev:

```bash
cd web/static
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser and:

1. Set global variables in dev tools console (or inline in `app.js`):
   ```js
   window.SUPABASE_URL = 'https://YOUR_PROJECT.supabase.co';
   window.SUPABASE_ANON_KEY = 'YOUR_ANON_KEY';
   window.BACKEND_BASE = 'http://localhost:8000';
   ```
2. Use the email box to send yourself a magic link.
3. After logging in, manage customers and trigger the pipeline.

Minimal Supabase schema (SQL):

```sql
create table if not exists public.customers (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null,
  status text default 'active',
  created_at timestamp with time zone default timezone('utc'::text, now())
);
```

You can add a `payments` table and link it by `customer_id` for billing.

---

### 5. n8n automation

1. Start n8n (Docker or local).
2. Import `automation/n8n/video_automation_flow.json`.
3. Adjust:
   - Backend URL (`http://backend:8000` → actual host)
   - SNS webhook URL (PowerAutomate HTTP trigger URL)
4. Trigger by:
   - HTTP webhook (with JSON body `{ "outline": "...", "query": "..." }`), or
   - Manual execution inside n8n.

The flow:

1. Receives outline.
2. Calls backend `/generate/script`.
3. Calls `/generate/video` with the script.
4. Posts payload to PowerAutomate which handles SNS posting.

---

### 6. Google Apps Script (free alternative to n8n for scraping)

- Open `automation/google_apps_script/scraper.gs` in Apps Script.
- Set `BACKEND_URL` to your public backend URL (`/ingest/script`).
- Attach it to a Google Sheet that contains reviews / SNS text.
- Run `scrapeAndSendToBackend()` manually or via time‑based trigger.

This gives you:

- Review / SNS text scraping.
- Free scheduled ingestion into the RAG vector DB.

You can create a **separate Apps Script project** for:

- SNS analytics (read stats from YouTube/LinkedIn and write to Sheet).
- Email sending (GmailApp) using the same data.

---

### 7. PowerAutomate (SNS auto‑posting)

See `automation/power_automate/README.md` for details.

High‑level:

- `When a HTTP request is received` → main trigger.
- Accept the JSON payload from backend/n8n.
- Add actions for:
  - LinkedIn post
  - YouTube upload from `long_video_url`
  - Email campaigns (Outlook/Gmail)
  - Any other SNS via connectors.

---

### 8. Extending for TradingView / finance data

For the *“sometimes need TradingView chart and financial data crawl”* part:

- Use Python or Apps Script to:
  - Fetch OHLC data from a broker/crypto API, or
  - Download chart images via TradingView chart URLs (respect TOS).
- Ingest text‑based analysis into `/ingest/script`.
- Save images under `backend/assets/` and pass them into `VideoPipeline` as extra scenes.

The core architecture is already ready to accept more sources – you just add
more ingestion and scene generation logic.

---

### 9. What’s left for deployment

This repo gives you a complete **working skeleton** with real code for:

- Backend API with RAG + B‑roll + FFmpeg video generation.
- Static dashboard connected to Supabase.
- n8n / Apps Script / PowerAutomate integration points.

To go to production, you still need to:

- Host the FastAPI app (e.g. on a VM with FFmpeg installed).
- Host static UI (any static hosting).
- Point Apps Script + PowerAutomate + n8n to the public backend URL.
- Add proper API keys and tighten auth (JWT, API keys, etc.).


-cd backend
-python -m venv .venv
-.\.venv\Scripts\activate
-pip install -r requirements.txt

-# Create backend/.env as described in README.md (OpenAI, Pixabay, Pexels, Supabase, etc.)

- uvicorn app.main:app --reload --port 8000

-cd web/static
-python -m http.server 5500

-window.SUPABASE_URL = 'https://YOUR_PROJECT.supabase.co';
-window.SUPABASE_ANON_KEY = 'YOUR_ANON_KEY';
-window.BACKEND_BASE = 'http://localhost:8000';



