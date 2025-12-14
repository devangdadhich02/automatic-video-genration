## Architecture Overview

This document explains how all moving parts fit together: **Apps Script / n8n / RAG /
video pipeline / Supabase / PowerAutomate**.

---

### 1. High‑level flow

1. **Data collection (Google Apps Script / scraping)**
   - Scrape reviews, SNS posts, financial commentary, etc.
   - Send consolidated text to backend `POST /ingest/script`.
   - Backend splits text, computes embeddings, and stores them in a **vector DB**.

2. **Outline → Script (RAG + LLM)**
   - n8n, UI, or any client sends an outline to `POST /generate/script` or `/pipeline/full`.
   - Backend retrieves most relevant chunks from the vector DB.
   - (Optionally) an LLM is used to write a full script for a ~1‑hour video.

3. **Script → Video (B‑roll + FFmpeg)**
   - Backend splits script into scenes.
   - For each scene, it calls Pixabay/Pexels APIs to download matching B‑roll.
   - Creates a long‑form video and a short‑form version using MoviePy + FFmpeg.

4. **SNS fan‑out (PowerAutomate / n8n)**
   - Backend builds a JSON payload containing:
     - Title, text body, tags
     - URLs or paths to long/short videos
   - Sends this payload to a **PowerAutomate HTTP trigger** (or n8n HTTP node).
   - PowerAutomate posts to LinkedIn, YouTube, email, etc.

5. **UI + customer/payments (Supabase)**
   - Static web dashboard uses Supabase for:
     - Auth (magic link sign‑in for operator/admin).
     - `customers` table (stores client accounts).
     - Optionally `payments` table for manual or Stripe‑integrated payments.
   - From the UI you can:
     - View customers
     - Create customers
     - Trigger the full pipeline for a given outline and SNS webhook URL.

---

### 2. Components

#### 2.1 Backend (`backend/app`)

- **`config.py`**
  - Centralizes environment variables (API keys, vector DB, FFmpeg path).

- **`rag.py`**
  - Uses LangChain + Chroma by default (`VECTOR_DB_PROVIDER=chroma`).
  - Responsibilities:
    - Split text into chunks.
    - Compute OpenAI embeddings.
    - Store / retrieve documents.
  - Can be swapped out for Pinecone or Weaviate by replacing `_init_vector_store`.

- **`broll.py`**
  - `BrollService.fetch_broll(query, max_items)`:
    - Queries Pixabay (`PIXABAY_API_KEY`) and Pexels (`PEXELS_API_KEY`).
    - Downloads images to `backend/assets/`.
    - (Optional) `upload_to_cloud` stub for Supabase storage / S3 / GCS.

- **`video_pipeline.py`**
  - `VideoPipeline.script_to_scenes(script)`:
    - Splits script into paragraphs.
    - For each paragraph, fetches a small set of B‑roll images.
  - `VideoPipeline.build_video_from_scenes(...)`:
    - Builds a slideshow‑style video using MoviePy.
    - Merges narration audio with FFmpeg if `narration_audio_path` is provided.
  - `VideoPipeline.generate_long_and_short(...)`:
    - Generates two videos:
      - Long‑form (configurable seconds per scene).
      - Short‑form (shorter scenes for reels/shorts).

- **`sns.py`**
  - `SNSService.build_payload(...)`:
    - Normalizes payload sent to external automation tools.
  - `SNSService.post_to_webhook(url, payload)`:
    - Posts JSON to PowerAutomate or n8n HTTP trigger.

- **`main.py` (FastAPI app)**
  - `/ingest/script`:
    - Ingests text from Apps Script/n8n into RAG.
  - `/generate/script`:
    - Uses RAG to stitch a draft script from outline + retrieved chunks.
    - Stub where you plug your own LLM call.
  - `/generate/video`:
    - Script → `{ long: path, short: path }`.
  - `/sns/post`:
    - Wraps `sns_service` to forward payloads to PowerAutomate webhook.
  - `/pipeline/full`:
    - Outline → script → videos → optional SNS webhook.
    - Designed for n8n or the web UI to fire everything in one call.

---

### 3. Automation layer

#### 3.1 n8n (`automation/n8n/video_automation_flow.json`)

Nodes:

1. **Webhook / Manual Trigger**
   - Accepts JSON body with at least `outline`.
2. **Generate Script (Backend)**
   - HTTP Request → `POST /generate/script`.
3. **Generate Videos**
   - HTTP Request → `POST /generate/video`.
4. **SNS Webhook**
   - HTTP Request → PowerAutomate HTTP trigger.

You can add more nodes, e.g.:

- Store metadata in a DB.
- Notify via Slack/Telegram when a pipeline run is complete.

#### 3.2 Google Apps Script (`automation/google_apps_script/scraper.gs`)

Use cases:

- Scrape text from:
  - Google Sheets (reviews, comments, SNS exports).
  - Websites or APIs (YouTube, LinkedIn, etc.).
- Send text to `/ingest/script` in one batch.

This gives you a **zero‑cost ingestion layer** that can be scheduled with time‑based
triggers instead of running n8n 24/7.

#### 3.3 PowerAutomate (`automation/power_automate/README.md`)

Acts as the **SNS dispatcher**:

- Receives JSON from backend/n8n.
- Fans out to:
  - LinkedIn posts
  - YouTube uploads
  - Emails & newsletters
  - Any supported SNS or CRM

This matches the requirement to **use PowerAutomate for SNS uploading** because
it has easy connectors and UI.

---

### 4. Web UI + Supabase

The UI is intentionally simple but functional to satisfy:

- Customer management
- A place to trigger 1‑hour video generation
- A start for payments integration

Files:

- `web/static/index.html` – layout & sections
- `web/static/styles.css` – modern dark theme styling
- `web/static/app.js` – logic

Core flows:

1. **Auth**
   - Uses Supabase magic link via `@supabase/supabase-js` CDN.
   - Only logged‑in users see the dashboard.

2. **Customers**
   - Fetches `customers` from Supabase.
   - Inserts new rows into `customers`.

3. **Pipeline trigger**
   - Reads outline + optional PowerAutomate webhook URL.
   - Calls backend `/pipeline/full`.
   - Displays returned script and file paths (or later URLs).

**Payments**:

- The UI is ready to be extended with:
  - A `payments` table in Supabase.
  - Integration of Stripe Checkout / payment links per customer.
- This keeps the current implementation simple while providing a clear path
  to full billing.

---

### 5. Data & file flows

- **Vector DB (RAG)**
  - Stored under `backend/data/chroma_db` (when using Chroma).
  - Ingested text is chunked and embedded; metadata includes source info.

- **Assets**
  - Downloaded B‑roll is stored under `backend/assets/`.
  - You can later sync this folder to cloud storage.

- **Outputs**
  - Generated videos are saved under `backend/outputs/` as `long_form.mp4` and `short_form.mp4`.
  - In production you would:
    - Upload them to cloud storage (Supabase storage, S3, etc.).
    - Use public URLs in the SNS payload instead of local paths.

---

### 6. How this maps to the client requirements

- **n8n automation setup + flow replication**
  - `automation/n8n/video_automation_flow.json` + `/pipeline/full` endpoint.

- **RAG + Vector DB integration**
  - `rag.py` + Chroma DB; easily extendable to Pinecone/Weaviate.

- **Script → image/video generation pipeline**
  - `video_pipeline.py` + `broll.py` + FFmpeg; long + short video generation.

- **Pixabay/Pexels API bulk B‑roll + cloud upload**
  - `broll.py` implements Pixabay/Pexels fetch and provides `upload_to_cloud` hook.

- **Long + short auto video rendering (FFmpeg)**
  - `VideoPipeline.generate_long_and_short` merges scenes into multiple outputs.

- **SNS auto‑posting (PowerAutomate hooks)**
  - `/sns/post` endpoint + PowerAutomate HTTP trigger (documented).

- **Basic Supabase UI/UX (customer mgmt + payments)**
  - Static dashboard (`web/static/`) with customers and pipeline trigger;
    payments easily added in Supabase.

- **Google Apps Script alternative + SNS analysis**
  - `scraper.gs` shows ingestion path; SNS analytics can be implemented as
    additional Apps Script functions reading SNS APIs and writing to Sheets,
    then ingested the same way.

This architecture is intentionally modular so you can **swap tools** (e.g.
use Apps Script instead of n8n, or Supabase storage instead of S3) without
changing the core flow.


