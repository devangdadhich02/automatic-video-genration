## Client Testing Guide (Frontend Dashboard) — Generation → Training

This document explains **exactly what to run/click** so your client can test the system from the frontend dashboard.

### 0) Quick prerequisites

- **Python already included** in this repo under `backend/venv/` (Windows).
- Your client needs to configure at least one LLM provider for “real AI output”:
  - **OpenAI**: set `OPENAI_API_KEY` and `OPENAI_MODEL=gpt-4o-mini`
  - or **OpenRouter**: set `OPENROUTER_API_KEY` (provider becomes `openrouter`)
  - or **Far.ai** (OpenAI-compatible): set `FARAI_API_KEY` (+ optional `FARAI_BASE_URL`)

If keys are missing, the system still runs end-to-end and returns a **fallback** script (so the UI doesn’t break), but quality will be lower.

### 1) One-time setup on client laptop (Windows)

#### 1.1 Backend `.env`

- Copy `backend/env.example` → `backend/.env`
- Fill required values:
  - `OPENAI_API_KEY=...`
  - `OPENAI_MODEL=gpt-4o-mini`
  - Optional:
    - `OPENROUTER_API_KEY=...`
    - `FARAI_API_KEY=...`
    - `GOOGLE_SHEETS_ID=...`
    - `GOOGLE_SERVICE_ACCOUNT_JSON=C:\\path\\to\\service-account.json`

#### 1.2 Install/update backend deps (important after updates)

Open PowerShell in the repo folder and run:

```bash
cd backend
.\venv\Scripts\pip.exe install -r requirements.txt
```

#### 1.3 Start backend

```bash
cd backend
.\venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
```

Confirm backend works:
- Open `http://localhost:8000/health` → should return `{"status":"ok"}`

#### 1.4 Start frontend (static dashboard)

Open a new PowerShell window:

```bash
cd web\static
python -m http.server 5500
```

Open dashboard:
- `http://localhost:5500`

If backend is on another machine/port, set in browser devtools console:

```js
window.BACKEND_BASE = "http://localhost:8000"
```

Refresh page.

---

## 2) GENERATION PHASE (client tests output creation)

Goal: Create a new row in the **single main table** (`content_id`) and generate output.

### 2.1 Create RAW input (OR-combination allowed)

In the dashboard card: **“Long-form System: Create RAW input”**

- Set `use_case = GENERATION`
- Fill any combination of:
  - `Long original text`
  - `PDF` (choose file) — PDF text will be extracted and appended to `raw_text`
  - `Web URL`
  - `YouTube URL`
  - `YouTube channel link` (bulk transcripts; best-effort)
- Click **Create RAW**

Expected:
- You get `content_id` in the response.
- Copy it.

### 2.2 Run steps (recommended order)

In **“Long-form System: Run pipeline steps”**

1) Paste `content_id`
2) Click **Clean**
   - Expected status becomes `CLEANED`
   - `merged_text` and `cleaned_text` get populated
3) Click **Classify**
   - Expected status becomes `CLASSIFIED`
   - `story_type` gets set (auto or manual)
4) Click **Generate**
   - Expected status becomes `OUTPUT`
   - `final_output` appears
   - `assets_json` contains sentence-level prompts with `parent_content_id`

Note:
- If you forget to click **Clean**, the backend now **auto-cleans during Generate** (no “content_not_cleaned” failure).

### 2.3 Suggested Generation options (fast test)

Open “Generation options” and set:
- `ai_provider`: `openai`
- `ai_model`: `gpt-4o-mini`
- `total_chars_target`: `2000` (fast)
- `steps`: `6` (fast)

### 2.4 Config + Export (optional)

Open “Config (stored in config_json column)”:

Example config:

```json
{"retry_count":2,"max_tokens_per_step":2000,"mark_key_terms":true}
```

Click:
- **Save config**
- After Generate completes, click:
  - **Export DOCX** or **Export PDF**

Expected:
- Export files saved in `backend/outputs/exports/`
- Export metadata is appended into `assets_json` (derivative of `content_id` only).

---

## 3) TRAINING PHASE (client tests “ingredients for writing”)

Goal: Store **story types, step prompts, examples, channel DNA** (rules/ingredients) so future generations become consistent.

### 3.1 Add story types

In **“Training DB: Story types”**:
- Click Refresh
- Add these examples (you can change later):
  - `Narrative`
  - `Problem-solving`
  - `News-type`

### 3.2 Add step prompts (5–15 steps per story type)

In **“Training DB: Step prompts”**:
- Set `type_id = Narrative`
- Add 6–8 steps (step_index 0..N)
- Optionally set `ratio` to control length distribution.

### 3.3 Upload examples (10+ ideal; 3 ok for quick test)

In **“Training DB: Examples”**:
- Upload example text for each story type (longer examples are better).
- System will auto-chunk and ingest into RAG for retrieval during generation.

### 3.4 Channel DNA (auto during TRAINING classify)

When you run TRAINING classify:
- `channel_dna_json` is extracted and stored (training-only)
- It is saved into training storage and used in later generation to keep tone consistent.

---

## 4) Troubleshooting (common issues)

### 4.1 “content_not_cleaned”
- Fix: click **Clean** first, or just **Generate** again (auto-clean is enabled).

### 4.2 Model errors (example: `gpt-5.2-mini`)
- Fix: use `gpt-4o-mini` for OpenAI.
- Backend also falls back automatically to `gpt-4o-mini` if OpenAI model is invalid.

### 4.3 YouTube channel link transcripts not working
- Requires `yt-dlp` installed (already in `requirements.txt`).
- Some channels/videos may not have transcripts available.

### 4.4 PDF extraction returns empty
- Scanned PDFs need OCR (not included by default).
- Try a text-based PDF first.


