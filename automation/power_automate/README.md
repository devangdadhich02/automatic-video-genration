## PowerAutomate Integration (SNS Auto-Posting)

This folder documents how to connect the backend and n8n to Microsoft PowerAutomate
for automatic posting to text-based (LinkedIn, X, etc.) and video-based (YouTube, etc.) SNS.

### 1. Create a new Flow with HTTP trigger

- **Trigger**: `When a HTTP request is received`
- Copy the generated URL – this is your **webhook URL**.
- This URL is passed to:
  - The backend `/sns/post` endpoint (`SNSPostRequest.webhook_url`), or
  - The n8n flow (`sns_webhook_url` field in `video_automation_flow.json`).

Expected JSON payload:

```json
{
  "title": "Video title",
  "text": "Long text body / script / summary",
  "long_video_url": "https://.../long_form.mp4",
  "short_video_url": "https://.../short_form.mp4",
  "tags": ["finance", "trading"]
}
```

### 2. Fan-out actions inside PowerAutomate

Inside the Flow, add actions like:

- **LinkedIn**: `Share an article` / `Create a post`
  - Use `title` + `text` + `long_video_url` or `short_video_url` in the message.
- **YouTube**: use a connector or custom API to upload video from `long_video_url`.
- **Email (Outlook/Gmail)**: send an email to scraped email lists with `title` and `text`.
- **Others (X, Facebook, Instagram)**: use corresponding connectors or APIs.

The idea: PowerAutomate becomes the “SNS fan-out layer”, while your backend +
n8n handle heavy work (RAG, video generation).


