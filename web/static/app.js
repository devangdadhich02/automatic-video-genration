(function () {
  // Versioned key so old saved UI state doesn't appear after updates.
  const STORAGE_KEY = 'video_automation_v2_state_v2';

  function saveUiState(partial) {
    try {
      const prev = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      const next = { ...prev, ...(partial || {}), _savedAt: Date.now() };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {}
  }

  function loadUiState() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') || {};
    } catch {
      return {};
    }
  }

  function wirePersist(el, key, { event = 'input' } = {}) {
    if (!el) return;
    el.addEventListener(event, () => {
      saveUiState({ [key]: el.value });
    });
  }

  // Prevent duplicate polling loops after multiple refreshes/clicks.
  // IMPORTANT: must auto-release on completion/failure, otherwise buttons appear "stuck".
  const pollPromises = new Map();

  const SUPABASE_URL = window.SUPABASE_URL || 'https://YOUR_SUPABASE_URL';
  const SUPABASE_ANON_KEY = window.SUPABASE_ANON_KEY || 'YOUR_SUPABASE_ANON_KEY';

  const isSupabaseConfigured =
    SUPABASE_URL &&
    SUPABASE_ANON_KEY &&
    !SUPABASE_URL.includes('YOUR_SUPABASE_URL') &&
    !SUPABASE_ANON_KEY.includes('YOUR_SUPABASE_ANON_KEY');

  const supabaseClient = isSupabaseConfigured
    ? window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
    : null;

  const emailInput = document.getElementById('email');
  const resetSavedStateBtn = document.getElementById('reset-saved-state');
  const loginBtn = document.getElementById('login-btn');
  const authStatus = document.getElementById('auth-status');
  const dashboard = document.getElementById('dashboard');
  const authSection = document.getElementById('auth-section');
  const userInfo = document.getElementById('user-info');

  const customersList = document.getElementById('customers-list');
  const refreshCustomersBtn = document.getElementById('refresh-customers');
  const createCustomerBtn = document.getElementById('create-customer');
  const custNameInput = document.getElementById('cust-name');
  const custEmailInput = document.getElementById('cust-email');
  const customersCard = document.getElementById('customers-card');
  const createCustomerCard = document.getElementById('create-customer-card');

  const outlineInput = document.getElementById('outline');
  const languageInput = document.getElementById('language');
  const targetMinutesInput = document.getElementById('target-minutes');
  const snsWebhookInput = document.getElementById('sns-webhook-url');
  const generateScriptBtn = document.getElementById('generate-script');
  const triggerPipelineBtn = document.getElementById('trigger-pipeline');
  const scriptResult = document.getElementById('script-result');
  const legacyScriptText = document.getElementById('legacy-script-text');
  const legacySaveToStudioBtn = document.getElementById('legacy-save-to-studio');
  const legacyVideoBtn = document.getElementById('legacy-video');
  const legacyExportTxtBtn = document.getElementById('legacy-export-txt');
  const legacyExportDocxBtn = document.getElementById('legacy-export-docx');
  const legacyExportPdfBtn = document.getElementById('legacy-export-pdf');
  const pipelineResult = document.getElementById('pipeline-result');

  // Long-form system (v2)
  const useCaseSelect = document.getElementById('use-case');
  const rawDomainInput = document.getElementById('raw-domain');
  const rawTopicInput = document.getElementById('raw-topic');
  const rawSubTopicInput = document.getElementById('raw-sub-topic');
  const rawChannelInput = document.getElementById('raw-channel');
  const rawTextInput = document.getElementById('raw-text');
  const rawPdfInput = document.getElementById('raw-pdf');
  const rawWebUrlInput = document.getElementById('raw-web-url');
  const rawYoutubeUrlInput = document.getElementById('raw-youtube-url');
  const rawChannelLinkInput = document.getElementById('raw-channel-link');
  const generateV2Btn = document.getElementById('generate-v2');
  const v2Progress = document.getElementById('v2-progress');

  const contentIdInput = document.getElementById('content-id');
  const scriptEditor = document.getElementById('script-editor');
  const saveScriptBtn = document.getElementById('save-script');
  const exportTxtBtn = document.getElementById('export-txt');
  const generateVideoBtn = document.getElementById('generate-video');
  const refreshCurrentBtn = document.getElementById('refresh-current');
  const clearCurrentBtn = document.getElementById('clear-current');
  const videoDurationInput = document.getElementById('video-duration');
  const actionResult = document.getElementById('action-result');
  const jobResult = document.getElementById('job-result');
  const contentResult = document.getElementById('content-result');

  const genStoryTypeInput = document.getElementById('gen-story-type');
  const genProviderInput = document.getElementById('gen-provider');
  const genModelInput = document.getElementById('gen-model');
  const genPromptVersionInput = document.getElementById('gen-prompt-version');
  const genTotalCharsInput = document.getElementById('gen-total-chars');
  const genStepsInput = document.getElementById('gen-steps');
  const contentConfigInput = document.getElementById('content-config');
  const saveConfigBtn = document.getElementById('save-config');
  const exportDocxBtn = document.getElementById('export-docx');
  const exportPdfBtn = document.getElementById('export-pdf');
  const configResult = document.getElementById('config-result');
  const advancedCreateRawBtn = document.getElementById('advanced-create-raw');
  const advancedRunGenerateBtn = document.getElementById('advanced-run-generate');

  const refreshStoryTypesBtn = document.getElementById('refresh-story-types');
  const storyTypesList = document.getElementById('story-types-list');
  const stTypeIdInput = document.getElementById('st-type-id');
  const stNameInput = document.getElementById('st-name');
  const stDescInput = document.getElementById('st-desc');
  const stStructureInput = document.getElementById('st-structure');
  const stRulesInput = document.getElementById('st-rules');
  const saveStoryTypeBtn = document.getElementById('save-story-type');
  const storyTypeResult = document.getElementById('story-type-result');

  const spTypeIdInput = document.getElementById('sp-type-id');
  const refreshStepPromptsBtn = document.getElementById('refresh-step-prompts');
  const stepPromptsList = document.getElementById('step-prompts-list');
  const spPromptIdInput = document.getElementById('sp-prompt-id');
  const spStepIndexInput = document.getElementById('sp-step-index');
  const spStepNameInput = document.getElementById('sp-step-name');
  const spObjectiveInput = document.getElementById('sp-objective');
  const spPromptTextInput = document.getElementById('sp-prompt-text');
  const spExampleRefInput = document.getElementById('sp-example-ref');
  const spRatioInput = document.getElementById('sp-ratio');
  const saveStepPromptBtn = document.getElementById('save-step-prompt');
  const stepPromptResult = document.getElementById('step-prompt-result');

  const refreshExamplesBtn = document.getElementById('refresh-examples');
  const examplesList = document.getElementById('examples-list');
  const exIdInput = document.getElementById('ex-id');
  const exTypeIdInput = document.getElementById('ex-type-id');
  const exChannelInput = document.getElementById('ex-channel');
  const exTitleInput = document.getElementById('ex-title');
  const exSourceUrlInput = document.getElementById('ex-source-url');
  const exRawTextInput = document.getElementById('ex-raw-text');
  const saveExampleBtn = document.getElementById('save-example');
  const exampleResult = document.getElementById('example-result');

  const videoPreview = document.getElementById('video-preview');
  const videoPreviewLegacy = document.getElementById('video-preview-legacy');

  const BACKEND_BASE = window.BACKEND_BASE || 'http://localhost:8000';

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    const text = await res.text();
    try {
      return { ok: res.ok, status: res.status, json: JSON.parse(text) };
    } catch {
      return { ok: res.ok, status: res.status, json: { raw: text } };
    }
  }

  function pathToOutputUrl(p) {
    if (!p || typeof p !== 'string') return null;
    const parts = p.split(/[\\/]/);
    const name = parts[parts.length - 1];
    if (!name) return null;
    return `${BACKEND_BASE}/outputs/${encodeURIComponent(name)}`;
  }

  function renderVideoPreview(container, videos) {
    if (!container) return;
    container.innerHTML = '';
    if (!videos) return;
    const longUrl = pathToOutputUrl(videos.long || videos.long_path);
    const shortUrl = pathToOutputUrl(videos.short || videos.short_path);
    const url = longUrl || shortUrl;
    if (!url) return;
    const label = longUrl && shortUrl ? 'Long version preview' : 'Video preview';
    container.innerHTML = `
      <div class="hint">${label}</div>
      <video controls src="${url}" preload="metadata"></video>
    `;
  }

  async function pollJob(jobId, { onTick } = {}) {
    if (!jobId) return { status: 'invalid', job_id: jobId };
    if (pollPromises.has(jobId)) return await pollPromises.get(jobId);

    const p = (async () => {
      const deadline = Date.now() + 30 * 60 * 1000; // 30 minutes
      try {
        while (Date.now() < deadline) {
          const { json } = await fetchJson(`${BACKEND_BASE}/v2/jobs/${jobId}`);
          if (onTick) onTick(json);
          if (json.status === 'completed' || json.status === 'failed') return json;
          await new Promise((r) => setTimeout(r, 2000));
        }
        return { status: 'timeout', job_id: jobId };
      } finally {
        pollPromises.delete(jobId);
      }
    })();

    pollPromises.set(jobId, p);
    return await p;
  }

  async function loadContent(contentId) {
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${contentId}`);
    contentResult.textContent = JSON.stringify(json, null, 2);
    try {
      if (contentConfigInput) {
        const cfg = json.config_json || {};
        contentConfigInput.value = Object.keys(cfg).length ? JSON.stringify(cfg, null, 2) : '';
      }
      if (scriptEditor) {
        const txt = json.final_output || '';
        if (txt && (!scriptEditor.value || scriptEditor.value.trim().length < 30)) {
          scriptEditor.value = txt;
        }
      }
    } catch {}
    return json;
  }

  function getRawInputBody() {
    const body = {
      use_case: (useCaseSelect?.value || 'TRAINING').trim(),
      domain: rawDomainInput?.value?.trim() || null,
      topic: rawTopicInput?.value?.trim() || null,
      sub_topic: rawSubTopicInput?.value?.trim() || null,
      channel: rawChannelInput?.value?.trim() || null,
      raw_text: rawTextInput?.value || null,
      web_url: rawWebUrlInput?.value?.trim() || null,
      youtube_url: rawYoutubeUrlInput?.value?.trim() || null,
      channel_link: rawChannelLinkInput?.value?.trim() || null
    };
    return body;
  }

  function getGenerateBody() {
    return {
      story_type: genStoryTypeInput?.value?.trim() || null,
      ai_provider: genProviderInput?.value?.trim() || null,
      ai_model: genModelInput?.value?.trim() || null,
      prompt_version: genPromptVersionInput?.value?.trim() || null,
      total_chars_target: Number(genTotalCharsInput?.value || 30000),
      steps: genStepsInput?.value ? Number(genStepsInput.value) : null,
      config: safeJsonParse(contentConfigInput?.value || '')
    };
  }

  function setHidden(el, hidden) {
    if (!el) return;
    if (hidden) el.classList.add('hidden');
    else el.classList.remove('hidden');
  }

  function setActionStatus(msg) {
    if (!actionResult) return;
    actionResult.textContent = msg || '';
  }

  function setButtonsDisabled(disabled) {
    const btns = [
      saveScriptBtn,
      exportDocxBtn,
      exportPdfBtn,
      exportTxtBtn,
      generateVideoBtn,
      refreshCurrentBtn,
      clearCurrentBtn,
      generateV2Btn
    ];
    btns.forEach((b) => {
      if (b) b.disabled = !!disabled;
    });
  }

  let currentContentId = null;

  function restoreUiState() {
    const s = loadUiState();

    // Legacy "Generate Script" / "Trigger Video Pipeline" inputs
    if (outlineInput && s.legacy_outline) outlineInput.value = s.legacy_outline;
    if (languageInput && s.legacy_language) languageInput.value = s.legacy_language;
    if (targetMinutesInput && s.legacy_target_minutes) targetMinutesInput.value = s.legacy_target_minutes;
    if (snsWebhookInput && s.legacy_webhook) snsWebhookInput.value = s.legacy_webhook;
    if (legacyScriptText && s.legacy_script_text) legacyScriptText.value = s.legacy_script_text;

    // Script Studio inputs
    if (useCaseSelect && s.use_case) useCaseSelect.value = s.use_case;
    if (rawDomainInput && s.raw_domain) rawDomainInput.value = s.raw_domain;
    if (rawTopicInput && s.raw_topic) rawTopicInput.value = s.raw_topic;
    if (rawSubTopicInput && s.raw_sub_topic) rawSubTopicInput.value = s.raw_sub_topic;
    if (rawChannelInput && s.raw_channel) rawChannelInput.value = s.raw_channel;
    if (rawTextInput && s.raw_text) rawTextInput.value = s.raw_text;
    if (rawWebUrlInput && s.raw_web_url) rawWebUrlInput.value = s.raw_web_url;
    if (rawYoutubeUrlInput && s.raw_youtube_url) rawYoutubeUrlInput.value = s.raw_youtube_url;
    if (rawChannelLinkInput && s.raw_channel_link) rawChannelLinkInput.value = s.raw_channel_link;

    // Generation options + config
    if (genStoryTypeInput && s.gen_story_type) genStoryTypeInput.value = s.gen_story_type;
    if (genProviderInput && s.gen_provider) genProviderInput.value = s.gen_provider;
    if (genModelInput && s.gen_model) genModelInput.value = s.gen_model;
    if (genPromptVersionInput && s.gen_prompt_version) genPromptVersionInput.value = s.gen_prompt_version;
    if (genTotalCharsInput && s.gen_total_chars) genTotalCharsInput.value = s.gen_total_chars;
    if (genStepsInput && s.gen_steps) genStepsInput.value = s.gen_steps;
    if (contentConfigInput && s.config_json) contentConfigInput.value = s.config_json;
    if (videoDurationInput && s.video_duration) videoDurationInput.value = s.video_duration;

    // Current working content/script
    if (s.current_content_id) {
      currentContentId = s.current_content_id;
      if (contentIdInput) {
        contentIdInput.value = s.current_content_id;
        // Only show content_id if a script already exists (so UX stays clean).
        setHidden(contentIdInput, !s.script_editor);
      }
    }
    if (scriptEditor && s.script_editor) {
      scriptEditor.value = s.script_editor;
    }
  }

  function renderV2Progress(j) {
    if (!v2Progress) return;
    const p = j.progress || {};
    const phase = p.phase || j.status || 'running';
    if (phase === 'auto_cleaning' || phase === 'cleaning') {
      v2Progress.textContent = 'Cleaning input text…';
      return;
    }
    if (phase === 'outline') {
      v2Progress.textContent = 'Creating outline…';
      return;
    }
    if (phase === 'generating') {
      const step = Number(p.step || 0);
      const total = Number(p.total_steps || 0);
      if (total > 0 && step > 0) {
        v2Progress.textContent = `Generating script… step ${step}/${total}`;
      } else {
        v2Progress.textContent = 'Generating script…';
      }
      return;
    }
    if (phase === 'polishing') {
      v2Progress.textContent = 'Polishing tone…';
      return;
    }
    if (j.status === 'completed') {
      v2Progress.textContent = 'Done.';
      return;
    }
    if (j.status === 'failed') {
      v2Progress.textContent = 'Failed.';
      return;
    }
    v2Progress.textContent = 'Working…';
  }

  function renderVideoProgress(j) {
    if (!jobResult) return;
    const p = j.progress || {};
    const phase = p.phase || j.status || 'running';
    if (phase === 'video') {
      jobResult.textContent = 'Generating video…';
      setActionStatus('Rendering video scenes…');
    } else if (phase === 'done' || j.status === 'completed') {
      jobResult.textContent = 'Video done.';
    } else if (j.status === 'failed') {
      jobResult.textContent = 'Video failed.';
      setActionStatus('Video failed. Open Debug for details.');
    } else {
      jobResult.textContent = 'Generating video…';
      setActionStatus('Generating video…');
    }
  }

  async function resumeInFlightJobs() {
    const s = loadUiState();

    // Resume Script Studio generation job (v2)
    if (s.v2_job_id && s.v2_job_status === 'running' && s.current_content_id) {
      try {
        if (v2Progress) v2Progress.textContent = 'Resuming generation…';
        const final = await pollJob(s.v2_job_id, { onTick: renderV2Progress });
        if (final.status === 'completed') {
          const cid = s.current_content_id;
          currentContentId = cid;
          if (contentIdInput) {
            contentIdInput.value = cid;
            setHidden(contentIdInput, false);
          }
          const content = await loadContent(cid);
          if (scriptEditor) scriptEditor.value = content.final_output || scriptEditor.value || '';
          saveUiState({ v2_job_id: null, v2_job_status: null, script_editor: scriptEditor?.value || '' });
          if (v2Progress) v2Progress.textContent = 'Script ready. You can edit and export now.';
        }
        if (final.status === 'failed') {
          saveUiState({ v2_job_status: 'failed' });
        }
      } catch {}
    }

    // Resume video job
    if (s.video_job_id && s.video_job_status === 'running' && s.current_content_id) {
      try {
        const final = await pollJob(s.video_job_id, { onTick: renderVideoProgress });
        if (final.status === 'completed') {
          saveUiState({ video_job_id: null, video_job_status: null });
          await loadContent(s.current_content_id);
        }
      } catch {}
    }

    // Resume legacy "Generate Script" job
    if (s.legacy_script_job_id && s.legacy_script_job_status === 'running') {
      try {
        scriptResult.textContent = 'Resuming script generation…';
        const deadline = Date.now() + 15 * 60 * 1000;
        while (Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 2000));
          const statusRes = await fetch(`${BACKEND_BASE}/generate/script/status/${s.legacy_script_job_id}`);
          const statusJson = await statusRes.json();
          if (statusJson.progress) {
            const p = statusJson.progress;
            const idx = Number(p.section_index || 0);
            const total = Number(p.total_sections || 0);
            const phase = p.phase || 'running';
            if (total > 0) {
              scriptResult.textContent = `Generating... (${phase}) section ${Math.min(idx + 1, total)}/${total}`;
            } else {
              scriptResult.textContent = `Generating... (${phase})`;
            }
          }
          if (statusJson.status === 'completed') {
            scriptResult.textContent = statusJson.result?.script || JSON.stringify(statusJson, null, 2);
            saveUiState({ legacy_script_job_id: null, legacy_script_job_status: null });
            break;
          }
          if (statusJson.status === 'failed') {
            scriptResult.textContent = 'Job failed: ' + (statusJson.error || JSON.stringify(statusJson, null, 2));
            saveUiState({ legacy_script_job_status: 'failed' });
            break;
          }
        }
      } catch {}
    }
  }

  function clearCurrentScript() {
    // UI-only clear. Does NOT delete backend rows. Keeps inputs so user can regenerate quickly.
    currentContentId = null;
    try {
      if (contentIdInput) {
        contentIdInput.value = '';
        setHidden(contentIdInput, true);
      }
      if (scriptEditor) scriptEditor.value = '';
      if (v2Progress) v2Progress.textContent = '';
      setActionStatus('Cleared. You can generate a new script now.');

      // Stop resume/polling on refresh by clearing job pointers.
      saveUiState({
        current_content_id: null,
        script_editor: '',
        v2_job_id: null,
        v2_job_status: null,
        video_job_id: null,
        video_job_status: null
      });
    } catch {}
  }

  async function createRawOnly() {
    const body = getRawInputBody();
    const { json, ok } = await fetchJson(`${BACKEND_BASE}/v2/content/raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!ok || !json.content_id) throw new Error(JSON.stringify(json));
    const cid = json.content_id;

    // Optional: PDF upload as an additional input source
    const file = rawPdfInput?.files?.[0] || null;
    if (file) {
      const fd = new FormData();
      fd.append('file', file, file.name);
      await fetch(`${BACKEND_BASE}/v2/content/${cid}/upload/pdf`, { method: 'POST', body: fd });
    }
    saveUiState({ current_content_id: cid });
    return cid;
  }

  async function generateOneClickV2() {
    if (!v2Progress) return;
    v2Progress.textContent = 'Starting…';
    setHidden(contentIdInput, true);
    if (scriptEditor) scriptEditor.value = '';
    setActionStatus('');
    setButtonsDisabled(true);

    const useCase = (useCaseSelect?.value || 'GENERATION').trim().toUpperCase();

    // Enforce that a channel is selected first – this keeps `content_id` rows
    // anchored to a channel_id, and allows domain/story_type logic to stay stable.
    const channelVal = rawChannelInput?.value?.trim() || '';
    if (!channelVal) {
      v2Progress.textContent = 'Please select / enter a channel first. Channel is required before creating input.';
      setButtonsDisabled(false);
      return;
    }

    // Lightweight domain auto-fill: if domain is empty, mirror channel into domain
    // and keep the input visually read-only so operators treat it as locked.
    if (rawDomainInput && !rawDomainInput.value.trim()) {
      rawDomainInput.value = channelVal;
    }
    if (rawDomainInput) {
      rawDomainInput.readOnly = true;
    }
    if (useCase !== 'GENERATION') {
      // Training: only create the row; no script generation.
      const cid = await createRawOnly();
      currentContentId = cid;
      contentIdInput.value = cid;
      setHidden(contentIdInput, false);
      v2Progress.textContent = `TRAINING row created. content_id: ${cid}\nNow use Training DB panels to add story types/prompts/examples.`;
      await loadContent(cid);
      setButtonsDisabled(false);
      return;
    }

    const cid = await createRawOnly();
    currentContentId = cid;
    saveUiState({ current_content_id: cid });

    v2Progress.textContent = 'Working… (collecting + cleaning + generating)';
    const body = getGenerateBody();
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/generation/${cid}/generate/async`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!json.job_id) {
      v2Progress.textContent = 'Failed to start job: ' + JSON.stringify(json, null, 2);
      setButtonsDisabled(false);
      return;
    }
    saveUiState({ v2_job_id: json.job_id, v2_job_status: 'running' });

    const final = await pollJob(json.job_id, {
      onTick: renderV2Progress
    });

    if (final.status !== 'completed') {
      v2Progress.textContent = 'Job failed: ' + JSON.stringify(final, null, 2);
      saveUiState({ v2_job_status: 'failed' });
      setButtonsDisabled(false);
      return;
    }

    // Only now reveal content_id
    contentIdInput.value = cid;
    setHidden(contentIdInput, false);
    saveUiState({ current_content_id: cid });

    const content = await loadContent(cid);
    if (scriptEditor) scriptEditor.value = content.final_output || '';
    saveUiState({ script_editor: (scriptEditor?.value || '') });
    saveUiState({ v2_job_id: null, v2_job_status: null });
    v2Progress.textContent = 'Script ready. You can edit and export now.';
    setActionStatus('Script ready. You can edit, save, export, or generate video.');
    setButtonsDisabled(false);
  }

  async function saveScript() {
    const cid = (contentIdInput?.value || currentContentId || '').trim();
    if (!cid) return;
    if (!scriptEditor) return;
    setButtonsDisabled(true);
    setActionStatus('Saving script…');
    const body = { final_output: scriptEditor.value || '' };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${cid}/final`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    jobResult.textContent = JSON.stringify(json, null, 2);
    saveUiState({ current_content_id: cid, script_editor: scriptEditor.value || '' });
    await loadContent(cid);
    setActionStatus('Saved. ✅');
    setButtonsDisabled(false);
  }

  async function generateVideo() {
    const cid = (contentIdInput?.value || currentContentId || '').trim();
    if (!cid) return;
    const script = scriptEditor?.value || '';
    // UI uses minutes (more natural). Backend expects seconds.
    const targetMinutes = videoDurationInput?.value ? Number(videoDurationInput.value) : null;
    const targetSeconds = targetMinutes ? Math.max(1, Math.floor(targetMinutes * 60)) : null;
    saveUiState({ video_duration: videoDurationInput?.value || '' });
    setActionStatus('Starting video generation… (this can take time)');
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${cid}/video/async`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ script, target_duration_seconds: targetSeconds })
    });
    if (!json.job_id) return;
    saveUiState({ video_job_id: json.job_id, video_job_status: 'running' });
    jobResult.textContent = `Video job started. job_id=${json.job_id}`;
    const final = await pollJob(json.job_id, {
      onTick: (j) => {
        renderVideoProgress(j);
      }
    });
    saveUiState({ video_job_id: null, video_job_status: null });
    await loadContent(cid);
    if (final.status === 'completed') {
      const vids = final.result?.videos || final.result?.result?.videos || final.result?.result || final.result;
      // Compact, user-friendly summary of output paths
      jobResult.textContent = JSON.stringify(vids, null, 2);
      renderVideoPreview(videoPreview, vids);
      const longPath = vids?.long || vids?.long_path || '';
      const shortPath = vids?.short || vids?.short_path || '';
      const lines = ['Video done ✅. Preview below. Files saved in backend/outputs/.'];
      if (longPath) lines.push(`Long: ${longPath}`);
      if (shortPath) lines.push(`Short: ${shortPath}`);
      setActionStatus(lines.join('\n'));
    } else {
      const err = final.error || 'Unknown error – open Debug for full JSON.';
      setActionStatus('Video failed: ' + err);
    }
  }

  async function refreshStoryTypes() {
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/training/story-types`);
    storyTypesList.innerHTML = '';
    (json.items || []).forEach((st) => {
      const li = document.createElement('li');
      li.textContent = `${st.type_id}: ${st.name}`;
      li.addEventListener('click', () => {
        stTypeIdInput.value = st.type_id || '';
        stNameInput.value = st.name || '';
        stDescInput.value = st.description || '';
        stStructureInput.value = st.structure_json ? JSON.stringify(st.structure_json, null, 2) : '';
        stRulesInput.value = st.rules_json ? JSON.stringify(st.rules_json, null, 2) : '';
      });
      storyTypesList.appendChild(li);
    });
  }

  function safeJsonParse(s) {
    const t = (s || '').trim();
    if (!t) return {};
    try {
      return JSON.parse(t);
    } catch {
      return {};
    }
  }

  async function saveConfig() {
    const contentId = contentIdInput.value.trim();
    if (!contentId) return;
    if (!configResult) return;
    configResult.textContent = 'Saving config...';
    const body = { config: safeJsonParse(contentConfigInput?.value || '') };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${contentId}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    configResult.textContent = JSON.stringify(json, null, 2);
    saveUiState({ config_json: contentConfigInput?.value || '' });
    await loadContent(contentId);
  }

  async function exportFormats(formats) {
    const contentId = (contentIdInput?.value || currentContentId || '').trim();
    if (!contentId) return;
    setButtonsDisabled(true);
    setActionStatus(`Exporting ${formats.join(', ')}…`);
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${contentId}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ formats })
    });
    if (configResult) configResult.textContent = JSON.stringify(json, null, 2);
    await loadContent(contentId);
    if (json && json.exports && json.exports.length) {
      const paths = json.exports.map((e) => e.path).filter(Boolean);
      setActionStatus(`Exported ✅\n${paths.join('\n')}`);
    } else {
      setActionStatus('Export finished (no files returned). Open Debug for details.');
    }
    setButtonsDisabled(false);
  }

  async function saveStoryType() {
    storyTypeResult.textContent = 'Saving...';
    const body = {
      type_id: stTypeIdInput.value.trim(),
      name: stNameInput.value.trim(),
      description: stDescInput.value || null,
      structure: safeJsonParse(stStructureInput.value),
      rules: safeJsonParse(stRulesInput.value)
    };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/training/story-types`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    storyTypeResult.textContent = JSON.stringify(json, null, 2);
    refreshStoryTypes();
  }

  async function refreshStepPrompts() {
    const typeId = spTypeIdInput.value.trim();
    if (!typeId) return;
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/training/step-prompts/${encodeURIComponent(typeId)}`);
    stepPromptsList.innerHTML = '';
    (json.items || []).forEach((p) => {
      const li = document.createElement('li');
      li.textContent = `#${p.step_index} ${p.step_name || ''}`.trim();
      li.addEventListener('click', () => {
        spPromptIdInput.value = p.prompt_id || '';
        spStepIndexInput.value = p.step_index;
        spStepNameInput.value = p.step_name || '';
        spObjectiveInput.value = p.objective || '';
        spPromptTextInput.value = p.prompt_text || '';
        spExampleRefInput.value = p.example_ref || '';
        spRatioInput.value = p.ratio != null ? String(p.ratio) : '';
      });
      stepPromptsList.appendChild(li);
    });
  }

  async function saveStepPrompt() {
    stepPromptResult.textContent = 'Saving...';
    const body = {
      prompt_id: spPromptIdInput.value.trim(),
      type_id: spTypeIdInput.value.trim(),
      step_index: Number(spStepIndexInput.value),
      step_name: spStepNameInput.value.trim() || null,
      objective: spObjectiveInput.value.trim() || null,
      prompt_text: spPromptTextInput.value || '',
      example_ref: spExampleRefInput.value.trim() || null,
      ratio: spRatioInput.value.trim() ? Number(spRatioInput.value) : null
    };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/training/step-prompts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    stepPromptResult.textContent = JSON.stringify(json, null, 2);
    refreshStepPrompts();
  }

  async function refreshExamples() {
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/training/examples?limit=50`);
    examplesList.innerHTML = '';
    (json.items || []).forEach((ex) => {
      const li = document.createElement('li');
      li.textContent = `${ex.example_id} (${ex.type_id || '-'}/${ex.channel || '-'})`;
      li.addEventListener('click', () => {
        exIdInput.value = ex.example_id || '';
        exTypeIdInput.value = ex.type_id || '';
        exChannelInput.value = ex.channel || '';
        exTitleInput.value = ex.title || '';
        exSourceUrlInput.value = ex.source_url || '';
        exRawTextInput.value = ex.raw_text || '';
      });
      examplesList.appendChild(li);
    });
  }

  async function saveExample() {
    exampleResult.textContent = 'Saving...';
    const body = {
      example_id: exIdInput.value.trim(),
      type_id: exTypeIdInput.value.trim() || null,
      channel: exChannelInput.value.trim() || null,
      title: exTitleInput.value.trim() || null,
      source_url: exSourceUrlInput.value.trim() || null,
      raw_text: exRawTextInput.value || ''
    };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/training/examples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    exampleResult.textContent = JSON.stringify(json, null, 2);
    refreshExamples();
  }

  async function sendMagicLink() {
    if (!supabaseClient) {
      authStatus.textContent = 'Supabase is not configured. Running in Guest Mode.';
      return;
    }
    const email = emailInput.value.trim();
    if (!email) {
      authStatus.textContent = 'Enter an email first.';
      return;
    }
    authStatus.textContent = 'Sending magic link...';
    const { error } = await supabaseClient.auth.signInWithOtp({ email });
    if (error) {
      authStatus.textContent = 'Error: ' + error.message;
    } else {
      authStatus.textContent = 'Magic link sent! Check your inbox.';
    }
  }

  async function checkSession() {
    // Guest Mode: allow script/pipeline testing without Supabase.
    if (!supabaseClient) {
      authSection.classList.add('hidden');
      dashboard.classList.remove('hidden');
      userInfo.textContent = 'Guest Mode (Supabase not configured)';
      if (customersCard) customersCard.classList.add('hidden');
      if (createCustomerCard) createCustomerCard.classList.add('hidden');
      return;
    }

    const {
      data: { session }
    } = await supabaseClient.auth.getSession();
    if (session) {
      authSection.classList.add('hidden');
      dashboard.classList.remove('hidden');
      userInfo.textContent = `Logged in as ${session.user.email}`;
      loadCustomers();
    }
  }

  async function loadCustomers() {
    if (!supabaseClient) return;
    customersList.innerHTML = 'Loading...';
    const { data, error } = await supabaseClient.from('customers').select('*').order('created_at', {
      ascending: false
    });
    if (error) {
      customersList.innerHTML = 'Error: ' + error.message;
      return;
    }
    customersList.innerHTML = '';
    data.forEach((row) => {
      const li = document.createElement('li');
      li.textContent = `${row.name} (${row.email}) - status: ${row.status || 'active'}`;
      customersList.appendChild(li);
    });
  }

  async function createCustomer() {
    if (!supabaseClient) return;
    const name = custNameInput.value.trim();
    const email = custEmailInput.value.trim();
    if (!name || !email) return;
    const { error } = await supabaseClient.from('customers').insert({
      name,
      email,
      status: 'active'
    });
    if (error) {
      alert('Error: ' + error.message);
      return;
    }
    custNameInput.value = '';
    custEmailInput.value = '';
    loadCustomers();
  }

  async function triggerPipeline() {
    const outline = outlineInput.value.trim();
    const cid = (contentIdInput?.value || currentContentId || '').trim();
    const editorText = (scriptEditor?.value || '').trim();

    // If no outline provided, but we have an edited script, generate video from that script (v2).
    if (!outline && cid && editorText) {
      pipelineResult.textContent = 'Starting video generation from Script Studio text…';
      try {
        const start = await fetchJson(`${BACKEND_BASE}/v2/contents/${cid}/video/async`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ script: editorText })
        });
        const jobId = start.json?.job_id;
        if (!jobId) {
          pipelineResult.textContent = 'Failed to start video job: ' + JSON.stringify(start.json, null, 2);
          return;
        }
        saveUiState({ video_job_id: jobId, video_job_status: 'running' });
        const final = await pollJob(jobId, {
          onTick: (j) => {
            const p = j.progress || {};
            pipelineResult.textContent = `Generating video… (${p.phase || j.status || 'running'})`;
          }
        });
        if (final.status === 'completed') {
          const vids = final.result?.videos || final.result?.result?.videos || final.result?.result || final.result;
          pipelineResult.textContent = `Video done ✅\n${JSON.stringify(vids, null, 2)}`;
          renderVideoPreview(videoPreviewLegacy, vids);
          await loadContent(cid);
          saveUiState({ video_job_id: null, video_job_status: null });
          return;
        }
        const err = final.error || 'Unknown error – see job JSON.';
        pipelineResult.textContent = 'Video job failed: ' + err;
        saveUiState({ video_job_status: 'failed' });
      } catch (err) {
        pipelineResult.textContent = 'Error: ' + err.message;
      }
      return;
    }

    if (!outline) {
      pipelineResult.textContent =
        'Provide an outline first.\n\nTip: If you already generated/edited a script in Script Studio, use the “Generate video” button there (or ensure content_id is available and try again).';
      return;
    }
    const webhookUrl = snsWebhookInput.value.trim() || null;
    pipelineResult.textContent = 'Starting pipeline…';
    try {
      const startRes = await fetch(`${BACKEND_BASE}/pipeline/full/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          outline,
          narration_audio_path: null,
          webhook_url: webhookUrl,
          tags: ['auto'],
          target_minutes: Number(targetMinutesInput?.value || 60),
          language: (languageInput?.value || 'English').trim() || 'English'
        })
      });
      const startJson = await startRes.json();
      const jobId = startJson.job_id;
      if (!jobId) {
        pipelineResult.textContent = 'Failed to start pipeline: ' + JSON.stringify(startJson, null, 2);
        return;
      }
      saveUiState({ legacy_pipeline_job_id: jobId, legacy_pipeline_job_status: 'running' });
      pipelineResult.textContent = `Running…`;

      const deadline = Date.now() + 300000; // 5 minutes UI wait
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 3000));
        const statusRes = await fetch(`${BACKEND_BASE}/pipeline/full/status/${jobId}`);
        const statusJson = await statusRes.json();
        if (statusJson.status === 'completed') {
          const result = statusJson.result || {};
          const script = result.script || '';
          const videos = result.videos || {};
          pipelineResult.textContent = `Pipeline done ✅\n\nVideos:\n${JSON.stringify(videos, null, 2)}`;
          renderVideoPreview(videoPreviewLegacy, videos);
          // Show script for editing in legacy area too
          if (legacyScriptText && script) {
            legacyScriptText.value = script;
            saveUiState({ legacy_script_text: script });
          }
          return;
        }
        if (statusJson.status === 'failed') {
          const err = statusJson.error || JSON.stringify(statusJson, null, 2);
          pipelineResult.textContent = 'Pipeline failed: ' + err;
          saveUiState({ legacy_pipeline_job_status: 'failed' });
          return;
        }
      }

      pipelineResult.textContent =
        'Still running in background.\n' +
        `Check status: ${BACKEND_BASE}/pipeline/full/status/${jobId}\n`;
    } catch (err) {
      pipelineResult.textContent = 'Error: ' + err.message;
    }
  }

  async function generateScript() {
    const outline = outlineInput.value.trim();
    if (!outline) {
      scriptResult.textContent = 'Provide an outline first.';
      return;
    }
    const language = (languageInput?.value || 'English').trim() || 'English';
    const targetMinutes = Number(targetMinutesInput?.value || 60);

    scriptResult.textContent = 'Starting…';
    try {
      const startRes = await fetch(`${BACKEND_BASE}/generate/script/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          outline,
          query: outline,
          target_minutes: targetMinutes,
          language
        })
      });

      const startJson = await startRes.json();
      const jobId = startJson.job_id;
      if (!jobId) {
        scriptResult.textContent = 'Failed to start job: ' + JSON.stringify(startJson, null, 2);
        return;
      }
      saveUiState({ legacy_script_job_id: jobId, legacy_script_job_status: 'running' });

      scriptResult.textContent = `Generating…`;

      const deadline = Date.now() + 900000; // 15 minutes max wait in UI
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2000));
        const statusRes = await fetch(`${BACKEND_BASE}/generate/script/status/${jobId}`);
        const statusJson = await statusRes.json();

        if (statusJson.progress) {
          const p = statusJson.progress;
          const idx = Number(p.section_index || 0);
          const total = Number(p.total_sections || 0);
          const phase = p.phase || 'running';
          if (total > 0) {
            scriptResult.textContent = `Generating... (${phase}) section ${Math.min(idx + 1, total)}/${total}\n(job: ${jobId})`;
          } else {
            scriptResult.textContent = `Generating... (${phase})\n(job: ${jobId})`;
          }
        }

        if (statusJson.status === 'completed') {
          const script = statusJson.result?.script || '';
          scriptResult.textContent = 'Done ✅';
          if (legacyScriptText && script) {
            legacyScriptText.value = script;
            saveUiState({ legacy_script_text: script });
          }
          saveUiState({ legacy_script_job_id: null, legacy_script_job_status: null });
          return;
        }
        if (statusJson.status === 'failed') {
          scriptResult.textContent = 'Job failed: ' + (statusJson.error || JSON.stringify(statusJson, null, 2));
          saveUiState({ legacy_script_job_status: 'failed' });
          return;
        }
      }

      scriptResult.textContent =
        'Still running. Your job will finish in background.\n' +
        `Job ID: ${jobId}\n` +
        `Check status: ${BACKEND_BASE}/generate/script/status/${jobId}\n`;
    } catch (err) {
      scriptResult.textContent = 'Error: ' + err.message;
    }
  }

  async function ensureStudioForLegacyScript() {
    const script = (legacyScriptText?.value || '').trim();
    if (!script) {
      alert('No script to save. Generate a script first.');
      return;
    }

    // Reuse an existing hidden studio content_id for legacy exports if available.
    const s = loadUiState();
    let cid = (s.legacy_studio_content_id || '').trim();
    if (!cid) {
      // Create a hidden content_id in Script Studio and store the script there so exports/video use the v2 pipeline.
      cid = await createRawOnly();
      saveUiState({ legacy_studio_content_id: cid });
    }

    currentContentId = cid;
    // Keep content_id hidden (user doesn't need it)
    if (contentIdInput) {
      contentIdInput.value = cid;
      setHidden(contentIdInput, true);
    }
    if (scriptEditor) scriptEditor.value = script;
    const saved = await fetchJson(`${BACKEND_BASE}/v2/contents/${cid}/final`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ final_output: script })
    });
    saveUiState({ current_content_id: cid, script_editor: script, legacy_script_text: script });
    if (!saved.ok) {
      throw new Error(JSON.stringify(saved.json));
    }
    setActionStatus('Saved to Script Studio ✅. Now use Export / Generate video.');
    return cid;
  }

  async function videoFromLegacyScript() {
    const script = (legacyScriptText?.value || '').trim();
    if (!script) {
      alert('No script available. Generate a script first.');
      return;
    }
    // Ensure script exists in Script Studio content so lineage is preserved
    if (!currentContentId) await ensureStudioForLegacyScript();
    
    // Use the same pipeline flow as triggerPipeline when generating from script
    const cid = (contentIdInput?.value || currentContentId || '').trim();
    if (!cid) {
      pipelineResult.textContent = 'Error: content_id not available. Please save script to Script Studio first.';
      return;
    }
    
    pipelineResult.textContent = 'Starting video generation from script…';
    try {
      const webhookUrl = snsWebhookInput?.value?.trim() || null;
      const start = await fetchJson(`${BACKEND_BASE}/v2/contents/${cid}/video/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script })
      });
      const jobId = start.json?.job_id;
      if (!jobId) {
        pipelineResult.textContent = 'Failed to start video job: ' + JSON.stringify(start.json, null, 2);
        return;
      }
      saveUiState({ video_job_id: jobId, video_job_status: 'running' });
      const final = await pollJob(jobId, {
        onTick: (j) => {
          const p = j.progress || {};
          pipelineResult.textContent = `Generating video… (${p.phase || j.status || 'running'})`;
        }
      });
      if (final.status === 'completed') {
        const vids = final.result?.videos || final.result?.result?.videos || final.result?.result || final.result;
        pipelineResult.textContent = `Video done ✅\n${JSON.stringify(vids, null, 2)}`;
        renderVideoPreview(videoPreviewLegacy, vids);
        
        // If webhook URL is provided, send notification (similar to pipeline flow)
        if (webhookUrl) {
          try {
            const payload = {
              title: (script.substring(0, 120) || 'Video Generated'),
              text: script.substring(0, 1500),
              long_video_url: vids?.long || vids?.long_path || null,
              short_video_url: vids?.short || vids?.short_path || null,
              tags: ['auto']
            };
            await fetch(webhookUrl, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            pipelineResult.textContent += '\n\nWebhook notification sent ✅';
          } catch (webhookErr) {
            pipelineResult.textContent += '\n\nWebhook error: ' + webhookErr.message;
          }
        }
        
        await loadContent(cid);
        saveUiState({ video_job_id: null, video_job_status: null });
        return;
      }
      const err = final.error || JSON.stringify(final, null, 2);
      pipelineResult.textContent = 'Video job failed: ' + err;
      saveUiState({ video_job_status: 'failed' });
    } catch (err) {
      pipelineResult.textContent = 'Error: ' + err.message;
    }
  }

  async function exportFromLegacy(format) {
    try {
      const cid = await ensureStudioForLegacyScript();
      if (!cid) return;
      scriptResult.textContent = `Exporting ${format.toUpperCase()}…`;
      const { ok, json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${cid}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ formats: [format] })
      });
      if (!ok) {
        scriptResult.textContent = 'Export failed: ' + JSON.stringify(json, null, 2);
        return;
      }
      const paths = (json.exports || []).map((e) => e.path).filter(Boolean);
      scriptResult.textContent = `Exported ✅\n${paths.join('\n') || JSON.stringify(json, null, 2)}`;
    } catch (e) {
      scriptResult.textContent = 'Export error: ' + String(e?.message || e);
    }
  }

  loginBtn.addEventListener('click', sendMagicLink);
  refreshCustomersBtn.addEventListener('click', loadCustomers);
  createCustomerBtn.addEventListener('click', createCustomer);
  generateScriptBtn.addEventListener('click', generateScript);
  triggerPipelineBtn.addEventListener('click', triggerPipeline);
  if (legacySaveToStudioBtn) legacySaveToStudioBtn.addEventListener('click', ensureStudioForLegacyScript);
  if (legacyVideoBtn) legacyVideoBtn.addEventListener('click', videoFromLegacyScript);
  if (legacyExportTxtBtn) legacyExportTxtBtn.addEventListener('click', () => exportFromLegacy('txt'));
  if (legacyExportDocxBtn) legacyExportDocxBtn.addEventListener('click', () => exportFromLegacy('docx'));
  if (legacyExportPdfBtn) legacyExportPdfBtn.addEventListener('click', () => exportFromLegacy('pdf'));

  if (generateV2Btn) generateV2Btn.addEventListener('click', generateOneClickV2);
  if (saveScriptBtn) saveScriptBtn.addEventListener('click', saveScript);
  if (generateVideoBtn) generateVideoBtn.addEventListener('click', generateVideo);
  if (refreshCurrentBtn)
    refreshCurrentBtn.addEventListener('click', async () => {
      const cid = (contentIdInput?.value || currentContentId || '').trim();
      if (!cid) return;
      setButtonsDisabled(true);
      setActionStatus('Refreshing…');
      await loadContent(cid);
      setActionStatus('Refreshed.');
      setButtonsDisabled(false);
    });

  if (clearCurrentBtn)
    clearCurrentBtn.addEventListener('click', () => {
      const ok = confirm('Clear the currently loaded script from the UI? (This will not delete anything from the backend.)');
      if (!ok) return;
      clearCurrentScript();
    });
  if (refreshStoryTypesBtn) refreshStoryTypesBtn.addEventListener('click', refreshStoryTypes);
  if (saveStoryTypeBtn) saveStoryTypeBtn.addEventListener('click', saveStoryType);
  if (refreshStepPromptsBtn) refreshStepPromptsBtn.addEventListener('click', refreshStepPrompts);
  if (saveStepPromptBtn) saveStepPromptBtn.addEventListener('click', saveStepPrompt);
  if (refreshExamplesBtn) refreshExamplesBtn.addEventListener('click', refreshExamples);
  if (saveExampleBtn) saveExampleBtn.addEventListener('click', saveExample);
  if (saveConfigBtn) saveConfigBtn.addEventListener('click', saveConfig);
  if (exportDocxBtn) exportDocxBtn.addEventListener('click', () => exportFormats(['docx']));
  if (exportPdfBtn) exportPdfBtn.addEventListener('click', () => exportFormats(['pdf']));
  if (exportTxtBtn) exportTxtBtn.addEventListener('click', () => exportFormats(['txt']));

  // Advanced actions (debugging only)
  if (advancedCreateRawBtn)
    advancedCreateRawBtn.addEventListener('click', async () => {
      try {
        const cid = await createRawOnly();
        currentContentId = cid;
        contentIdInput.value = cid;
        setHidden(contentIdInput, false);
        if (v2Progress) v2Progress.textContent = `Created RAW only. content_id: ${cid}`;
        await loadContent(cid);
      } catch (e) {
        if (v2Progress) v2Progress.textContent = 'Advanced create failed: ' + String(e?.message || e);
      }
    });

  if (advancedRunGenerateBtn)
    advancedRunGenerateBtn.addEventListener('click', async () => {
      const cid = (contentIdInput?.value || '').trim();
      if (!cid) return;
      if (v2Progress) v2Progress.textContent = 'Starting generate for existing content_id...';
      const body = getGenerateBody();
      const { json } = await fetchJson(`${BACKEND_BASE}/v2/generation/${cid}/generate/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!json.job_id) {
        if (v2Progress) v2Progress.textContent = 'Failed to start job: ' + JSON.stringify(json, null, 2);
        return;
      }
      const final = await pollJob(json.job_id, {
        onTick: (j) => {
          if (v2Progress) v2Progress.textContent = JSON.stringify(j, null, 2);
        }
      });
      if (final.status === 'completed') {
        const content = await loadContent(cid);
        if (scriptEditor) scriptEditor.value = content.final_output || '';
      }
    });

  // Initial session check
  checkSession();

  // Persist / restore UI state so hard refresh doesn't wipe the operator flow.
  restoreUiState();
  // Resume any in-flight jobs after refresh (so generation doesn't “disappear”).
  resumeInFlightJobs();

  if (resetSavedStateBtn) {
    resetSavedStateBtn.addEventListener('click', () => {
      const ok = confirm('Reset saved UI state? This will clear saved inputs/scripts from this browser only.');
      if (!ok) return;
      try {
        // Clear current version + older versions (best-effort)
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem('video_automation_v2_state');
        localStorage.removeItem('video_automation_v2_state_v1');
      } catch {}
      location.reload();
    });
  }

  // Persist legacy inputs so refresh doesn't wipe them.
  wirePersist(outlineInput, 'legacy_outline');
  wirePersist(languageInput, 'legacy_language');
  wirePersist(targetMinutesInput, 'legacy_target_minutes');
  wirePersist(snsWebhookInput, 'legacy_webhook');
  wirePersist(legacyScriptText, 'legacy_script_text');
  wirePersist(useCaseSelect, 'use_case', { event: 'change' });
  wirePersist(rawDomainInput, 'raw_domain');
  wirePersist(rawTopicInput, 'raw_topic');
  wirePersist(rawSubTopicInput, 'raw_sub_topic');
  wirePersist(rawChannelInput, 'raw_channel');
  wirePersist(rawTextInput, 'raw_text');
  wirePersist(rawWebUrlInput, 'raw_web_url');
  wirePersist(rawYoutubeUrlInput, 'raw_youtube_url');
  wirePersist(rawChannelLinkInput, 'raw_channel_link');
  wirePersist(genStoryTypeInput, 'gen_story_type');
  wirePersist(genProviderInput, 'gen_provider');
  wirePersist(genModelInput, 'gen_model');
  wirePersist(genPromptVersionInput, 'gen_prompt_version');
  wirePersist(genTotalCharsInput, 'gen_total_chars');
  wirePersist(genStepsInput, 'gen_steps');
  wirePersist(contentConfigInput, 'config_json');
  wirePersist(videoDurationInput, 'video_duration');
  if (scriptEditor) {
    // Save script edits locally so refresh doesn't wipe them
    scriptEditor.addEventListener('input', () => saveUiState({ script_editor: scriptEditor.value || '' }));
  }

  // Load training metadata in background (non-blocking)
  try {
    refreshStoryTypes();
    refreshExamples();
  } catch {}

  // Simple tab switching for a cleaner UX
  const tabButtons = document.querySelectorAll('[data-tab-target]');
  const tabPanels = document.querySelectorAll('.tab-panel');
  tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab-target');
      tabPanels.forEach((panel) => {
        panel.classList.toggle('active', panel.id === targetId);
      });
      tabButtons.forEach((b) => {
        if (b === btn) b.classList.add('active');
        else b.classList.remove('active');
      });
    });
  });
})();


