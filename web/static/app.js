(function () {
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
  const pipelineResult = document.getElementById('pipeline-result');

  // Long-form system (v2)
  const useCaseSelect = document.getElementById('use-case');
  const rawDomainInput = document.getElementById('raw-domain');
  const rawTopicInput = document.getElementById('raw-topic');
  const rawSubTopicInput = document.getElementById('raw-sub-topic');
  const rawChannelInput = document.getElementById('raw-channel');
  const rawTextInput = document.getElementById('raw-text');
  const rawWebUrlInput = document.getElementById('raw-web-url');
  const rawYoutubeUrlInput = document.getElementById('raw-youtube-url');
  const rawChannelLinkInput = document.getElementById('raw-channel-link');
  const createRawBtn = document.getElementById('create-raw');
  const rawResult = document.getElementById('raw-result');

  const contentIdInput = document.getElementById('content-id');
  const runCleanBtn = document.getElementById('run-clean');
  const runClassifyBtn = document.getElementById('run-classify');
  const runGenerateBtn = document.getElementById('run-generate');
  const jobResult = document.getElementById('job-result');
  const contentResult = document.getElementById('content-result');

  const genStoryTypeInput = document.getElementById('gen-story-type');
  const genProviderInput = document.getElementById('gen-provider');
  const genModelInput = document.getElementById('gen-model');
  const genPromptVersionInput = document.getElementById('gen-prompt-version');
  const genTotalCharsInput = document.getElementById('gen-total-chars');
  const genStepsInput = document.getElementById('gen-steps');

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

  async function pollJob(jobId, { onTick } = {}) {
    const deadline = Date.now() + 30 * 60 * 1000; // 30 minutes
    while (Date.now() < deadline) {
      const { json } = await fetchJson(`${BACKEND_BASE}/v2/jobs/${jobId}`);
      if (onTick) onTick(json);
      if (json.status === 'completed' || json.status === 'failed') return json;
      await new Promise((r) => setTimeout(r, 2000));
    }
    return { status: 'timeout', job_id: jobId };
  }

  async function loadContent(contentId) {
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/contents/${contentId}`);
    contentResult.textContent = JSON.stringify(json, null, 2);
    return json;
  }

  async function createRaw() {
    rawResult.textContent = 'Creating...';
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
    const { json, ok } = await fetchJson(`${BACKEND_BASE}/v2/content/raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    rawResult.textContent = JSON.stringify(json, null, 2);
    if (ok && json.content_id) {
      contentIdInput.value = json.content_id;
      await loadContent(json.content_id);
    }
  }

  async function runClean() {
    const contentId = contentIdInput.value.trim();
    if (!contentId) return;
    jobResult.textContent = 'Starting clean job...';
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/content/${contentId}/clean/async`, {
      method: 'POST'
    });
    jobResult.textContent = JSON.stringify(json, null, 2);
    const jobId = json.job_id;
    if (!jobId) return;
    const final = await pollJob(jobId, {
      onTick: (j) => {
        jobResult.textContent = JSON.stringify(j, null, 2);
      }
    });
    jobResult.textContent = JSON.stringify(final, null, 2);
    await loadContent(contentId);
  }

  async function runClassify() {
    const contentId = contentIdInput.value.trim();
    if (!contentId) return;
    jobResult.textContent = 'Starting classify job...';
    const body = {
      manual_story_type: genStoryTypeInput?.value?.trim() || null
    };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/content/${contentId}/classify/async`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    jobResult.textContent = JSON.stringify(json, null, 2);
    const jobId = json.job_id;
    if (!jobId) return;
    const final = await pollJob(jobId, {
      onTick: (j) => {
        jobResult.textContent = JSON.stringify(j, null, 2);
      }
    });
    jobResult.textContent = JSON.stringify(final, null, 2);
    await loadContent(contentId);
  }

  async function runGenerate() {
    const contentId = contentIdInput.value.trim();
    if (!contentId) return;
    jobResult.textContent = 'Starting generate job...';
    const body = {
      story_type: genStoryTypeInput?.value?.trim() || null,
      ai_provider: genProviderInput?.value?.trim() || null,
      ai_model: genModelInput?.value?.trim() || null,
      prompt_version: genPromptVersionInput?.value?.trim() || null,
      total_chars_target: Number(genTotalCharsInput?.value || 30000),
      steps: genStepsInput?.value ? Number(genStepsInput.value) : null
    };
    const { json } = await fetchJson(`${BACKEND_BASE}/v2/generation/${contentId}/generate/async`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    jobResult.textContent = JSON.stringify(json, null, 2);
    const jobId = json.job_id;
    if (!jobId) return;
    const final = await pollJob(jobId, {
      onTick: (j) => {
        jobResult.textContent = JSON.stringify(j, null, 2);
      }
    });
    jobResult.textContent = JSON.stringify(final, null, 2);
    await loadContent(contentId);
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
    if (!outline) {
      pipelineResult.textContent = 'Provide an outline first.';
      return;
    }
    const webhookUrl = snsWebhookInput.value.trim() || null;
    pipelineResult.textContent = 'Starting pipeline job...';
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

      pipelineResult.textContent = `Running... (job: ${jobId})`;

      const deadline = Date.now() + 300000; // 5 minutes UI wait
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 3000));
        const statusRes = await fetch(`${BACKEND_BASE}/pipeline/full/status/${jobId}`);
        const statusJson = await statusRes.json();
        if (statusJson.status === 'completed') {
          pipelineResult.textContent = JSON.stringify(statusJson.result, null, 2);
          return;
        }
        if (statusJson.status === 'failed') {
          pipelineResult.textContent = 'Pipeline failed: ' + (statusJson.error || JSON.stringify(statusJson, null, 2));
          return;
        }
      }

      pipelineResult.textContent =
        'Still running in background.\n' +
        `Job ID: ${jobId}\n` +
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

    scriptResult.textContent = 'Starting job...';
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

      scriptResult.textContent = `Generating... (job: ${jobId})`;

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
          scriptResult.textContent = statusJson.result?.script || JSON.stringify(statusJson, null, 2);
          return;
        }
        if (statusJson.status === 'failed') {
          scriptResult.textContent = 'Job failed: ' + (statusJson.error || JSON.stringify(statusJson, null, 2));
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

  loginBtn.addEventListener('click', sendMagicLink);
  refreshCustomersBtn.addEventListener('click', loadCustomers);
  createCustomerBtn.addEventListener('click', createCustomer);
  generateScriptBtn.addEventListener('click', generateScript);
  triggerPipelineBtn.addEventListener('click', triggerPipeline);

  if (createRawBtn) createRawBtn.addEventListener('click', createRaw);
  if (runCleanBtn) runCleanBtn.addEventListener('click', runClean);
  if (runClassifyBtn) runClassifyBtn.addEventListener('click', runClassify);
  if (runGenerateBtn) runGenerateBtn.addEventListener('click', runGenerate);
  if (refreshStoryTypesBtn) refreshStoryTypesBtn.addEventListener('click', refreshStoryTypes);
  if (saveStoryTypeBtn) saveStoryTypeBtn.addEventListener('click', saveStoryType);
  if (refreshStepPromptsBtn) refreshStepPromptsBtn.addEventListener('click', refreshStepPrompts);
  if (saveStepPromptBtn) saveStepPromptBtn.addEventListener('click', saveStepPrompt);
  if (refreshExamplesBtn) refreshExamplesBtn.addEventListener('click', refreshExamples);
  if (saveExampleBtn) saveExampleBtn.addEventListener('click', saveExample);

  // Initial session check
  checkSession();

  // Load training metadata in background (non-blocking)
  try {
    refreshStoryTypes();
    refreshExamples();
  } catch {}
})();


