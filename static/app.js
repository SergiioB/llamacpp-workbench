const state = {
  config: null,
  chats: [],
  currentChatId: null,
  generating: false,
  stopRequested: false,
  models: [],
  downloads: [],
  modelPresets: [],
  browserEngine: null,
  browserModelId: null,
  browserMode: false,
  _pendingRender: null,
  _rafId: null,
};

const markdown = window.markdownit({
  html: false,
  linkify: true,
  breaks: true,
});

const PROMPT_PRESETS = [
  { label: "Latency Smoke", text: "Reply with only: latency smoke test passed." },
  { label: "Reasoning Check", text: "Explain in 5 numbered steps how KV cache quantization helps local inference." },
  { label: "Coding Check", text: "Write a Python function that streams JSON lines from a HTTP response." },
  { label: "Long Answer", text: "Give me 12 numbered points about what makes a local LLM web UI production-ready." },
  { label: "REAP Probe", text: "Summarize how expert pruning could help a personal MoE model in 8 concise bullets." },
];

const fields = [
  "llama_binary", "model_path", "cpu_mask", "llama_host", "llama_port", "ctx_size", "threads", "gpu_layers",
  "parallel", "batch_size", "ubatch_size", "temperature", "top_p", "top_k", "min_p",
  "repeat_penalty", "presence_penalty", "max_tokens", "custom_args", "system_prompt"
];

/* ── helpers ── */

function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return await response.json();
}

async function parseErrorResponse(response) {
  const raw = await response.text();
  try {
    const parsed = JSON.parse(raw);
    const detail = parsed?.detail || parsed?.error?.message || raw;
    if (typeof detail === "string" && detail.includes("llama.cpp server is not running")) {
      return "Model is loading or offline. Try again in a few seconds.";
    }
    return String(detail);
  } catch {
    if (raw.includes("llama.cpp server is not running")) {
      return "Model is loading or offline. Try again in a few seconds.";
    }
    return raw;
  }
}

/* ── config form ── */

function readConfigForm() {
  const config = {};
  for (const field of fields) {
    const element = document.getElementById(field);
    const value = element.value;
    config[field] = ["llama_port", "ctx_size", "threads", "gpu_layers", "parallel", "batch_size", "ubatch_size", "top_k", "max_tokens"].includes(field)
      ? Number(value)
      : ["temperature", "top_p", "min_p", "repeat_penalty", "presence_penalty"].includes(field)
        ? Number(value)
        : value;
  }
  return config;
}

function writeConfigForm(config, candidateModels = [], modelPresets = []) {
  state.config = config;
  state.modelPresets = modelPresets;
  for (const field of fields) {
    const element = document.getElementById(field);
    if (element) element.value = config[field] ?? "";
  }

  const select = document.getElementById("candidate_models");
  select.innerHTML = `<option value="">Select a scanned model...</option>`;
  for (const model of candidateModels) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  }

  const presetSelect = document.getElementById("model_presets");
  presetSelect.innerHTML = `<option value="">Select a saved preset...</option>`;
  for (const preset of modelPresets) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.label} - ${preset.description}`;
    option.dataset.presetId = preset.id;
    presetSelect.appendChild(option);
  }

  renderLoadedModelSummary();
}

function formatGiB(sizeBytes) {
  return `${(sizeBytes / (1024 ** 3)).toFixed(2)} GiB`;
}

function basename(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

function renderLoadedModelSummary() {
  const nameEl = document.getElementById("loaded-model-name");
  const metaEl = document.getElementById("loaded-model-meta");
  if (!nameEl || !metaEl || !state.config) return;

  if (state.browserMode) {
    nameEl.textContent = state.browserModelId || "Browser Model";
    metaEl.textContent = "WebGPU · in-browser";
    return;
  }

  const modelPath = String(state.config.model_path || "");
  nameEl.textContent = basename(modelPath) || "Unknown";

  const ctx = state.config.ctx_size ? `ctx ${state.config.ctx_size}` : "";
  const temp = Number.isFinite(Number(state.config.temperature)) ? `temp ${Number(state.config.temperature).toFixed(2)}` : "";
  const maxTokens = state.config.max_tokens ? `max ${state.config.max_tokens}` : "";
  const parts = [ctx, temp, maxTokens].filter(Boolean);
  metaEl.textContent = parts.join(" · ");
}

/* ── model library ── */

function renderModelLibrary() {
  const container = document.getElementById("model-library");
  container.innerHTML = "";
  const selectedPath = document.getElementById("model_path").value || "";

  const frag = document.createDocumentFragment();
  for (const model of state.models) {
    const card = document.createElement("div");
    card.className = `model-card ${selectedPath === model.path ? "active" : ""}`;

    const title = document.createElement("div");
    title.className = "card-title";
    title.textContent = model.name;

    const meta = document.createElement("div");
    meta.className = "card-meta";
    const tags = [formatGiB(model.size_bytes)];
    if (model.is_reap) tags.push("REAP");
    if (model.is_a3b) tags.push("A3B");
    if (model.is_moe) tags.push("MoE");
    meta.textContent = tags.join(" · ");

    const vramEl = document.createElement("div");
    vramEl.className = "card-meta";
    if (model.vram) {
      const v = model.vram;
      const fits = [];
      if (v.fits_4gb) fits.push("4GB");
      if (v.fits_8gb) fits.push("8GB");
      if (v.fits_12gb) fits.push("12GB");
      if (v.fits_16gb) fits.push("16GB");
      if (v.fits_24gb) fits.push("24GB");
      vramEl.textContent = `~${v.min_vram_gib} GiB VRAM · fits: ${fits.length ? fits.join(", ") : "needs >24GB"}`;
    } else {
      vramEl.textContent = "";
    }

    const actions = document.createElement("div");
    actions.className = "card-actions";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Use This";
    btn.onclick = () => {
      document.getElementById("model_path").value = model.path;
      renderModelLibrary();
      setStatus(`Selected ${model.name}`);
    };
    actions.appendChild(btn);

    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(vramEl);
    card.appendChild(actions);
    frag.appendChild(card);
  }
  container.appendChild(frag);
}

function renderDownloads() {
  const container = document.getElementById("download-jobs");
  container.innerHTML = "";
  if (state.downloads.length === 0) {
    container.innerHTML = `<div class="card-meta">No downloads yet.</div>`;
    return;
  }

  const frag = document.createDocumentFragment();
  for (const job of state.downloads) {
    const card = document.createElement("div");
    card.className = "download-card";
    const logTail = (job.log_tail || []).slice(-6).join("\n");

    const title = document.createElement("div");
    title.className = "card-title";
    title.textContent = job.status.toUpperCase();

    const meta = document.createElement("div");
    meta.className = "card-meta";
    meta.textContent = job.destination_path;

    const size = document.createElement("div");
    size.className = "card-meta";
    size.textContent = `${job.downloaded_bytes ? formatGiB(job.downloaded_bytes) : "0.00 GiB"} downloaded`;

    const log = document.createElement("div");
    log.className = "log-tail";
    log.textContent = logTail || "Waiting for log output...";

    const actions = document.createElement("div");
    actions.className = "card-actions";
    if (job.status === "running") {
      const cancel = document.createElement("button");
      cancel.textContent = "Cancel";
      cancel.onclick = async () => {
        await api(`/api/models/download/${job.job_id}/cancel`, { method: "POST" });
        await refreshModels();
      };
      actions.appendChild(cancel);
    }

    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(size);
    card.appendChild(log);
    card.appendChild(actions);
    frag.appendChild(card);
  }
  container.appendChild(frag);
}

/* ── chat list ── */

function renderChats() {
  const chatList = document.getElementById("chat-list");
  chatList.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const chat of state.chats) {
    const item = document.createElement("div");
    item.className = `chat-item ${chat.chat_id === state.currentChatId ? "active" : ""}`;
    item.textContent = chat.title;
    item.onclick = () => loadChat(chat.chat_id);
    frag.appendChild(item);
  }
  chatList.appendChild(frag);
}

/* ── messages: performance-critical rendering ── */

function renderMessages(chat) {
  document.getElementById("chat-title").textContent = chat.title;
  const container = document.getElementById("messages");
  container.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const message of chat.messages) {
    const el = document.createElement("div");
    el.className = `message ${message.role}`;
    renderMessageContent(el, message.content);
    frag.appendChild(el);
  }
  container.appendChild(frag);
  requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
}

function appendLiveMessage(role, content = "", options = {}) {
  const container = document.getElementById("messages");
  const el = document.createElement("div");
  el.className = `message ${role}`;
  if (options.pending) {
    el.dataset.pending = "true";
  }
  renderMessageContent(el, content);
  container.appendChild(el);
  requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
  return el;
}

function renderMessageContent(element, content) {
  const safeContent = content || "";
  if (!safeContent.trim()) {
    if (element.dataset.pending === "true") {
      element.innerHTML = `<p class="message-placeholder">Generating<span class="dot-1">.</span><span class="dot-2">.</span><span class="dot-3">.</span></p>`;
    } else {
      element.textContent = safeContent;
    }
    return;
  }
  delete element.dataset.pending;
  element.innerHTML = markdown.render(safeContent);
}

/* Streaming: fast-path that only appends new text instead of re-rendering everything */
let _streamBuffer = "";
let _streamElement = null;
let _streamRenderScheduled = false;

function streamAppendDelta(delta) {
  _streamBuffer += delta;
  if (!_streamRenderScheduled) {
    _streamRenderScheduled = true;
    requestAnimationFrame(renderStreamBuffer);
  }
}

function renderStreamBuffer() {
  _streamRenderScheduled = false;
  if (!_streamElement) return;
  delete _streamElement.dataset.pending;
  _streamElement.innerHTML = markdown.render(_streamBuffer);
  const container = document.getElementById("messages");
  container.scrollTop = container.scrollHeight;
}

function streamReset() {
  _streamBuffer = "";
  _streamElement = null;
  _streamRenderScheduled = false;
}

/* ── status helpers ── */

function formatLatencyMs(ms) {
  return `${(Number(ms) / 1000).toFixed(2)} s`;
}

function setStatus(text, ok = true) {
  const status = document.getElementById("status-text");
  status.textContent = text;
  status.style.color = ok ? "#5ee17f" : "#ff948c";
}

function setChatStatus(text, ok = true) {
  const status = document.getElementById("chat-status-text");
  status.textContent = text;
  status.style.color = ok ? "#5ee17f" : "#ff948c";
}

function setGenerating(active) {
  state.generating = active;
  const button = document.getElementById("send-message");
  button.textContent = active ? "Stop" : "Send";
  button.classList.toggle("stop", active);
}

/* ── prompt presets ── */

function loadPromptPresets() {
  const select = document.getElementById("prompt-presets");
  select.innerHTML = `<option value="">Choose a test prompt...</option>`;
  for (const preset of PROMPT_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.text;
    option.textContent = preset.label;
    select.appendChild(option);
  }
}

/* ── API refresh calls ── */

async function refreshConfig() {
  const data = await api("/api/config");
  writeConfigForm(data.config, data.candidate_models, data.model_presets || []);
}

async function refreshModels() {
  if (document.hidden) return;
  const data = await api("/api/models");
  state.models = data.models;
  state.downloads = data.downloads;
  renderModelLibrary();
  renderDownloads();
}

async function refreshChats() {
  const data = await api("/api/chats");
  state.chats = data.chats;
  renderChats();
}

async function refreshServerStatus() {
  const data = await api("/api/server/status");
  const indicator = document.getElementById("server-indicator");
  indicator.textContent = data.status.healthy ? "online" : "offline";
  indicator.className = `badge ${data.status.healthy ? "online" : "offline"}`;
  if (data.config) {
    state.config = data.config;
    renderLoadedModelSummary();
  }
}

async function loadChat(chatId) {
  const data = await api(`/api/chats/${chatId}`);
  state.currentChatId = chatId;
  renderChats();
  renderMessages(data.chat);
}

async function createChat() {
  const data = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({}),
  });
  await refreshChats();
  await loadChat(data.chat.chat_id);
}

/* ── server mode actions ── */

async function saveConfig() {
  const config = readConfigForm();
  await api("/api/config", {
    method: "POST",
    body: JSON.stringify({ config }),
  });
  setStatus("Config saved");
  await refreshConfig();
}

async function startServer() {
  const config = readConfigForm();
  setStatus("Starting llama.cpp...", true);
  await api("/api/server/start", {
    method: "POST",
    body: JSON.stringify({ config }),
  });
  await refreshServerStatus();
  setStatus("llama.cpp online");
  await refreshConfig();
}

async function stopServer() {
  await api("/api/server/stop", { method: "POST" });
  await refreshServerStatus();
  setStatus("llama.cpp stopped");
}

async function loadSelectedModel() {
  const modelPath = document.getElementById("model_path").value.trim();
  if (!modelPath) return;
  setStatus("Loading selected model...", true);
  await api("/api/models/load", {
    method: "POST",
    body: JSON.stringify({ model_path: modelPath }),
  });
  await refreshServerStatus();
  await refreshConfig();
  await refreshModels();
  setStatus("Selected model loaded");
}

function selectedPreset() {
  const presetId = document.getElementById("model_presets").value;
  return state.modelPresets.find((preset) => preset.id === presetId) || null;
}

function applyModelPreset() {
  const preset = selectedPreset();
  if (!preset) return;
  writeConfigForm(preset.config, state.models.map((model) => model.path), state.modelPresets);
  renderModelLibrary();
  setStatus(`Applied preset: ${preset.label}`);
}

async function loadModelPreset() {
  const preset = selectedPreset();
  if (!preset) return;
  writeConfigForm(preset.config, state.models.map((model) => model.path), state.modelPresets);
  renderModelLibrary();
  setStatus(`Loading preset: ${preset.label}`, true);
  await api("/api/server/start", {
    method: "POST",
    body: JSON.stringify({ config: preset.config }),
  });
  await refreshServerStatus();
  await refreshConfig();
  await refreshModels();
  setStatus(`Preset loaded: ${preset.label}`);
}

async function startDownload() {
  const url = document.getElementById("download_url").value.trim();
  const destinationPath = document.getElementById("download_destination").value.trim();
  if (!url) return;
  await api("/api/models/download", {
    method: "POST",
    body: JSON.stringify({ url, destination_path: destinationPath || null }),
  });
  document.getElementById("download_url").value = "";
  await refreshModels();
  setStatus("Download started");
}

async function stopGeneration() {
  if (!state.generating) return;
  state.stopRequested = true;
  setChatStatus("Stopping...", true);
  if (state.browserMode) return;
  await api("/api/generation/stop", { method: "POST" });
}

/* ── streaming: server mode ── */

async function streamChatServer(chatId, content, assistantEl) {
  const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!response.ok || !response.body) {
    throw new Error(await parseErrorResponse(response) || "stream request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const rawLine = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (rawLine) {
        const event = JSON.parse(rawLine);
        if (event.type === "delta") {
          streamAppendDelta(event.delta);
          setChatStatus(`Generating... ${_streamBuffer.length} chars`, true);
        } else if (event.type === "done") {
          renderStreamBuffer();
          streamReset();
          await refreshChats();
          renderMessages(event.chat);
          setChatStatus(event.cancelled ? `Stopped after ${formatLatencyMs(event.latency_ms)}` : `Done in ${formatLatencyMs(event.latency_ms)}`);
          return;
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      }
      newlineIndex = buffer.indexOf("\n");
    }
  }
}

/* ── streaming: browser mode via WebLLM ── */

async function streamChatBrowser(content, assistantEl) {
  if (!state.browserEngine) throw new Error("No browser model loaded");
  const config = state.config || {};
  const messages = [];
  const sys = String(config.system_prompt || "").trim();
  if (sys) messages.push({ role: "system", content: sys });
  messages.push({ role: "user", content });

  const chunks = await state.browserEngine.chat.completions.create({
    messages,
    temperature: Number(config.temperature) || 1.0,
    top_p: Number(config.top_p) || 0.95,
    stream: true,
  });

  for await (const chunk of chunks) {
    const delta = chunk.choices[0]?.delta?.content || "";
    if (delta) {
      streamAppendDelta(delta);
      setChatStatus(`Generating... ${_streamBuffer.length} chars`, true);
    }
    if (state.stopRequested) {
      state.browserEngine?.interruptGenerate();
      break;
    }
  }

  renderStreamBuffer();
  const finalContent = _streamBuffer;
  streamReset();
  setChatStatus("Done (browser)");
  return finalContent;
}

/* ── send message ── */

async function sendMessage() {
  const input = document.getElementById("message-input");
  if (state.generating) {
    await stopGeneration();
    return;
  }

  const content = input.value.trim();
  if (!content) return;

  if (!state.browserMode && !state.currentChatId) {
    await createChat();
  }

  appendLiveMessage("user", content);
  const assistantEl = appendLiveMessage("assistant", "", { pending: true });
  streamReset();
  _streamElement = assistantEl;

  input.value = "";
  state.stopRequested = false;
  setGenerating(true);
  setChatStatus("Generating...", true);

  try {
    if (state.browserMode) {
      await streamChatBrowser(content, assistantEl);
    } else {
      await streamChatServer(state.currentChatId, content, assistantEl);
    }
  } catch (error) {
    const cancelledByUser = state.stopRequested;
    renderStreamBuffer();
    streamReset();
    setChatStatus(cancelledByUser ? "Stopped" : error.message, !cancelledByUser);
    if (!state.browserMode && state.currentChatId) {
      await loadChat(state.currentChatId);
    }
  } finally {
    state.stopRequested = false;
    setGenerating(false);
  }
}

function applyPresetToComposer() {
  const select = document.getElementById("prompt-presets");
  const input = document.getElementById("message-input");
  if (!select.value) return;
  input.value = select.value;
  input.focus();
}

async function runPreset() {
  applyPresetToComposer();
  await sendMessage();
}

/* ── browser inference: WebLLM ── */

const BROWSER_MODELS = [
  { id: "SmolLM2-135M-Instruct-q4f16_1-MLC", label: "SmolLM2 135M (test)", size: "~100MB" },
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2 360M", size: "~220MB" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B", size: "~700MB" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", label: "Llama 3.2 3B", size: "~1.8GB" },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 1.5B", size: "~1GB" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 3B", size: "~1.8GB" },
  { id: "Qwen2.5-7B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 7B", size: "~4GB" },
  { id: "Phi-3.5-mini-instruct-q4f16_1-MLC", label: "Phi 3.5 Mini 3.8B", size: "~2.3GB" },
  { id: "Llama-3.1-8B-Instruct-q4f16_1-MLC", label: "Llama 3.1 8B", size: "~4.5GB" },
  { id: "gemma-2-2b-it-q4f16_1-MLC", label: "Gemma 2 2B", size: "~1.4GB" },
];

function populateBrowserModels() {
  const select = document.getElementById("browser-model-select");
  select.innerHTML = '<option value="">Choose a model...</option>';
  for (const m of BROWSER_MODELS) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.size})`;
    select.appendChild(opt);
  }
}

function detectWebGPU() {
  return !!navigator.gpu;
}

async function initBrowserMode() {
  if (!detectWebGPU()) {
    setBrowserStatus("WebGPU not available in this browser", false);
    return;
  }

  setBrowserStatus("Loading WebLLM engine...", true);

  try {
    const webllm = await import("https://esm.run/@mlc-ai/web-llm");

    state.browserEngine = new webllm.MLCEngine({
      initProgressCallback: (progress) => {
        const msg = progress.text || `Loading: ${Math.round((progress.progress || 0) * 100)}%`;
        setBrowserStatus(msg, true);
      },
    });

    state.browserMode = true;
    populateBrowserModels();
    setBrowserStatus("WebLLM ready. Select a model to load.", true);
    document.getElementById("browser-section").classList.add("active");
    renderLoadedModelSummary();
  } catch (err) {
    setBrowserStatus(`WebLLM load failed: ${err.message}`, false);
  }
}

async function loadBrowserModel() {
  const modelId = document.getElementById("browser-model-select").value;
  if (!modelId || !state.browserEngine) return;

  setBrowserStatus(`Downloading ${modelId}...`, true);
  setGenerating(true);

  try {
    await state.browserEngine.reload(modelId);
    state.browserModelId = modelId;
    setBrowserStatus(`Loaded: ${modelId}`, true);
    renderLoadedModelSummary();
    setStatus(`Browser model: ${modelId}`);
  } catch (err) {
    setBrowserStatus(`Failed: ${err.message}`, false);
  } finally {
    setGenerating(false);
  }
}

function setBrowserStatus(text, ok) {
  const el = document.getElementById("browser-status");
  if (!el) return;
  el.textContent = text;
  el.style.color = ok ? "#5ee17f" : "#ff948c";
}

/* ── event bindings ── */

document.getElementById("new-chat").onclick = createChat;
document.getElementById("save-config").onclick = saveConfig;
document.getElementById("apply-model-preset").onclick = applyModelPreset;
document.getElementById("load-model-preset").onclick = loadModelPreset;
document.getElementById("start-server").onclick = startServer;
document.getElementById("stop-server").onclick = stopServer;
document.getElementById("refresh-models").onclick = refreshModels;
document.getElementById("load-selected-model").onclick = loadSelectedModel;
document.getElementById("start-download").onclick = startDownload;
document.getElementById("send-message").onclick = sendMessage;
document.getElementById("apply-preset").onclick = applyPresetToComposer;
document.getElementById("run-preset").onclick = runPreset;
document.getElementById("candidate_models").onchange = (event) => {
  if (event.target.value) {
    document.getElementById("model_path").value = event.target.value;
    renderModelLibrary();
  }
};

document.getElementById("message-input").addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    await sendMessage();
  }
});

document.getElementById("init-browser").onclick = initBrowserMode;
document.getElementById("load-browser-model").onclick = loadBrowserModel;

/* ── init ── */

(async function init() {
  loadPromptPresets();

  if (detectWebGPU()) {
    document.getElementById("browser-section").classList.add("available");
  }

  await refreshConfig();
  await refreshModels();
  await refreshChats();
  await refreshServerStatus();
  if (state.chats.length > 0) {
    await loadChat(state.chats[0].chat_id);
  }

  setInterval(refreshModels, 5000);
})();
