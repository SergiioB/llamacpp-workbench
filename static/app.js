const state = {
  config: null,
  chats: [],
  currentChatId: null,
  generating: false,
  stopRequested: false,
  models: [],
  downloads: [],
  modelPresets: [],
  preflight: null,
  preflightAutoRefresh: true,
  serverStarting: false,
  browserEngine: null,
  browserModelId: null,
  browserMode: false,
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
  "llama_binary", "model_path", "cpu_mask", "llama_host", "llama_port", "runtime_mode", "rpc_host", "rpc_port",
  "rpc_tensor_split", "ctx_size", "threads", "gpu_layers", "parallel", "batch_size", "ubatch_size", "temperature", "top_p", "top_k", "min_p",
  "repeat_penalty", "presence_penalty", "max_tokens", "custom_args", "system_prompt"
];

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   API helpers
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Settings Drawer
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function openDrawer() {
  document.getElementById("settings-drawer").classList.add("open");
}

function closeDrawer() {
  document.getElementById("settings-drawer").classList.remove("open");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Sidebar Toggle (mobile)
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Config Form
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function readConfigForm() {
  const config = {};
  for (const field of fields) {
    const element = document.getElementById(field);
    const value = element.value;
    config[field] = ["llama_port", "rpc_port", "ctx_size", "threads", "gpu_layers", "parallel", "batch_size", "ubatch_size", "top_k", "max_tokens"].includes(field)
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
    option.textContent = `${preset.label} â€” ${preset.description}`;
    presetSelect.appendChild(option);
  }

  renderLoadedModelSummary();
  updateRpcGuide();
}

function rpcConfigFromForm() {
  const mode = document.getElementById("runtime_mode")?.value || "local";
  const host = document.getElementById("rpc_host")?.value?.trim() || "";
  const port = document.getElementById("rpc_port")?.value || "50052";
  const split = document.getElementById("rpc_tensor_split")?.value?.trim() || "";
  return { mode, host, port, split };
}

function updateRpcGuide() {
  const guide = document.getElementById("rpc-guide");
  const command = document.getElementById("rpc-guide-command");
  const localNote = document.getElementById("local-mode-note");
  const rpcNote = document.getElementById("rpc-mode-note");
  const health = document.getElementById("rpc-health");
  if (!guide || !command) return;

  const { mode, host, port, split } = rpcConfigFromForm();
  const rpcEnabled = mode === "rpc";
  guide.hidden = !rpcEnabled;
  command.textContent = `rpc-server.exe -H 0.0.0.0 -p ${port || "50052"} -c`;
  localNote?.classList.toggle("active", !rpcEnabled);
  rpcNote?.classList.toggle("active", rpcEnabled);
  if (health) {
    health.textContent = rpcEnabled ? "Not checked" : "Local mode";
    health.className = `health-badge ${rpcEnabled ? "neutral" : "ok"}`;
  }
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Utility
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function formatGiB(sizeBytes) {
  return `${(sizeBytes / (1024 ** 3)).toFixed(2)} GiB`;
}

function basename(path) {
  if (!path) return "";
  const parts = path.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path;
}

function renderLoadedModelSummary() {
  const nameEl = document.getElementById("loaded-model-name");
  const metaEl = document.getElementById("loaded-model-meta");
  if (!nameEl || !metaEl || !state.config) return;

  if (state.browserMode) {
    nameEl.textContent = state.browserModelId || "Browser model";
    metaEl.textContent = state.browserModelId ? "WebGPU in-browser" : "WebGPU ready";
    return;
  }

  const modelPath = String(state.config.model_path || "");
  nameEl.textContent = basename(modelPath) || "No model loaded";

  const ctx = state.config.ctx_size ? `ctx ${state.config.ctx_size}` : "";
  const temp = Number.isFinite(Number(state.config.temperature)) ? `temp ${Number(state.config.temperature).toFixed(2)}` : "";
  const maxTokens = state.config.max_tokens ? `max ${state.config.max_tokens}` : "";
  const rpcEndpoint = state.config.rpc_host ? `${state.config.rpc_host}:${state.config.rpc_port}` : "";
  const split = state.config.rpc_tensor_split ? ` split ${state.config.rpc_tensor_split}` : "";
  const mode = state.config.runtime_mode === "rpc" ? `rpc ${rpcEndpoint}${split}` : "";
  const parts = [mode, ctx, temp, maxTokens].filter(Boolean);
  metaEl.textContent = parts.join(" · ");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Rendering
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function renderModelLibrary() {
  const container = document.getElementById("model-library");
  container.innerHTML = "";
  const selectedPath = document.getElementById("model_path").value || "";

  for (const model of state.models) {
    const card = document.createElement("div");
    card.className = `model-card ${selectedPath === model.path ? "active" : ""}`;
    const validationClass = model.validation_status || "unvalidated";
    const validationLabel = model.validation_label || "Unvalidated";
    const validationNote = model.validation_note || "";
    card.innerHTML = `
      <div class="model-card-top">
        <div class="card-title">${model.name}</div>
        <span class="validation-badge ${validationClass}">${validationLabel}</span>
      </div>
      <div class="card-meta">${formatGiB(model.size_bytes)}${model.is_reap ? " · REAP" : ""}${model.is_a3b ? " · A3B" : ""}</div>
      <div class="card-meta">${validationNote}</div>
      <div class="card-meta">${model.path}</div>
      <div class="card-actions">
        <button type="button" class="btn btn-secondary btn-sm">Use This</button>
      </div>
    `;
    card.querySelector("button").onclick = () => {
      document.getElementById("model_path").value = model.path;
      renderModelLibrary();
      setStatus(`Selected ${model.name}`);
    };
    container.appendChild(card);
  }
}

function renderDownloads() {
  const container = document.getElementById("download-jobs");
  container.innerHTML = "";
  if (state.downloads.length === 0) {
    container.innerHTML = `<div class="card-meta">No downloads yet.</div>`;
    return;
  }

  for (const job of state.downloads) {
    const card = document.createElement("div");
    card.className = "download-card";
    const logTail = (job.log_tail || []).slice(-6).join("\n");
    card.innerHTML = `
      <div class="card-title">${job.status.toUpperCase()}</div>
      <div class="card-meta">${job.destination_path}</div>
      <div class="card-meta">${job.downloaded_bytes ? formatGiB(job.downloaded_bytes) : "0.00 GiB"} downloaded</div>
      <div class="log-tail">${logTail || "Waiting for log output..."}</div>
      <div class="card-actions"></div>
    `;
    const actions = card.querySelector(".card-actions");
    if (job.status === "running") {
      const cancel = document.createElement("button");
      cancel.className = "btn btn-secondary btn-sm";
      cancel.textContent = "Cancel";
      cancel.onclick = async () => {
        await api(`/api/models/download/${job.job_id}/cancel`, { method: "POST" });
        await refreshModels();
      };
      actions.appendChild(cancel);
    }
    container.appendChild(card);
  }
}

function renderChats() {
  const chatList = document.getElementById("chat-list");
  chatList.innerHTML = "";
  for (const chat of state.chats) {
    const item = document.createElement("div");
    item.className = `chat-item ${chat.chat_id === state.currentChatId ? "active" : ""}`;
    item.textContent = chat.title;
    item.onclick = () => loadChat(chat.chat_id);
    chatList.appendChild(item);
  }
}

function renderMessages(chat) {
  document.getElementById("chat-title").textContent = chat.title;
  const container = document.getElementById("messages");
  container.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const message of chat.messages) {
    const el = createMessageElement(message.role, message.content);
    fragment.appendChild(el);
  }
  container.appendChild(fragment);
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
}

function createMessageElement(role, content = "", options = {}) {
  const el = document.createElement("div");
  el.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "role-label";
  label.textContent = role === "user" ? "You" : "llama.cpp";
  el.appendChild(label);

  const contentEl = document.createElement("div");
  el.appendChild(contentEl);

  renderMessageContent(contentEl, content, options.pending);

  return el;
}

function appendLiveMessage(role, content = "", options = {}) {
  const container = document.getElementById("messages");
  const el = createMessageElement(role, content, options);
  if (options.pending) {
    el.dataset.pending = "true";
  }
  container.appendChild(el);
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
  return el;
}

function renderMessageContent(element, content, pending = false) {
  const safeContent = content || "";
  if (!safeContent.trim()) {
    if (pending) {
      element.innerHTML = `<p class="message-placeholder">Generating<span class="dot-1">.</span><span class="dot-2">.</span><span class="dot-3">.</span></p>`;
    } else {
      element.textContent = safeContent;
    }
    return;
  }
  delete element.dataset.pending;
  element.innerHTML = markdown.render(safeContent);
}

let streamBuffer = "";
let streamElement = null;
let streamRenderScheduled = false;

function streamAppendDelta(delta) {
  streamBuffer += delta;
  if (!streamRenderScheduled) {
    streamRenderScheduled = true;
    requestAnimationFrame(renderStreamBuffer);
  }
}

function renderStreamBuffer() {
  streamRenderScheduled = false;
  if (!streamElement) return;
  renderMessageContent(streamElement, streamBuffer);
  const container = document.getElementById("messages");
  container.scrollTop = container.scrollHeight;
}

function streamReset() {
  streamBuffer = "";
  streamElement = null;
  streamRenderScheduled = false;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Status helpers
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function formatLatencyMs(ms) {
  return `${(Number(ms) / 1000).toFixed(2)}s`;
}

function setStatus(text, ok = true) {
  const status = document.getElementById("status-text");
  status.textContent = text;
  status.style.color = ok ? "var(--green)" : "var(--red)";
}

function setChatStatus(text, ok = true) {
  const status = document.getElementById("chat-status-text");
  status.textContent = text;
  status.style.color = ok ? "var(--text-tertiary)" : "var(--red)";
}

function setGenerating(active) {
  state.generating = active;
  const button = document.getElementById("send-message");
  button.classList.toggle("stop", active);
  // Swap icon
  const icon = button.querySelector("i, svg");
  if (icon) {
    button.innerHTML = active
      ? '<i data-lucide="square" class="icon-md"></i>'
      : '<i data-lucide="arrow-up" class="icon-md"></i>';
    lucide.createIcons();
  }
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Prompt Presets
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function loadPromptPresets() {
  const select = document.getElementById("prompt-presets");
  select.innerHTML = `<option value="">Presets...</option>`;
  for (const preset of PROMPT_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.text;
    option.textContent = preset.label;
    select.appendChild(option);
  }
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Data refresh
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

async function refreshConfig() {
  const data = await api("/api/config");
  writeConfigForm(data.config, data.candidate_models, data.model_presets || []);
}

async function refreshModels() {
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
  const isOnline = data.status.healthy;
  indicator.className = `server-status ${isOnline ? "online" : "offline"}`;
  indicator.querySelector(".status-text").textContent = isOnline ? "Online" : "Offline";
  if (data.config) {
    state.config = data.config;
    renderLoadedModelSummary();
  }
  const serverManaged = Boolean(data.status?.managed || data.status?.pid || state.serverStarting);
  // Auto-refresh preflight only when no managed llama-server is loading.
  if (!isOnline && state.preflightAutoRefresh && !serverManaged) {
    runPreflight();
  }
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Actions
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

async function loadChat(chatId) {
  const data = await api(`/api/chats/${chatId}`);
  state.currentChatId = chatId;
  renderChats();
  renderMessages(data.chat);
}

async function runPreflight() {
  if (state.serverStarting) {
    const statusEl = document.getElementById("preflight-status");
    if (statusEl) {
      statusEl.textContent = "Paused while model loads";
      statusEl.className = "preflight-status neutral";
    }
    return;
  }
  const config = readConfigForm();
  const statusEl = document.getElementById("preflight-status");
  const checksEl = document.getElementById("preflight-checks");
  const warningsEl = document.getElementById("preflight-warnings");
  const startBtn = document.getElementById("start-server");

  statusEl.textContent = "Checking...";
  statusEl.className = "preflight-status neutral";

  try {
    const result = await api("/api/preflight", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    state.preflight = result;
    renderPreflight(result, statusEl, checksEl, warningsEl);

    // Disable Start only when preflight ran and found blocking issues
    if (startBtn) {
      startBtn.disabled = !result.ready;
      startBtn.title = result.ready ? "Start llama.cpp" : "Fix blocking issues first";
    }
  } catch (error) {
    // 404 means server hasn't been updated yet — don't block Start
    const msg = String(error.message || "");
    const notFound = msg.includes("Not Found") || msg.includes("404");
    statusEl.textContent = notFound ? "Preflight unavailable (restart WebUI to enable)" : `Check failed: ${msg}`;
    statusEl.className = "preflight-status neutral";
    checksEl.innerHTML = "";
    warningsEl.innerHTML = "";
    // Never disable Start on preflight failure — the old start path still works
  }
}

function renderPreflight(result, statusEl, checksEl, warningsEl) {
  if (result.ready) {
    statusEl.textContent = "Ready to launch";
    statusEl.className = "preflight-status ok";
  } else {
    statusEl.textContent = `${result.blocking_issues.length} blocking issue(s)`;
    statusEl.className = "preflight-status bad";
  }

  const checks = result.checks || {};
  let html = "";

  if (checks.server) {
    const server = checks.server;
    const label = server.healthy
      ? `Online${server.pid ? ` (PID ${server.pid})` : ""}`
      : server.pid ? `Loading (PID ${server.pid})` : "Offline";
    html += renderCheckItem("llama-server", server.healthy ? true : null, label);
    checksEl.innerHTML = html;
    warningsEl.innerHTML = (result.warnings || [])
      .map(w => `<div class="preflight-warning">\u26A0\uFE0F ${escapeHtml(w)}</div>`)
      .join("");
    if (window.lucide) lucide.createIcons();
    return;
  }

  // Binary check
  html += renderCheckItem("llama-server binary", checks.binary_exists, checks.binary_exists ? "Found" : "Not found");

  // Model check
  html += renderCheckItem("Model file", checks.model_exists, checks.model_exists ? "Found" : "Not found");

  // VRAM check
  if (checks.vram) {
    const vram = checks.vram;
    if (vram.local_vram && vram.local_vram.available) {
      const freeGib = vram.local_vram.free_gib;
      const neededGib = vram.estimated_local_gib;
      const label = vram.fits === true ? `OK (${freeGib} GiB free, ~${neededGib} GiB needed)`
        : vram.fits === false ? `Too low (${freeGib} GiB free, ~${vram.needed_gib} GiB needed)`
        : `Unknown (free: ${freeGib} GiB)`;
      html += renderCheckItem("Local VRAM", vram.fits, label);
    } else {
      html += renderCheckItem("Local VRAM", null, "Skipped (no NVIDIA GPU)");
    }
  }

  // RPC check
  if (checks.rpc) {
    const rpc = checks.rpc;
    const rpcStatus = rpc.reachable && rpc.tensor_split_ok !== false;
    const rpcLabel = !rpc.reachable ? `Unreachable: ${rpc.error}`
      : rpc.tensor_split_ok === false ? "Missing tensor split"
      : "Reachable";
    html += renderCheckItem("RPC endpoint", rpcStatus, rpcLabel);
  }

  checksEl.innerHTML = html;

  // Warnings and log diagnoses
  const allWarnings = [...(result.warnings || [])];
  if (result.log_diagnoses) {
    for (const d of result.log_diagnoses) {
      allWarnings.push(`${d.title}: ${d.suggestion}`);
    }
  }
  if (allWarnings.length > 0) {
    warningsEl.innerHTML = allWarnings.map(w => `<div class="preflight-warning">\u26A0\uFE0F ${escapeHtml(w)}</div>`).join("");
  } else {
    warningsEl.innerHTML = "";
  }

  // Re-create Lucide icons for the new elements
  if (window.lucide) lucide.createIcons();
}

function renderCheckItem(name, status, label) {
  const icon = status === true ? "check-circle" : status === false ? "x-circle" : "help-circle";
  const cls = status === true ? "check-ok" : status === false ? "check-bad" : "check-unknown";
  return `<div class="check-item ${cls}"><i data-lucide="${icon}" class="icon-xs"></i><span class="check-name">${escapeHtml(name)}</span><span class="check-label">${escapeHtml(label)}</span></div>`;
}

function escapeHtml(text) {
  const el = document.createElement("span");
  el.textContent = String(text || "");
  return el.innerHTML;
}

async function refreshDiagnostics() {
  try {
    const data = await api("/api/server/logs?lines=80");
    document.getElementById("diag-start-error").textContent = data.start_error || "No error recorded.";
    document.getElementById("diag-log-tail").textContent = data.log_tail || "No log output.";
    const diagEl = document.getElementById("diag-diagnoses");
    if (data.diagnoses && data.diagnoses.length > 0) {
      diagEl.innerHTML = data.diagnoses.map(d =>
        `<div class="diag-item diag-${d.severity}"><strong>${escapeHtml(d.title)}</strong><p>${escapeHtml(d.detail)}</p><p class="diag-suggestion">${escapeHtml(d.suggestion)}</p></div>`
      ).join("");
    } else {
      diagEl.textContent = "No issues found.";
    }
    if (window.lucide) lucide.createIcons();
  } catch {
    // silent
  }
}

async function createChat() {
  const data = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({}),
  });
  await refreshChats();
  await loadChat(data.chat.chat_id);
}

async function saveConfig() {
  const config = readConfigForm();
  try {
    await api("/api/config", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    setStatus("Config saved");
    await refreshConfig();
  } catch (error) {
    setStatus(error.message, false);
  }
}

async function checkRpcEndpoint() {
  const config = readConfigForm();
  const health = document.getElementById("rpc-health");
  if (config.runtime_mode !== "rpc") {
    if (health) {
      health.textContent = "Local mode";
      health.className = "health-badge ok";
    }
    setStatus("RPC check skipped in single-host mode");
    return;
  }
  if (!String(config.rpc_host || "").trim()) {
    if (health) {
      health.textContent = "Missing host";
      health.className = "health-badge bad";
    }
    setStatus("Enter an RPC host before checking", false);
    return;
  }
  setStatus("Checking RPC endpoint...", true);
  if (health) {
    health.textContent = "Checking...";
    health.className = "health-badge neutral";
  }
  const data = await api("/api/rpc/preflight", {
    method: "POST",
    body: JSON.stringify({ config }),
  });
  if (data.reachable) {
    if (health) {
      health.textContent = "Reachable";
      health.className = "health-badge ok";
    }
    setStatus(`RPC ready: ${data.endpoint || "endpoint"}`);
    return;
  }
  if (health) {
    health.textContent = "Unreachable";
    health.className = "health-badge bad";
  }
  setStatus(`RPC unreachable: ${data.error || data.endpoint || "unknown error"}`, false);
}

async function startServer() {
  const config = readConfigForm();
  state.serverStarting = true;
  setStatus("Starting llama.cpp...", true);
  try {
    const result = await api("/api/server/start", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    if (result.status === "starting") {
      pollUntilReady();
    } else {
      await refreshServerStatus();
      setStatus("llama.cpp online");
    }
    await refreshConfig();
  } catch (error) {
    state.serverStarting = false;
    setStatus(error.message, false);
  }
}

async function pollUntilReady() {
  const maxAttempts = 120;
  for (let i = 0; i < maxAttempts; i++) {
    setStatus(`Loading model... (${i + 1}s)`, true);
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const data = await api("/api/server/status");
      if (data.status.healthy) {
        state.serverStarting = false;
        state.config = data.config;
        renderLoadedModelSummary();
        const indicator = document.getElementById("server-indicator");
        indicator.className = "server-status online";
        indicator.querySelector(".status-text").textContent = "Online";
        setStatus("llama.cpp online");
        await refreshConfig();
        return;
      }
      // Check if the process died â€” health() returns managed pid info
      const pid = data.status.pid;
      if (pid === null && i > 3) {
        state.serverStarting = false;
        const rawError = data.status.start_error || data.status.error || "llama-server exited unexpectedly";
        const firstLine = String(rawError).split("\n").find(Boolean) || rawError;
        setStatus(firstLine.replace(/^Error:\s*/, ""), false);
        refreshDiagnostics();
        runPreflight();
        return;
      }
    } catch {
      // keep polling
    }
  }
  state.serverStarting = false;
  setStatus("Timed out waiting for llama.cpp", false);
}

async function stopServer() {
  state.serverStarting = false;
  await api("/api/server/stop", { method: "POST" });
  await refreshServerStatus();
  setStatus("llama.cpp stopped");
}

async function loadSelectedModel() {
  const modelPath = document.getElementById("model_path").value.trim();
  if (!modelPath) return;
  state.serverStarting = true;
  setStatus("Loading selected model...", true);
  try {
    const result = await api("/api/models/load", {
      method: "POST",
      body: JSON.stringify({ model_path: modelPath }),
    });
    if (result.status === "starting") {
      pollUntilReady();
    }
    await refreshModels();
  } catch (error) {
    state.serverStarting = false;
    setStatus(error.message, false);
  }
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
  try {
    const result = await api("/api/server/start", {
      method: "POST",
      body: JSON.stringify({ config: preset.config }),
    });
    if (result.status === "starting") {
      pollUntilReady();
    }
    await refreshModels();
  } catch (error) {
    setStatus(error.message, false);
  }
}

async function startDownload() {
  const url = document.getElementById("download_url").value.trim();
  const destinationPath = document.getElementById("download_destination").value.trim();
  if (!url) return;
  try {
    await api("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ url, destination_path: destinationPath || null }),
    });
    document.getElementById("download_url").value = "";
    await refreshModels();
    setStatus("Download started");
  } catch (error) {
    setStatus(error.message, false);
  }
}

async function stopGeneration() {
  if (!state.generating) return;
  state.stopRequested = true;
  setChatStatus("Stopping...");
  if (state.browserMode) {
    state.browserEngine?.interruptGenerate?.();
    return;
  }
  await api("/api/generation/stop", { method: "POST" });
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Chat Streaming
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

async function streamChatServer(chatId, content) {
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
          streamAppendDelta(event.delta || event.content || "");
          setChatStatus(`Generating... ${streamBuffer.length} chars`);
        } else if (event.type === "done") {
          renderStreamBuffer();
          streamReset();
          await refreshChats();
          renderMessages(event.chat);
          // Handle empty response (thinking-only output sanitized to nothing)
          const msgs = event.chat?.messages || [];
          const lastMsg = msgs[msgs.length - 1];
          if (lastMsg && lastMsg.role === "user") {
            const container = document.getElementById("messages");
            const note = document.createElement("div");
            note.className = "message assistant";
            note.innerHTML = '<div class="role-label">llama.cpp</div><div><p class="message-placeholder">Model returned thinking-only output â€” no visible response.</p></div>';
            container.appendChild(note);
            container.scrollTop = container.scrollHeight;
          }
          setChatStatus(event.cancelled ? `Stopped · ${formatLatencyMs(event.latency_ms)}` : `Done · ${formatLatencyMs(event.latency_ms)}`);
          return;
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      }
      newlineIndex = buffer.indexOf("\n");
    }
  }
}

async function streamChatBrowser(content) {
  if (!state.browserEngine || !state.browserModelId) {
    throw new Error("Load a browser model first.");
  }

  const config = state.config || {};
  const messages = [];
  const systemPrompt = String(config.system_prompt || "").trim();
  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }
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
      setChatStatus(`Generating... ${streamBuffer.length} chars`);
    }
    if (state.stopRequested) {
      state.browserEngine?.interruptGenerate?.();
      break;
    }
  }

  renderStreamBuffer();
  streamReset();
  setChatStatus(state.stopRequested ? "Stopped" : "Done in browser");
}

async function sendMessage() {
  const input = document.getElementById("message-input");
  if (state.generating) {
    await stopGeneration();
    return;
  }

  const content = input.value.trim();
  if (!content) return;
  const indicator = document.getElementById("server-indicator");
  if (!state.browserMode && !indicator.classList.contains("online")) {
    setChatStatus("Server is offline â€” start llama.cpp first", false);
    return;
  }
  if (!state.browserMode && !state.currentChatId) {
    await createChat();
  }

  appendLiveMessage("user", content);
  const assistantEl = appendLiveMessage("assistant", "", { pending: true });
  streamReset();
  streamElement = assistantEl.querySelector("div:last-child") || assistantEl;
  input.value = "";
  autoResizeTextarea(input);
  state.stopRequested = false;
  setGenerating(true);
  setChatStatus("Generating...");
  try {
    if (state.browserMode) {
      await streamChatBrowser(content);
    } else {
      await streamChatServer(state.currentChatId, content);
    }
  } catch (error) {
    const cancelledByUser = state.stopRequested;
    renderStreamBuffer();
    streamReset();
    if (!cancelledByUser) {
      const contentEl = assistantEl.querySelector("div:last-child") || assistantEl;
      contentEl.innerHTML = `<p style="color: var(--red);">${error.message}</p>`;
    }
    setChatStatus(cancelledByUser ? "Stopped" : error.message);
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
  autoResizeTextarea(input);
  input.focus();
}

async function runPreset() {
  applyPresetToComposer();
  await sendMessage();
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Auto-resize textarea
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

const BROWSER_MODELS = [
  { id: "SmolLM2-135M-Instruct-q4f16_1-MLC", label: "SmolLM2 135M", size: "~100 MB" },
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2 360M", size: "~220 MB" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B", size: "~700 MB" },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen2.5 1.5B", size: "~1 GB" },
  { id: "gemma-2-2b-it-q4f16_1-MLC", label: "Gemma 2 2B", size: "~1.4 GB" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", label: "Llama 3.2 3B", size: "~1.8 GB" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", label: "Qwen2.5 3B", size: "~1.8 GB" },
  { id: "Phi-3.5-mini-instruct-q4f16_1-MLC", label: "Phi 3.5 Mini", size: "~2.3 GB" },
];

function detectWebGPU() {
  return Boolean(navigator.gpu);
}

function populateBrowserModels() {
  const select = document.getElementById("browser-model-select");
  if (!select) return;
  select.innerHTML = '<option value="">Choose a browser model...</option>';
  for (const model of BROWSER_MODELS) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.label} (${model.size})`;
    select.appendChild(option);
  }
}

function setBrowserStatus(text, ok = true) {
  const status = document.getElementById("browser-status");
  const badge = document.getElementById("browser-mode-badge");
  if (status) {
    status.textContent = text;
    status.className = `browser-status ${ok ? "ok" : "bad"}`;
  }
  if (badge) {
    badge.textContent = ok ? "Available" : "Unavailable";
    badge.className = `health-badge ${ok ? "ok" : "bad"}`;
  }
}

function setBrowserControlsEnabled(enabled) {
  const select = document.getElementById("browser-model-select");
  const load = document.getElementById("load-browser-model");
  if (select) select.disabled = !enabled;
  if (load) load.disabled = !enabled;
}

async function initBrowserMode() {
  if (!detectWebGPU()) {
    setBrowserStatus("WebGPU is not available in this browser.", false);
    return;
  }

  setBrowserStatus("Loading WebLLM engine...");
  try {
    const webllm = await import("https://esm.run/@mlc-ai/web-llm");
    state.browserEngine = new webllm.MLCEngine({
      initProgressCallback: (progress) => {
        const pct = Number.isFinite(progress.progress) ? ` ${Math.round(progress.progress * 100)}%` : "";
        setBrowserStatus(progress.text || `Preparing WebLLM${pct}`);
      },
    });
    state.browserMode = true;
    populateBrowserModels();
    setBrowserControlsEnabled(true);
    setBrowserStatus("WebLLM ready. Choose a model to load.");
    document.getElementById("browser-section")?.classList.add("active");
    renderLoadedModelSummary();
  } catch (error) {
    state.browserMode = false;
    setBrowserControlsEnabled(false);
    setBrowserStatus(`WebLLM failed to load: ${error.message}`, false);
  }
}

async function loadBrowserModel() {
  const select = document.getElementById("browser-model-select");
  const modelId = select?.value || "";
  if (!modelId || !state.browserEngine) return;

  setGenerating(true);
  setBrowserStatus(`Loading ${modelId}...`);
  try {
    await state.browserEngine.reload(modelId);
    state.browserModelId = modelId;
    state.browserMode = true;
    setBrowserStatus(`Loaded ${modelId}`);
    setStatus(`Browser model loaded: ${modelId}`);
    renderLoadedModelSummary();
  } catch (error) {
    setBrowserStatus(`Model load failed: ${error.message}`, false);
  } finally {
    setGenerating(false);
  }
}

function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  const maxHeight = 200;
  textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + "px";
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Event Bindings
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

document.getElementById("new-chat").onclick = createChat;
document.getElementById("save-config").onclick = saveConfig;
document.getElementById("apply-model-preset").onclick = applyModelPreset;
document.getElementById("load-model-preset").onclick = loadModelPreset;
document.getElementById("start-server").onclick = startServer;
document.getElementById("stop-server").onclick = stopServer;
document.getElementById("check-rpc").onclick = checkRpcEndpoint;
document.getElementById("run-preflight").onclick = runPreflight;
document.getElementById("refresh-models").onclick = refreshModels;
document.getElementById("load-selected-model").onclick = loadSelectedModel;
document.getElementById("start-download").onclick = startDownload;
document.getElementById("send-message").onclick = sendMessage;
document.getElementById("apply-preset").onclick = applyPresetToComposer;
document.getElementById("run-preset").onclick = runPreset;
document.getElementById("init-browser").onclick = initBrowserMode;
document.getElementById("load-browser-model").onclick = loadBrowserModel;

// Settings drawer
document.getElementById("settings-btn").onclick = openDrawer;
document.getElementById("open-settings-btn").onclick = openDrawer;
document.getElementById("close-drawer").onclick = closeDrawer;
document.getElementById("drawer-backdrop").onclick = closeDrawer;

// Sidebar toggle (mobile)
document.getElementById("toggle-sidebar").onclick = toggleSidebar;

// Candidate model select
document.getElementById("candidate_models").onchange = (event) => {
  if (event.target.value) {
    document.getElementById("model_path").value = event.target.value;
    renderModelLibrary();
  }
};

// RPC field change handlers
for (const field of ["runtime_mode", "rpc_host", "rpc_port", "rpc_tensor_split"]) {
  const element = document.getElementById(field);
  element.addEventListener("input", updateRpcGuide);
  element.addEventListener("change", updateRpcGuide);
}

// Composer: Ctrl+Enter to send
document.getElementById("message-input").addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    await sendMessage();
  }
});

// Composer: auto-resize on input
document.getElementById("message-input").addEventListener("input", function () {
  autoResizeTextarea(this);
});

// Escape to close drawer
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeDrawer();
    document.getElementById("sidebar").classList.remove("open");
  }
});

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Initialize
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

(async function init() {
  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  loadPromptPresets();
  if (detectWebGPU()) {
    populateBrowserModels();
    setBrowserStatus("WebGPU available. Enable Browser Inference to load WebLLM.");
  } else {
    setBrowserStatus("WebGPU is not available in this browser.", false);
  }
  await refreshConfig();
  await refreshModels();
  await refreshChats();
  await refreshServerStatus();
  await runPreflight();
  if (state.chats.length > 0) {
    await loadChat(state.chats[0].chat_id);
  }
  setInterval(refreshModels, 5000);
  setInterval(refreshServerStatus, 5000);
  setInterval(refreshDiagnostics, 5000);
})();
