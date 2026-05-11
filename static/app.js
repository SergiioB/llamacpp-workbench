/* ═══════════════════════════════════════════════════════════════
   llama-webui — Frontend v2
   ═══════════════════════════════════════════════════════════════ */

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
  lastGenStats: null,
  knowledgeMode: false,
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

/* ═══════════════════════════════════════════════════════════════
   Toast Notification System
   ═══════════════════════════════════════════════════════════════ */

const TOAST_ICONS = {
  success: "check-circle",
  error: "alert-circle",
  info: "info",
  warning: "alert-triangle",
};

function showToast(message, type = "info", duration = 3500) {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i data-lucide="${TOAST_ICONS[type] || 'info'}" class="icon-md toast-icon"></i>
    <span class="toast-message">${escapeHtml(message)}</span>
    <button class="toast-close" title="Dismiss"><i data-lucide="x" class="icon-xs"></i></button>
  `;
  toast.querySelector(".toast-close").onclick = () => dismissToast(toast);
  container.appendChild(toast);
  if (window.lucide) lucide.createIcons({ nodes: [toast] });
  if (duration > 0) {
    setTimeout(() => dismissToast(toast), duration);
  }
  return toast;
}

function dismissToast(toast) {
  if (!toast || !toast.parentNode) return;
  toast.classList.add("leaving");
  setTimeout(() => toast.remove(), 200);
}

/* ═══════════════════════════════════════════════════════════════
   Confirm Dialog
   ═══════════════════════════════════════════════════════════════ */

function confirmDialog(title, message) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "confirm-overlay";
    overlay.innerHTML = `
      <div class="confirm-dialog">
        <div class="confirm-title">${escapeHtml(title)}</div>
        <div class="confirm-message">${escapeHtml(message)}</div>
        <div class="confirm-actions">
          <button class="btn btn-ghost btn-sm" data-action="cancel">Cancel</button>
          <button class="btn btn-danger btn-sm" data-action="confirm">Delete</button>
        </div>
      </div>
    `;
    overlay.onclick = (e) => {
      if (e.target === overlay) { overlay.remove(); resolve(false); }
    };
    overlay.querySelector('[data-action="cancel"]').onclick = () => { overlay.remove(); resolve(false); };
    overlay.querySelector('[data-action="confirm"]').onclick = () => { overlay.remove(); resolve(true); };
    document.body.appendChild(overlay);
  });
}

/* ═══════════════════════════════════════════════════════════════
   API helpers
   ═══════════════════════════════════════════════════════════════ */

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

/* ═══════════════════════════════════════════════════════════════
   Settings Drawer
   ═══════════════════════════════════════════════════════════════ */

function openDrawer() {
  document.getElementById("settings-drawer").classList.add("open");
}

function closeDrawer() {
  document.getElementById("settings-drawer").classList.remove("open");
}

/* ═══════════════════════════════════════════════════════════════
   Sidebar Toggle (mobile)
   ═══════════════════════════════════════════════════════════════ */

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

/* ═══════════════════════════════════════════════════════════════
   Config Form
   ═══════════════════════════════════════════════════════════════ */

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
    option.textContent = `${preset.label} — ${preset.description}`;
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

/* ═══════════════════════════════════════════════════════════════
   Utility
   ═══════════════════════════════════════════════════════════════ */

function formatGiB(sizeBytes) {
  return `${(sizeBytes / (1024 ** 3)).toFixed(2)} GiB`;
}

function basename(path) {
  if (!path) return "";
  const parts = path.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path;
}

function escapeHtml(text) {
  const el = document.createElement("span");
  el.textContent = String(text || "");
  return el.innerHTML;
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(
    () => showToast("Copied to clipboard", "success", 2000),
    () => showToast("Failed to copy", "error", 2000)
  );
}

function formatLatencyMs(ms) {
  return `${(Number(ms) / 1000).toFixed(2)}s`;
}

function renderLoadedModelSummary() {
  const nameEl = document.getElementById("loaded-model-name");
  const metaEl = document.getElementById("loaded-model-meta");
  if (!nameEl || !metaEl || !state.config) return;

  if (state.browserMode) {
    nameEl.textContent = state.browserModelId || "Browser model";
    nameEl.classList.toggle("unloaded", !state.browserModelId);
    metaEl.textContent = state.browserModelId ? "WebGPU in-browser" : "WebGPU ready";
    updateSidebarMetrics();
    return;
  }

  const modelPath = String(state.config.model_path || "");
  const modelName = basename(modelPath);
  nameEl.textContent = modelName || "No model loaded";
  nameEl.classList.toggle("unloaded", !modelName);

  const ctx = state.config.ctx_size ? `ctx ${state.config.ctx_size}` : "";
  const temp = Number.isFinite(Number(state.config.temperature)) ? `temp ${Number(state.config.temperature).toFixed(2)}` : "";
  const maxTokens = state.config.max_tokens ? `max ${state.config.max_tokens}` : "";
  const rpcEndpoint = state.config.rpc_host ? `${state.config.rpc_host}:${state.config.rpc_port}` : "";
  const split = state.config.rpc_tensor_split ? ` split ${state.config.rpc_tensor_split}` : "";
  const mode = state.config.runtime_mode === "rpc" ? `rpc ${rpcEndpoint}${split}` : "";
  const parts = [mode, ctx, temp, maxTokens].filter(Boolean);
  metaEl.textContent = parts.join(" · ");

  updateSidebarMetrics();
}

/* ═══════════════════════════════════════════════════════════════
   Sidebar Metrics
   ═══════════════════════════════════════════════════════════════ */

function updateSidebarMetrics() {
  const config = state.config;
  if (!config) return;

  // Model name
  const modelVal = document.getElementById("metric-model-value");
  const modelPath = String(config.model_path || "");
  const modelName = basename(modelPath);
  if (modelVal) {
    modelVal.textContent = modelName || "—";
    modelVal.className = `metric-value ${modelName ? "active" : ""}`;
  }

  // Runtime / backend
  const backendVal = document.getElementById("metric-backend-value");
  if (backendVal) {
    const mode = config.runtime_mode === "rpc" ? "rpc" : "local";
    const gpuLayers = Number(config.gpu_layers || 0);
    let backend = mode;
    if (gpuLayers > 0) backend = `${mode}:cuda`;
    else if (gpuLayers === 0) backend = `${mode}:cpu`;
    backendVal.textContent = backend;
    backendVal.className = `metric-value ${gpuLayers > 0 ? "active" : ""}`;
  }

  // Context size
  const ctxVal = document.getElementById("metric-ctx-value");
  if (ctxVal) {
    const ctx = Number(config.ctx_size || 0);
    ctxVal.textContent = ctx > 0 ? ctx.toLocaleString() : "—";
  }

  // GPU layers
  const layersVal = document.getElementById("metric-layers-value");
  if (layersVal) {
    const layers = Number(config.gpu_layers || 0);
    layersVal.textContent = layers > 0 ? `${layers} layers` : "off";
    layersVal.className = `metric-value ${layers > 0 ? "active" : ""}`;
  }
}

function updateSidebarFooterStats() {
  const chatCount = document.getElementById("sidebar-chat-count");
  const modelCount = document.getElementById("sidebar-model-count");
  if (chatCount) chatCount.textContent = `${state.chats.length} chat${state.chats.length !== 1 ? "s" : ""}`;
  if (modelCount) modelCount.textContent = `${state.models.length} model${state.models.length !== 1 ? "s" : ""}`;
}

/* ═══════════════════════════════════════════════════════════════
   Thinking Block Parser
   ═══════════════════════════════════════════════════════════════ */

function parseThinkingBlocks(content) {
  // Match <think...>...</think > or <thinking...>...</thinking> blocks
  const thinkRegex = /<think(?:ing)?(?:\s[^>]*)?>([\s\S]*?)<\/think(?:ing)?>/gi;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = thinkRegex.exec(content)) !== null) {
    // Text before the thinking block
    if (match.index > lastIndex) {
      parts.push({ type: "text", content: content.slice(lastIndex, match.index) });
    }
    // The thinking block content
    const thinkContent = match[1].trim();
    if (thinkContent) {
      parts.push({ type: "thinking", content: thinkContent });
    }
    lastIndex = match.index + match[0].length;
  }

  // Remaining text after last thinking block
  if (lastIndex < content.length) {
    parts.push({ type: "text", content: content.slice(lastIndex) });
  }

  return parts;
}

function renderContentWithThinking(element, content, pending = false) {
  const safeContent = content || "";
  if (!safeContent.trim()) {
    if (pending) {
      element.innerHTML = `<p class="message-placeholder">Generating<span class="dot-1">.</span><span class="dot-2">.</span><span class="dot-3">.</span></p>`;
    } else {
      element.textContent = safeContent;
    }
    return;
  }

  const parts = parseThinkingBlocks(safeContent);

  if (parts.length === 0 || (parts.length === 1 && parts[0].type === "text")) {
    // No thinking blocks — render as normal markdown
    element.innerHTML = markdown.render(safeContent.replace(/<think(?:ing)?(?:\s[^>]*)?>[\s\S]*?<\/think(?:ing)?>/gi, "").trim() || safeContent);
  } else {
    let html = "";
    for (const part of parts) {
      if (part.type === "thinking") {
        const id = "think-" + Math.random().toString(36).slice(2, 8);
        html += `<div class="thinking-block collapsed" id="${id}">
          <div class="thinking-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <i data-lucide="chevron-down" class="collapse-icon"></i>
            <span>Thinking…</span>
          </div>
          <div class="thinking-content">${markdown.render(part.content)}</div>
        </div>`;
      } else {
        const trimmed = part.content.trim();
        if (trimmed) {
          html += markdown.render(trimmed);
        }
      }
    }
    element.innerHTML = html;
  }

  // Add copy buttons to code blocks
  addCodeBlockCopyButtons(element);
  // Re-init lucide icons for new elements
  if (window.lucide) lucide.createIcons({ nodes: [element] });
}

function addCodeBlockCopyButtons(container) {
  // Find all pre > code blocks and add copy buttons with language labels
  const codeBlocks = container.querySelectorAll("pre");
  for (const pre of codeBlocks) {
    if (pre.querySelector(".code-block-header")) continue;

    const code = pre.querySelector("code");
    if (!code) continue;

    // Detect language from class
    const langClass = code.className?.match(/language-(\w+)/)?.[1] || "";

    // Create header
    const header = document.createElement("div");
    header.className = "code-block-header";
    header.innerHTML = `
      <span>${langClass || "code"}</span>
      <button class="code-copy-btn" title="Copy code">
        <i data-lucide="copy" style="width:12px;height:12px;"></i>
        Copy
      </button>
    `;

    const copyBtn = header.querySelector(".code-copy-btn");
    copyBtn.onclick = (e) => {
      e.stopPropagation();
      const text = code.textContent || "";
      navigator.clipboard.writeText(text).then(() => {
        copyBtn.classList.add("copied");
        copyBtn.innerHTML = `<i data-lucide="check" style="width:12px;height:12px;"></i> Copied`;
        if (window.lucide) lucide.createIcons({ nodes: [copyBtn] });
        setTimeout(() => {
          copyBtn.classList.remove("copied");
          copyBtn.innerHTML = `<i data-lucide="copy" style="width:12px;height:12px;"></i> Copy`;
          if (window.lucide) lucide.createIcons({ nodes: [copyBtn] });
        }, 2000);
      });
    };

    pre.parentNode.insertBefore(header, pre);
  }
}

/* ═══════════════════════════════════════════════════════════════
   Rendering
   ═══════════════════════════════════════════════════════════════ */

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

    // Calculate progress percentage
    let progressHtml = "";
    if (job.status === "running" && job.downloaded_bytes && job.total_bytes) {
      const pct = Math.round((job.downloaded_bytes / job.total_bytes) * 100);
      progressHtml = `<div class="download-progress"><div class="download-progress-bar" style="width:${pct}%"></div></div>`;
    }

    card.innerHTML = `
      <div class="card-title">${job.status.toUpperCase()}</div>
      <div class="card-meta">${job.destination_path}</div>
      <div class="card-meta">${job.downloaded_bytes ? formatGiB(job.downloaded_bytes) : "0.00 GiB"}${job.total_bytes ? ` / ${formatGiB(job.total_bytes)}` : ""} downloaded</div>
      ${progressHtml}
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
    item.innerHTML = `
      <span class="chat-item-title">${escapeHtml(chat.title)}</span>
      <div class="chat-item-actions">
        <button class="chat-action-btn edit-btn" title="Rename"><i data-lucide="pencil" style="width:13px;height:13px;"></i></button>
        <button class="chat-action-btn delete-btn" title="Delete"><i data-lucide="trash-2" style="width:13px;height:13px;"></i></button>
      </div>
    `;

    // Click to load chat
    item.querySelector(".chat-item-title").onclick = () => loadChat(chat.chat_id);

    // Rename
    item.querySelector(".edit-btn").onclick = (e) => {
      e.stopPropagation();
      startRenameChat(item, chat);
    };

    // Delete
    item.querySelector(".delete-btn").onclick = async (e) => {
      e.stopPropagation();
      const confirmed = await confirmDialog("Delete chat?", `"${chat.title}" will be permanently deleted.`);
      if (confirmed) {
        await api(`/api/chats/${chat.chat_id}`, { method: "DELETE" });
        if (state.currentChatId === chat.chat_id) {
          state.currentChatId = null;
          updateEmptyState();
        }
        showToast("Chat deleted", "success", 2000);
        await refreshChats();
      }
    };

    chatList.appendChild(item);
  }
  if (window.lucide) lucide.createIcons({ nodes: [chatList] });
  updateSidebarFooterStats();
}

function startRenameChat(itemEl, chat) {
  itemEl.classList.add("editing");
  const input = document.createElement("input");
  input.className = "chat-rename-input";
  input.value = chat.title;
  input.onclick = (e) => e.stopPropagation();

  const finishRename = async () => {
    const newTitle = input.value.trim();
    itemEl.classList.remove("editing");
    if (newTitle && newTitle !== chat.title) {
      try {
        await api(`/api/chats/${chat.chat_id}`, {
          method: "PATCH",
          body: JSON.stringify({ title: newTitle }),
        });
      } catch (e) {
        showToast("Failed to rename chat", "error");
      }
      chat.title = newTitle;
      renderChats();
      showToast("Chat renamed", "success", 2000);
    } else {
      renderChats();
    }
  };

  input.onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); finishRename(); }
    if (e.key === "Escape") { itemEl.classList.remove("editing"); renderChats(); }
  };
  input.onblur = finishRename;

  // Replace content with input
  itemEl.innerHTML = "";
  itemEl.appendChild(input);
  input.focus();
  input.select();
}

function renderMessages(chat) {
  document.getElementById("chat-title").textContent = chat.title;
  const container = document.getElementById("messages");
  container.innerHTML = "";
  container.classList.remove("hidden");
  updateEmptyState(false);
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

  renderContentWithThinking(contentEl, content, options.pending);

  // Add message-level action buttons
  if (role === "assistant" && content) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    actions.innerHTML = `
      <button class="msg-action-btn copy-msg-btn" title="Copy message">
        <i data-lucide="copy" style="width:14px;height:14px;"></i>
      </button>
    `;
    actions.querySelector(".copy-msg-btn").onclick = () => {
      copyToClipboard(content.replace(/<think(?:ing)?(?:\s[^>]*)?>[\s\S]*?<\/think(?:ing)?>/gi, "").trim() || content);
    };
    el.appendChild(actions);
  }

  return el;
}

function appendLiveMessage(role, content = "", options = {}) {
  const container = document.getElementById("messages");
  container.classList.remove("hidden");
  updateEmptyState(false);
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

let streamBuffer = "";
let streamElement = null;
let streamRenderScheduled = false;
let streamStartTime = null;
let streamTokenCount = 0;

function streamAppendDelta(delta) {
  streamBuffer += delta;
  streamTokenCount++;
  if (!streamRenderScheduled) {
    streamRenderScheduled = true;
    requestAnimationFrame(renderStreamBuffer);
  }
}

function renderStreamBuffer() {
  streamRenderScheduled = false;
  if (!streamElement) return;
  renderContentWithThinking(streamElement, streamBuffer);
  const container = document.getElementById("messages");
  container.scrollTop = container.scrollHeight;
}

function streamReset() {
  streamBuffer = "";
  streamElement = null;
  streamRenderScheduled = false;
  streamTokenCount = 0;
}

/* ═══════════════════════════════════════════════════════════════
   Empty State Management
   ═══════════════════════════════════════════════════════════════ */

function updateEmptyState(show = true) {
  const emptyState = document.getElementById("empty-state");
  const messages = document.getElementById("messages");
  if (show) {
    emptyState.classList.remove("hidden");
    messages.classList.add("hidden");
  } else {
    emptyState.classList.add("hidden");
    messages.classList.remove("hidden");
  }
}

/* ═══════════════════════════════════════════════════════════════
   Scroll-to-Bottom Button
   ═══════════════════════════════════════════════════════════════ */

function setupScrollToBottom() {
  const messages = document.getElementById("messages");
  const btn = document.getElementById("scroll-bottom");

  messages.addEventListener("scroll", () => {
    const atBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 100;
    btn.classList.toggle("visible", !atBottom);
  });

  btn.onclick = () => {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  };
}

/* ═══════════════════════════════════════════════════════════════
   Status helpers
   ═══════════════════════════════════════════════════════════════ */

function setStatus(text, ok = true) {
  const status = document.getElementById("status-text");
  status.textContent = text;
  status.style.color = ok ? "var(--green)" : "var(--red)";
}

function currentRpcEndpoint() {
  const config = readConfigForm();
  const host = String(config.rpc_host || "").trim();
  const port = Number(config.rpc_port || 0);
  return host && port > 0 ? `${host}:${port}` : "";
}

function clearStaleRpcStatus() {
  const status = document.getElementById("status-text");
  const endpoint = currentRpcEndpoint();
  if (!status || !endpoint) return;
  const text = status.textContent || "";
  if (text.includes("RPC endpoint") && !text.includes(endpoint)) {
    setStatus(`RPC ${endpoint} is reachable`);
  }
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
  const icon = button.querySelector("i, svg");
  if (icon) {
    button.innerHTML = active
      ? '<i data-lucide="square" class="icon-md"></i>'
      : '<i data-lucide="arrow-up" class="icon-md"></i>';
    lucide.createIcons();
  }
}

/* ═══════════════════════════════════════════════════════════════
   Generation Stats
   ═══════════════════════════════════════════════════════════════ */

function addGenStats(messageEl, latencyMs, tokenCount, cancelled) {
  if (!messageEl || !latencyMs) return;

  const statsEl = document.createElement("div");
  statsEl.className = "gen-stats";

  const latencySec = (latencyMs / 1000);
  const tokPerSec = tokenCount && latencySec > 0 ? (tokenCount / latencySec).toFixed(1) : "—";
  const tokens = tokenCount || "—";

  statsEl.innerHTML = `
    <span class="gen-stat">
      <i data-lucide="clock" style="width:12px;height:12px;"></i>
      <span class="gen-stat-value">${formatLatencyMs(latencyMs)}</span>
    </span>
    <span class="gen-stat">
      <i data-lucide="hash" style="width:12px;height:12px;"></i>
      <span class="gen-stat-value">${tokens} tokens</span>
    </span>
    <span class="gen-stat">
      <i data-lucide="gauge" style="width:12px;height:12px;"></i>
      <span class="gen-stat-value">${tokPerSec} tok/s</span>
    </span>
    ${cancelled ? '<span class="gen-stat" style="color:var(--accent)">Stopped</span>' : ''}
  `;

  // Insert after content div, before message-actions
  const actionsEl = messageEl.querySelector(".message-actions");
  if (actionsEl) {
    messageEl.insertBefore(statsEl, actionsEl);
  } else {
    messageEl.appendChild(statsEl);
  }

  if (window.lucide) lucide.createIcons({ nodes: [statsEl] });
}

/* ═══════════════════════════════════════════════════════════════
   Prompt Presets
   ═══════════════════════════════════════════════════════════════ */

function loadPromptPresets() {
  const select = document.getElementById("prompt-presets");
  select.innerHTML = `<option value="">Presets…</option>`;
  for (const preset of PROMPT_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.text;
    option.textContent = preset.label;
    select.appendChild(option);
  }
}

/* ═══════════════════════════════════════════════════════════════
   Data refresh
   ═══════════════════════════════════════════════════════════════ */

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
  updateSidebarFooterStats();
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
  indicator.className = `server-status ${state.serverStarting ? "starting" : isOnline ? "online" : "offline"}`;
  indicator.querySelector(".status-text").textContent = state.serverStarting ? "Loading" : isOnline ? "Online" : "OFF";
  if (data.config) {
    state.config = data.config;
    renderLoadedModelSummary();
  }

  // Show/hide unload button
  const unloadBtn = document.getElementById("unload-model-btn");
  if (unloadBtn) {
    unloadBtn.style.display = isOnline ? "flex" : "none";
  }

  const serverManaged = Boolean(data.status?.managed || data.status?.pid || state.serverStarting);
  if (!isOnline && state.preflightAutoRefresh && !serverManaged) {
    runPreflight();
  }
}

/* ═══════════════════════════════════════════════════════════════
   Actions
   ═══════════════════════════════════════════════════════════════ */

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

  statusEl.textContent = "Checking…";
  statusEl.className = "preflight-status neutral";

  try {
    const result = await api("/api/preflight", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    state.preflight = result;
    renderPreflight(result, statusEl, checksEl, warningsEl);
    if (startBtn) {
      startBtn.disabled = !result.ready;
      startBtn.title = result.ready ? "Start llama.cpp" : "Fix blocking issues first";
    }
  } catch (error) {
    const msg = String(error.message || "");
    const notFound = msg.includes("Not Found") || msg.includes("404");
    statusEl.textContent = notFound ? "Preflight unavailable (restart WebUI to enable)" : `Check failed: ${msg}`;
    statusEl.className = "preflight-status neutral";
    checksEl.innerHTML = "";
    warningsEl.innerHTML = "";
  }
}

function renderPreflight(result, statusEl, checksEl, warningsEl) {
  if (result.ready) {
    statusEl.textContent = "Ready to launch";
    statusEl.className = "preflight-status ok";
    clearStaleRpcStatus();
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
      .map(w => `<div class="preflight-warning">⚠️ ${escapeHtml(w)}</div>`)
      .join("");
    if (window.lucide) lucide.createIcons();
    return;
  }

  html += renderCheckItem("llama-server binary", checks.binary_exists, checks.binary_exists ? "Found" : "Not found");
  html += renderCheckItem("Model file", checks.model_exists, checks.model_exists ? "Found" : "Not found");

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

  if (checks.rpc) {
    const rpc = checks.rpc;
    const rpcStatus = rpc.reachable && rpc.tensor_split_ok !== false;
    const rpcLabel = !rpc.reachable ? `Unreachable: ${rpc.error}`
      : rpc.tensor_split_ok === false ? "Missing tensor split"
      : "Reachable";
    html += renderCheckItem("RPC endpoint", rpcStatus, rpcLabel);
  }

  checksEl.innerHTML = html;

  const allWarnings = [...(result.warnings || [])];
  if (result.log_diagnoses) {
    for (const d of result.log_diagnoses) {
      allWarnings.push(`${d.title}: ${d.suggestion}`);
    }
  }
  warningsEl.innerHTML = allWarnings.length > 0
    ? allWarnings.map(w => `<div class="preflight-warning">⚠️ ${escapeHtml(w)}</div>`).join("")
    : "";

  if (window.lucide) lucide.createIcons();
}

function renderCheckItem(name, status, label) {
  const icon = status === true ? "check-circle" : status === false ? "x-circle" : "help-circle";
  const cls = status === true ? "check-ok" : status === false ? "check-bad" : "check-unknown";
  return `<div class="check-item ${cls}"><i data-lucide="${icon}" class="icon-xs"></i><span class="check-name">${escapeHtml(name)}</span><span class="check-label">${escapeHtml(label)}</span></div>`;
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
    showToast("Configuration saved", "success", 2000);
    await refreshConfig();
  } catch (error) {
    setStatus(error.message, false);
    showToast(error.message, "error");
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
  setStatus("Checking RPC endpoint…", true);
  if (health) {
    health.textContent = "Checking…";
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
    showToast(`RPC endpoint reachable: ${data.endpoint}`, "success", 3000);
    return;
  }
  if (health) {
    health.textContent = "Unreachable";
    health.className = "health-badge bad";
  }
  setStatus(`RPC unreachable: ${data.error || data.endpoint || "unknown error"}`, false);
  showToast(`RPC unreachable: ${data.error || data.endpoint}`, "error");
}

async function startServer() {
  const config = readConfigForm();
  state.serverStarting = true;
  setStatus("Starting llama.cpp…", true);
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
      showToast("llama.cpp is online", "success");
    }
    await refreshConfig();
  } catch (error) {
    state.serverStarting = false;
    setStatus(error.message, false);
    showToast(error.message, "error");
  }
}

async function pollUntilReady() {
  const maxAttempts = 120;
  for (let i = 0; i < maxAttempts; i++) {
    setStatus(`Loading model… (${i + 1}s)`, true);
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
        showToast("Model loaded — llama.cpp is online", "success");
        await refreshConfig();
        return;
      }
      const pid = data.status.pid;
      if (pid === null && i > 3) {
        state.serverStarting = false;
        const rawError = data.status.start_error || data.status.error || "llama-server exited unexpectedly";
        const firstLine = String(rawError).split("\n").find(Boolean) || rawError;
        setStatus(firstLine.replace(/^Error:\s*/, ""), false);
        showToast(firstLine.replace(/^Error:\s*/, ""), "error");
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
  showToast("Timed out waiting for llama.cpp", "error");
}

async function stopServer() {
  state.serverStarting = false;
  await api("/api/server/stop", { method: "POST" });
  await refreshServerStatus();
  setStatus("llama.cpp stopped");
  showToast("llama.cpp stopped", "info", 2000);
}

async function loadSelectedModel() {
  const modelPath = document.getElementById("model_path").value.trim();
  if (!modelPath) return;
  state.serverStarting = true;
  setStatus("Loading selected model…", true);
  showToast("Loading model…", "info", 0);
  try {
    await api("/api/config", {
      method: "POST",
      body: JSON.stringify({ config: readConfigForm() }),
    });
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
    showToast(error.message, "error");
  }
}

function selectedPreset() {
  const presetId = document.getElementById("model_presets").value;
  return state.modelPresets.find((preset) => preset.id === presetId) || null;
}

async function applyModelPreset() {
  const preset = selectedPreset();
  if (!preset) return;
  writeConfigForm(preset.config, state.models.map((model) => model.path), state.modelPresets);
  renderModelLibrary();
  try {
    await api("/api/config", {
      method: "POST",
      body: JSON.stringify({ config: readConfigForm() }),
    });
    setStatus(`Applied and saved preset: ${preset.label}`);
    showToast(`Applied preset: ${preset.label}`, "success", 2500);
    await refreshConfig();
    await runPreflight();
  } catch (error) {
    setStatus(error.message, false);
    showToast(error.message, "error");
  }
}

async function loadModelPreset() {
  const preset = selectedPreset();
  if (!preset) return;
  writeConfigForm(preset.config, state.models.map((model) => model.path), state.modelPresets);
  renderModelLibrary();
  setStatus(`Loading preset: ${preset.label}`, true);
  showToast(`Loading preset: ${preset.label}…`, "info", 0);
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
    state.serverStarting = false;
    setStatus(error.message, false);
    showToast(error.message, "error");
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
    showToast("Download started", "success", 2500);
  } catch (error) {
    setStatus(error.message, false);
    showToast(error.message, "error");
  }
}

async function stopGeneration() {
  if (!state.generating) return;
  state.stopRequested = true;
  setChatStatus("Stopping…");
  if (state.browserMode) {
    state.browserEngine?.interruptGenerate?.();
    return;
  }
  await api("/api/generation/stop", { method: "POST" });
}

/* ═══════════════════════════════════════════════════════════════
   Chat Streaming
   ═══════════════════════════════════════════════════════════════ */

async function streamChatServer(chatId, content, knowledgeContext = null) {
  const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, knowledge_context: knowledgeContext || undefined }),
  });
  if (!response.ok || !response.body) {
    throw new Error(await parseErrorResponse(response) || "stream request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  streamStartTime = streamStartTime || Date.now();

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
          setChatStatus(`Generating… ${streamBuffer.length} chars`);
        } else if (event.type === "done") {
          renderStreamBuffer();
          streamReset();
          await refreshChats();
          renderMessages(event.chat);
          // Handle empty response
          const msgs = event.chat?.messages || [];
          const lastMsg = msgs[msgs.length - 1];
          if (lastMsg && lastMsg.role === "user") {
            const container = document.getElementById("messages");
            const note = document.createElement("div");
            note.className = "message assistant";
            note.innerHTML = '<div class="role-label">llama.cpp</div><div><p class="message-placeholder">Model returned thinking-only output — no visible response.</p></div>';
            container.appendChild(note);
            container.scrollTop = container.scrollHeight;
          }

          // Add generation stats to last assistant message
          const container = document.getElementById("messages");
          const assistantMsgs = container.querySelectorAll(".message.assistant");
          const lastAssistantEl = assistantMsgs[assistantMsgs.length - 1];
          if (lastAssistantEl && event.latency_ms) {
            // Estimate token count from response content
            const responseContent = event.assistant_message?.content || streamBuffer || "";
            const estimatedTokens = estimateTokens(responseContent);
            addGenStats(lastAssistantEl, event.latency_ms, estimatedTokens, event.cancelled);
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

function estimateTokens(text) {
  if (!text) return 0;
  // Rough estimation: ~4 chars per token for English, ~2 for code
  return Math.round(text.length / 3.5);
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

  streamStartTime = Date.now();
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
      setChatStatus(`Generating… ${streamBuffer.length} chars`);
    }
    if (state.stopRequested) {
      state.browserEngine?.interruptGenerate?.();
      break;
    }
  }

  renderStreamBuffer();
  const elapsed = Date.now() - streamStartTime;
  const estimatedTokens = estimateTokens(streamBuffer);

  // Add stats to the message element
  if (streamElement) {
    const msgEl = streamElement.closest(".message");
    if (msgEl) {
      addGenStats(msgEl, elapsed, estimatedTokens, state.stopRequested);
    }
  }

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
    setChatStatus("Server is offline — start llama.cpp first", false);
    showToast("Server is offline — open Settings to start llama.cpp", "warning");
    return;
  }
  if (!state.browserMode && !state.currentChatId) {
    await createChat();
  }

  appendLiveMessage("user", content);
  const assistantEl = appendLiveMessage("assistant", "", { pending: true });
  streamReset();
  streamElement = assistantEl.querySelector("div:last-child") || assistantEl;
  streamStartTime = Date.now();
  input.value = "";
  autoResizeTextarea(input);
  state.stopRequested = false;
  setGenerating(true);
  setChatStatus("Generating…");

  // Knowledge retrieval
  let knowledgeContext = null;
  if (state.knowledgeMode && !state.browserMode) {
    setChatStatus("Retrieving knowledge…");
    try {
      const kData = await api("/api/knowledge/query", {
        method: "POST",
        body: JSON.stringify({ query: content, top_k: 5, use_vectors: true }),
      });
      if (kData.results && kData.results.length > 0) {
        knowledgeContext = kData.results
          .map((r, i) => `[${i + 1}] (${r.source}/${r.category}) ${r.text}`)
          .join("\n\n");
        showContextIndicator(kData.results.length, kData.stats?.elapsed_ms);
      }
    } catch (e) {
      // Silently continue without context
    }
  }

  try {
    if (state.browserMode) {
      await streamChatBrowser(content);
    } else {
      await streamChatServer(state.currentChatId, content, knowledgeContext);
    }
  } catch (error) {
    const cancelledByUser = state.stopRequested;
    renderStreamBuffer();
    streamReset();
    if (!cancelledByUser) {
      const contentEl = assistantEl.querySelector("div:last-child") || assistantEl;
      contentEl.innerHTML = `<p style="color: var(--red);">${escapeHtml(error.message)}</p>`;
      showToast(error.message, "error");
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

/* ═══════════════════════════════════════════════════════════════
   Browser Inference
   ═══════════════════════════════════════════════════════════════ */

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
  select.innerHTML = '<option value="">Choose a browser model…</option>';
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

  setBrowserStatus("Loading WebLLM engine…");
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
    showToast("WebLLM engine ready", "success", 2500);
  } catch (error) {
    state.browserMode = false;
    setBrowserControlsEnabled(false);
    setBrowserStatus(`WebLLM failed to load: ${error.message}`, false);
    showToast("WebLLM failed to load", "error");
  }
}

async function loadBrowserModel() {
  const select = document.getElementById("browser-model-select");
  const modelId = select?.value || "";
  if (!modelId || !state.browserEngine) return;

  setGenerating(true);
  setBrowserStatus(`Loading ${modelId}…`);
  try {
    await state.browserEngine.reload(modelId);
    state.browserModelId = modelId;
    state.browserMode = true;
    setBrowserStatus(`Loaded ${modelId}`);
    setStatus(`Browser model loaded: ${modelId}`);
    renderLoadedModelSummary();
    showToast(`Browser model loaded: ${modelId}`, "success");
  } catch (error) {
    setBrowserStatus(`Model load failed: ${error.message}`, false);
    showToast(`Model load failed: ${error.message}`, "error");
  } finally {
    setGenerating(false);
  }
}

/* ═══════════════════════════════════════════════════════════════
   Auto-resize textarea
   ═══════════════════════════════════════════════════════════════ */

function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  const maxHeight = 200;
  textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + "px";
}

/* ═══════════════════════════════════════════════════════════════
   Event Bindings
   ═══════════════════════════════════════════════════════════════ */

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

// Knowledge panel
document.getElementById("knowledge-tab-btn").onclick = toggleKnowledgePanel;
document.getElementById("knowledge-back-btn").onclick = hideKnowledgePanel;
document.getElementById("knowledge-search-btn").onclick = searchKnowledge;
document.getElementById("knowledge-refresh-sources").onclick = refreshKnowledgeSources;
document.getElementById("knowledge-embed-btn").onclick = embedKnowledge;
document.getElementById("knowledge-clear-btn").onclick = clearKnowledge;
document.getElementById("knowledge-toggle-btn").onclick = toggleKnowledgeMode;

document.getElementById("knowledge-search-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    searchKnowledge();
  }
});

// Unload model button
document.getElementById("unload-model-btn").onclick = async () => {
  const confirmed = await confirmDialog("Unload model?", "This will stop llama-server and free VRAM on all devices.");
  if (!confirmed) return;
  await stopServer();
  showToast("Model unloaded — VRAM freed", "success");
};

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

// Escape to close drawer or knowledge panel
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (knowledgeState.active) {
      hideKnowledgePanel();
    } else {
      closeDrawer();
    }
    document.getElementById("sidebar").classList.remove("open");
  }
});

// Empty state shortcut chips
document.getElementById("empty-shortcuts").addEventListener("click", (e) => {
  const chip = e.target.closest(".shortcut-chip");
  if (!chip) return;
  const action = chip.dataset.action;

  if (action === "settings") {
    openDrawer();
  } else if (action === "preset-latency") {
    const input = document.getElementById("message-input");
    input.value = PROMPT_PRESETS[0].text;
    autoResizeTextarea(input);
    input.focus();
  } else if (action === "preset-code") {
    const input = document.getElementById("message-input");
    input.value = PROMPT_PRESETS[2].text;
    autoResizeTextarea(input);
    input.focus();
  } else if (action === "preset-reason") {
    const input = document.getElementById("message-input");
    input.value = PROMPT_PRESETS[1].text;
    autoResizeTextarea(input);
    input.focus();
  }
});

/* ═══════════════════════════════════════════════════════════════
   Knowledge Base
   ═══════════════════════════════════════════════════════════════ */

const knowledgeState = {
  active: false,
  stats: null,
  sources: [],
  results: [],
};

function showKnowledgePanel() {
  knowledgeState.active = true;
  document.getElementById("knowledge-panel").classList.remove("hidden");
  document.getElementById("chat-panel").style.display = "none";
  document.getElementById("knowledge-tab-btn").classList.add("active");
  refreshKnowledgeStats();
  refreshKnowledgeSources();
}

function hideKnowledgePanel() {
  knowledgeState.active = false;
  document.getElementById("knowledge-panel").classList.add("hidden");
  document.getElementById("chat-panel").style.display = "";
  document.getElementById("knowledge-tab-btn").classList.remove("active");
}

function toggleKnowledgePanel() {
  if (knowledgeState.active) {
    hideKnowledgePanel();
  } else {
    showKnowledgePanel();
  }
}

async function refreshKnowledgeStats() {
  try {
    const data = await api("/api/knowledge/stats");
    knowledgeState.stats = data;
    document.getElementById("kstat-sources").textContent = data.sources || 0;
    document.getElementById("kstat-records").textContent = data.records || 0;
    document.getElementById("kstat-chunks").textContent = data.chunks || 0;
    document.getElementById("kstat-embedded").textContent = data.embedded || 0;

    // Update embed button state
    const pending = (data.chunks || 0) - (data.embedded || 0);
    const embedBtn = document.getElementById("knowledge-embed-btn");
    const embedStatus = document.getElementById("knowledge-embed-status");
    if (pending > 0) {
      embedBtn.disabled = false;
      embedStatus.textContent = `${pending} chunks pending`;
    } else {
      embedBtn.disabled = true;
      embedStatus.textContent = data.chunks > 0 ? "All chunks embedded" : "No chunks to embed";
    }
  } catch (error) {
    // Silent
  }
}

async function refreshKnowledgeSources() {
  try {
    const data = await api("/api/knowledge/sources");
    knowledgeState.sources = data.sources || [];
    renderKnowledgeSources(data.sources);
  } catch (error) {
    document.getElementById("knowledge-sources-list").innerHTML =
      `<div class="knowledge-empty-hint">Failed to load sources.</div>`;
  }
}

const SOURCE_COLORS = {
  pi: "#f5a623",
  claude: "#bc8cff",
  codex: "#34d058",
  factory: "#f85149",
  opencode: "#58a6ff",
  qwen_code: "#ff7b72",
};

function renderKnowledgeSources(sources) {
  const container = document.getElementById("knowledge-sources-list");
  if (!sources.length) {
    container.innerHTML = `<div class="knowledge-empty-hint">No AI session directories found.</div>`;
    return;
  }

  container.innerHTML = sources.map((s) => {
    const color = SOURCE_COLORS[s.source] || "var(--text-tertiary)";
    const statusClass = s.available ? "available" : "unavailable";
    const statusText = s.available ? `${s.files} file${s.files !== 1 ? "s" : ""}` : "Not found";
    return `
      <div class="ksource">
        <div class="ksource-info">
          <span class="source-dot" style="background:${color};width:8px;height:8px;border-radius:50%;flex-shrink:0;"></span>
          <div>
            <div class="ksource-name">${escapeHtml(s.source)}</div>
            <div class="ksource-path">${escapeHtml(s.path)}</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="ksource-status ${statusClass}">${statusText}</span>
          ${s.available && s.files > 0 ? `<button class="btn btn-secondary btn-sm ingest-source-btn" data-source="${escapeHtml(s.source)}" data-path="${escapeHtml(s.path)}">
            <i data-lucide="upload" class="icon-xs"></i>
            Ingest
          </button>` : ""}
        </div>
      </div>
    `;
  }).join("");

  if (window.lucide) lucide.createIcons({ nodes: [container] });

  container.querySelectorAll(".ingest-source-btn").forEach((btn) => {
    btn.onclick = async () => {
      const source = btn.dataset.source;
      const path = btn.dataset.path;
      btn.disabled = true;
      btn.innerHTML = '<i data-lucide="loader" class="icon-xs" style="animation:spin 1s linear infinite;"></i> Ingesting…';
      if (window.lucide) lucide.createIcons({ nodes: [btn] });
      try {
        const result = await api("/api/knowledge/ingest", {
          method: "POST",
          body: JSON.stringify({ source, path, embed: true }),
        });
        const summary = result.summary || {};
        showToast(
          `Ingested ${summary.total_records || 0} records, ${summary.total_chunks || 0} chunks`,
          "success",
          4000
        );
        await refreshKnowledgeStats();
      } catch (error) {
        showToast(`Ingestion failed: ${error.message}`, "error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="upload" class="icon-xs"></i> Ingest';
        if (window.lucide) lucide.createIcons({ nodes: [btn] });
      }
    };
  });
}

async function searchKnowledge() {
  const query = document.getElementById("knowledge-search-input").value.trim();
  if (!query) return;

  const useVectors = document.getElementById("knowledge-use-vectors").checked;
  const resultsContainer = document.getElementById("knowledge-results");
  resultsContainer.innerHTML = `<div class="knowledge-empty-state"><div class="knowledge-empty-title">Searching…</div></div>`;

  try {
    const data = await api("/api/knowledge/query", {
      method: "POST",
      body: JSON.stringify({ query, top_k: 15, use_vectors: useVectors }),
    });

    knowledgeState.results = data.results || [];
    const searchStats = data.stats || {};

    if (!knowledgeState.results.length) {
      resultsContainer.innerHTML = `
        <div class="knowledge-empty-state">
          <i data-lucide="search" style="width:48px;height:48px;color:var(--text-tertiary);"></i>
          <div class="knowledge-empty-title">No results found</div>
          <div class="knowledge-empty-subtitle">Try different keywords or ingest more sources.</div>
        </div>`;
      if (window.lucide) lucide.createIcons({ nodes: [resultsContainer] });
      return;
    }

    resultsContainer.innerHTML = knowledgeState.results.map((r, i) => {
      const color = SOURCE_COLORS[r.source] || "var(--text-tertiary)";
      const scoreBreakdown = [];
      if (r.bm25_score > 0) scoreBreakdown.push(`BM25 ${r.bm25_score.toFixed(2)}`);
      if (r.vector_score > 0) scoreBreakdown.push(`Vec ${r.vector_score.toFixed(2)}`);
      const scoreText = scoreBreakdown.length > 0 ? scoreBreakdown.join(" · ") : `Score ${r.score.toFixed(2)}`;

      // Highlight matched terms
      let text = escapeHtml(r.text);
      if (r.matched_terms && r.matched_terms.length > 0) {
        for (const term of r.matched_terms) {
          const regex = new RegExp(`(${escapeRegex(term)})`, "gi");
          text = text.replace(regex, "<mark>$1</mark>");
        }
      }

      return `
        <div class="kresult" style="animation-delay:${i * 30}ms;">
          <div class="kresult-header">
            <span class="kresult-source">
              <span class="source-dot" style="background:${color}"></span>
              ${escapeHtml(r.source)}
            </span>
            <span class="kresult-score">${scoreText}</span>
          </div>
          <div class="kresult-title">${escapeHtml(r.title)}</div>
          <div class="kresult-text">${text}</div>
          <div class="kresult-footer">
            ${r.category ? `<span class="kresult-tag">${escapeHtml(r.category)}</span>` : ""}
            <span>${searchStats.elapsed_ms ? searchStats.elapsed_ms.toFixed(0) + "ms" : ""}</span>
          </div>
        </div>
      `;
    }).join("");

    if (window.lucide) lucide.createIcons({ nodes: [resultsContainer] });
  } catch (error) {
    resultsContainer.innerHTML = `
      <div class="knowledge-empty-state">
        <div class="knowledge-empty-title" style="color:var(--red);">Search failed</div>
        <div class="knowledge-empty-subtitle">${escapeHtml(error.message)}</div>
      </div>`;
  }
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function embedKnowledge() {
  const embedBtn = document.getElementById("knowledge-embed-btn");
  const progress = document.getElementById("knowledge-embed-progress");
  embedBtn.disabled = true;
  progress.textContent = "Generating embeddings…";

  try {
    const data = await api("/api/knowledge/embed", {
      method: "POST",
      body: JSON.stringify({ batch_size: 32 }),
    });
    const embedded = data.embedded || 0;
    const failed = data.failed || 0;
    if (embedded > 0) {
      showToast(`Embedded ${embedded} chunks${failed > 0 ? ` (${failed} failed)` : ""}`, "success");
    } else {
      showToast("No new chunks to embed", "info", 2000);
    }
    await refreshKnowledgeStats();
  } catch (error) {
    showToast(`Embedding failed: ${error.message}`, "error");
  } finally {
    progress.textContent = "";
    const pending = (knowledgeState.stats?.chunks || 0) - (knowledgeState.stats?.embedded || 0);
    embedBtn.disabled = pending <= 0;
  }
}

async function clearKnowledge() {
  const confirmed = await confirmDialog(
    "Clear knowledge base?",
    "All sources, records, chunks, and embeddings will be permanently deleted."
  );
  if (!confirmed) return;

  try {
    await api("/api/knowledge", { method: "DELETE" });
    showToast("Knowledge base cleared", "success", 2000);
    await refreshKnowledgeStats();
    document.getElementById("knowledge-results").innerHTML = `
      <div class="knowledge-empty-state">
        <i data-lucide="brain" style="width:48px;height:48px;color:var(--text-tertiary);"></i>
        <div class="knowledge-empty-title">No results yet</div>
        <div class="knowledge-empty-subtitle">Search across your indexed AI conversations, or ingest new sources below.</div>
      </div>`;
    if (window.lucide) lucide.createIcons();
  } catch (error) {
    showToast(`Failed to clear: ${error.message}`, "error");
  }
}

function toggleKnowledgeMode() {
  state.knowledgeMode = !state.knowledgeMode;
  const btn = document.getElementById("knowledge-toggle-btn");
  btn.classList.toggle("active", state.knowledgeMode);
  if (state.knowledgeMode) {
    showToast("Knowledge mode on — context will be injected into messages", "info", 2500);
  }
}

function showContextIndicator(count, elapsedMs) {
  // Remove any existing indicator
  const existing = document.querySelector(".context-injected");
  if (existing) existing.remove();

  const indicator = document.createElement("div");
  indicator.className = "context-injected";
  indicator.innerHTML = `
    <i data-lucide="brain" style="width:14px;height:14px;"></i>
    <span>Injected ${count} passage${count !== 1 ? "s" : ""} from knowledge base${elapsedMs ? ` (${elapsedMs.toFixed(0)}ms)` : ""}</span>
  `;
  const messagesContainer = document.getElementById("messages");
  messagesContainer.prepend(indicator);
  if (window.lucide) lucide.createIcons({ nodes: [indicator] });
}

(async function init() {
  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  setupScrollToBottom();
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

  // Show empty state or load last chat
  if (state.chats.length > 0) {
    await loadChat(state.chats[0].chat_id);
  } else {
    updateEmptyState(true);
  }

  // Periodic refresh
  setInterval(refreshModels, 5000);
  setInterval(refreshServerStatus, 5000);
  setInterval(refreshDiagnostics, 5000);
  setInterval(() => {
    if (knowledgeState.active) refreshKnowledgeStats();
  }, 10000);
})();
