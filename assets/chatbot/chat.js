/* HeartGuard AI — Chat Widget JavaScript */
(function () {
  "use strict";

  const API_CHAT     = "/chatbot/api/chat/";
  const API_HISTORY  = "/chatbot/api/history/";
  const API_NEW      = "/chatbot/api/new-session/";
  const API_SESSIONS = "/chatbot/api/sessions/";
  const LS_KEY       = "hg_session_id";

  let sessionId = null;
  let streaming = false;
  let inHistory = false;

  /* ── DOM References ─────────────────────────────────────── */
  const toggleBtn      = document.getElementById("hg-chat-toggle");
  const panel          = document.getElementById("hg-chat-panel");
  const closeBtn       = document.getElementById("hg-close-btn");
  const newBtn         = document.getElementById("hg-new-btn");
  const msgArea        = document.getElementById("hg-chat-messages");
  const input          = document.getElementById("hg-chat-input");
  const sendBtn        = document.getElementById("hg-chat-send");
  const statusBar      = document.getElementById("hg-chat-status");
  const histBtn        = document.getElementById("hg-history-btn");
  const histOverlay    = document.getElementById("hg-history-overlay");
  const histCloseBtn   = document.getElementById("hg-history-close-btn");
  const histList       = document.getElementById("hg-history-list");
  const inputArea      = document.getElementById("hg-chat-input-area");

  /* ── Helpers ───────────────────────────────────────────── */
  function getCsrfToken() {
    const cookie = document.cookie.split(";").find(c => c.trim().startsWith("csrftoken="));
    return cookie ? cookie.split("=")[1].trim() : "";
  }

  function setStatus(text) {
    if (statusBar) statusBar.textContent = text;
  }

  function scrollBottom() {
    if (msgArea) {
      setTimeout(() => { msgArea.scrollTop = msgArea.scrollHeight; }, 40);
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatTime(dateObj) {
    return new Date(dateObj).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatDate(isoStr) {
    const d = new Date(isoStr);
    const now = new Date();
    const diffDays = Math.floor((now - d) / 86400000);
    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function saveSession(id) {
    sessionId = id;
    if (id) {
      localStorage.setItem(LS_KEY, String(id));
    } else {
      localStorage.removeItem(LS_KEY);
    }
  }

  /* ── History Overlay Controls ──────────────────────────── */
  function openHistory() {
    inHistory = true;
    if (histOverlay) histOverlay.style.display = "flex";
    if (histBtn) histBtn.setAttribute("title", "Close History");
    loadSessionList();
  }

  function closeHistory() {
    inHistory = false;
    if (histOverlay) histOverlay.style.display = "none";
    if (histBtn) histBtn.setAttribute("title", "Chat History");
  }

  function toggleHistory() {
    if (inHistory) {
      closeHistory();
    } else {
      openHistory();
    }
  }

  async function loadSessionList() {
    if (!histList) return;
    histList.innerHTML = '<div class="hg-hist-loading">Loading conversations...</div>';
    try {
      const resp = await fetch(API_SESSIONS);
      if (!resp.ok) throw new Error("Failed to load sessions");
      const data = await resp.json();
      renderSessionList(data.sessions || []);
    } catch (e) {
      histList.innerHTML = '<div class="hg-hist-loading">Could not load previous chats.</div>';
    }
  }

  function renderSessionList(sessions) {
    if (!histList) return;
    if (!sessions || sessions.length === 0) {
      histList.innerHTML = '<div class="hg-hist-empty">No previous chats yet.<br>Start a conversation to see it saved here!</div>';
      return;
    }

    histList.innerHTML = "";
    sessions.forEach(s => {
      const item = document.createElement("div");
      const isActive = (s.session_id === sessionId);
      item.className = "hg-session-item" + (isActive ? " active" : "");
      item.innerHTML = `
        <div class="hg-session-preview">${escapeHtml(s.preview)}</div>
        <div class="hg-session-meta">
          <span>${formatDate(s.updated_at)}</span>
          <span>${s.message_count} message${s.message_count !== 1 ? "s" : ""}</span>
        </div>
      `;
      item.addEventListener("click", () => switchSession(s.session_id));
      histList.appendChild(item);
    });
  }

  async function switchSession(id) {
    if (id === sessionId) {
      closeHistory();
      return;
    }
    saveSession(id);
    closeHistory();
    if (msgArea) msgArea.innerHTML = "";
    setStatus("Loading conversation...");
    await loadHistory(id);
    setStatus("Ready");
  }

  /* ── Message Rendering ─────────────────────────────────── */
  function appendMessage(role, text, sources, isEmergency, timestamp) {
    if (!msgArea) return null;

    const wrap = document.createElement("div");
    wrap.className = "hg-msg " + (isEmergency ? "emergency" : role);

    const bubble = document.createElement("div");
    bubble.className = "hg-bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);

    if (sources && sources.length > 0) {
      const srcDiv = document.createElement("div");
      srcDiv.className = "hg-sources";
      srcDiv.innerHTML = "&#128218; Sources: " + sources.map(s => `<span>${escapeHtml(s)}</span>`).join("");
      wrap.appendChild(srcDiv);
    }

    const tsDiv = document.createElement("div");
    tsDiv.className = "hg-ts";
    tsDiv.textContent = formatTime(timestamp || new Date());
    wrap.appendChild(tsDiv);

    msgArea.appendChild(wrap);
    scrollBottom();
    return bubble;
  }

  function showTypingIndicator() {
    if (!msgArea) return;
    const wrap = document.createElement("div");
    wrap.className = "hg-msg assistant hg-typing";
    wrap.id = "hg-typing-indicator";
    wrap.innerHTML = '<div class="hg-bubble"><div class="hg-dots"><span></span><span></span><span></span></div></div>';
    msgArea.appendChild(wrap);
    scrollBottom();
  }

  function removeTypingIndicator() {
    const typing = document.getElementById("hg-typing-indicator");
    if (typing) typing.remove();
  }

  /* ── Welcome Screen ────────────────────────────────────── */
  function buildWelcome() {
    const card = document.createElement("div");
    card.className = "hg-welcome";
    card.innerHTML = `
      <div class="hg-welcome-icon">&#10084;&#65039;</div>
      <strong>HeartGuard AI</strong>
      Your personal cardiovascular health assistant.
      <div class="hg-suggestions">
        <div class="hg-suggestion">What are the symptoms of a heart attack?</div>
        <div class="hg-suggestion">How can I lower my blood pressure?</div>
        <div class="hg-suggestion">Explain my heart disease prediction result.</div>
        <div class="hg-suggestion">What foods are good for heart health?</div>
      </div>
    `;

    card.querySelectorAll(".hg-suggestion").forEach(btn => {
      btn.addEventListener("click", () => {
        if (input) {
          input.value = btn.textContent.trim();
          input.focus();
        }
        card.remove();
      });
    });

    return card;
  }

  /* ── Load History Messages ─────────────────────────────── */
  async function loadHistory(sid) {
    if (!sid || !msgArea) return;
    try {
      const resp = await fetch(`${API_HISTORY}?session_id=${sid}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const messages = data.messages || [];

      msgArea.innerHTML = "";
      if (messages.length === 0) {
        msgArea.appendChild(buildWelcome());
      } else {
        messages.forEach(m => {
          appendMessage(m.role, m.content, m.sources || [], false, m.timestamp);
        });
      }
    } catch (e) {
      console.error("[HeartGuard] History load error:", e);
    }
  }

  /* ── Start New Session ─────────────────────────────────── */
  async function startNewSession() {
    if (streaming) return;
    closeHistory();
    try {
      const resp = await fetch(API_NEW, {
        method: "POST",
        headers: { "X-CSRFToken": getCsrfToken() },
      });
      const data = await resp.json();
      saveSession(data.session_id);

      if (msgArea) {
        msgArea.innerHTML = "";
        msgArea.appendChild(buildWelcome());
      }
      setStatus("New conversation started");
    } catch (e) {
      setStatus("Could not start new chat");
    }
  }

  /* ── Send Message & SSE Stream ─────────────────────────── */
  async function sendMessage() {
    if (!input) return;
    const text = input.value.trim();
    if (!text || streaming) return;

    // Remove welcome card if present
    const welcome = msgArea ? msgArea.querySelector(".hg-welcome") : null;
    if (welcome) welcome.remove();

    closeHistory();

    input.value = "";
    input.style.height = "auto";
    if (sendBtn) sendBtn.disabled = true;
    streaming = true;

    appendMessage("user", text, null, false, new Date());
    showTypingIndicator();
    setStatus("HeartGuard AI is thinking…");

    try {
      const resp = await fetch(API_CHAT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCsrfToken(),
        },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        removeTypingIndicator();
        appendMessage("assistant", "Error: " + (err.error || "Something went wrong. Please try again."), null, false);
        setStatus("Error");
        return;
      }

      const contentType = resp.headers.get("content-type") || "";

      // ── JSON Response (Emergency Alert or non-streaming error) ──
      if (contentType.includes("application/json")) {
        const data = await resp.json();
        if (data.session_id) saveSession(data.session_id);
        removeTypingIndicator();
        appendMessage("assistant", data.response || data.error, null, !!data.is_emergency);
        setStatus(data.is_emergency ? "⚠️ Emergency alert shown" : "Ready");
        return;
      }

      // ── SSE Streaming Response ──
      removeTypingIndicator();

      const wrap = document.createElement("div");
      wrap.className = "hg-msg assistant";
      const bubble = document.createElement("div");
      bubble.className = "hg-bubble";
      wrap.appendChild(bubble);
      msgArea.appendChild(wrap);
      scrollBottom();

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamSources = [];

      streamLoop:
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();

          if (payload === "[DONE]") {
            if (streamSources.length > 0) {
              const srcDiv = document.createElement("div");
              srcDiv.className = "hg-sources";
              srcDiv.innerHTML = "&#128218; Sources: " + streamSources.map(s => `<span>${escapeHtml(s)}</span>`).join("");
              wrap.appendChild(srcDiv);
            }
            const tsDiv = document.createElement("div");
            tsDiv.className = "hg-ts";
            tsDiv.textContent = formatTime(new Date());
            wrap.appendChild(tsDiv);
            setStatus("Ready");
            scrollBottom();
            break streamLoop;
          }

          try {
            const parsed = JSON.parse(payload);
            if (parsed.session_id) saveSession(parsed.session_id);
            if (parsed.sources)    streamSources = parsed.sources;
            if (parsed.token)      { bubble.textContent += parsed.token; scrollBottom(); }
            if (parsed.error)      { bubble.textContent = "Error: " + parsed.error; }
          } catch (_) { /* ignore partial chunks */ }
        }
      }

    } catch (err) {
      removeTypingIndicator();
      appendMessage("assistant", "Connection error. Make sure the server and Ollama are running.", null, false);
      setStatus("Connection error");
    } finally {
      streaming = false;
      if (sendBtn) sendBtn.disabled = false;
      if (input) input.focus();
    }
  }

  /* ── Event Listeners Binding ────────────────────────────── */
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      if (panel) {
        panel.classList.toggle("open");
        if (panel.classList.contains("open")) {
          if (input) input.focus();
        }
      }
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      if (panel) panel.classList.remove("open");
      closeHistory();
    });
  }

  if (newBtn) {
    newBtn.addEventListener("click", startNewSession);
  }

  if (histBtn) {
    histBtn.addEventListener("click", toggleHistory);
  }

  if (histCloseBtn) {
    histCloseBtn.addEventListener("click", closeHistory);
  }

  if (sendBtn) {
    sendBtn.addEventListener("click", sendMessage);
  }

  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 90) + "px";
    });
  }

  /* ── Initialization ────────────────────────────────────── */
  window.addEventListener("load", async () => {
    const saved = localStorage.getItem(LS_KEY);
    if (saved) {
      saveSession(parseInt(saved, 10));
      await loadHistory(sessionId);
    } else {
      if (msgArea) msgArea.appendChild(buildWelcome());
    }
    setStatus("Ready");
  });

})();
