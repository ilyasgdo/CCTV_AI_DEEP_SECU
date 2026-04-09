async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function renderEventFeed(items) {
  const feed = document.getElementById("event-feed");
  if (!feed) return;
  feed.innerHTML = "";
  for (const item of items.slice(0, 20)) {
    const div = document.createElement("div");
    div.className = "event-item";
    const ts = item.timestamp || "-";
    const message = item.message || item.event_type || "event";
    div.textContent = `${ts} - ${message}`;
    feed.appendChild(div);
  }
}

async function refreshStatus() {
  const status = await fetchJson("/api/status");
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  setText("m-fps", Number(status.fps || 0).toFixed(1));
  setText("m-persons", String(status.persons || 0));
  setText("m-llm", status.llm_connected ? "connecte" : "deconnecte");
  setText("m-cpu", `${Number(status.cpu_percent || 0).toFixed(1)}%`);
  setText("m-ram", `${Number(status.ram_percent || 0).toFixed(1)}%`);
  if (status.last_ai_action) {
    setText("last-ai-action", status.last_ai_action);
  }
}

async function refreshEventsPage() {
  const table = document.getElementById("events-table");
  if (!table) return;
  const data = await fetchJson("/api/events?per_page=50");
  table.innerHTML = "";
  for (const event of data.items) {
    const row = document.createElement("div");
    row.className = "event-item";
    row.textContent = `${event.timestamp || "-"} - ${event.event_type || "event"} - ${event.message || ""}`;
    table.appendChild(row);
  }
}

async function refreshWhitelist() {
  const container = document.getElementById("whitelist-list");
  if (!container) return;

  const data = await fetchJson("/api/whitelist");
  container.innerHTML = "";

  for (const person of data.items) {
    const card = document.createElement("div");
    card.className = "event-item";
    card.innerHTML = `<strong>${person.name}</strong> (${person.role}) - ${person.access_level}`;

    const del = document.createElement("button");
    del.textContent = "Supprimer";
    del.onclick = async () => {
      await fetchJson(`/api/whitelist/${person.id}`, { method: "DELETE" });
      await refreshWhitelist();
    };
    card.appendChild(document.createTextNode(" "));
    card.appendChild(del);
    container.appendChild(card);
  }
}

async function setupWhitelistForm() {
  const form = document.getElementById("add-person-form");
  if (!form) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const fd = new FormData(form);
    const payload = {
      name: fd.get("name"),
      role: fd.get("role"),
      access_level: fd.get("access_level"),
    };
    await fetchJson("/api/whitelist", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    form.reset();
    await refreshWhitelist();
  });
}

async function refreshSettings() {
  const pre = document.getElementById("settings-json");
  if (!pre) return;
  const settings = await fetchJson("/api/settings");
  pre.textContent = JSON.stringify(settings, null, 2);
}

async function setupLlmTestButton() {
  const button = document.getElementById("btn-llm-test");
  const result = document.getElementById("llm-test-result");
  if (!button || !result) return;

  button.addEventListener("click", async () => {
    result.textContent = "test en cours...";
    try {
      const data = await fetchJson("/api/settings/llm/test", { method: "POST" });
      result.textContent = data.ok ? "✅ LLM connecte" : "❌ LLM inaccessible";
    } catch (err) {
      result.textContent = `Erreur: ${err.message}`;
    }
  });
}

async function refreshSnapshots() {
  const grid = document.getElementById("snapshots-grid");
  if (!grid) return;
  const data = await fetchJson("/api/snapshots");
  grid.innerHTML = "";
  for (const item of data.items) {
    const card = document.createElement("div");
    card.className = "snapshot-card";
    card.innerHTML = `<strong>${item.name}</strong><br><small>${item.modified}</small><br><small>${item.size} bytes</small>`;
    grid.appendChild(card);
  }
}

function setupSocketRealtime() {
  if (typeof io === "undefined") return;
  const socket = io();

  socket.on("new_event", () => {
    fetchJson("/api/events?per_page=20").then((data) => renderEventFeed(data.items));
  });

  socket.on("system_status", () => {
    refreshStatus().catch(() => {});
  });

  socket.on("llm_response", (payload) => {
    const el = document.getElementById("last-ai-action");
    if (el && payload && payload.action_vocale) {
      el.textContent = payload.action_vocale;
    }
  });
}

async function boot() {
  const page = document.body.dataset.page || "";

  if (page.includes("live")) {
    await refreshStatus();
    const events = await fetchJson("/api/events?per_page=20");
    renderEventFeed(events.items);
    setInterval(() => refreshStatus().catch(() => {}), 3000);
  }

  if (page.includes("whitelist")) {
    await setupWhitelistForm();
    await refreshWhitelist();
  }

  if (page.includes("events")) {
    await refreshEventsPage();
  }

  if (page.includes("settings")) {
    await refreshSettings();
    await setupLlmTestButton();
  }

  if (page.includes("snapshots")) {
    await refreshSnapshots();
  }

  setupSocketRealtime();
}

document.addEventListener("DOMContentLoaded", () => {
  boot().catch((err) => {
    console.error(err);
  });
});
