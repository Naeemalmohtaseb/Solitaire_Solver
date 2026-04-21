const state = {
  presets: [],
  session: null,
  sessionJson: null,
  replay: null,
  replayIndex: 0,
  generatedMetadata: null,
  diagnostics: null,
  lastBenchmark: null,
  activeTask: null,
};

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", async () => {
  bindNavigation();
  bindActions();
  await loadStatus();
  await loadPresets();
  setStatus("Ready.");
});

function bindNavigation() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      activateTab(button.dataset.tab);
    });
  });
}

function activateTab(tab) {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
  $(`tab-${tab}`)?.classList.add("active");
}

function bindActions() {
  $("quick-new-game").addEventListener("click", () => {
    activateTab("solve");
    $("new-seed").focus();
  });
  $("quick-open").addEventListener("click", () => $("session-file").click());
  $("inspect-load").addEventListener("click", () => $("session-file").click());
  $("quick-save").addEventListener("click", saveSessionDownload);
  $("inspect-save").addEventListener("click", saveSessionDownload);
  $("quick-recommend").addEventListener("click", recommendMove);
  $("solve-recommend").addEventListener("click", recommendMove);
  $("solve-one-step").addEventListener("click", solveOneStep);
  $("solve-run-end").addEventListener("click", solveRunToEnd);
  $("quick-autoplay").addEventListener("click", () => startBenchmark("autoplay"));
  $("quick-benchmark").addEventListener("click", () => startBenchmark());
  $("generate-game").addEventListener("click", () => generateGame(false));
  $("generate-autoplay").addEventListener("click", () => generateGame(true));
  $("analyze-session").addEventListener("click", analyzeCurrentSession);
  $("export-diagnostics").addEventListener("click", () => downloadJson("diagnostics.json", state.diagnostics));
  $("start-benchmark").addEventListener("click", () => startBenchmark());
  $("session-file").addEventListener("change", loadSessionFromPicker);
  $("first-step").addEventListener("click", () => showReplayStep(0));
  $("prev-step").addEventListener("click", () => showReplayStep(Math.max(0, state.replayIndex - 1)));
  $("next-step").addEventListener("click", () => showReplayStep(Math.min(replayLastIndex(), state.replayIndex + 1)));
  $("last-step").addEventListener("click", () => showReplayStep(replayLastIndex()));
  $("export-trace").addEventListener("click", () => downloadJson("trace.json", state.replay));
  $("export-bench-json").addEventListener("click", () => downloadJson("benchmark.json", state.lastBenchmark));
  $("export-bench-csv").addEventListener("click", () => downloadText("benchmark.csv", benchmarkCsv(state.lastBenchmark)));
  $("export-game-csv").addEventListener("click", () => downloadText("benchmark-games.csv", gameCsv(state.lastBenchmark)));
}

async function loadStatus() {
  const status = await apiGet("/api/status");
  $("backend-status").textContent = status.backend_status;
}

async function loadPresets() {
  state.presets = await apiGet("/api/presets");
  for (const id of ["setting-preset", "new-preset", "bench-preset", "bench-baseline", "bench-candidate"]) {
    const select = $(id);
    select.innerHTML = "";
    state.presets.forEach((preset) => {
      const option = document.createElement("option");
      option.value = preset;
      option.textContent = preset;
      select.appendChild(option);
    });
  }
  $("setting-preset").value = state.presets.includes("balanced_benchmark") ? "balanced_benchmark" : state.presets[0];
  $("new-preset").value = state.presets.includes("balanced_benchmark") ? "balanced_benchmark" : state.presets[0];
  $("bench-preset").value = state.presets.includes("fast_benchmark") ? "fast_benchmark" : state.presets[0];
  $("bench-baseline").value = state.presets.includes("pimc_baseline") ? "pimc_baseline" : state.presets[0];
  $("bench-candidate").value = state.presets.includes("belief_uct_late_exact") ? "belief_uct_late_exact" : state.presets[0];
  $("bench-presets").value = "fast_benchmark,balanced_benchmark";
}

async function loadSessionFromPicker(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const contents = await file.text();
  const response = await apiPost("/api/session/parse", { contents, label: file.name });
  loadSessionPayload(response, file.name);
  setStatus(`Loaded ${file.name}`);
}

function loadSessionPayload(response, label) {
  state.session = response.session;
  state.sessionJson = JSON.stringify(response.session, null, 2);
  state.replay = response.replay;
  state.replayIndex = 0;
  state.generatedMetadata = response.generated || null;
  state.diagnostics = null;
  clearDiagnostics();
  $("loaded-label").textContent = label || response.replay?.label || "Loaded session";
  showReplayStep(0);
}

async function generateGame(autoplay) {
  const seed = numberValue("new-seed", 0);
  if (seed < 0) {
    setStatus("Seed must be zero or greater.");
    return;
  }
  const settings = collectGameSetupSettings();
  applyGameSetupToGlobalSettings(settings);
  setStatus(autoplay ? "Generating and autoplaying game..." : "Generating game...");
  try {
    const response = await apiPost(autoplay ? "/api/game/generate_autoplay" : "/api/game/generate", {
      seed,
      settings,
    });
    const label = autoplay ? `Autoplay seed ${seed}` : `Generated seed ${seed}`;
    loadSessionPayload(response, label);
    if (autoplay) {
      activateTab("replay");
    }
    const kind = autoplay ? "Generated autoplay trace" : "Generated game";
    setStatus(`${kind} for seed ${seed}. Open Diagnostics to audit the result.`);
  } catch (err) {
    setStatus(err.message);
  }
}

async function analyzeCurrentSession() {
  if (!state.session) {
    setStatus("No analyzable session loaded.");
    return;
  }
  setStatus("Analyzing session diagnostics...");
  try {
    const report = await apiPost("/api/diagnostics/analyze_session", {
      session: state.session,
      settings: collectSettings(),
    });
    state.diagnostics = report;
    renderDiagnostics(report);
    setStatus(`Diagnostics complete: ${report.overall_status}.`);
  } catch (err) {
    setStatus(err.message);
  }
}

function saveSessionDownload() {
  if (!state.session) {
    setStatus("No session loaded.");
    return;
  }
  downloadText(`${state.replay?.label || "session"}.json`, JSON.stringify(state.session, null, 2));
}

function showReplayStep(index) {
  if (!state.replay?.steps?.length) return;
  state.replayIndex = index;
  const step = state.replay.steps[index];
  renderBoard($("replay-board"), step.board);
  renderBoard($("solve-board"), step.board);
  $("move-annotation").textContent = step.move_text
    ? `Move ${index}: ${step.move_text}${step.revealed_card ? `; revealed ${prettyCard(step.revealed_card)}` : ""}`
    : "Initial position.";
  updateInspector(step);
}

function replayLastIndex() {
  return Math.max(0, (state.replay?.steps?.length || 1) - 1);
}

async function recommendMove() {
  if (!state.session) {
    setStatus("Load a session first.");
    return;
  }
  setStatus("Requesting recommendation...");
  const result = await apiPost("/api/recommend", {
    session: state.session,
    settings: collectSettings(),
  });
  renderRecommendation(result);
  setStatus("Recommendation ready.");
}

async function solveOneStep() {
  if (!state.session) {
    setStatus("Load a session first.");
    return;
  }
  setStatus("Solving one step...");
  try {
    const result = await apiPost("/api/solve/one_step", {
      session: state.session,
      settings: collectSettings(),
    });
    state.session = result.session;
    state.sessionJson = JSON.stringify(result.session, null, 2);
    state.replay = result.replay;
    renderRecommendation(result.recommendation);
    showReplayStep(replayLastIndex());
    setStatus("Applied one solver step.");
  } catch (err) {
    setStatus(err.message);
  }
}

async function solveRunToEnd() {
  if (!state.session) {
    setStatus("Load a session first.");
    return;
  }
  setStatus("Running autoplay from current true state...");
  try {
    const result = await apiPost("/api/solve/run_to_end", {
      session: state.session,
      settings: collectSettings(),
    });
    $("recommendation-output").innerHTML = `<pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    setStatus(`Run finished: ${result.termination || "done"}`);
  } catch (err) {
    setStatus(err.message);
  }
}

async function startBenchmark(forcedType) {
  const benchmarkType = forcedType || $("bench-type").value;
  const request = {
    benchmark_type: benchmarkType,
    preset: $("bench-preset").value,
    baseline: $("bench-baseline").value,
    candidate: $("bench-candidate").value,
    presets: $("bench-presets").value.split(",").map((x) => x.trim()).filter(Boolean),
    games: numberValue("bench-games", 5),
    repetitions: numberValue("bench-repetitions", 3),
    seed: numberValue("bench-seed", 0),
    rank_by: $("bench-rank").value,
    max_steps: numberValue("bench-max-steps", 20),
    settings: collectSettings(),
  };
  $("progress-line").textContent = "Starting benchmark...";
  $("benchmark-output").textContent = "";
  setStatus("Benchmark running...");
  const start = await apiPost("/api/benchmark/start", request);
  state.activeTask = start.task_id;
  pollTask(start.task_id);
}

async function pollTask(taskId) {
  const task = await apiGet(`/api/task/${taskId}`);
  if (task.progress) {
    $("progress-line").textContent = progressText(task.progress);
  }
  if (task.error) {
    $("benchmark-output").textContent = task.error;
    setStatus("Benchmark failed.");
    return;
  }
  if (task.done) {
    state.lastBenchmark = task.result;
    renderBenchmarkResult(task.result);
    setStatus("Benchmark complete.");
    return;
  }
  setTimeout(() => pollTask(taskId), 600);
}

function collectSettings() {
  return {
    preset: $("setting-preset").value,
    backend: $("setting-backend").value,
    leaf_eval_mode: $("setting-leaf").value,
    vnet_model_path: $("setting-vnet").value || null,
    max_steps: null,
    late_exact_enabled: $("setting-late-exact").checked,
    root_parallel: $("setting-root-parallel").checked,
    root_workers: numberValue("setting-workers", 2),
    worker_sim_budget: numberValue("setting-worker-budget", 8),
    worker_seed_stride: numberValue("setting-seed-stride", 1000003),
  };
}

function collectGameSetupSettings() {
  const vnetPath = $("new-vnet").value || $("setting-vnet").value || null;
  return {
    preset: $("new-preset").value,
    backend: $("new-backend").value,
    leaf_eval_mode: vnetPath ? "vnet" : $("setting-leaf").value,
    vnet_model_path: vnetPath,
    max_steps: numberOrNull("new-max-steps"),
    late_exact_enabled: $("setting-late-exact").checked,
    root_parallel: $("new-root-parallel").checked,
    root_workers: numberValue("new-workers", 2),
    worker_sim_budget: numberValue("new-worker-budget", 8),
    worker_seed_stride: numberValue("setting-seed-stride", 1000003),
  };
}

function applyGameSetupToGlobalSettings(settings) {
  $("setting-preset").value = settings.preset;
  $("setting-backend").value = settings.backend;
  $("setting-root-parallel").checked = Boolean(settings.root_parallel);
  $("setting-workers").value = settings.root_workers;
  $("setting-worker-budget").value = settings.worker_sim_budget;
  if (settings.vnet_model_path) {
    $("setting-leaf").value = "vnet";
    $("setting-vnet").value = settings.vnet_model_path;
  }
}

function renderBoard(container, board) {
  if (!board) {
    container.className = "board empty-board";
    container.textContent = "No board loaded.";
    return;
  }
  container.className = "board";
  const foundations = board.foundations.map((card) => card ? cardHtml(card) : `<div class="slot">F</div>`).join("");
  const stock = `
    <div>
      <div class="stock-row">
        ${board.stock.stock_len > 0 ? `<div class="card-back"></div>` : `<div class="slot">Stock</div>`}
        ${board.stock.accessible_card ? cardHtml(board.stock.accessible_card) : `<div class="slot">Waste</div>`}
      </div>
      <div class="stock-meta">stock ${board.stock.stock_len} · waste ${board.stock.waste_len} · pass ${board.stock.pass_index}</div>
    </div>`;
  const columns = board.tableau.map((column) => {
    const hidden = Array.from({ length: column.hidden_count }, () => `<div class="card-back"></div>`).join("");
    const face = column.face_up.map(cardHtml).join("");
    return `<div class="tableau-column">${hidden}${face || (column.hidden_count === 0 ? `<div class="slot">Empty</div>` : "")}</div>`;
  }).join("");
  container.innerHTML = `
    <div class="board-top">
      ${stock}
      <div class="foundation-row">${foundations}</div>
    </div>
    <div class="tableau-row">${columns}</div>`;
}

function cardHtml(card) {
  const parsed = parseCard(card);
  const color = parsed.red ? "red" : "black";
  return `<div class="card ${color}"><span class="rank">${parsed.rank}</span><span class="suit">${parsed.suit}</span><span class="corner">${parsed.rank}${parsed.suit}</span></div>`;
}

function parseCard(card) {
  const rank = card.slice(0, 1);
  const suitCode = card.slice(1, 2).toLowerCase();
  const suits = { c: "♣", d: "♦", h: "♥", s: "♠" };
  return {
    rank,
    suit: suits[suitCode] || suitCode,
    red: suitCode === "d" || suitCode === "h",
  };
}

function prettyCard(card) {
  const parsed = parseCard(card);
  return `${parsed.rank}${parsed.suit}`;
}

function updateInspector(step) {
  $("inspect-preset").textContent = step.preset || state.replay?.metadata?.preset_name || "n/a";
  $("inspect-backend").textContent = step.backend || state.replay?.metadata?.backend || "n/a";
  $("inspect-step").textContent = `${state.replayIndex} / ${replayLastIndex()}`;
  $("inspect-hidden").textContent = step.board?.hidden_count ?? "n/a";
  $("inspect-move").textContent = step.move_text || "Initial";
  $("inspect-reveal").textContent = step.revealed_card ? prettyCard(step.revealed_card) : "none";
  $("inspect-stats").textContent = step.planner
    ? `value ${step.planner.best_value.toFixed(3)}, nodes ${step.planner.deterministic_nodes}, visits ${step.planner.root_visits}`
    : generatedStatsText();
}

function generatedStatsText() {
  if (!state.generatedMetadata) {
    return "n/a";
  }
  const meta = state.generatedMetadata;
  const maxSteps = meta.max_steps == null ? "n/a" : meta.max_steps;
  return `seed ${meta.seed}, ${meta.kind}, max steps ${maxSteps}`;
}

function renderRecommendation(result) {
  const rows = result.alternatives.map((alt) => `
    <tr><td>${escapeHtml(alt.move_text)}</td><td>${alt.visits}</td><td>${alt.mean_value.toFixed(3)}</td><td>${alt.stderr.toFixed(3)}</td></tr>
  `).join("");
  $("recommendation-output").innerHTML = `
    <h3>Recommendation</h3>
    <p><strong>${escapeHtml(result.best_move || "No legal move")}</strong> · value ${result.best_value.toFixed(3)} · ${result.elapsed_ms}ms · nodes ${result.deterministic_nodes}</p>
    <table class="summary-table"><thead><tr><th>Move</th><th>Visits</th><th>Mean</th><th>StdErr</th></tr></thead><tbody>${rows}</tbody></table>`;
  $("inspect-stats").textContent = `value ${result.best_value.toFixed(3)}, nodes ${result.deterministic_nodes}, visits ${result.root_visits}`;
}

function clearDiagnostics() {
  $("diagnostic-summary").innerHTML = `<div class="empty-panel">Load or generate a session, then run diagnostics.</div>`;
  $("diagnostic-step-list").textContent = "No diagnostic report yet.";
  $("terminal-audit").textContent = "No diagnostic report yet.";
  $("action-audit").textContent = "Select a diagnostic step.";
}

function renderDiagnostics(report) {
  $("diagnostic-summary").innerHTML = [
    summaryCard("Overall", report.overall_status),
    summaryCard("Consistency", report.consistency_held ? "held" : "issue detected"),
    summaryCard("Suspicious flags", report.total_suspicious_flags),
    summaryCard("Final status", report.final_termination_reason),
  ].join("");

  $("terminal-audit").innerHTML = `
    <p><strong>${escapeHtml(report.terminal_audit.terminal_status_valid ? "Terminal shape valid" : "Not terminal")}</strong></p>
    <p class="audit-note">${escapeHtml(report.terminal_audit.note)}</p>
    <table class="summary-table"><tbody>
      <tr><td>Structural win</td><td>${report.terminal_audit.structural_win}</td></tr>
      <tr><td>Legal moves remaining</td><td>${report.terminal_audit.legal_moves_remaining}</td></tr>
      <tr><td>Likely bug</td><td>${report.terminal_audit.likely_bug}</td></tr>
      <tr><td>Likely weak play</td><td>${report.terminal_audit.likely_weak_decision_making}</td></tr>
    </tbody></table>`;

  if (!report.steps.length) {
    $("diagnostic-step-list").textContent = "No recorded moves; only terminal/current-state audit is available.";
    $("action-audit").textContent = "No steps to audit.";
    return;
  }

  $("diagnostic-step-list").innerHTML = report.steps.map((step) => diagnosticRowHtml(step)).join("");
  document.querySelectorAll(".diagnostic-row").forEach((row) => {
    row.addEventListener("click", () => {
      const index = Number(row.dataset.step);
      const step = report.steps.find((candidate) => candidate.step_index === index);
      if (step?.action_audit) {
        renderActionAudit(step.action_audit);
      } else {
        $("action-audit").textContent = "No local action audit was persisted for this step.";
      }
      showReplayStep(Math.min(index + 1, replayLastIndex()));
      activateTab("replay");
      setStatus(`Synced Replay to diagnostic step ${index}.`);
    });
  });

  const firstInteresting = report.steps.find((step) => step.flags.length > 0) || report.steps[0];
  if (firstInteresting?.action_audit) {
    renderActionAudit(firstInteresting.action_audit);
  } else {
    $("action-audit").textContent = "No local action audit was persisted for the selected step.";
  }
}

function summaryCard(label, value) {
  return `<div class="summary-card"><span>${escapeHtml(String(label))}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function diagnosticRowHtml(step) {
  const severityClass = String(step.severity || "Ok").toLowerCase();
  const badges = step.flags.length
    ? step.flags.map((flag) => `<span class="badge ${badgeClass(flag.code)}" title="${escapeHtml(flag.message)}">${escapeHtml(flag.code)}</span>`).join("")
    : `<span class="badge">OK</span>`;
  const reveal = step.revealed_card ? ` · reveal ${prettyCard(step.revealed_card)}` : "";
  return `
    <button class="diagnostic-row ${severityClass}" data-step="${step.step_index}">
      <div><strong>#${step.step_index}</strong><div class="diagnostic-move">${step.legal_move_count} legal${reveal}</div></div>
      <div>
        <strong>${escapeHtml(step.consistency_ok ? "Consistency ok" : "Inspect")}</strong>
        <div class="diagnostic-move">${escapeHtml(step.chosen_move)}</div>
      </div>
      <div class="badge-row">${badges}</div>
    </button>`;
}

function renderActionAudit(audit) {
  const rows = audit.top_alternatives?.length
    ? tableFromRows(["Move", "Visits", "Mean", "StdErr"], audit.top_alternatives.map((alt) => [alt.move_text, alt.visits, fmt(alt.mean_value), fmt(alt.stderr)]))
    : "";
  $("action-audit").innerHTML = `
    <p><strong>${escapeHtml(audit.chosen_move)}</strong></p>
    <p class="audit-note">Planner value: ${audit.chosen_planner_value == null ? "n/a" : fmt(audit.chosen_planner_value)}</p>
    <p class="audit-note">${escapeHtml(audit.note)}</p>
    ${rows}`;
}

function badgeClass(code) {
  if (["BeliefFullMismatch", "IllegalTerminal", "RevealUpdateMismatch", "ReplayMismatch"].includes(code)) {
    return "error";
  }
  return "warning";
}

function renderBenchmarkResult(result) {
  if (!result) {
    $("benchmark-output").textContent = "No result.";
    return;
  }
  if (result.entries) {
    $("benchmark-output").innerHTML = tableFromRows(["Preset", "Backend", "Wins", "Win rate", "Time/game", "Efficiency"],
      result.entries.map((e) => [e.preset_name, e.backend, e.wins, fmt(e.win_rate), `${fmt(e.average_time_per_game_ms)}ms`, fmt(e.win_rate_per_second)]));
    return;
  }
  if (result.baseline && result.candidate) {
    $("benchmark-output").innerHTML = tableFromRows(["Config", "Wins", "Win rate", "Time/game"],
      [["Baseline", result.baseline.wins, fmt(result.baseline.win_rate), `${fmt(result.baseline.average_total_planner_time_per_game_ms)}ms`],
       ["Candidate", result.candidate.wins, fmt(result.candidate.win_rate), `${fmt(result.candidate.average_total_planner_time_per_game_ms)}ms`]])
      + `<p>Paired delta: ${fmt(result.paired_win_rate_delta)} · stderr ${fmt(result.paired_standard_error || 0)}</p>`;
    return;
  }
  $("benchmark-output").innerHTML = tableFromRows(["Preset", "Games", "Wins", "Win rate", "Avg moves", "Avg time/game"],
    [[result.config?.name || "n/a", result.games, result.wins, fmt(result.win_rate), fmt(result.average_moves_per_game), `${fmt(result.average_total_planner_time_per_game_ms)}ms`]]);
}

function tableFromRows(headers, rows) {
  return `<table class="summary-table"><thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(String(cell))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function progressText(progress) {
  const preset = progress.preset_name || "n/a";
  const rep = progress.repetition_index ? ` rep ${progress.repetition_index}/${progress.repetition_total}` : "";
  const presetIndex = progress.preset_index ? ` preset ${progress.preset_index}/${progress.preset_total}` : "";
  return `${progress.command}${presetIndex}${rep}: ${preset} game ${progress.game_index}/${progress.game_total}, wins/losses ${progress.wins}/${progress.losses}, elapsed ${formatMs(progress.elapsed_ms)}, ETA ${progress.eta_ms == null ? "n/a" : formatMs(progress.eta_ms)}`;
}

async function apiGet(path) {
  const response = await fetch(path);
  return unwrapApi(response);
}

async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return unwrapApi(response);
}

async function unwrapApi(response) {
  const payload = await response.json();
  if (payload.ok === false) {
    throw new Error(payload.error || "Workbench command failed");
  }
  return payload.data ?? payload;
}

function downloadJson(name, value) {
  if (!value) {
    setStatus("Nothing to export.");
    return;
  }
  downloadText(name, JSON.stringify(value, null, 2));
}

function downloadText(name, text) {
  const blob = new Blob([text || ""], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function benchmarkCsv(result) {
  if (!result) return "";
  if (result.entries) {
    return ["preset,backend,wins,win_rate,time_per_game_ms,efficiency",
      ...result.entries.map((e) => [e.preset_name, e.backend, e.wins, e.win_rate, e.average_time_per_game_ms, e.win_rate_per_second].join(","))].join("\n");
  }
  if (result.baseline && result.candidate) {
    return ["side,preset,wins,win_rate,time_per_game_ms",
      ["baseline", result.baseline.config.name, result.baseline.wins, result.baseline.win_rate, result.baseline.average_total_planner_time_per_game_ms].join(","),
      ["candidate", result.candidate.config.name, result.candidate.wins, result.candidate.win_rate, result.candidate.average_total_planner_time_per_game_ms].join(",")].join("\n");
  }
  return ["preset,games,wins,win_rate,time_per_game_ms",
    [result.config?.name || "", result.games, result.wins, result.win_rate, result.average_total_planner_time_per_game_ms].join(",")].join("\n");
}

function gameCsv(result) {
  if (!result?.records) return "";
  return ["seed,won,termination,moves,total_planner_time_ms,root_visits",
    ...result.records.map((r) => [r.seed?.[0] ?? r.seed, r.won, r.termination, r.moves_played, r.total_planner_time_ms, r.root_visits].join(","))].join("\n");
}

function numberValue(id, fallback) {
  const value = Number($(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function numberOrNull(id) {
  const raw = $(id).value;
  if (raw == null || raw.trim() === "") {
    return null;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function fmt(value) {
  return Number(value || 0).toFixed(3);
}

function formatMs(ms) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function setStatus(message) {
  $("statusbar").textContent = message;
}
