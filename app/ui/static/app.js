const form = document.getElementById("task-form");
const submitButton = document.getElementById("submit-button");
const statusPill = document.getElementById("task-status");
const summary = document.getElementById("task-summary");
const parsedIntent = document.getElementById("parsed-intent");
const retrievedKnowledge = document.getElementById("retrieved-knowledge");
const agentTrace = document.getElementById("agent-trace");
const gitStatus = document.getElementById("git-status");
const candidateContainer = document.getElementById("candidate-container");
const repairHistory = document.getElementById("repair-history");
const graphSnapshot = document.getElementById("graph-snapshot");
const graphView = document.getElementById("graph-view");
const graphMeta = document.getElementById("graph-meta");

let activeTaskId = null;
let pollTimer = null;

const STATUS_TEXT = {
  idle: "空闲",
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败"
};

function normalizeConstraintLabel(value) {
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  if (normalized.includes("隐式") || normalized.includes("implicit")) {
    return "implicit feedback";
  }
  if ((normalized.includes("本地") && normalized.includes("cpu")) || normalized === "本地cpu" || normalized === "local cpu") {
    return "local cpu";
  }
  return String(value).trim();
}

function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function setStatus(value) {
  const normalized = String(value).toLowerCase();
  const styleKey = normalized === "queued" ? "running" : normalized;
  statusPill.textContent = STATUS_TEXT[normalized] ?? value;
  statusPill.className = `status-pill ${styleKey}`;
}

function buildCandidateCard(candidate, bestCandidateId) {
  const card = document.createElement("article");
  card.className = `candidate-card ${candidate.candidate_id === bestCandidateId ? "best" : ""}`;

  const report = candidate.validation ?? {};
  const metrics = report.metrics ?? {};

  card.innerHTML = `
    <div class="candidate-header">
      <div>
        <h3>${candidate.display_name}</h3>
        <div class="status-pill ${report.success ? "completed" : "failed"}">${report.success ? "验证通过" : "验证失败"}</div>
      </div>
      <strong>算法键：${candidate.algorithm_key}</strong>
    </div>
    <p class="candidate-meta">${candidate.plan_summary}</p>
    <pre class="code-block">${prettyJson({
      score: report.score,
      metrics,
      errors: report.errors,
      warnings: report.warnings,
      repairs: candidate.repair_attempts
    })}</pre>
    <pre class="code-block">${candidate.source_code}</pre>
  `;
  return card;
}

function renderTask(task) {
  setStatus(task.status);
  if (task.best_candidate) {
    summary.textContent = `最佳候选方案：${task.best_candidate.display_name}，得分 ${task.best_candidate.validation?.score ?? 0}。`;
  } else if (task.status === "failed") {
    summary.textContent = task.error_message ? `任务失败：${task.error_message}` : "任务执行失败。";
  } else {
    summary.textContent = "任务仍在执行中。";
  }
  parsedIntent.textContent = prettyJson(task.parsed_intent);
  retrievedKnowledge.textContent = prettyJson(task.retrieved_knowledge);
  agentTrace.textContent = prettyJson(task.agent_trace ?? []);
  gitStatus.textContent = prettyJson(task.git_context ?? {});
  repairHistory.textContent = prettyJson(task.repair_history ?? []);

  candidateContainer.replaceChildren();
  for (const candidate of task.candidates ?? []) {
    candidateContainer.appendChild(buildCandidateCard(candidate, task.best_candidate?.candidate_id));
  }
}

async function fetchGraph() {
  const [graphResponse, gitResponse] = await Promise.all([
    fetch("/recommendation/graph"),
    fetch("/git/status")
  ]);
  const graphPayload = await graphResponse.json();
  const gitPayload = await gitResponse.json();
  graphSnapshot.textContent = prettyJson(graphPayload);
  gitStatus.textContent = prettyJson(gitPayload);
  graphMeta.textContent = `${graphPayload.dataset.source} · ${graphPayload.dataset.users} 位用户 · ${graphPayload.dataset.movies} 部电影 · ${graphPayload.dataset.positive_interactions} 条高评分记录`;
  renderGraph(graphPayload);
}

function renderGraph(layoutPayload) {
  while (graphView.firstChild) {
    graphView.removeChild(graphView.firstChild);
  }
  const width = 960;
  const height = 580;
  const nodeMap = new Map(layoutPayload.nodes.map((node) => [node.id, node]));

  const toScreen = (value, size) => value * size;

  for (const edge of layoutPayload.edges) {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) {
      continue;
    }
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", `graph-edge ${edge.type}`);
    line.setAttribute("x1", String(toScreen(source.x, width)));
    line.setAttribute("y1", String(toScreen(source.y, height)));
    line.setAttribute("x2", String(toScreen(target.x, width)));
    line.setAttribute("y2", String(toScreen(target.y, height)));
    graphView.appendChild(line);
  }

  for (const node of layoutPayload.nodes) {
    const x = toScreen(node.x, width);
    const y = toScreen(node.y, height);

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const typeClass = String(node.type).toLowerCase();
    group.setAttribute("class", `graph-node-group ${typeClass}${node.recommended ? " recommended" : ""}`);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "graph-node");
    circle.setAttribute("cx", String(x));
    circle.setAttribute("cy", String(y));
    circle.setAttribute("r", node.type === "User" ? "18" : node.type === "Genre" ? "14" : "12");
    group.appendChild(circle);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "graph-label");
    label.setAttribute("x", String(x));
    label.setAttribute("y", String(y + 31));
    label.setAttribute("text-anchor", "middle");
    label.textContent = String(node.label).slice(0, 22);
    group.appendChild(label);

    const detail = document.createElementNS("http://www.w3.org/2000/svg", "text");
    detail.setAttribute("class", "graph-detail");
    detail.setAttribute("x", String(x));
    detail.setAttribute("y", String(y + 48));
    detail.setAttribute("text-anchor", "middle");
    detail.textContent = String(node.detail ?? "").slice(0, 26);
    group.appendChild(detail);

    graphView.appendChild(group);
  }
}

async function pollTask(taskId) {
  const response = await fetch(`/tasks/${taskId}`);
  const payload = await response.json();
  renderTask(payload);
  if (payload.status === "completed" || payload.status === "failed") {
    submitButton.disabled = false;
    clearInterval(pollTimer);
    pollTimer = null;
    await fetchGraph();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  setStatus("running");
  summary.textContent = "正在提交任务并启动后台工作流。";

  const payload = {
    task_name: document.getElementById("task-name").value.trim(),
    capability_description: document.getElementById("capability-description").value.trim(),
    top_k: Number(document.getElementById("top-k").value),
    candidate_count: Number(document.getElementById("candidate-count").value),
    constraints: document
      .getElementById("constraints")
      .value
      .split(",")
      .map((part) => normalizeConstraintLabel(part))
      .filter(Boolean)
  };

  const response = await fetch("/tasks/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const task = await response.json();
  activeTaskId = task.task_id;
  summary.textContent = `任务 ${activeTaskId} 已进入队列，正在轮询执行结果。`;

  if (pollTimer) {
    clearInterval(pollTimer);
  }
  await pollTask(activeTaskId);
  pollTimer = setInterval(() => pollTask(activeTaskId), 700);
});

fetchGraph();
