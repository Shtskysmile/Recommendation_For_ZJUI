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

let activeTaskId = null;
let pollTimer = null;

function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function setStatus(value) {
  statusPill.textContent = value;
  statusPill.className = `status-pill ${String(value).toLowerCase()}`;
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
        <div class="status-pill ${report.success ? "completed" : "failed"}">${report.success ? "validated" : "failed"}</div>
      </div>
      <strong>${candidate.algorithm_key}</strong>
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
  summary.textContent = task.best_candidate
    ? `Best candidate: ${task.best_candidate.display_name} with score ${task.best_candidate.validation?.score ?? 0}.`
    : task.error_message || "Task is still running.";
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
  const [graphResponse, layoutResponse, gitResponse] = await Promise.all([
    fetch("/knowledge/graph"),
    fetch("/knowledge/graph/layout"),
    fetch("/git/status")
  ]);
  const graphPayload = await graphResponse.json();
  const layoutPayload = await layoutResponse.json();
  const gitPayload = await gitResponse.json();
  graphSnapshot.textContent = prettyJson(graphPayload);
  gitStatus.textContent = prettyJson(gitPayload);
  renderGraph(layoutPayload);
}

function renderGraph(layoutPayload) {
  while (graphView.firstChild) {
    graphView.removeChild(graphView.firstChild);
  }
  const width = 520;
  const height = 320;
  const nodeMap = new Map(layoutPayload.nodes.map((node) => [node.id, node]));

  const toScreen = (value, size) => ((value + 1) / 2) * (size - 48) + 24;

  for (const edge of layoutPayload.edges) {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) {
      continue;
    }
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", "graph-edge");
    line.setAttribute("x1", String(toScreen(source.x, width)));
    line.setAttribute("y1", String(toScreen(source.y, height)));
    line.setAttribute("x2", String(toScreen(target.x, width)));
    line.setAttribute("y2", String(toScreen(target.y, height)));
    graphView.appendChild(line);
  }

  for (const node of layoutPayload.nodes) {
    const x = toScreen(node.x, width);
    const y = toScreen(node.y, height);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "graph-node");
    circle.setAttribute("cx", String(x));
    circle.setAttribute("cy", String(y));
    circle.setAttribute("r", "7");
    graphView.appendChild(circle);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "graph-label");
    label.setAttribute("x", String(x + 10));
    label.setAttribute("y", String(y - 8));
    label.textContent = String(node.label).slice(0, 26);
    graphView.appendChild(label);
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
  summary.textContent = "Submitting task and starting background workflow.";

  const payload = {
    task_name: document.getElementById("task-name").value.trim(),
    capability_description: document.getElementById("capability-description").value.trim(),
    top_k: Number(document.getElementById("top-k").value),
    candidate_count: Number(document.getElementById("candidate-count").value),
    constraints: document
      .getElementById("constraints")
      .value
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean)
  };

  const response = await fetch("/tasks/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const task = await response.json();
  activeTaskId = task.task_id;
  summary.textContent = `Task ${activeTaskId} queued. Polling for completion.`;

  if (pollTimer) {
    clearInterval(pollTimer);
  }
  await pollTask(activeTaskId);
  pollTimer = setInterval(() => pollTask(activeTaskId), 700);
});

fetchGraph();
