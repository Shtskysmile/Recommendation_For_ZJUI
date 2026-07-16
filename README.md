# AI Capability Factory for Recommendation Workflows

This repository implements a recommendation-oriented `AI Agent` capability factory prototype for the assignment in [docs/requirements.pdf](docs/requirements.pdf). It converts a natural-language recommendation capability into runnable code, validates the result, compares multiple candidates, writes the outcome back into a knowledge graph, and now also supports `Beam Search`, `real repository extraction`, `Git-aware version management`, and a `visual graph UI`.

The system is built around `MovieLens-derived implicit feedback` and `Top-K recommendation`, with `硅基流动 (SiliconFlow)` as the default real LLM backend through its `OpenAI-compatible` API.

## 1. Project Background And Goal

The goal is to demonstrate the full loop:

`capability understanding -> knowledge retrieval -> candidate search -> code generation -> validation -> repair -> writeback`

Scenario choice:

- business scenario: movie homepage recommendation
- task form: `Top-K recall / ranking`
- data style: `MovieLens-derived implicit feedback`
- user-facing result: generated recommender code, validation report, best candidate, graph writeback

## 2. System Architecture And Module Design

High-level workflow:

```text
Natural-language request
  -> Planner Agent (LLM / mock)
  -> Retriever Agent (knowledge graph)
  -> Beam Search Planner
  -> Codegen Agent (template-based candidates)
  -> Verifier Agent (safety + metrics + resource analysis)
  -> Repair Agent (multi-round fallback repair)
  -> Writer Agent (report + graph writeback + version context)
```

Core modules:

- `app/main.py`: FastAPI routes and web entrypoints
- `app/workflow.py`: end-to-end orchestration
- `app/search.py`: `Beam Search` candidate selection
- `app/llm.py`: SiliconFlow / OpenAI-compatible provider plus mock fallback
- `app/codegen.py`: safe code-template generation
- `app/validation.py`: interface checks, safety scan, deterministic validation, latency, and memory analysis
- `app/repo_mining.py`: extraction of files, functions, classes, imports, and dependencies from real code repositories
- `app/gitops.py`: git status and snapshot management
- `app/knowledge.py`: graph load, retrieve, layout export, repo ingestion, and writeback
- `app/ui/`: single-page UI with candidate view, graph visualization, agent trace, and git status

## 3. Capability Knowledge Graph Schema And Example

Schema details are documented in [knowledge/schema.md](knowledge/schema.md).

Main entity types:

- `Capability`
- `TaskType`
- `Algorithm`
- `Metric`
- `Constraint`
- `CodeTemplate`
- `FailurePattern`
- `ValidationResult`
- `Repository`
- `SourceFile`
- `CodeSymbol`
- `Dependency`

Main relation types:

- `supports`
- `uses`
- `evaluated_by`
- `fits`
- `implemented_by`
- `suggests_fix`
- `verifies`
- `evaluated`
- `contains`
- `defines`
- `depends_on`

Artifacts:

- seed graph: [knowledge/seed_catalog.json](knowledge/seed_catalog.json)
- exported graph: [knowledge/exports/knowledge_graph.json](knowledge/exports/knowledge_graph.json)
- GraphML export: [knowledge/exports/knowledge_graph.graphml](knowledge/exports/knowledge_graph.graphml)
- extracted repo knowledge: `knowledge/extracted/*.json`

## 4. Agent Workflow Design

The implementation is explicitly multi-stage rather than a single prompt.

1. `Planner Agent`
   - parses user input into `CapabilitySpec`
   - real mode uses `SiliconFlow OpenAI-compatible API`
   - fallback mode uses a deterministic local mock
2. `Retriever Agent`
   - retrieves candidate algorithms, metrics, constraints, and failure patterns from the graph
3. `Beam Search Planner`
   - ranks algorithms using request intent, graph retrieval order, and heuristic strengths
   - selects a final candidate set with diversity-aware beam expansion
4. `Codegen Agent`
   - renders known-safe templates into runnable Python implementations
5. `Verifier Agent`
   - runs AST-based safety checks
   - validates `fit / recommend / batch_recommend`
   - computes `Recall@K`, `NDCG@K`, `Coverage`, latency, and peak allocated memory
6. `Repair Agent`
   - retries a failed candidate via safe template regeneration
7. `Writer Agent`
   - writes JSON and Markdown reports
   - records graph writeback and git version context

Current algorithm candidates:

- `itemcf`
- `popularity`
- `genre_affinity`

## 5. Environment Setup And Running

### Prerequisites

- Python `3.14`
- Git

### Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

### SiliconFlow configuration

The project defaults to `SiliconFlow` when an API key is present.

Edit `.env`:

```bash
LLM_PROVIDER=siliconflow
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_API_KEY=your_siliconflow_key
OPENAI_MODEL=deepseek-ai/DeepSeek-V3
OPENAI_TIMEOUT_SECONDS=60
```

If no API key is available, the app falls back to the deterministic mock provider.

### Start the web app

```bash
./.venv/bin/python -m app.cli serve --host 127.0.0.1 --port 8000
```

Open:

- UI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

### CLI examples

Run a generation task:

```bash
./.venv/bin/python -m app.cli run-task \
  --task-name movie_homepage \
  --description "Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu." \
  --top-k 5 \
  --constraints "implicit feedback" "local cpu" \
  --candidate-count 3
```

Extract a real repository:

```bash
./.venv/bin/python -m app.cli ingest-repo --repo-path .
```

Inspect git status:

```bash
./.venv/bin/python -m app.cli git-status
```

Create a git snapshot:

```bash
./.venv/bin/python -m app.cli git-snapshot --message "feat: update capability factory"
```

Export the graph:

```bash
./.venv/bin/python -m app.cli export-graph
```

## 6. Example Data And Test Task

Sample data:

- [data/movielens_sample/items.csv](data/movielens_sample/items.csv)
- [data/movielens_sample/interactions.csv](data/movielens_sample/interactions.csv)

Design:

- ratings `>= 4` are treated as positive implicit feedback
- evaluation uses leave-last-one-out per user
- the sample is intentionally compact and deterministic for local validation

Example request:

- [examples/sample_request.json](examples/sample_request.json)

## 7. Generated Algorithm Code Example

Sample candidates are generated under `generated/<task_id>/`.

Representative example files:

- [generated/sample-topk-movielens/itemcf-1.py](generated/sample-topk-movielens/itemcf-1.py)
- [generated/sample-topk-movielens/popularity-2.py](generated/sample-topk-movielens/popularity-2.py)
- [generated/sample-topk-movielens/genre_affinity-3.py](generated/sample-topk-movielens/genre_affinity-3.py)

Each candidate implements:

- `fit(train_df)`
- `recommend(user_id, k)`
- `batch_recommend(user_ids, k)`

## 8. Validation Results And Report Samples

Sample reports:

- [reports/tasks/sample-topk-movielens.json](reports/tasks/sample-topk-movielens.json)
- [reports/tasks/sample-topk-movielens.md](reports/tasks/sample-topk-movielens.md)

The report now includes:

- parsed intent
- multi-agent trace
- candidate comparison
- latency and peak memory metrics
- repair history
- git context
- knowledge writeback id

## 9. Challenges And Solutions

### Real LLM variability vs deterministic evaluation

Challenge:
Real LLM outputs can be slow or structurally inconsistent.

Solution:
The SiliconFlow provider normalizes structured outputs and automatically falls back to local deterministic planning/explanation if a remote call times out or returns nonconforming fields.

### Candidate search quality

Challenge:
A single prompt does not make candidate search explicit or inspectable.

Solution:
The workflow now inserts a `Beam Search` stage with traceable search steps and candidate scores, which directly addresses the search-oriented bonus requirement.

### Knowledge extraction from real repositories

Challenge:
Algorithm capability often lives in historical repositories rather than clean business docs.

Solution:
The system can extract Python files, imports, functions, classes, README context, and dependencies from a real repository and ingest them into the graph as reusable capability knowledge.

### Version management and reproducibility

Challenge:
Generated outputs and validation reports should be tied to a versioned code state.

Solution:
Git status is exposed through API, UI, and CLI, task reports record git context, and the system provides a snapshot command for structured commits.

## 10. Future Extensions

- add process-level sandbox isolation instead of in-process validation only
- add matrix factorization, sequence models, and hybrid rerankers
- add plugin registration for metrics and code templates
- generalize validation profiles to tabular classification and text classification
- persist tasks and graph writebacks in SQLite or Neo4j
- add deployment presets for cloud demos and multi-user serving

## Deployment

Minimal container config is included:

- [Dockerfile](/Users/shtsky/PycharmProjects/ZJUI/Dockerfile:1)

Build and run:

```bash
docker build -t capability-factory .
docker run --rm -p 8000:8000 --env-file .env capability-factory
```

## Testing

Run the test suite:

```bash
./.venv/bin/pytest
```

Current automated test status:

- `13` tests passing
- total coverage `82%`
