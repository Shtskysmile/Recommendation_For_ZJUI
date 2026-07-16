# Validation Report: sample-tabular-churn

- Status: completed
- Provider: mock
- Capability version: 67051b24
- Task: customer_churn
- Task type: tabular_classification
- Dataset: churn_sample
- Top-K: 5

## Request

Build a customer churn classification workflow. Use local cpu, compare baseline and score-based methods, and optimize f1.

## Parsed Intent

```json
{
  "task_name": "customer_churn",
  "objective": "Build a customer churn classification workflow. Use local cpu, compare baseline and score-based methods, and optimize f1.",
  "task_type": "tabular_classification",
  "dataset_name": "churn_sample",
  "top_k": 5,
  "constraints": [
    "local cpu"
  ],
  "success_metrics": [
    "accuracy",
    "precision",
    "recall",
    "f1"
  ],
  "preferred_algorithms": [
    "risk_score",
    "majority_class"
  ],
  "data_profile": "customer churn tabular classification"
}
```

## Agent Trace

```json
[
  {
    "agent": "IntentParser",
    "summary": "Parsed natural-language capability into a structured task specification.",
    "score": 0.0,
    "state": {
      "task_name": "customer_churn",
      "task_type": "tabular_classification",
      "dataset_name": "churn_sample"
    }
  },
  {
    "agent": "KnowledgeRetriever",
    "summary": "Retrieved graph-backed capability knowledge and failure patterns.",
    "score": 3.0,
    "state": {
      "algorithms": [
        "algorithm::risk_score",
        "algorithm::majority_class",
        "algorithm::genre_affinity"
      ],
      "metrics": [
        "metric::f1",
        "metric::accuracy",
        "metric::precision"
      ],
      "profile_key": "tabular_classification"
    }
  },
  {
    "agent": "BeamSearchPlanner",
    "summary": "Expanded beam at depth 1",
    "score": 27.0,
    "state": {
      "depth": 1,
      "beam_width": 3,
      "best_sequence": [
        "risk_score"
      ]
    }
  },
  {
    "agent": "BeamSearchPlanner",
    "summary": "Expanded beam at depth 2",
    "score": 46.2,
    "state": {
      "depth": 2,
      "beam_width": 3,
      "best_sequence": [
        "majority_class",
        "risk_score"
      ]
    }
  }
]
```

## Candidates

### Risk Score Classifier (risk_score)

- Candidate ID: risk_score-1
- Repairs: 0
- Summary: Build a lightweight linear risk score from class-conditional feature means for churn prediction.
- Rationale: Provides a stronger tabular classification candidate while remaining dependency-light and explainable.
- Validation profile: tabular_classification
- Execution mode: subprocess_sandbox

```json
{
  "candidate_id": "risk_score-1",
  "algorithm_key": "risk_score",
  "profile_name": "tabular_classification",
  "success": true,
  "interface_ok": true,
  "stability_ok": true,
  "code_path": "/Users/shtsky/PycharmProjects/ZJUI/generated/sample-tabular-churn/risk_score-1.py",
  "execution_mode": "subprocess_sandbox",
  "score": 1.0,
  "metrics": {
    "recall_at_k": 0.0,
    "ndcg_at_k": 0.0,
    "coverage": 0.0,
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "average_latency_ms": 0.197401,
    "peak_allocated_kb": 19.669922
  },
  "errors": [],
  "warnings": []
}
```

### Majority Classifier (majority_class)

- Candidate ID: majority_class-2
- Repairs: 0
- Summary: Predict the majority label from the training set as a classification baseline.
- Rationale: Provides a deterministic baseline for tabular classification and sanity-checks the validation profile.
- Validation profile: tabular_classification
- Execution mode: subprocess_sandbox

```json
{
  "candidate_id": "majority_class-2",
  "algorithm_key": "majority_class",
  "profile_name": "tabular_classification",
  "success": true,
  "interface_ok": true,
  "stability_ok": true,
  "code_path": "/Users/shtsky/PycharmProjects/ZJUI/generated/sample-tabular-churn/majority_class-2.py",
  "execution_mode": "subprocess_sandbox",
  "score": 0.175,
  "metrics": {
    "recall_at_k": 0.0,
    "ndcg_at_k": 0.0,
    "coverage": 0.0,
    "accuracy": 0.5,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "average_latency_ms": 0.059453,
    "peak_allocated_kb": 19.854492
  },
  "errors": [],
  "warnings": []
}
```

## Best Candidate

- Risk Score Classifier (risk_score)

## Git Context

```json
{
  "is_git_repo": true,
  "branch": "main",
  "head_commit": "67051b244170833d5435f38e7653100b1f5b9fcc",
  "is_dirty": true,
  "staged_files": [],
  "unstaged_files": [
    ".env.example",
    "README.md",
    "app/cli.py",
    "app/codegen.py",
    "app/data.py",
    "app/knowledge.py",
    "app/llm.py",
    "app/models.py",
    "app/reporting.py",
    "app/search.py",
    "app/settings.py",
    "app/validation.py",
    "app/workflow.py",
    "knowledge/exports/knowledge_graph.graphml",
    "knowledge/exports/knowledge_graph.json",
    "knowledge/seed_catalog.json",
    "reports/tasks/sample-topk-movielens.json",
    "reports/tasks/sample-topk-movielens.md",
    "tests/test_data.py",
    "tests/test_knowledge.py",
    "tests/test_search.py",
    "tests/test_validation.py",
    "tests/test_workflow.py"
  ],
  "untracked_files": [
    "app/plugin_loader.py",
    "app/plugins/",
    "app/profile_library.py",
    "app/sandbox_worker.py",
    "app/template_library.py",
    "data/churn_sample/",
    "scripts/",
    "tests/test_plugins.py"
  ]
}
```
