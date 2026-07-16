# Validation Report: sample-topk-movielens

- Status: completed
- Provider: mock
- Capability version: 87dbc0ec
- Task: movie_homepage
- Top-K: 5

## Request

Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu.

## Parsed Intent

```json
{
  "task_name": "movie_homepage",
  "objective": "Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu.",
  "task_type": "top_k_recommendation",
  "top_k": 5,
  "constraints": [
    "implicit feedback",
    "local cpu"
  ],
  "success_metrics": [
    "recall@k",
    "ndcg@k",
    "coverage"
  ],
  "preferred_algorithms": [
    "itemcf",
    "popularity"
  ],
  "data_profile": "movielens implicit feedback"
}
```

## Agent Trace

```json
[
  {
    "agent": "IntentParser",
    "summary": "Parsed natural-language capability into a structured recommendation spec.",
    "score": 0.0,
    "state": {
      "task_name": "movie_homepage",
      "task_type": "top_k_recommendation"
    }
  },
  {
    "agent": "KnowledgeRetriever",
    "summary": "Retrieved graph-backed capability knowledge and failure patterns.",
    "score": 3.0,
    "state": {
      "algorithms": [
        "algorithm::itemcf",
        "algorithm::genre_affinity",
        "algorithm::popularity"
      ],
      "metrics": [
        "metric::ndcg_at_k",
        "metric::recall_at_k",
        "metric::coverage"
      ]
    }
  },
  {
    "agent": "BeamSearchPlanner",
    "summary": "Expanded beam at depth 1",
    "score": 24.5,
    "state": {
      "depth": 1,
      "beam_width": 3,
      "best_sequence": [
        "itemcf"
      ]
    }
  },
  {
    "agent": "BeamSearchPlanner",
    "summary": "Expanded beam at depth 2",
    "score": 38.2,
    "state": {
      "depth": 2,
      "beam_width": 3,
      "best_sequence": [
        "itemcf",
        "popularity"
      ]
    }
  },
  {
    "agent": "BeamSearchPlanner",
    "summary": "Expanded beam at depth 3",
    "score": 41.9,
    "state": {
      "depth": 3,
      "beam_width": 3,
      "best_sequence": [
        "genre_affinity",
        "itemcf",
        "popularity"
      ]
    }
  }
]
```

## Candidates

### Item-Based Collaborative Filtering (itemcf)

- Candidate ID: itemcf-1
- Repairs: 0
- Summary: Score candidate items by similarity to each user's interacted items.
- Rationale: Better aligned with Top-K recommendation quality metrics such as Recall@K and NDCG@K.

```json
{
  "candidate_id": "itemcf-1",
  "algorithm_key": "itemcf",
  "success": true,
  "interface_ok": true,
  "stability_ok": true,
  "code_path": "/Users/shtsky/PycharmProjects/ZJUI/generated/sample-topk-movielens/itemcf-1.py",
  "score": 1.0,
  "metrics": {
    "recall_at_k": 1.0,
    "ndcg_at_k": 1.0,
    "coverage": 1.0,
    "average_latency_ms": 0.009444,
    "peak_allocated_kb": 1.703125
  },
  "errors": [],
  "warnings": []
}
```

### Popularity Baseline (popularity)

- Candidate ID: popularity-2
- Repairs: 0
- Summary: Use global item frequency as a strong local baseline with clear behavior.
- Rationale: Good for cold-start fallback, coverage checks, and interface validation.

```json
{
  "candidate_id": "popularity-2",
  "algorithm_key": "popularity",
  "success": true,
  "interface_ok": true,
  "stability_ok": true,
  "code_path": "/Users/shtsky/PycharmProjects/ZJUI/generated/sample-topk-movielens/popularity-2.py",
  "score": 0.301667,
  "metrics": {
    "recall_at_k": 0.333333,
    "ndcg_at_k": 0.166667,
    "coverage": 0.6,
    "average_latency_ms": 0.001788,
    "peak_allocated_kb": 1.617188
  },
  "errors": [],
  "warnings": []
}
```

### Genre Affinity Recommender (genre_affinity)

- Candidate ID: genre_affinity-3
- Repairs: 0
- Summary: Estimate user preference from interaction genres and rank unseen items by genre affinity.
- Rationale: Adds a content-oriented candidate that is easy to explain and transfers better to sparse histories.

```json
{
  "candidate_id": "genre_affinity-3",
  "algorithm_key": "genre_affinity",
  "success": true,
  "interface_ok": true,
  "stability_ok": true,
  "code_path": "/Users/shtsky/PycharmProjects/ZJUI/generated/sample-topk-movielens/genre_affinity-3.py",
  "score": 0.985417,
  "metrics": {
    "recall_at_k": 1.0,
    "ndcg_at_k": 0.958333,
    "coverage": 1.0,
    "average_latency_ms": 0.019257,
    "peak_allocated_kb": 1.820312
  },
  "errors": [],
  "warnings": []
}
```

## Best Candidate

- Item-Based Collaborative Filtering (itemcf)

## Git Context

```json
{
  "is_git_repo": true,
  "branch": "main",
  "head_commit": "87dbc0ec47d96b6f6c77e62883a67899b2c89bd3",
  "is_dirty": false,
  "staged_files": [],
  "unstaged_files": [],
  "untracked_files": []
}
```
