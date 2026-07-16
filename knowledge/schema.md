# Knowledge Graph Schema

## Entities

| Entity | Purpose | Example |
| --- | --- | --- |
| `Capability` | Abstract algorithm capability to be replicated | `topk_recommendation` |
| `TaskType` | Business scenario or usage surface | `movie_homepage` |
| `Algorithm` | Candidate implementation family | `popularity`, `itemcf`, `genre_affinity` |
| `Metric` | Validation target | `recall_at_k`, `ndcg_at_k`, `coverage` |
| `Constraint` | Execution or data constraint | `implicit_feedback`, `local_cpu` |
| `CodeTemplate` | Known-safe code template | `popularity_py`, `itemcf_py`, `genre_affinity_py` |
| `FailurePattern` | Reusable failure knowledge | `seen_items`, `empty_list`, `unsafe_code` |
| `ValidationResult` | Historical verification writeback | `validation::sample-topk-movielens` |
| `Repository` | Extracted external code repository | `repo::sample_repo` |
| `SourceFile` | File extracted from a real code repository | `repo_file::sample_repo::service.py` |
| `CodeSymbol` | Function or class extracted from code | `repo_symbol::sample_repo::service.py::score` |
| `Dependency` | Dependency discovered from repository manifests | `dependency::fastapi` |

## Relations

| Relation | Meaning |
| --- | --- |
| `supports` | Capability supports a task type |
| `uses` | Capability may use an algorithm |
| `evaluated_by` | Algorithm is judged by a metric |
| `fits` | Algorithm matches a constraint |
| `implemented_by` | Algorithm is backed by a code template |
| `suggests_fix` | Failure pattern suggests a recovery target |
| `verifies` | Validation result verifies a capability |
| `evaluated` | Validation result records an algorithm outcome |
| `contains` | Repository contains a source file |
| `defines` | Source file defines a symbol |
| `depends_on` | Repository depends on a package |

## Writeback Payload

Each `ValidationResult` node stores:

- `best_candidate_id`
- `task_name`
- `capability_version`
- structured `reports` containing candidate score and metrics

The exported graph is available in:

- `knowledge/exports/knowledge_graph.json`
- `knowledge/exports/knowledge_graph.graphml`

Repository extraction outputs are written to:

- `knowledge/extracted/<repo_name>.json`
