from __future__ import annotations

from app.algorithm_registry import ALGORITHM_REGISTRY
from app.models import AlgorithmCandidate, CapabilitySpec, CandidatePlan
from app.settings import AppSettings


class CandidateGenerator:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def generate_candidate(
        self,
        task_id: str,
        candidate_id: str,
        algorithm_key: str,
        spec: CapabilitySpec,
        plan: CandidatePlan | None = None,
        version_suffix: str = "",
    ) -> AlgorithmCandidate:
        source_code = self.render_template(algorithm_key=algorithm_key, spec=spec, task_id=task_id)
        target_dir = self.settings.generated_dir / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{candidate_id}{version_suffix}.py" if version_suffix else f"{candidate_id}.py"
        code_path = target_dir / filename
        code_path.write_text(source_code, encoding="utf-8")

        metadata = ALGORITHM_REGISTRY[algorithm_key]
        return AlgorithmCandidate(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            display_name=plan.display_name if plan else metadata.display_name,
            rationale=plan.rationale if plan else metadata.rationale,
            plan_summary=plan.plan_summary if plan else metadata.summary,
            code_path=str(code_path),
            source_code=source_code,
        )

    def render_template(self, algorithm_key: str, spec: CapabilitySpec, task_id: str) -> str:
        if algorithm_key == "itemcf":
            return self._itemcf_template(spec=spec, task_id=task_id)
        if algorithm_key == "genre_affinity":
            return self._genre_affinity_template(spec=spec, task_id=task_id)
        return self._popularity_template(spec=spec, task_id=task_id)

    def _popularity_template(self, spec: CapabilitySpec, task_id: str) -> str:
        return f'''from __future__ import annotations

import pandas as pd

TASK_ID = "{task_id}"
OBJECTIVE = {spec.objective!r}


class GeneratedRecommender:
    algorithm_key = "popularity"
    algorithm_name = "Popularity Baseline"

    def __init__(self) -> None:
        self.user_history = {{}}
        self.ranked_items = []

    def fit(self, train_df: pd.DataFrame):
        grouped = train_df.groupby("user_id")["item_id"].apply(list)
        self.user_history = {{
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in grouped.items()
        }}
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self.ranked_items = [int(item_id) for item_id in counts.index.tolist()]
        return self

    def recommend(self, user_id: int, k: int = {spec.top_k}) -> list[int]:
        seen = set(self.user_history.get(int(user_id), []))
        recommendations = []
        for item_id in self.ranked_items:
            if item_id in seen:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= k:
                break
        return recommendations

    def batch_recommend(self, user_ids: list[int], k: int = {spec.top_k}) -> dict[int, list[int]]:
        return {{int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}}
'''

    def _itemcf_template(self, spec: CapabilitySpec, task_id: str) -> str:
        return f'''from __future__ import annotations

from math import sqrt

import pandas as pd

TASK_ID = "{task_id}"
OBJECTIVE = {spec.objective!r}


class GeneratedRecommender:
    algorithm_key = "itemcf"
    algorithm_name = "Item-Based Collaborative Filtering"

    def __init__(self) -> None:
        self.user_history = {{}}
        self.item_users = {{}}
        self.item_similarity = {{}}
        self.popularity = []

    def fit(self, train_df: pd.DataFrame):
        grouped = train_df.groupby("user_id")["item_id"].apply(list)
        self.user_history = {{
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in grouped.items()
        }}
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self.popularity = [int(item_id) for item_id in counts.index.tolist()]
        item_groups = train_df.groupby("item_id")["user_id"].apply(list)
        self.item_users = {{
            int(item_id): set(int(user_id) for user_id in users)
            for item_id, users in item_groups.items()
        }}
        self.item_similarity = {{item_id: {{}} for item_id in self.item_users}}

        item_ids = sorted(self.item_users)
        for index, left_item in enumerate(item_ids):
            left_users = self.item_users[left_item]
            for right_item in item_ids[index + 1:]:
                right_users = self.item_users[right_item]
                overlap = len(left_users & right_users)
                if overlap == 0:
                    continue
                similarity = overlap / sqrt(len(left_users) * len(right_users))
                self.item_similarity[left_item][right_item] = similarity
                self.item_similarity[right_item][left_item] = similarity
        return self

    def recommend(self, user_id: int, k: int = {spec.top_k}) -> list[int]:
        seen = set(self.user_history.get(int(user_id), []))
        scores = {{}}
        for seen_item in seen:
            for candidate_item, similarity in self.item_similarity.get(seen_item, {{}}).items():
                if candidate_item in seen:
                    continue
                scores[candidate_item] = scores.get(candidate_item, 0.0) + similarity

        ranked = [
            item_id
            for item_id, _score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
        for item_id in self.popularity:
            if item_id not in seen and item_id not in ranked:
                ranked.append(item_id)
        return ranked[:k]

    def batch_recommend(self, user_ids: list[int], k: int = {spec.top_k}) -> dict[int, list[int]]:
        return {{int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}}
'''

    def _genre_affinity_template(self, spec: CapabilitySpec, task_id: str) -> str:
        return f'''from __future__ import annotations

from collections import Counter

import pandas as pd

TASK_ID = "{task_id}"
OBJECTIVE = {spec.objective!r}


class GeneratedRecommender:
    algorithm_key = "genre_affinity"
    algorithm_name = "Genre Affinity Recommender"

    def __init__(self) -> None:
        self.user_history = {{}}
        self.user_genre_scores = {{}}
        self.item_metadata = {{}}
        self.popularity = []

    def fit(self, train_df: pd.DataFrame):
        grouped = train_df.groupby("user_id")["item_id"].apply(list)
        self.user_history = {{
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in grouped.items()
        }}
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self.popularity = [int(item_id) for item_id in counts.index.tolist()]
        self.item_metadata = {{
            int(row.item_id): str(row.genres).split("|")
            for row in train_df.drop_duplicates("item_id").itertuples(index=False)
        }}

        self.user_genre_scores = {{}}
        for user_id, frame in train_df.groupby("user_id"):
            counter = Counter()
            for genres in frame["genres"].tolist():
                for genre in str(genres).split("|"):
                    counter[genre] += 1
            self.user_genre_scores[int(user_id)] = counter
        return self

    def recommend(self, user_id: int, k: int = {spec.top_k}) -> list[int]:
        user_id = int(user_id)
        seen = set(self.user_history.get(user_id, []))
        genre_scores = self.user_genre_scores.get(user_id, Counter())
        ranked_pairs = []

        for item_id, genres in self.item_metadata.items():
            if item_id in seen:
                continue
            score = sum(genre_scores.get(genre, 0) for genre in genres)
            if score > 0:
                ranked_pairs.append((item_id, float(score)))

        ranked = [
            item_id
            for item_id, _score in sorted(ranked_pairs, key=lambda pair: (-pair[1], pair[0]))
        ]
        for item_id in self.popularity:
            if item_id not in seen and item_id not in ranked:
                ranked.append(item_id)
        return ranked[:k]

    def batch_recommend(self, user_ids: list[int], k: int = {spec.top_k}) -> dict[int, list[int]]:
        return {{int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}}
'''
