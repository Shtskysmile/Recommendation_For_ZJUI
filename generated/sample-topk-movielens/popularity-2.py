from __future__ import annotations

import pandas as pd

TASK_ID = "sample-topk-movielens"
OBJECTIVE = 'Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu.'


class GeneratedRecommender:
    algorithm_key = "popularity"
    algorithm_name = "Popularity Baseline"

    def __init__(self) -> None:
        self.user_history = {}
        self.ranked_items = []

    def fit(self, train_df: pd.DataFrame):
        grouped = train_df.groupby("user_id")["item_id"].apply(list)
        self.user_history = {
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in grouped.items()
        }
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self.ranked_items = [int(item_id) for item_id in counts.index.tolist()]
        return self

    def recommend(self, user_id: int, k: int = 5) -> list[int]:
        seen = set(self.user_history.get(int(user_id), []))
        recommendations = []
        for item_id in self.ranked_items:
            if item_id in seen:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= k:
                break
        return recommendations

    def batch_recommend(self, user_ids: list[int], k: int = 5) -> dict[int, list[int]]:
        return {int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}
