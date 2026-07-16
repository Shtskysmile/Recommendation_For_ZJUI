from __future__ import annotations

from math import sqrt

import pandas as pd

TASK_ID = "sample-topk-movielens"
OBJECTIVE = 'Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu.'


class GeneratedRecommender:
    algorithm_key = "itemcf"
    algorithm_name = "Item-Based Collaborative Filtering"

    def __init__(self) -> None:
        self.user_history = {}
        self.item_users = {}
        self.item_similarity = {}
        self.popularity = []

    def fit(self, train_df: pd.DataFrame):
        grouped = train_df.groupby("user_id")["item_id"].apply(list)
        self.user_history = {
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in grouped.items()
        }
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self.popularity = [int(item_id) for item_id in counts.index.tolist()]
        item_groups = train_df.groupby("item_id")["user_id"].apply(list)
        self.item_users = {
            int(item_id): set(int(user_id) for user_id in users)
            for item_id, users in item_groups.items()
        }
        self.item_similarity = {item_id: {} for item_id in self.item_users}

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

    def recommend(self, user_id: int, k: int = 5) -> list[int]:
        seen = set(self.user_history.get(int(user_id), []))
        scores = {}
        for seen_item in seen:
            for candidate_item, similarity in self.item_similarity.get(seen_item, {}).items():
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

    def batch_recommend(self, user_ids: list[int], k: int = 5) -> dict[int, list[int]]:
        return {int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}
