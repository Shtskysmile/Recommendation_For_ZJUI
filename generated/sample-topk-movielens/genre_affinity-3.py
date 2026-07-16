from __future__ import annotations

from collections import Counter

import pandas as pd

TASK_ID = "sample-topk-movielens"
OBJECTIVE = 'Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu.'


class GeneratedRecommender:
    algorithm_key = "genre_affinity"
    algorithm_name = "Genre Affinity Recommender"

    def __init__(self) -> None:
        self.user_history = {}
        self.user_genre_scores = {}
        self.item_metadata = {}
        self.popularity = []

    def fit(self, train_df: pd.DataFrame):
        grouped = train_df.groupby("user_id")["item_id"].apply(list)
        self.user_history = {
            int(user_id): [int(item_id) for item_id in items]
            for user_id, items in grouped.items()
        }
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self.popularity = [int(item_id) for item_id in counts.index.tolist()]
        self.item_metadata = {
            int(row.item_id): str(row.genres).split("|")
            for row in train_df.drop_duplicates("item_id").itertuples(index=False)
        }

        self.user_genre_scores = {}
        for user_id, frame in train_df.groupby("user_id"):
            counter = Counter()
            for genres in frame["genres"].tolist():
                for genre in str(genres).split("|"):
                    counter[genre] += 1
            self.user_genre_scores[int(user_id)] = counter
        return self

    def recommend(self, user_id: int, k: int = 5) -> list[int]:
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

    def batch_recommend(self, user_ids: list[int], k: int = 5) -> dict[int, list[int]]:
        return {int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}
