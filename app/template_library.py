from __future__ import annotations

from app.models import CapabilitySpec


def render_popularity_template(spec: CapabilitySpec, task_id: str) -> str:
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


def render_itemcf_template(spec: CapabilitySpec, task_id: str) -> str:
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


def render_genre_affinity_template(spec: CapabilitySpec, task_id: str) -> str:
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


def render_majority_class_template(spec: CapabilitySpec, task_id: str) -> str:
    return f'''from __future__ import annotations

import pandas as pd

TASK_ID = "{task_id}"
OBJECTIVE = {spec.objective!r}


class GeneratedClassifier:
    algorithm_key = "majority_class"
    algorithm_name = "Majority Classifier"

    def __init__(self) -> None:
        self.label_column = "churned"
        self.majority_label = 0

    def fit(self, train_df: pd.DataFrame):
        self.label_column = "churned" if "churned" in train_df.columns else train_df.columns[-1]
        self.majority_label = int(train_df[self.label_column].mode().iloc[0])
        return self

    def predict(self, features_df: pd.DataFrame) -> list[int]:
        return [int(self.majority_label) for _ in range(len(features_df))]

    def batch_predict(self, features_df: pd.DataFrame) -> list[int]:
        return self.predict(features_df)
'''


def render_risk_score_template(spec: CapabilitySpec, task_id: str) -> str:
    return f'''from __future__ import annotations

import pandas as pd

TASK_ID = "{task_id}"
OBJECTIVE = {spec.objective!r}


class GeneratedClassifier:
    algorithm_key = "risk_score"
    algorithm_name = "Risk Score Classifier"

    def __init__(self) -> None:
        self.label_column = "churned"
        self.feature_columns = []
        self.weights = {{}}
        self.threshold = 0.0

    def fit(self, train_df: pd.DataFrame):
        self.label_column = "churned" if "churned" in train_df.columns else train_df.columns[-1]
        self.feature_columns = [
            column
            for column in train_df.columns
            if column not in {{self.label_column, "customer_id", "split"}}
        ]
        positive = train_df[train_df[self.label_column] == 1]
        negative = train_df[train_df[self.label_column] == 0]
        self.weights = {{}}
        for column in self.feature_columns:
            pos_mean = float(positive[column].mean())
            neg_mean = float(negative[column].mean())
            self.weights[column] = pos_mean - neg_mean
        train_scores = self._score_frame(train_df[self.feature_columns])
        positive_scores = train_scores[train_df[self.label_column] == 1]
        negative_scores = train_scores[train_df[self.label_column] == 0]
        self.threshold = float((positive_scores.mean() + negative_scores.mean()) / 2)
        return self

    def _score_frame(self, features_df: pd.DataFrame) -> pd.Series:
        scores = pd.Series([0.0] * len(features_df), index=features_df.index, dtype=float)
        for column, weight in self.weights.items():
            scores = scores + (features_df[column].astype(float) * float(weight))
        return scores

    def predict(self, features_df: pd.DataFrame) -> list[int]:
        scores = self._score_frame(features_df[self.feature_columns])
        return [1 if float(score) >= self.threshold else 0 for score in scores.tolist()]

    def batch_predict(self, features_df: pd.DataFrame) -> list[int]:
        return self.predict(features_df)
'''
