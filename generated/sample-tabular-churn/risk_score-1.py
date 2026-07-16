from __future__ import annotations

import pandas as pd

TASK_ID = "sample-tabular-churn"
OBJECTIVE = 'Build a customer churn classification workflow. Use local cpu, compare baseline and score-based methods, and optimize f1.'


class GeneratedClassifier:
    algorithm_key = "risk_score"
    algorithm_name = "Risk Score Classifier"

    def __init__(self) -> None:
        self.label_column = "churned"
        self.feature_columns = []
        self.weights = {}
        self.threshold = 0.0

    def fit(self, train_df: pd.DataFrame):
        self.label_column = "churned" if "churned" in train_df.columns else train_df.columns[-1]
        self.feature_columns = [
            column
            for column in train_df.columns
            if column not in {self.label_column, "customer_id", "split"}
        ]
        positive = train_df[train_df[self.label_column] == 1]
        negative = train_df[train_df[self.label_column] == 0]
        self.weights = {}
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
