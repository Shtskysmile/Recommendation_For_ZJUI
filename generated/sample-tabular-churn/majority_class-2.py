from __future__ import annotations

import pandas as pd

TASK_ID = "sample-tabular-churn"
OBJECTIVE = 'Build a customer churn classification workflow. Use local cpu, compare baseline and score-based methods, and optimize f1.'


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
