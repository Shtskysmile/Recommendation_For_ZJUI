from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class RecommendationDataset:
    interactions_df: pd.DataFrame
    train_df: pd.DataFrame
    test_truth: dict[int, list[int]]
    catalog: dict[int, dict[str, str]]
    user_histories: dict[int, list[int]]
    user_count: int
    item_count: int

    @property
    def catalog_item_ids(self) -> list[int]:
        return sorted(self.catalog)


class MovieLensDatasetLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load(self) -> RecommendationDataset:
        interactions_path = self.data_dir / "interactions.csv"
        items_path = self.data_dir / "items.csv"
        interactions = pd.read_csv(interactions_path)
        items = pd.read_csv(items_path)
        merged = interactions.merge(items, on="item_id", how="left")
        filtered = merged[merged["rating"] >= 4].sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        train_frames: list[pd.DataFrame] = []
        test_truth: dict[int, list[int]] = {}

        for user_id, frame in filtered.groupby("user_id", sort=True):
            ordered = frame.sort_values("timestamp").reset_index(drop=True)
            if len(ordered) < 2:
                continue
            test_row = ordered.iloc[-1]
            train_frame = ordered.iloc[:-1].copy()
            if train_frame.empty:
                continue
            train_frames.append(train_frame)
            test_truth[int(user_id)] = [int(test_row["item_id"])]

        if not train_frames:
            raise ValueError("No valid train/test split could be created from the dataset.")

        train_df = pd.concat(train_frames, ignore_index=True)
        user_histories = {
            int(user_id): [int(item_id) for item_id in frame["item_id"].tolist()]
            for user_id, frame in train_df.groupby("user_id", sort=True)
        }
        catalog = {
            int(row.item_id): {
                "title": str(row.title),
                "genres": str(row.genres),
            }
            for row in items.itertuples(index=False)
        }

        return RecommendationDataset(
            interactions_df=filtered,
            train_df=train_df,
            test_truth=test_truth,
            catalog=catalog,
            user_histories=user_histories,
            user_count=len(test_truth),
            item_count=len(catalog),
        )
