from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.data import MovieLensDatasetLoader


GENRE_LABELS = {
    "Action": "动作",
    "Adventure": "冒险",
    "Animation": "动画",
    "Children": "儿童",
    "Comedy": "喜剧",
    "Crime": "犯罪",
    "Documentary": "纪录片",
    "Drama": "剧情",
    "Fantasy": "奇幻",
    "Film-Noir": "黑色电影",
    "Horror": "恐怖",
    "Musical": "歌舞",
    "Mystery": "悬疑",
    "Romance": "爱情",
    "Sci-Fi": "科幻",
    "Thriller": "惊悚",
    "War": "战争",
    "Western": "西部",
}


class MovieRecommendationGraph:
    """Build a compact recommendation-domain graph from MovieLens interactions."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def payload(self, movie_limit: int = 12, genre_limit: int = 6) -> dict[str, Any]:
        dataset = MovieLensDatasetLoader(self.dataset_dir).load()
        interactions = dataset.interactions_df
        movie_counts = interactions.groupby("item_id").size().sort_values(ascending=False)
        movie_ids = [int(item_id) for item_id in movie_counts.head(movie_limit).index]

        selected = interactions[interactions["item_id"].isin(movie_ids)]
        user_overlap = selected.groupby("user_id")["item_id"].nunique().sort_values(ascending=False)
        eligible = user_overlap[(user_overlap >= 2) & (user_overlap < movie_limit - 2)]
        representative_user = int((eligible if not eligible.empty else user_overlap).index[0])
        liked_movie_ids = {
            int(item_id)
            for item_id in selected[selected["user_id"] == representative_user]["item_id"].tolist()
        }

        user_history = interactions[interactions["user_id"] == representative_user]
        genre_affinity: Counter[str] = Counter()
        for genres in user_history["genres"].tolist():
            genre_affinity.update(str(genres).split("|"))

        unseen_movie_ids = [item_id for item_id in movie_ids if item_id not in liked_movie_ids]
        recommended_movie_ids = set(
            sorted(
                unseen_movie_ids,
                key=lambda item_id: (
                    -sum(genre_affinity[genre] for genre in dataset.catalog[item_id]["genres"].split("|")),
                    -int(movie_counts.get(item_id, 0)),
                    item_id,
                ),
            )[:3]
        )

        genre_counts: Counter[str] = Counter()
        for item_id in movie_ids:
            genre_counts.update(dataset.catalog[item_id]["genres"].split("|"))
        selected_genres = [genre for genre, _count in genre_counts.most_common(genre_limit)]

        nodes: list[dict[str, Any]] = [
            {
                "id": f"user::{representative_user}",
                "label": f"用户 {representative_user}",
                "type": "User",
                "x": 0.08,
                "y": 0.5,
                "detail": f"{len(user_history)} 条高评分记录",
            }
        ]
        edges: list[dict[str, str]] = []

        for index, item_id in enumerate(movie_ids):
            catalog_item = dataset.catalog[item_id]
            column = index % 2
            row = index // 2
            is_recommended = item_id in recommended_movie_ids
            nodes.append(
                {
                    "id": f"movie::{item_id}",
                    "label": catalog_item["title"],
                    "type": "Movie",
                    "x": 0.36 + (column * 0.26),
                    "y": 0.1 + (row * 0.16),
                    "detail": f"{int(movie_counts.get(item_id, 0))} 次高评分",
                    "recommended": is_recommended,
                }
            )
            if item_id in liked_movie_ids:
                edges.append(
                    {"source": f"user::{representative_user}", "target": f"movie::{item_id}", "type": "liked"}
                )
            if is_recommended:
                edges.append(
                    {
                        "source": f"user::{representative_user}",
                        "target": f"movie::{item_id}",
                        "type": "recommended",
                    }
                )
            for genre in catalog_item["genres"].split("|"):
                if genre in selected_genres:
                    edges.append(
                        {"source": f"movie::{item_id}", "target": f"genre::{genre}", "type": "belongs_to"}
                    )

        genre_step = 0.8 / max(len(selected_genres) - 1, 1)
        for index, genre in enumerate(selected_genres):
            nodes.append(
                {
                    "id": f"genre::{genre}",
                    "label": GENRE_LABELS.get(genre, genre),
                    "type": "Genre",
                    "x": 0.92,
                    "y": 0.1 + (index * genre_step),
                    "detail": genre,
                }
            )

        return {
            "dataset": {
                "source": "GroupLens MovieLens Latest Small",
                "source_url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
                "license": "Research use; see bundled README.txt",
                "users": dataset.user_count,
                "movies": dataset.item_count,
                "positive_interactions": len(interactions),
            },
            "focus": {
                "user_id": representative_user,
                "liked_movies": len(liked_movie_ids),
                "recommended_movies": len(recommended_movie_ids),
            },
            "nodes": nodes,
            "edges": edges,
        }
