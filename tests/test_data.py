from app.data import MovieLensDatasetLoader


def test_loader_creates_train_test_split(test_settings):
    dataset = MovieLensDatasetLoader(test_settings.data_dir / "movielens_sample").load()

    assert not dataset.train_df.empty
    assert dataset.user_count == 12
    assert dataset.item_count == 15
    assert len(dataset.test_truth) == 12
    assert set(dataset.train_df.columns) >= {"user_id", "item_id", "rating", "timestamp", "title", "genres"}
    assert all(dataset.test_truth[user_id] for user_id in dataset.test_truth)
