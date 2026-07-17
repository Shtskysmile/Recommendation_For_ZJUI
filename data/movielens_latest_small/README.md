# MovieLens Latest Small derived runtime dataset

Source: GroupLens MovieLens Latest Small

- Official download: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- Downloaded archive: `data/ml-latest-small.zip`
- Archive SHA-256: `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`
- Original dataset: 100,836 ratings, 9,742 movies, 610 anonymized users
- Runtime subset: 300 most-rated movies, retaining users with at least two ratings of 4.0 or higher

The runtime subset preserves the original user ids, movie ids, ratings, timestamps,
titles, and genres. It keeps local candidate validation responsive while using real
MovieLens interactions rather than synthetic records.

The original dataset README and its research-use license are included inside
`data/ml-latest-small.zip`. Publications must acknowledge the MovieLens dataset:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems 5, 4.
https://doi.org/10.1145/2827872
