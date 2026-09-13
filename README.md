# Expedia Hotel Search — Learning to Rank

Ranking hotel listings for a search query using XGBoost's LambdaMART
implementation, on the
[Expedia personalised sort](https://www.kaggle.com/competitions/expedia-personalized-sort)
dataset. Built as coursework in data mining.

Ranking is not classification with extra steps. A classifier asks *"will this
hotel be booked?"* one row at a time; a ranker asks *"given these 30 hotels
returned for search #48291, what order should they be in?"* The unit of
prediction is the whole result list, and a model that scores every hotel
plausibly but orders them badly is useless.

## How it works

**Group structure is the whole thing.** The dataset is grouped by `srch_id` —
one search, many candidate properties. The ranker has to be told those
boundaries explicitly:

```python
groups_size_train = OrderedDict(...)   # rows per srch_id, in order

model = XGBRanker(
    objective='rank:ndcg',
    learning_rate=.1,
    max_depth=7,
    n_estimators=1500)

model.fit(X_train, y_train, group=list(groups_size_train.values()))
```

The `group` argument tells XGBoost which rows compete with each other. Get it
wrong — or leave the frame unsorted so the group sizes no longer line up with
the rows — and training silently optimises comparisons between hotels from
unrelated searches. It won't error; it will just produce a bad model.

**`rank:ndcg`** makes the objective position-aware. Normalised Discounted
Cumulative Gain discounts relevance logarithmically by rank, which encodes
something true about search: users read from the top, so an error at position 2
costs far more than the same error at position 25.

**Prediction is a sort, not a label.** The model emits a relevance score per
row, and the ranking comes from ordering within each search:

```python
df_pred.sort_values(['srch_id', 'score'], ascending=[True, False])
```

**Supporting work**, in the other two notebooks:

- `PCA.ipynb` — dimensionality reduction across the feature set, which spans
  property attributes, price, location desirability scores and competitor
  pricing signals.
- `Resampling.ipynb` — the dataset is severely imbalanced, since almost every
  hotel shown is not booked. Undersampling the majority class with
  `RandomUnderSampler` keeps the bookings from being drowned out entirely.

## Results

Not published. The notebooks train the ranker and produce ordered predictions,
but no NDCG evaluation was recorded against a held-out split, and the dataset
is not committed, so there is nothing here to reproduce a score from. The
`rank:ndcg` above is the *training objective*, not a measured result — worth
saying plainly, because the two are easy to confuse.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Ranking.ipynb
```

Download the training data from the
[competition page](https://www.kaggle.com/competitions/expedia-personalized-sort/data)
and update the path in the loading cell. Note the ordering requirement above:
the frame must be sorted by `srch_id` before group sizes are computed.

## Stack

Python, XGBoost (`XGBRanker`), scikit-learn, imbalanced-learn, pandas, NumPy,
matplotlib, seaborn.
