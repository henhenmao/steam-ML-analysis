# Steam Video Game Data Analysis

A machine learning analysis of ~83,000 Steam games, exploring what measurable game
features (price, genre, playtime, reviews, release timing) can predict about a game's
success and player engagement.

Final project for a data science course, completed by a 4-person team

Full write-up: [`Data Science Final Project.pdf`](./Data%20Science%20Final%20Project.pdf)

## Dataset

[Steam Games Dataset 2025 (Cleaned)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) — Kaggle, published by Artermiloff.

~83,000 games with 30+ attributes; we used a subset including `price`, `genres`,
`release_date`, `positive`/`negative` reviews, `estimated_owners`, `average_playtime_forever`,
`average_playtime_2weeks`, and `peak_ccu`. Rows missing key fields (price, reviews, playtime,
CCU) were dropped. `estimated_owners` (a string range like `"20000-50000"`) was converted to
its numeric midpoint; `genres` was one-hot/multi-label encoded; numeric features were
standard-scaled.

## Questions & Results

For each question, we built a Linear Regression/Logistic Regression baseline, then tuned a KNN model with `GridSearchCV` / cross-validation, and compared.

| # | Question | Best model | Result |
|---|----------|-----------|--------|
| 1 | Predict positive vs. negative review (≥70% positive ratio) from price, playtime, CCU, ratings | KNN (manhattan, k=3) | 96.71% accuracy vs. 68.69% for Logistic Regression |
| 2 | Predict release year from estimated owners, price, playtime | KNN (euclidean, k=19) | RMSE 2.77 vs. 3.53 for Linear Regression |
| 3 | Predict peak concurrent users (CCU) from price, genre, reviews, playtime, release date | KNN (manhattan, k=2) | RMSE 532.65, MAE 33.97 vs. RMSE 1046.07, MAE 210.59 for Linear Regression |
| 4 | Predict total owners from reviews, price, genre, platform support | KNN (euclidean, k=3) | Lower error than Linear Regression, but both models had very high absolute error. Ownership is poorly explained by these features alone |
| 5 | Predict total review count from price and genre | KNN (k=12) | RMSE ~21,779 |

Takeaway: KNN consistently outperformed linear models, suggesting the relationships
between game features and engagement/success metrics are non-linear and locally structured.
Ownership and review-count prediction remained hard, likely because popularity is driven by
factors outside this dataset (marketing, franchise reputation, social trends).

## Repo

```
Data Science Final Project.pdf   # Full written report
project-code/
├── q1.py   # Review sentiment classification (Logistic Regression vs. KNN)
├── q2.py   # Release year prediction (Linear Regression vs. KNN)
├── q3.py   # Peak CCU prediction (Linear Regression vs. KNN)
├── q4.py   # Total owners prediction (Linear Regression vs. KNN)
└── q5.py   # Review count prediction (KNN)
```

Each script was exported from a Google Colab notebook and fetches the dataset directly from Kaggle via `kagglehub`.

## Running

```bash
pip install pandas numpy scikit-learn matplotlib kagglehub
python project-code/q1.py   # q2.py, q3.py, etc.
```
