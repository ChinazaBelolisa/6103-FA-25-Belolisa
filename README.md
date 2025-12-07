# Movie Recommender System

A comparative study of recommendation approaches using the MovieLens 100K dataset. This project demonstrates why collaborative filtering outperforms traditional classification for personalized movie recommendations.

## Key Results

| Method | Hit Rate@10 | Precision@10 | Recall@10 |
|--------|-------------|--------------|-----------|
| Popularity Baseline | 64.8% | 13.3% | 14.1% |
| User-KNN | 80.8% | 22.7% | 24.5% |
| **Item-KNN** | **81.3%** | **23.0%** | **24.4%** |

**Item-KNN achieved 81.3% hit rate** — a 16.5 percentage point improvement over the popularity baseline.

## Project Overview

### The Problem
Given a user's rating history, can we predict which movies they'll enjoy? A naive approach (recommending popular movies to everyone) works okay, but fails to personalize.

### Two Approaches Compared

**1. Classification (70% accuracy)**
- Engineered 23 features per user-movie pair
- Trained Logistic Regression, SVM, and KNN classifiers
- **Result:** Models learned "popular movies get liked by generous raters" — not actual personalization

**2. Collaborative Filtering (81.3% hit rate)**
- Used rating matrix directly with Item-KNN
- No feature engineering needed
- **Result:** Captured personal taste through rating pattern similarity

### Why Classification Failed

The logistic regression coefficients revealed the problem:

| Feature | Coefficient |
|---------|-------------|
| movie_avg_rating | 1.72 |
| user_avg_rating | 1.58 |
| Genre features | ~0.01 |

The model took a shortcut — it learned that highly-rated movies get liked by users who rate generously, ignoring actual preferences.

### Why Collaborative Filtering Worked

Item-KNN doesn't need to "understand" movies. It discovers that users who liked Movie A also tend to like Movie B, without knowing why. The algorithm doesn't care that Star Wars is sci-fi — it just observes that Toy Story fans rate it highly too.

## Project Structure

```
movie-recommender-system/
├── data/
│   ├── raw/                    # Original MovieLens files
│   │   ├── u.data              # 100K ratings
│   │   ├── u.item              # Movie metadata
│   │   ├── u.user              # User demographics
│   │   └── ...
│   └── processed/
│       └── movielens.db        # SQLite database
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_modeling.ipynb
├── src/
│   └── run_pipeline.py         # Item-KNN recommender pipeline
└── README.md
```

## Quick Start

### Run the Recommender

```bash
python src/run_pipeline.py
```

This launches an interactive CLI where you can:
1. Find movies similar to a given movie
2. Get personalized recommendations for any user

### Example Output

```
Movies similar to 'Toy Story (1995)':
  1. Aladdin (1992)                    (similarity: 0.874)
  2. Lion King, The (1994)             (similarity: 0.861)
  3. Beauty and the Beast (1991)       (similarity: 0.847)
  ...

Recommendations for User 42:
  1. Shawshank Redemption, The (1994)  (score: 0.923)
  2. Pulp Fiction (1994)               (score: 0.891)
  3. Forrest Gump (1994)               (score: 0.856)
  ...
```

## Technical Details

### Item-KNN Algorithm

1. Build user-item matrix (movies × users)
2. Compute cosine similarity between movie rating vectors
3. For recommendations: find movies similar to what the user already likes
4. Aggregate similarity scores and rank

### Evaluation Protocol

- 80/20 train/test split
- For each user: identify test set "likes" (rating ≥ 4)
- Generate top-10 recommendations using only training data
- Measure hit rate, precision, and recall

### Why Item-KNN Beat User-KNN

Movie relationships are stable — The Godfather will always be similar to Goodfellas. User preferences evolve over time, making user-user similarities less reliable.

## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

### Acknowledgments

The MovieLens dataset was collected by the **GroupLens Research Project** at the University of Minnesota. We thank the GroupLens team for making this data freely available for research and education.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19 (December 2015), 19 pages. DOI: https://doi.org/10.1145/2827872

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Ratings | 100,000 |
| Users | 943 |
| Movies | 1,682 |
| Sparsity | 93.7% |
| Rating Scale | 1-5 stars |
| Collection Period | 1997-1998 |

## Requirements

```
pandas
numpy
scipy
scikit-learn
```

Install with:
```bash
pip install pandas numpy scipy scikit-learn
```

## Future Work

- **Neural Collaborative Filtering:** Deep learning for complex user-item interactions
- **Hybrid Methods:** Combine CF with content features to address cold-start
- **Explainability:** Add reasoning for why movies are recommended

## Author

**Chinaza Belolisa**  
The George Washington University  
M.S. Data Science

## License

This project is for educational purposes. The MovieLens dataset is provided by GroupLens under their [terms of use](https://grouplens.org/datasets/movielens/).
