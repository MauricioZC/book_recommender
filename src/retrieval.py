import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def similarity_score(
    query_embedding: np.ndarray,
    embeddings_matrix: np.ndarray,
) -> np.ndarray:
    """
    Cosine similarity between a single query vector and a (n_books, dim) matrix.
    Returns a 1-D array of scores aligned with embeddings_matrix row order.
    """
    scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)[0]
    return scores


def get_table(
    books_df: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Returns the top-N rows from books_df ranked by cosine similarity score.
    Adds a 'similarity_score' column (float, 0–1).
    """
    top_indices = np.argsort(scores)[::-1][:top_n]
    result = books_df.iloc[top_indices].copy()
    result["similarity_score"] = scores[top_indices].round(4)
    return result.reset_index(drop=True)
