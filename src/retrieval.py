import numpy as np
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity


def similarity_score(query_embedding: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
    """
    Compares the user's query embedding with every book embedding.
    Returns one similarity score per book.
    """
    query_embedding = np.asarray(query_embedding)
    embeddings_matrix = np.asarray(embeddings_matrix)

    scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings_matrix
    )[0]

    return scores


def get_table(
    books_df: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Takes the similarity scores and returns the top-N matching books.
    """
    top_indices = np.argsort(scores)[::-1][:top_n]

    result = books_df.iloc[top_indices].copy()
    result["similarity_score"] = scores[top_indices].round(4)

    max_score = result["similarity_score"].max()
    result["match_score"] = (
        result["similarity_score"] / max_score * 100
    ).round(0).astype(int)

    return result.reset_index(drop=True)


def add_reason(top_books: pd.DataFrame, user_query: str, model="gpt-5.4-nano") -> pd.DataFrame:
    result = top_books.copy()

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None

    reasons = []

    for _, row in result.iterrows():
        fallback_reason = (
            f"This book matches your request because it is related to "
            f"{row.get('categories', 'the requested topic')} and its description aligns with your search."
        )

        if client is None:
            reasons.append(fallback_reason)
            continue

        prompt = f"""
User query: {user_query}

Book:
Title: {row.get("title", "")}
Author: {row.get("authors", "")}
Genre: {row.get("categories", "")}
Description: {row.get("description", "")}

Write one short reason explaining why this book matches the user query.
Only use the book information provided.
"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You explain book recommendations clearly and briefly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=80,
            )

            reason = response.choices[0].message.content.strip()
            reasons.append(reason)

        except Exception:
            reasons.append(fallback_reason)

    result["reason"] = reasons
    return result