import numpy as np
import pandas as pd

from src import (
    receive_prompt,
    enhance_prompt,
    encode_prompt,
    similarity_score,
    get_table,
    add_reason,
)


def main() -> None:
    # --- 1. Receive & validate user input ---
    user_prompt = receive_prompt(
        "A dark fantasy about power and sacrifice",
        genre="fantasy",
        min_rating=4.0,
    )

    # --- 2. Enhance the prompt (LLM expands query for richer semantic matching) ---
    enhanced = enhance_prompt(user_prompt)
    print(f"Enhanced prompt:\n{enhanced}\n")

    # --- 3. Encode the enhanced prompt with the same model used to build embeddings ---
    query_embedding = encode_prompt(enhanced)

    # --- 4. Load books metadata + pre-built embeddings matrix ---
    books_df = pd.read_csv("books.csv").fillna('')
    embeddings_matrix = np.load("embeddings.npy")   # shape: (6810, 384)

    # --- 5. Compute cosine similarity and get top-N books ---
    scores = similarity_score(query_embedding, embeddings_matrix)
    top_books = get_table(books_df, scores, top_n=5)

    # --- Teammates: pass top_books to add_reason() → generate_response() → print/Gradio ---
    print(top_books[['title', 'authors', 'categories', 'similarity_score']].to_string(index=False))


if __name__ == "__main__":
    main()
