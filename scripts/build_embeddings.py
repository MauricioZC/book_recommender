"""
One-time script: reads books.csv, computes sentence-transformer embeddings
for each book, and saves the result as a NumPy .npy file.

The same model and clean_text logic used here MUST match encoder.py so that
query vectors and book vectors live in the same embedding space.

Usage:
    uv run scripts/build_embeddings.py
    uv run scripts/build_embeddings.py --input books.csv --output embeddings.npy
"""

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def build_combined_text(row: pd.Series) -> str:
    parts = [
        str(row.get('title', '') or ''),
        str(row.get('authors', '') or ''),
        str(row.get('categories', '') or ''),
        str(row.get('description', '') or ''),
    ]
    return ' '.join(p for p in parts if p.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="books.csv")
    parser.add_argument("--output", default="embeddings.npy")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.fillna('')
    print(f"Loaded {len(df)} books from '{args.input}'")

    texts = df.apply(build_combined_text, axis=1).tolist()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings (this may take a minute)...")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    np.save(args.output, embeddings)
    print(f"Saved {embeddings.shape} matrix to '{args.output}'")
    print(f"  → Load with: np.load('{args.output}')")


if __name__ == "__main__":
    main()
