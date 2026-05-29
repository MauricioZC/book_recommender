"""Visualise the book_recommender embeddings with Plotly.

Loads the pre-computed ``embeddings.npy`` (shape (N, 384)) alongside
``books.csv`` metadata, reduces the vectors to 2-D or 3-D, and renders an
interactive scatter plot coloured by book category.

Two public functions:

    plot_embeddings_2d(...)   ->  go.Figure
    plot_embeddings_3d(...)   ->  go.Figure

Mirrors the t-SNE + plotly.graph_objects approach from the course notebook,
adapted to the project's CSV + NPY data store. Run directly to open both:

    uv run python visualize_embeddings.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BOOKS_CSV = Path("books.csv")
EMBEDDINGS_NPY = Path("embeddings.npy")

DEFAULT_MAX_CATEGORIES = 10
DEFAULT_RANDOM_STATE = 42

# A calm qualitative palette; cycled if there are more categories than colours.
PALETTE = [
    "#e95a0c", "#1d9e75", "#534ab7", "#d4537e", "#185fa5",
    "#ba7517", "#0f6e56", "#993c1d", "#72243e", "#444441",
]
OTHER_COLOR = "#b4b2a9"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(
    books_csv: Path = BOOKS_CSV,
    embeddings_npy: Path = EMBEDDINGS_NPY,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load metadata + vectors and check row-count parity."""
    if not books_csv.exists():
        raise FileNotFoundError(f"{books_csv} not found.")
    if not embeddings_npy.exists():
        raise FileNotFoundError(
            f"{embeddings_npy} not found. Build it with "
            "scripts/build_embeddings.py first."
        )

    df = pd.read_csv(books_csv).fillna("")
    vectors = np.load(embeddings_npy)

    if len(df) != len(vectors):
        raise ValueError(
            f"{books_csv} has {len(df)} rows but {embeddings_npy} has "
            f"{len(vectors)} vectors. Regenerate the embeddings."
        )
    return df, vectors


# --------------------------------------------------------------------------- #
# Dimensionality reduction
# --------------------------------------------------------------------------- #
def reduce_dimensions(
    vectors: np.ndarray,
    n_components: int,
    method: str = "tsne",
    perplexity: float = 30.0,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Project high-dim vectors down to ``n_components`` dimensions.

    method='tsne' (default) preserves local neighbourhood structure — best for
    the final visual. method='pca' is fast and deterministic — handy while
    iterating on a large corpus.
    """
    method = method.lower()
    if method == "pca":
        return PCA(n_components=n_components, random_state=random_state).fit_transform(vectors)
    if method == "tsne":
        # perplexity must be < n_samples; clamp for safety on small corpora.
        safe_perplexity = float(min(perplexity, max(5, len(vectors) - 1)))
        tsne = TSNE(
            n_components=n_components,
            perplexity=safe_perplexity,
            init="pca",
            random_state=random_state,
        )
        return tsne.fit_transform(vectors)
    raise ValueError(f"Unknown method {method!r}; use 'tsne' or 'pca'.")


# --------------------------------------------------------------------------- #
# Category grouping + hover text
# --------------------------------------------------------------------------- #
def _primary_category(value: Any) -> str:
    """Take the first comma-separated category as the book's primary genre."""
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    return text.split(",")[0].strip() or "Unknown"


def grouped_categories(df: pd.DataFrame, max_categories: int) -> pd.Series:
    """Return a per-row category label, collapsing the long tail into 'Other'.

    Book taxonomies are large, so plotting every raw category yields an
    unreadable legend. We keep the most common ``max_categories`` and bucket
    the rest as 'Other'.
    """
    column = "categories" if "categories" in df.columns else None
    if column is None:
        return pd.Series(["All books"] * len(df), index=df.index)

    primary = df[column].map(_primary_category)
    top = primary.value_counts().head(max_categories).index
    return primary.where(primary.isin(top), other="Other")


def _truncate(text: Any, limit: int = 120) -> str:
    text = str(text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def hover_texts(df: pd.DataFrame) -> np.ndarray:
    """Build an HTML hover string per book (title / author / category / blurb)."""
    titles = df.get("title", pd.Series([""] * len(df)))
    authors = df.get("authors", pd.Series([""] * len(df)))
    cats = df.get("categories", pd.Series([""] * len(df)))
    descs = df.get("description", pd.Series([""] * len(df)))

    out = []
    for title, author, cat, desc in zip(titles, authors, cats, descs):
        lines = [f"<b>{str(title).strip() or 'Untitled'}</b>"]
        if str(author).strip():
            lines.append(f"by {str(author).strip()}")
        if str(cat).strip():
            lines.append(f"<i>{_truncate(cat, 60)}</i>")
        if str(desc).strip():
            lines.append(_truncate(desc, 140))
        out.append("<br>".join(lines))
    return np.array(out, dtype=object)


def _color_map(ordered_categories: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    palette_idx = 0
    for cat in ordered_categories:
        if cat == "Other":
            mapping[cat] = OTHER_COLOR
        else:
            mapping[cat] = PALETTE[palette_idx % len(PALETTE)]
            palette_idx += 1
    return mapping


def _ordered_categories(labels: pd.Series) -> list[str]:
    """Most-frequent first, with 'Other' always pushed to the end."""
    counts = labels.value_counts()
    ordered = [c for c in counts.index if c != "Other"]
    if "Other" in counts.index:
        ordered.append("Other")
    return ordered


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_embeddings_2d(
    df: pd.DataFrame | None = None,
    vectors: np.ndarray | None = None,
    method: str = "tsne",
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    marker_size: int = 5,
    show: bool = True,
    save_html: str | None = None,
) -> go.Figure:
    """2-D scatter of the embeddings, one legend entry per category."""
    if df is None or vectors is None:
        df, vectors = load_data()

    coords = reduce_dimensions(vectors, n_components=2, method=method)
    labels = grouped_categories(df, max_categories)
    hover = hover_texts(df)
    ordered = _ordered_categories(labels)
    colors = _color_map(ordered)

    fig = go.Figure()
    for cat in ordered:
        mask = (labels == cat).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=f"{cat} ({int(mask.sum())})",
                marker=dict(size=marker_size, color=colors[cat], opacity=0.75),
                text=hover[mask],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Book embeddings — 2-D ({method.upper()})",
        template="plotly_white",
        paper_bgcolor="#f5f0ea",
        plot_bgcolor="#f5f0ea",
        width=900,
        height=650,
        legend_title_text="Category",
        margin=dict(r=20, b=10, l=10, t=50),
    )
    fig.update_xaxes(title_text="dim 1", zeroline=False)
    fig.update_yaxes(title_text="dim 2", zeroline=False)

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig


def plot_embeddings_3d(
    df: pd.DataFrame | None = None,
    vectors: np.ndarray | None = None,
    method: str = "tsne",
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    marker_size: int = 4,
    show: bool = True,
    save_html: str | None = None,
) -> go.Figure:
    """3-D scatter of the embeddings, one legend entry per category."""
    if df is None or vectors is None:
        df, vectors = load_data()

    coords = reduce_dimensions(vectors, n_components=3, method=method)
    labels = grouped_categories(df, max_categories)
    hover = hover_texts(df)
    ordered = _ordered_categories(labels)
    colors = _color_map(ordered)

    fig = go.Figure()
    for cat in ordered:
        mask = (labels == cat).to_numpy()
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=f"{cat} ({int(mask.sum())})",
                marker=dict(size=marker_size, color=colors[cat], opacity=0.8),
                text=hover[mask],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Book embeddings — 3-D ({method.upper()})",
        template="plotly_white",
        paper_bgcolor="#f5f0ea",
        plot_bgcolor="#f5f0ea",
        width=1000,
        height=750,
        legend_title_text="Category",
        scene=dict(xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3"),
        margin=dict(r=10, b=10, l=10, t=50),
    )

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig


if __name__ == "__main__":
    books_df, embedding_vectors = load_data()
    print(f"Loaded {len(books_df):,} books with {embedding_vectors.shape[1]}-dim vectors.")
    plot_embeddings_2d(books_df, embedding_vectors)
    plot_embeddings_3d(books_df, embedding_vectors)