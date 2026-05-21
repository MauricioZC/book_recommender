"""
Gradio frontend for the Book Recommender project.

Wraps the existing pipeline in src/:
    receive_prompt -> enhance_prompt -> encode_prompt
    -> similarity_score -> get_table
    -> (optional) add_reason -> (optional) generate_response

Run with:
    uv run frontend.py
"""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from src import (
    receive_prompt,
    enhance_prompt,
    encode_prompt,
    similarity_score,
    get_table,
)

# These two functions are part of the planned pipeline but may not be
# implemented yet by the team. Import lazily so the UI works either way.
try:
    from src import add_reason  # type: ignore
except ImportError:
    add_reason = None

try:
    from src import generate_response  # type: ignore
except ImportError:
    generate_response = None


# ---------------------------------------------------------------------------
# Load data once at startup (avoid reloading on every request)
# ---------------------------------------------------------------------------
BOOKS_CSV = Path("books.csv")
EMBEDDINGS_NPY = Path("embeddings.npy")

_LOAD_ERROR: str | None = None
books_df: pd.DataFrame | None = None
embeddings_matrix: np.ndarray | None = None

try:
    if not BOOKS_CSV.exists():
        raise FileNotFoundError(f"{BOOKS_CSV} not found in working directory.")
    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_NPY} not found. Run: uv run build_embeddings.py"
        )
    books_df = pd.read_csv(BOOKS_CSV).fillna("")
    embeddings_matrix = np.load(EMBEDDINGS_NPY)
except Exception as e:  # noqa: BLE001
    _LOAD_ERROR = str(e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fallback_answer(top_books: pd.DataFrame, query: str) -> str:
    """Used until generate_response() is implemented by the team."""
    if top_books is None or len(top_books) == 0:
        return "No matching books found. Try a different query."

    lines = [f"Here are {len(top_books)} books that match **\"{query}\"**:\n"]
    for i, row in top_books.iterrows():
        title = row.get("title", "Unknown title")
        author = row.get("authors", "Unknown author")
        score = row.get("similarity_score", None)
        score_str = f" _(similarity: {score:.3f})_" if score is not None else ""
        lines.append(f"{i + 1}. **{title}** — {author}{score_str}")
    lines.append(
        "\n_Note: this is a placeholder summary. "
        "The LLM-generated answer will appear here once "
        "`generate_response()` is added to the backend._"
    )
    return "\n".join(lines)


def _select_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the columns most useful to show in the UI, in a friendly order."""
    preferred = [
        "title",
        "authors",
        "categories",
        "average_rating",
        "published_year",
        "similarity_score",
        "reason",  # added by add_reason() if available
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols] if cols else df


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------
def recommend(
    query: str,
    top_n: float,
    genre: str,
    language: str,
    min_year,
    max_year,
    min_rating,
):
    if _LOAD_ERROR:
        return f"❌ Failed to load data: {_LOAD_ERROR}", pd.DataFrame(), ""

    if not query or not query.strip():
        return "⚠️ Please enter a query.", pd.DataFrame(), ""

    # Build filter kwargs (skip blanks / zeros so they don't become hard filters)
    filters: dict = {}
    if genre and genre.strip():
        filters["genre"] = genre.strip()
    if language and language.strip():
        filters["language"] = language.strip()
    if min_year:
        filters["min_year"] = int(min_year)
    if max_year:
        filters["max_year"] = int(max_year)
    if min_rating:
        filters["min_rating"] = float(min_rating)

    try:
        # 1. Receive / validate
        user_prompt = receive_prompt(query, **filters)

        # 2. Enhance (LLM expands keywords)
        enhanced = enhance_prompt(user_prompt)

        # 3. Encode the enhanced query
        query_embedding = encode_prompt(enhanced)

        # 4. Cosine similarity vs. precomputed book embeddings
        scores = similarity_score(query_embedding, embeddings_matrix)

        # 5. Top-N table
        top_books = get_table(books_df, scores, top_n=int(top_n))

        # 6. (Optional) add a per-book explanation
        if add_reason is not None:
            try:
                top_books = add_reason(top_books, user_prompt.query)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] add_reason failed, continuing without reasons: {e}")

        # 7. (Optional) final LLM-written answer
        if generate_response is not None:
            try:
                final_answer = generate_response(top_books, user_prompt)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] generate_response failed, using fallback: {e}")
                final_answer = _fallback_answer(top_books, query)
        else:
            final_answer = _fallback_answer(top_books, query)

        display_df = _select_display_columns(top_books)
        status = f"✓ Enhanced query: _{enhanced}_"

        return final_answer, display_df, status

    except ValueError as e:
        return f"⚠️ {e}", pd.DataFrame(), ""
    except Exception as e:  # noqa: BLE001
        return f"❌ Something went wrong: {e}", pd.DataFrame(), ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Book Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Book Recommender")
    gr.Markdown(
        "*Find books based on your mood, interests, and reading preferences.*"
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="e.g. A dark fantasy about power and sacrifice",
                lines=3,
            )
        with gr.Column(scale=1):
            top_n = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of recommendations",
            )

    with gr.Accordion("Optional filters", open=False):
        with gr.Row():
            genre = gr.Textbox(label="Genre", placeholder="fantasy")
            language = gr.Textbox(label="Language", placeholder="en")
        with gr.Row():
            min_year = gr.Number(label="Min year", value=None, precision=0)
            max_year = gr.Number(label="Max year", value=None, precision=0)
            min_rating = gr.Number(
                label="Min rating (0–5)", value=None, minimum=0, maximum=5
            )

    submit = gr.Button("Recommend books", variant="primary", size="lg")

    gr.Markdown("### 💡 AI recommendation")
    answer_output = gr.Markdown()

    gr.Markdown("### 📖 Recommended books")
    table_output = gr.Dataframe(interactive=False, wrap=True)

    status_output = gr.Markdown()

    submit.click(
        fn=recommend,
        inputs=[
            query_input,
            top_n,
            genre,
            language,
            min_year,
            max_year,
            min_rating,
        ],
        outputs=[answer_output, table_output, status_output],
    )

    gr.Examples(
        examples=[
            ["A dark fantasy about power and sacrifice", 5, "fantasy", "", None, None, 4.0],
            ["A heartbreaking love story set in a small town", 5, "", "", None, None, None],
            ["Mind-bending sci-fi with time travel", 5, "science fiction", "", None, None, None],
            ["A cozy mystery for a rainy weekend", 5, "mystery", "", None, None, None],
        ],
        inputs=[query_input, top_n, genre, language, min_year, max_year, min_rating],
    )


if __name__ == "__main__":
    demo.launch()