from __future__ import annotations

import csv
import tempfile
from html import escape
from pathlib import Path
from typing import Any

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

try:
    from src import add_reason  # type: ignore
except ImportError:
    add_reason = None  # type: ignore


BOOKS_CSV = Path("books.csv")
EMBEDDINGS_NPY = Path("embeddings.npy")
DEFAULT_TOP_N = 5
MAX_RESULT_SLOTS = 5

_LOAD_ERROR: str | None = None
books_df: pd.DataFrame | None = None
embeddings_matrix: np.ndarray | None = None

try:
    if not BOOKS_CSV.exists():
        raise FileNotFoundError(f"{BOOKS_CSV} not found in working directory.")
    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            "embeddings.npy not found. Run: "
            "uv run python scripts/build_embeddings.py --input books.csv --output embeddings.npy"
        )

    books_df = pd.read_csv(BOOKS_CSV).fillna("")
    embeddings_matrix = np.load(EMBEDDINGS_NPY)

    if len(books_df) != len(embeddings_matrix):
        raise ValueError(
            f"books.csv has {len(books_df)} rows but embeddings.npy has "
            f"{len(embeddings_matrix)} vectors. Regenerate embeddings.npy."
        )
except Exception as e:  # noqa: BLE001
    _LOAD_ERROR = str(e)


PLACEHOLDER_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 180'>"
    "<rect width='120' height='180' rx='10' fill='%23ede6d6'/>"
    "<rect x='30' y='38' width='60' height='104' fill='none' "
    "stroke='%23b8a98a' stroke-width='2.5'/>"
    "<line x1='40' y1='62' x2='80' y2='62' stroke='%23b8a98a' stroke-width='2'/>"
    "<line x1='40' y1='78' x2='80' y2='78' stroke='%23b8a98a' stroke-width='2'/>"
    "<line x1='40' y1='94' x2='70' y2='94' stroke='%23b8a98a' stroke-width='2'/>"
    "</svg>"
)


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return not pd.isna(value)
    except Exception:  # noqa: BLE001
        return True


def _safe_float(value: Any) -> float | None:
    try:
        if not _has_value(value):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _clean_int(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return str(int(number))


def _book_key(book: dict | pd.Series) -> str:
    title = str(book.get("title", "")).strip().lower()
    authors = str(book.get("authors", "")).strip().lower()
    isbn = str(book.get("isbn13", "") or book.get("isbn10", "")).strip().lower()
    return isbn or f"{title}::{authors}"


def render_stars(rating: Any) -> str:
    r = _safe_float(rating)
    if r is None or r <= 0:
        return ""
    r = max(0.0, min(5.0, r))
    full = int(r)
    half = 1 if (r - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def truncate(text: Any, limit: int = 300) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def render_badge(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    return f'<span class="bc-pill">{escape(text)}</span>'


def render_card(idx: int, row: pd.Series | dict, compact: bool = False) -> str:
    title = escape(str(row.get("title", "Untitled")).strip() or "Untitled")
    subtitle_raw = str(row.get("subtitle", "") or "").strip()
    subtitle = escape(subtitle_raw)
    authors = escape(str(row.get("authors", "Unknown author")).strip() or "Unknown author")
    categories_raw = str(row.get("categories", "") or "").strip()
    description = escape(truncate(row.get("description", ""), 220 if compact else 300))
    year = _clean_int(row.get("published_year"))
    pages = _clean_int(row.get("num_pages"))
    ratings_count = _clean_int(row.get("ratings_count"))
    thumbnail = str(row.get("thumbnail", "") or "").strip() or PLACEHOLDER_SVG

    rating_val = _safe_float(row.get("average_rating"))
    rating_num = f"{rating_val:.2f}" if rating_val is not None else ""
    stars = render_stars(rating_val)

    match_val = _safe_float(row.get("match_score"))
    sim_val = _safe_float(row.get("similarity_score"))

    match_html = ""
    if match_val is not None:
        match_html = f"""
        <div class="bc-match" aria-label="Match score">
          <strong>{int(round(match_val))}%</strong>
          <span>match</span>
        </div>
        """

    sim_html = f'<span class="bc-chip">similarity {sim_val:.3f}</span>' if sim_val is not None else ""

    meta_items = []
    if stars:
        meta_items.append(
            f'<span class="bc-stars">{stars}</span><span class="bc-rating">{rating_num}</span>'
        )
    if year:
        meta_items.append(f'<span class="bc-meta-item">{escape(year)}</span>')
    if pages:
        meta_items.append(f'<span class="bc-meta-item">{escape(pages)} pp</span>')
    if ratings_count:
        meta_items.append(f'<span class="bc-meta-item">{escape(ratings_count)} ratings</span>')
    meta_html = '<span class="bc-dot">·</span>'.join(meta_items)

    tags_html = ""
    if categories_raw:
        tags = [c.strip() for c in categories_raw.split(",") if c.strip()][:3]
        tags_html = "".join(render_badge(tag) for tag in tags)

    reason = str(row.get("reason", "") or "").strip()
    if not reason:
        cat_phrase = categories_raw.lower() if categories_raw else "the themes in its description"
        reason = f"This recommendation aligns with your request through {cat_phrase} and related semantic signals."
    reason = escape(reason)

    subtitle_html = f'<div class="bc-subtitle">{subtitle}</div>' if subtitle else ""
    compact_class = " bc-card--compact" if compact else ""

    return f"""
    <article class="bc-card{compact_class}">
      <div class="bc-rank">{idx:02d}</div>
      <div class="bc-cover">
        <img src="{escape(thumbnail, quote=True)}" alt="{title} cover" loading="lazy"
             onerror="this.onerror=null;this.src='{PLACEHOLDER_SVG}';" />
      </div>
      <div class="bc-main">
        <header class="bc-card-head">
          <div>
            <h3>{title}</h3>
            {subtitle_html}
            <p class="bc-author">by <em>{authors}</em></p>
          </div>
          {match_html}
        </header>

        <div class="bc-meta">{meta_html}</div>
        <div class="bc-tags">{tags_html}</div>
        <p class="bc-desc">{description}</p>

        <section class="bc-reason">
          <div class="bc-reason-top">
            <span>Why it matches</span>
            {sim_html}
          </div>
          <p>{reason}</p>
        </section>
      </div>
    </article>
    """


def render_results_header(top_books: pd.DataFrame, original_query: str, enhanced_query: str) -> str:
    original_query = str(original_query or "").strip()
    enhanced_query = str(enhanced_query or "").strip()

    enhanced_html = ""
    if enhanced_query and enhanced_query.lower() != original_query.lower():
        enhanced_html = f"""
        <section class="bc-understood">
          <span>Enhanced prompt</span>
          <p>{escape(enhanced_query)}</p>
        </section>
        """

    return f"""
    <div class="bc-results" id="results">
      {enhanced_html}
      <div class="bc-results-head">
        <h2>Top {len(top_books)} recommendations</h2>
      </div>
    </div>
    """


def render_library(library: list[dict]) -> str:
    if not library:
        return """
        <div class="bc-empty">
          <div class="bc-empty-icon">📚</div>
          <h3>Library is empty.</h3>
          <p>Save books you like.</p>
        </div>
        """

    cards = "\n".join(render_card(i + 1, item, compact=True) for i, item in enumerate(library))
    return f"""
    <div class="bc-results">
      <div class="bc-results-head">
        <h2>My Library</h2>
        <p>{len(library)} saved</p>
      </div>
      <div class="bc-cards">{cards}</div>
    </div>
    """


def render_status(message: str, kind: str = "ok") -> str:
    if not message:
        return ""
    return f'<div class="bc-status bc-status--{kind}">{escape(message)}</div>'


def _dropdown_choices_from_library(library: list[dict]) -> list[tuple[str, int]]:
    choices: list[tuple[str, int]] = []
    for i, book in enumerate(library or []):
        title = str(book.get("title", "Untitled")).strip() or "Untitled"
        authors = str(book.get("authors", "Unknown author")).strip() or "Unknown author"
        choices.append((f"{title} — {authors}", int(i)))
    return choices


def apply_filters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    genre: str | None,
    language: str | None,
    min_year: Any,
    max_year: Any,
    min_rating: Any,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = pd.Series(True, index=df.index)

    if genre and str(genre).strip():
        genre_query = str(genre).strip()
        if "categories" in df.columns:
            mask &= df["categories"].astype(str).str.contains(genre_query, case=False, na=False)

    if language and str(language).strip() and "language" in df.columns:
        lang_query = str(language).strip()
        mask &= df["language"].astype(str).str.contains(lang_query, case=False, na=False)

    if _has_value(min_year) and "published_year" in df.columns:
        years = pd.to_numeric(df["published_year"], errors="coerce")
        mask &= years >= float(min_year)

    if _has_value(max_year) and "published_year" in df.columns:
        years = pd.to_numeric(df["published_year"], errors="coerce")
        mask &= years <= float(max_year)

    if _has_value(min_rating) and "average_rating" in df.columns:
        ratings = pd.to_numeric(df["average_rating"], errors="coerce")
        mask &= ratings >= float(min_rating)

    filtered_df = df.loc[mask].copy()
    filtered_embeddings = embeddings[mask.to_numpy()]
    return filtered_df.reset_index(drop=True), filtered_embeddings


def _empty_result_outputs(message: str, kind: str = "warn"):
    card_updates = []
    for _ in range(MAX_RESULT_SLOTS):
        card_updates.extend([
            gr.update(value="", visible=False),
            gr.update(visible=False),
        ])
    return (
        gr.update(value=render_status(message, kind), visible=True),
        pd.DataFrame(),
        *card_updates,
        gr.update(value="", visible=False),
    )


def recommend(
    query: str,
    genre: str,
    language: str,
    min_year: Any,
    max_year: Any,
    min_rating: Any,
):
    if _LOAD_ERROR:
        return _empty_result_outputs(_LOAD_ERROR, "error")

    if not query or not str(query).strip():
        return _empty_result_outputs("Write what you're in the mood to read first.", "warn")

    assert books_df is not None
    assert embeddings_matrix is not None

    try:
        filters: dict[str, Any] = {}
        if genre and str(genre).strip():
            filters["genre"] = str(genre).strip()
        if language and str(language).strip():
            filters["language"] = str(language).strip()
        if _has_value(min_year):
            filters["min_year"] = int(float(min_year))
        if _has_value(max_year):
            filters["max_year"] = int(float(max_year))
        if _has_value(min_rating):
            filters["min_rating"] = float(min_rating)

        filtered_df, filtered_embeddings = apply_filters(
            books_df,
            embeddings_matrix,
            genre=genre,
            language=language,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
        )

        if len(filtered_df) == 0:
            return _empty_result_outputs(
                "No books matched those filters. Try a broader genre, year range, or rating.",
                "warn",
            )

        user_prompt = receive_prompt(str(query).strip(), **filters)

        try:
            enhanced = enhance_prompt(user_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] enhance_prompt failed, using raw query: {e}")
            enhanced = user_prompt.query

        query_embedding = encode_prompt(enhanced)
        scores = similarity_score(query_embedding, filtered_embeddings)
        top_books = get_table(filtered_df, scores, top_n=min(DEFAULT_TOP_N, len(filtered_df)))

        if add_reason is not None and len(top_books) > 0:
            try:
                top_books = add_reason(top_books, user_prompt.query)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] add_reason failed, using fallback reasons: {e}")

        result_updates: list[Any] = [
            gr.update(value=render_results_header(top_books, query, enhanced), visible=True),
            top_books,
        ]

        for i in range(MAX_RESULT_SLOTS):
            if i < len(top_books):
                result_updates.extend([
                    gr.update(value=render_card(i + 1, top_books.reset_index(drop=True).iloc[i]), visible=True),
                    gr.update(value="Save to Library", visible=True, interactive=True),
                ])
            else:
                result_updates.extend([
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                ])

        result_updates.append(gr.update(value="", visible=False))
        return tuple(result_updates)

    except Exception as e:  # noqa: BLE001
        return _empty_result_outputs(str(e), "error")


def save_book_at(slot: int, last_results: pd.DataFrame, library: list[dict]):
    library = library or []

    if last_results is None or len(last_results) == 0 or slot >= len(last_results):
        return (
            library,
            render_library(library),
            gr.update(choices=_dropdown_choices_from_library(library), value=None),
            gr.update(value=render_status("Search first.", "warn"), visible=True),
        )

    row = last_results.reset_index(drop=True).iloc[slot]
    existing = {_book_key(book) for book in library}

    if _book_key(row) in existing:
        message = "Already saved."
        kind = "warn"
    else:
        library = library + [{k: ("" if pd.isna(v) else v) for k, v in row.items()}]
        message = "Saved."
        kind = "ok"

    return (
        library,
        render_library(library),
        gr.update(choices=_dropdown_choices_from_library(library), value=None),
        gr.update(value=render_status(message, kind), visible=True),
    )


def remove_selected_book(selected_idx: Any, library: list[dict]):
    library = library or []

    if selected_idx is None or selected_idx == "":
        return (
            library,
            render_library(library),
            gr.update(choices=_dropdown_choices_from_library(library), value=None),
            gr.update(value=render_status("Choose a book first.", "warn"), visible=True),
        )

    try:
        i = int(selected_idx)
        if not (0 <= i < len(library)):
            raise IndexError
        library = library[:i] + library[i + 1:]
        message = "Removed."
        kind = "ok"
    except Exception:  # noqa: BLE001
        message = "Invalid selection."
        kind = "error"

    return (
        library,
        render_library(library),
        gr.update(choices=_dropdown_choices_from_library(library), value=None),
        gr.update(value=render_status(message, kind), visible=True),
    )


def export_library(library: list[dict]):
    library = library or []
    if not library:
        return gr.update(value=None, visible=False)

    out_path = Path(tempfile.gettempdir()) / "my_book_library.csv"
    pd.DataFrame(library).to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return gr.update(value=str(out_path), visible=True)


def toggle_options(is_open: bool):
    next_open = not bool(is_open)
    return next_open, gr.update(visible=next_open), gr.update(value="Options ▲" if next_open else "Options")


CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,400;1,9..144,500&family=Manrope:wght@400;500;600;700;800;900&display=swap');

:root {
  --bc-bg: #f5efe2;
  --bc-panel: #fffdf8;
  --bc-ink: #1f1a14;
  --bc-muted: #5f523f;
  --bc-soft: #7b6b52;
  --bc-border: #deceb0;
  --bc-border-soft: #eadcc3;
  --bc-accent: #a8714a;
  --bc-orange: #e95a0c;
  --bc-shadow: 0 22px 45px -34px rgba(31,26,20,.35);
}

html, body, .gradio-container {
  min-height: 100%;
  background: var(--bc-bg) !important;
  color: var(--bc-ink) !important;
  font-family: 'Manrope', system-ui, sans-serif !important;
}

.gradio-container {
  max-width: none !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 clamp(22px, 6vw, 92px) 72px !important;
}

.gradio-container label,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container button {
  font-family: 'Manrope', system-ui, sans-serif !important;
}

footer, .gradio-container footer { display: none !important; }

/* Hero */
.bc-hero {
  min-height: 28vh;
  max-width: 1180px;
  margin: 0 auto 12px;
  padding: clamp(52px, 9vh, 96px) 0 20px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
  text-align: center;
}

.bc-eyebrow {
  color: var(--bc-accent);
  font-size: 11px;
  font-weight: 900;
  letter-spacing: .24em;
  text-transform: uppercase;
  margin-bottom: 12px;
}

.bc-hero h1 {
  margin: 0 0 14px;
  color: var(--bc-ink);
  font-family: 'Fraunces', serif;
  font-size: clamp(56px, 7vw, 104px);
  line-height: .94;
  letter-spacing: -0.055em;
  font-weight: 500;
}

.bc-hero h1 em {
  color: var(--bc-accent);
  font-style: italic;
  font-weight: 400;
}

.bc-hero p {
  margin: 0;
  max-width: 520px;
  color: var(--bc-muted);
  font-size: 17px;
  line-height: 1.5;
}

/* Tabs */
.gradio-container [role="tablist"] {
  max-width: 1180px !important;
  margin: 0 auto 24px !important;
  background: transparent !important;
  border-bottom: 1px solid var(--bc-border) !important;
  padding: 0 !important;
}

.gradio-container [role="tab"],
.gradio-container [role="tab"] * {
  color: var(--bc-soft) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  font-weight: 900 !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  font-size: 12px !important;
}

.gradio-container [role="tab"] {
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 12px 4px 10px !important;
  margin-right: 28px !important;
}

.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [role="tab"][aria-selected="true"] * {
  color: var(--bc-orange) !important;
  border-bottom-color: var(--bc-orange) !important;
}

/* Search box */
.bc-search-wrap {
  width: min(100%, 1080px);
  margin: 58px auto 52px !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bc-search-box {
  background: var(--bc-panel) !important;
  border: 1.4px solid var(--bc-border) !important;
  border-radius: 28px !important;
  box-shadow: 0 26px 70px -58px rgba(31,26,20,.55) !important;
  padding: 18px 18px 14px !important;
}

.bc-search-box,
.bc-search-box > div,
.bc-search-box .gradio-row,
.bc-search-box .gradio-column,
.bc-search-box .block,
.bc-search-box .form,
.bc-search-box .wrap {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bc-search-box textarea {
  min-height: 132px !important;
  width: 100% !important;
  resize: vertical !important;
  background: transparent !important;
  color: var(--bc-ink) !important;
  border: 0 !important;
  padding: 12px 14px 10px !important;
  font-family: 'Fraunces', serif !important;
  font-size: 28px !important;
  line-height: 1.35 !important;
  box-shadow: none !important;
}

.bc-search-box textarea::placeholder {
  color: #89775d !important;
  opacity: .9 !important;
  font-style: normal !important;
}

.bc-tool-row {
  align-items: center !important;
  gap: 10px !important;
  padding: 8px 6px 0 !important;
}

.bc-tool-row,
.bc-tool-row > div,
.bc-tool-row .gradio-column,
.bc-tool-row .block,
.bc-tool-row .wrap,
.bc-tool-row .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.bc-enhanced-chip {
  display: inline-flex;
  align-items: center;
  height: 38px;
  padding: 0 14px;
  border: 1px solid var(--bc-border);
  border-radius: 999px;
  background: #f8f0de;
  color: var(--bc-accent);
  font-size: 10px;
  font-weight: 950;
  letter-spacing: .13em;
  text-transform: uppercase;
  white-space: nowrap;
}

.bc-options-btn button,
.bc-recommend-btn button,
.bc-library-btn button {
  border-radius: 999px !important;
  min-height: 42px !important;
  font-weight: 850 !important;
  box-shadow: none !important;
}

.bc-options-btn button,
.bc-options-btn button *,
.bc-options-btn button span {
  background: #fffdf8 !important;
  color: var(--bc-ink) !important;
  border: 1px solid var(--bc-border) !important;
  font-size: 13px !important;
}

.bc-options-btn button:hover,
.bc-options-btn button:hover *,
.bc-options-btn button:hover span {
  color: var(--bc-accent) !important;
  border-color: var(--bc-accent) !important;
}

.bc-recommend-btn button,
.bc-recommend-btn button *,
.bc-recommend-btn button span {
  background: var(--bc-orange) !important;
  color: white !important;
  border: 1px solid var(--bc-orange) !important;
  font-size: 14px !important;
}

.bc-recommend-btn button:hover,
.bc-recommend-btn button:hover *,
.bc-recommend-btn button:hover span {
  background: var(--bc-ink) !important;
  color: var(--bc-bg) !important;
  border-color: var(--bc-ink) !important;
}

/* Horizontal options popover */
.bc-options-popover {
  margin: 12px 0 0 !important;
  padding: 14px !important;
  background: #fffdf8 !important;
  border: 1px solid var(--bc-border) !important;
  border-radius: 22px !important;
  box-shadow: 0 18px 38px -34px rgba(31,26,20,.38) !important;
}

.bc-options-popover,
.bc-options-popover > div,
.bc-options-popover .gradio-row,
.bc-options-popover .gradio-column,
.bc-options-popover .block,
.bc-options-popover .form,
.bc-options-popover .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.bc-options-row {
  display: flex !important;
  flex-wrap: nowrap !important;
  align-items: flex-end !important;
  gap: 12px !important;
  overflow-x: auto !important;
}

.bc-filter-field {
  min-width: 118px !important;
  flex: 1 1 0 !important;
}
.bc-filter-field:first-child { min-width: 190px !important; flex: 1.5 1 0 !important; }

.bc-filter-field label {
  display: block !important;
  margin: 0 0 7px 3px !important;
  color: var(--bc-accent) !important;
  font-size: 9.5px !important;
  font-weight: 950 !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
}

.bc-filter-field input,
.bc-filter-field textarea,
.bc-filter-field select {
  width: 100% !important;
  min-height: 40px !important;
  max-height: 42px !important;
  border-radius: 999px !important;
  border: 1px solid var(--bc-border-soft) !important;
  background: #fffaf0 !important;
  color: var(--bc-ink) !important;
  padding: 8px 13px !important;
  font-family: 'Manrope', system-ui, sans-serif !important;
  font-size: 13px !important;
  font-style: normal !important;
  box-shadow: none !important;
}

/* Status */
.bc-status {
  max-width: 1080px;
  margin: 18px auto;
  padding: 13px 16px;
  border-radius: 14px;
  font-size: 14px;
  font-weight: 800;
}
.bc-status--ok { background: #edf7e9; border: 1px solid #bed7b4; color: #2d4423; }
.bc-status--warn { background: #fff6df; border: 1px solid #ead09b; color: #674b17; }
.bc-status--error { background: #fff0ea; border: 1px solid #e8bda9; color: #7a341a; }

/* Results */
.bc-results {
  width: 100%;
  margin: 38px auto 0;
  animation: bc-rise .42s ease-out both;
  scroll-margin-top: 30px;
}
@keyframes bc-rise {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}

.bc-results-head {
  max-width: 1260px;
  margin: 0 auto 22px;
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--bc-border);
  padding-bottom: 14px;
}

.bc-results-head h2 {
  margin: 0;
  font-family: 'Fraunces', serif;
  color: var(--bc-ink);
  font-size: 38px;
  font-weight: 500;
  letter-spacing: -.02em;
}

.bc-results-head p {
  margin: 0;
  color: var(--bc-soft);
  font-size: 12px;
  font-weight: 900;
  letter-spacing: .16em;
  text-transform: uppercase;
}

.bc-understood {
  max-width: 1080px;
  margin: 0 auto 24px;
  background: var(--bc-panel);
  border: 1px solid var(--bc-border);
  border-left: 4px solid var(--bc-accent);
  border-radius: 16px;
  padding: 14px 16px;
}
.bc-understood span {
  display: block;
  color: var(--bc-accent);
  font-size: 10px;
  font-weight: 950;
  letter-spacing: .16em;
  text-transform: uppercase;
  margin-bottom: 7px;
}
.bc-understood p {
  margin: 0;
  color: var(--bc-muted);
  font-size: 14px;
  line-height: 1.5;
}

.bc-card-slot {
  max-width: 1260px;
  margin: 0 auto 26px !important;
  align-items: flex-start !important;
  gap: 16px !important;
}
.bc-card-slot > div,
.bc-card-slot .gradio-column,
.bc-card-slot .block,
.bc-card-slot .form,
.bc-card-slot .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.bc-save-side {
  min-width: 142px !important;
  max-width: 160px !important;
  padding-top: 64px !important;
  display: flex !important;
  justify-content: center !important;
}

.bc-save-side button,
[id^="bc-save-btn-"] button {
  opacity: 1 !important;
  visibility: visible !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-width: 126px !important;
  height: 38px !important;
  padding: 0 18px !important;
  border-radius: 999px !important;
  background: #fffdf8 !important;
  border: 1.5px solid var(--bc-border) !important;
  color: var(--bc-ink) !important;
  font-family: 'Manrope', system-ui, sans-serif !important;
  font-size: 13px !important;
  font-weight: 850 !important;
  letter-spacing: .01em !important;
  box-shadow: 0 12px 28px -24px rgba(31,26,20,.55) !important;
  text-shadow: none !important;
}
.bc-save-side button *,
[id^="bc-save-btn-"] button * {
  color: var(--bc-ink) !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.bc-save-side button:hover,
[id^="bc-save-btn-"] button:hover {
  background: var(--bc-ink) !important;
  border-color: var(--bc-ink) !important;
  color: var(--bc-bg) !important;
  transform: translateY(-1px);
}
.bc-save-side button:hover *,
[id^="bc-save-btn-"] button:hover * {
  color: var(--bc-bg) !important;
}

.bc-card {
  position: relative;
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 26px;
  background: var(--bc-panel);
  border: 1px solid var(--bc-border);
  border-radius: 22px;
  padding: 24px 28px 24px 58px;
  box-shadow: 0 18px 38px -32px rgba(31,26,20,.28);
  width: 100%;
}
.bc-card, .bc-card * { box-sizing: border-box; }
.bc-card--compact { grid-template-columns: 110px minmax(0, 1fr); padding-left: 50px; }
.bc-rank {
  position: absolute;
  top: 24px;
  left: 20px;
  color: #907c5c !important;
  font-family: 'Fraunces', serif;
  font-size: 15px;
  font-style: italic;
}
.bc-cover {
  width: 150px;
  height: 225px;
  border-radius: 11px;
  overflow: hidden;
  background: #ede6d6;
  box-shadow: 0 16px 28px -22px rgba(31,26,20,.45);
}
.bc-card--compact .bc-cover { width: 110px; height: 165px; }
.bc-cover img { width: 100%; height: 100%; object-fit: cover; display: block; }
.bc-card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.bc-card h3 {
  margin: 0 0 4px;
  color: var(--bc-ink) !important;
  font-family: 'Fraunces', serif;
  font-size: 25px;
  line-height: 1.18;
  font-weight: 500;
  letter-spacing: -.01em;
}
.bc-card--compact h3 { font-size: 21px; }
.bc-subtitle { margin-bottom: 5px; color: var(--bc-muted) !important; font-family: 'Fraunces', serif; font-size: 15px; font-style: italic; }
.bc-author { margin: 0 0 12px; color: var(--bc-muted) !important; font-size: 13px; }
.bc-author em { color: var(--bc-ink) !important; font-family: 'Fraunces', serif; font-size: 14px; }
.bc-match {
  flex: 0 0 auto;
  width: 70px;
  height: 70px;
  border-radius: 50%;
  background: var(--bc-ink);
  color: var(--bc-bg) !important;
  display: grid;
  place-items: center;
  text-align: center;
  line-height: 1;
}
.bc-match strong { color: var(--bc-bg) !important; font-family: 'Fraunces', serif; font-size: 22px; }
.bc-match span { color: var(--bc-bg) !important; display: block; margin-top: -11px; font-size: 8px; font-weight: 900; letter-spacing: .18em; text-transform: uppercase; opacity: .82; }
.bc-meta, .bc-meta span, .bc-meta .bc-meta-item { color: var(--bc-muted) !important; }
.bc-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; margin: 10px 0; font-size: 13px; font-weight: 800; }
.bc-dot { color: #8f7b5c !important; font-weight: 900; }
.bc-stars { color: #c98524 !important; letter-spacing: 1px; }
.bc-rating { color: var(--bc-ink) !important; font-weight: 950; }
.bc-tags { display: flex; gap: 7px; flex-wrap: wrap; margin: 10px 0 12px; }
.bc-pill { background: #f1e8d4; color: #654123 !important; border-radius: 999px; padding: 5px 10px; font-size: 11px; font-weight: 800; }
.bc-desc { margin: 0 0 14px; color: #2f281f !important; font-family: 'Fraunces', serif; font-size: 15px; line-height: 1.58; }
.bc-reason { background: #f8f0de; border: 1px solid #eadcc3; border-left: 4px solid var(--bc-accent); border-radius: 12px; padding: 13px 15px; }
.bc-reason-top { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 7px; }
.bc-reason-top > span:first-child { color: var(--bc-accent) !important; font-size: 10px; font-weight: 950; letter-spacing: .18em; text-transform: uppercase; }
.bc-chip { background: var(--bc-panel); border: 1px solid var(--bc-border); border-radius: 999px; color: var(--bc-muted) !important; padding: 3px 8px; font-size: 11px; font-weight: 800; }
.bc-reason p { margin: 0; color: #2f281f !important; font-family: 'Fraunces', serif; font-size: 14px; font-style: italic; line-height: 1.5; }

.bc-empty {
  margin: 38px auto 0;
  max-width: 760px;
  background: rgba(255,253,248,.72);
  border: 1px dashed var(--bc-border);
  border-radius: 20px;
  text-align: center;
  padding: 56px 24px;
}
.bc-empty-icon { font-size: 34px; margin-bottom: 10px; }
.bc-empty h3 { margin: 0 0 8px; color: var(--bc-ink) !important; font-family: 'Fraunces', serif; font-size: 25px; font-style: italic; font-weight: 500; }
.bc-empty p { margin: 0; color: var(--bc-muted) !important; }

/* Library toolbar */
.bc-library-actions {
  max-width: 760px;
  margin: 46px auto 30px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.bc-library-toolbar { justify-content: center !important; align-items: center !important; gap: 10px !important; }
.bc-library-actions,
.bc-library-actions > div,
.bc-library-actions .gradio-row,
.bc-library-actions .gradio-column,
.bc-library-actions .block,
.bc-library-actions .form,
.bc-library-actions .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.bc-library-actions .gradio-dropdown { min-width: 230px !important; }
.bc-library-actions [role="combobox"],
.bc-library-actions input,
.bc-library-actions textarea,
.bc-library-actions select {
  background: #fffdf8 !important;
  color: var(--bc-ink) !important;
  border: 1px solid var(--bc-border) !important;
  border-radius: 999px !important;
  min-height: 42px !important;
  box-shadow: none !important;
}
.bc-library-btn button {
  min-width: 112px !important;
  min-height: 42px !important;
  border-radius: 999px !important;
  background: #fffdf8 !important;
  color: var(--bc-ink) !important;
  border: 1px solid var(--bc-border) !important;
  font-weight: 850 !important;
}
.bc-library-btn button:hover {
  background: var(--bc-ink) !important;
  color: var(--bc-bg) !important;
  border-color: var(--bc-ink) !important;
}


.bc-tool-row {
  align-items: center !important;
}
.bc-enhanced-chip {
  background: #f8f0de !important;
  color: var(--bc-accent) !important;
  border-color: var(--bc-border) !important;
}
.bc-recommend-btn button,
.bc-recommend-btn button *,
.bc-recommend-btn button span {
  background: var(--bc-orange) !important;
  color: #fffaf0 !important;
  border-color: var(--bc-orange) !important;
}
.bc-save-side {
  min-width: 170px !important;
  max-width: 190px !important;
  padding-top: 62px !important;
}
.bc-save-side button,
.bc-save-button button,
[id^="bc-save-btn-"] button {
  background: var(--bc-ink) !important;
  color: var(--bc-bg) !important;
  border: 1.5px solid var(--bc-ink) !important;
  opacity: 1 !important;
  visibility: visible !important;
  min-width: 150px !important;
  height: 40px !important;
  border-radius: 999px !important;
  font-weight: 850 !important;
  letter-spacing: .01em !important;
  box-shadow: 0 14px 26px -22px rgba(31,26,20,.55) !important;
}
.bc-save-side button *,
.bc-save-button button *,
[id^="bc-save-btn-"] button * {
  color: var(--bc-bg) !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.bc-save-side button:hover,
.bc-save-button button:hover,
[id^="bc-save-btn-"] button:hover {
  background: var(--bc-orange) !important;
  border-color: var(--bc-orange) !important;
  color: #fffaf0 !important;
  transform: translateY(-1px);
}
.bc-save-side button:hover *,
.bc-save-button button:hover *,
[id^="bc-save-btn-"] button:hover * {
  color: #fffaf0 !important;
}

@media (max-width: 900px) {
  .gradio-container { padding: 0 16px 52px !important; }
  .bc-hero { min-height: 22vh; padding-top: 44px; }
  .bc-hero h1 { font-size: 46px; }
  .bc-search-wrap { margin-top: 30px !important; }
  .bc-tool-row { flex-direction: column !important; align-items: stretch !important; }
  .bc-options-row { flex-wrap: wrap !important; }
  .bc-filter-field, .bc-filter-field:first-child { flex: 1 1 42% !important; min-width: 140px !important; }
  .bc-card-slot { display: block !important; }
  .bc-save-side { max-width: none !important; min-width: 0 !important; justify-content: flex-end !important; padding-top: 0 !important; margin: -16px 18px 26px 0 !important; }
  .bc-card, .bc-card--compact { grid-template-columns: 1fr; padding: 56px 18px 20px; }
  .bc-cover, .bc-card--compact .bc-cover { width: 120px; height: 180px; margin: 0 auto; }
  .bc-card-head { flex-direction: column; }
  .bc-match { align-self: flex-end; }
  .bc-library-toolbar { flex-wrap: wrap !important; }
}
"""


HERO_HTML = """
<div class="bc-hero">
  <span class="bc-eyebrow">Book Recommender</span>
  <h1>Find your next <em>great read</em>.</h1>
  <p>Describe a mood, a theme, or a book you loved.</p>
</div>
"""

ENHANCED_HTML = """<span class="bc-enhanced-chip">Enhanced · On</span>"""

THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="stone")


with gr.Blocks(title="Book Recommender") as demo:
    library_state = gr.State([])
    last_results_state = gr.State(pd.DataFrame())

    gr.HTML(HERO_HTML)

    with gr.Tabs():
        with gr.Tab("Discover"):
            with gr.Group(elem_classes="bc-search-wrap"):
                with gr.Group(elem_classes="bc-search-box"):
                    query_input = gr.Textbox(
                        placeholder="Tell me what you want to read…",
                        lines=3,
                        max_lines=6,
                        show_label=False,
                        container=False,
                    )
                    with gr.Row(elem_classes="bc-tool-row"):
                        with gr.Column(scale=1, min_width=160):
                            gr.HTML(ENHANCED_HTML)
                        with gr.Column(scale=5):
                            pass
                        with gr.Column(scale=1, min_width=190):
                            submit = gr.Button(
                                "Recommend →",
                                variant="primary",
                                elem_classes="bc-recommend-btn",
                            )

            genre = gr.Textbox(value="", visible=False)
            language = gr.Textbox(value="", visible=False)
            min_year = gr.Number(value=None, visible=False)
            max_year = gr.Number(value=None, visible=False)
            min_rating = gr.Number(value=None, visible=False)

            results_header = gr.HTML(visible=False)

            card_htmls = []
            save_buttons = []
            for i in range(MAX_RESULT_SLOTS):
                with gr.Row(elem_classes="bc-card-slot"):
                    with gr.Column(scale=12):
                        card = gr.HTML(visible=False)
                    with gr.Column(scale=2, min_width=140, elem_classes="bc-save-side"):
                        btn = gr.Button("Save to Library", visible=False, elem_id=f"bc-save-btn-{i}", elem_classes="bc-save-button")
                card_htmls.append(card)
                save_buttons.append(btn)

            save_status = gr.HTML(visible=False)

        with gr.Tab("My Library"):
            with gr.Group(elem_classes="bc-library-actions"):
                with gr.Row(elem_classes="bc-library-toolbar"):
                    remove_picker = gr.Dropdown(
                        label="Saved books",
                        choices=[],
                        value=None,
                        interactive=True,
                        show_label=False,
                    )
                    remove_btn = gr.Button("Remove", elem_classes="bc-library-btn")
                    export_btn = gr.Button("Export", elem_classes="bc-library-btn")
                library_status = gr.HTML(visible=False)
                export_file = gr.File(label="Download", visible=False, interactive=False)

            library_html = gr.HTML(value=render_library([]))

    recommend_inputs = [query_input, genre, language, min_year, max_year, min_rating]
    recommend_outputs: list[Any] = [results_header, last_results_state]
    for i in range(MAX_RESULT_SLOTS):
        recommend_outputs.extend([card_htmls[i], save_buttons[i]])
    recommend_outputs.append(save_status)

    submit.click(recommend, inputs=recommend_inputs, outputs=recommend_outputs)
    query_input.submit(recommend, inputs=recommend_inputs, outputs=recommend_outputs)


    save_outputs = [library_state, library_html, remove_picker, save_status]
    for i, btn in enumerate(save_buttons):
        btn.click(
            fn=(lambda slot: lambda last_df, lib: save_book_at(slot, last_df, lib))(i),
            inputs=[last_results_state, library_state],
            outputs=save_outputs,
        )

    remove_btn.click(
        remove_selected_book,
        inputs=[remove_picker, library_state],
        outputs=[library_state, library_html, remove_picker, library_status],
    )

    export_btn.click(export_library, inputs=[library_state], outputs=[export_file])


CSS += r'''

.bc-search-wrap {
  width: min(100%, 1040px) !important;
  margin: 70px auto 60px !important;
}

.bc-search-box {
  background: #fffdf8 !important;
  border: 1.4px solid #deceb0 !important;
  border-radius: 26px !important;
  box-shadow: 0 28px 74px -58px rgba(31,26,20,.42) !important;
  padding: 20px 22px 18px !important;
}

.bc-search-box textarea {
  min-height: 150px !important;
  color: #1f1a14 !important;
  background: transparent !important;
  font-family: 'Fraunces', serif !important;
  font-size: 30px !important;
  line-height: 1.32 !important;
  padding: 14px 16px !important;
}

.bc-search-box textarea::placeholder {
  color: #8b7a61 !important;
  opacity: .92 !important;
}

.bc-tool-row {
  padding: 12px 8px 0 !important;
  align-items: center !important;
}

.bc-enhanced-chip {
  background: #fbf3e2 !important;
  border: 1px solid #dfcaaa !important;
  color: #a8714a !important;
  height: 36px !important;
  padding: 0 15px !important;
  border-radius: 999px !important;
  font-size: 10px !important;
  font-weight: 900 !important;
  letter-spacing: .14em !important;
  text-transform: uppercase !important;
}

.bc-recommend-btn button,
.bc-recommend-btn button *,
.bc-recommend-btn button span {
  background: #e95a0c !important;
  border-color: #e95a0c !important;
  color: #fffaf0 !important;
}

.bc-recommend-btn button {
  min-width: 190px !important;
  height: 44px !important;
  border-radius: 999px !important;
  font-size: 14px !important;
  font-weight: 850 !important;
  box-shadow: 0 14px 28px -22px rgba(233,90,12,.65) !important;
}

.bc-recommend-btn button:hover,
.bc-recommend-btn button:hover *,
.bc-recommend-btn button:hover span {
  background: #1f1a14 !important;
  border-color: #1f1a14 !important;
  color: #f5efe2 !important;
}

/* Save button */
.bc-save-side {
  min-width: 170px !important;
  max-width: 185px !important;
  padding-top: 76px !important;
  display: flex !important;
  justify-content: center !important;
}

.bc-save-side button,
.bc-save-button button,
[id^="bc-save-btn-"] button {
  background: #fffdf8 !important;
  color: #1f1a14 !important;
  border: 1.5px solid #deceb0 !important;
  opacity: 1 !important;
  visibility: visible !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-width: 144px !important;
  height: 40px !important;
  border-radius: 999px !important;
  font-family: 'Manrope', system-ui, sans-serif !important;
  font-size: 13px !important;
  font-weight: 850 !important;
  letter-spacing: .01em !important;
  box-shadow: 0 16px 30px -24px rgba(31,26,20,.45) !important;
  text-shadow: none !important;
}
.bc-save-side button *,
.bc-save-button button *,
[id^="bc-save-btn-"] button * {
  color: #1f1a14 !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.bc-save-side button:hover,
.bc-save-button button:hover,
[id^="bc-save-btn-"] button:hover {
  background: #1f1a14 !important;
  border-color: #1f1a14 !important;
  color: #f5efe2 !important;
  transform: translateY(-1px);
}
.bc-save-side button:hover *,
.bc-save-button button:hover *,
[id^="bc-save-btn-"] button:hover * {
  color: #f5efe2 !important;
}

/* Library toolbar */
.bc-library-actions {
  max-width: 720px !important;
  margin: 56px auto 34px !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.bc-library-toolbar {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 12px !important;
  flex-wrap: wrap !important;
}

.bc-library-toolbar > div,
.bc-library-actions .gradio-column,
.bc-library-actions .block,
.bc-library-actions .form,
.bc-library-actions .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.bc-library-actions .gradio-dropdown {
  min-width: 260px !important;
  max-width: 340px !important;
}

.bc-library-actions [role="combobox"],
.bc-library-actions input,
.bc-library-actions textarea,
.bc-library-actions select {
  background: #fffdf8 !important;
  color: #1f1a14 !important;
  border: 1px solid #deceb0 !important;
  border-radius: 999px !important;
  min-height: 42px !important;
  box-shadow: none !important;
  padding-left: 14px !important;
}

.bc-library-btn button,
.bc-library-btn button *,
.bc-library-btn button span {
  color: #1f1a14 !important;
}

.bc-library-btn button {
  min-width: 118px !important;
  min-height: 42px !important;
  border-radius: 999px !important;
  background: #fffdf8 !important;
  border: 1px solid #deceb0 !important;
  font-weight: 850 !important;
  box-shadow: 0 12px 24px -22px rgba(31,26,20,.38) !important;
}

.bc-library-btn button:hover,
.bc-library-btn button:hover *,
.bc-library-btn button:hover span {
  background: #1f1a14 !important;
  border-color: #1f1a14 !important;
  color: #f5efe2 !important;
}

.bc-empty {
  max-width: 780px !important;
  margin: 46px auto 0 !important;
  background: rgba(255,253,248,.72) !important;
  border: 1px dashed #deceb0 !important;
  border-radius: 22px !important;
  padding: 58px 28px !important;
}

.bc-options-btn,
.bc-options-popover,
.bc-options-row,
.bc-filter-field {
  display: none !important;
}

@media (max-width: 900px) {
  .bc-search-wrap { margin-top: 36px !important; }
  .bc-search-box textarea { font-size: 23px !important; min-height: 120px !important; }
  .bc-tool-row { flex-direction: column !important; align-items: stretch !important; }
  .bc-recommend-btn button { width: 100% !important; }
  .bc-save-side { max-width: none !important; min-width: 0 !important; padding-top: 0 !important; margin: -14px 18px 28px 0 !important; justify-content: flex-end !important; }
}
'''

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME)
    
#uv run python frontend.py