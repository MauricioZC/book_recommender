from dataclasses import dataclass, field


@dataclass
class UserPrompt:
    query: str
    genre: str | None = None
    language: str | None = None
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float | None = None

    def to_filter_description(self) -> str:
        parts = []
        if self.genre:
            parts.append(f"genre: {self.genre}")
        if self.language:
            parts.append(f"language: {self.language}")
        if self.min_year or self.max_year:
            year_range = f"{self.min_year or ''}–{self.max_year or ''}"
            parts.append(f"published: {year_range}")
        if self.min_rating:
            parts.append(f"minimum rating: {self.min_rating}")
        return ", ".join(parts)


def receive_prompt(query: str, **filters) -> UserPrompt:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    valid_fields = {f.name for f in UserPrompt.__dataclass_fields__.values()} - {"query"}
    unknown = set(filters) - valid_fields
    if unknown:
        raise ValueError(f"Unknown filters: {unknown}. Valid filters: {valid_fields}")

    return UserPrompt(query=query, **{k: v for k, v in filters.items() if v is not None})
