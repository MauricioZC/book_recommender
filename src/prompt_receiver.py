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


