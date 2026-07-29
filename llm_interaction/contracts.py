from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


SystemOutputType = Literal["message", "api_call"]


@dataclass(frozen=True)
class ApiCall:
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedDialog:
    dialog_id: str
    log: list[str] = field(default_factory=list)
    api_calls: list[Any] = field(default_factory=list)
    predicted_api_calls: list[Any] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    trait_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricOutput:
    name: str
    value: float | int | str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraitOutput:
    trait: str
    score: float | None = None
    coherence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
