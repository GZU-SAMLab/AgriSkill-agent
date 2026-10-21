from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalConfig:
    semantic_top_k: int = 20
    keyword_top_k: int = 20
    candidate_top_k: int = 30
    final_top_k: int = 5
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    max_chunks_per_source: int = 2


@dataclass(slots=True)
class AgentConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    include_kg: bool = True
    severity_boundaries: tuple[float, ...] = (0.01, 0.05, 0.15, 0.30)
    severity_labels: tuple[str, ...] = (
        "healthy_or_trace",
        "mild",
        "moderate",
        "severe",
        "very_severe",
    )
