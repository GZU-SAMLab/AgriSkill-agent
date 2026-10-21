from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    USER = "user"
    VISUAL = "visual"
    RETRIEVAL = "retrieval"
    KNOWLEDGE_GRAPH = "knowledge_graph"


@dataclass(slots=True)
class UserCase:
    query: str
    image_path: str | None = None
    crop: str | None = None
    organ: str | None = None
    location: str | None = None
    growth_stage: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedInput:
    original_query: str
    crop: str | None
    organ: str | None
    symptoms: list[str]
    intents: list[str]
    location: str | None
    growth_stage: str | None
    unknown_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolResult:
    tool: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    evidence_type: EvidenceType
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhenotypeCard:
    crop: str | None
    organ: str | None
    leaf_valid: bool | None = None
    leaf_color: str | None = None
    lesion_color: str | None = None
    lesion_shape: str | None = None
    lesion_distribution: str | None = None
    affected_region: str | None = None
    leaf_curling: str | None = None
    surface_signs: list[str] = field(default_factory=list)
    lesion_count: int | None = None
    lesion_area_ratio: float | None = None
    severity: str | None = None
    classifier_candidates: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Diagnosis:
    primary_disease: str | None
    differential_diagnoses: list[str]
    rationale: list[str]
    limitations: list[str]
    evidence_ids: list[str]
    status: str = "diagnosed"


@dataclass(slots=True)
class ManagementPlan:
    cultural_controls: list[str]
    biological_controls: list[str]
    chemical_controls: list[str]
    monitoring: list[str]
    safety_notes: list[str]
    evidence_ids: list[str]


@dataclass(slots=True)
class SkillTrace:
    skill: str
    steps: list[str]
    tool_calls: list[str]
    rules_applied: list[str]


@dataclass(slots=True)
class AgentResponse:
    normalized_input: NormalizedInput
    phenotype_card: PhenotypeCard
    diagnosis: Diagnosis
    management: ManagementPlan
    report: str
    evidence: list[Evidence]
    traces: list[SkillTrace]
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
