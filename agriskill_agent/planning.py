from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .models import NormalizedInput, UserCase
from .skill_definitions import SkillCatalog
from .terminology import extract_disease_from_text, resolve_disease


ALLOWED_GOALS = {"phenotype", "diagnosis", "management"}
ALLOWED_MANAGEMENT_OPERATIONS = {
    "disease_management",
    "pesticide_detail",
    "pesticide_uses",
    "crop_disease_list",
}
def _validated_entity_text(value: Any, original_query: str) -> str | None:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    normalized_text = text.casefold().strip(" \t\r\n!?.,;:，。！？；：")
    normalized_query = " ".join(original_query.split()).casefold().strip(
        " \t\r\n!?.,;:，。！？；："
    )
    if normalized_text == normalized_query:
        return None
    if len(text) > 120 or len(text.split()) > 12:
        return None
    return text


@dataclass(frozen=True, slots=True)
class TaskPlan:
    mode: str
    goals: tuple[str, ...]
    selected_skills: tuple[str, ...]
    rationale: str
    planner: str
    management_operation: str | None = None
    entities: dict[str, str | None] = field(default_factory=dict)
    excluded_goals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "goals": list(self.goals),
            "selected_skills": list(self.selected_skills),
            "rationale": self.rationale,
            "planner": self.planner,
            "management_operation": self.management_operation,
            "entities": dict(self.entities),
            "excluded_goals": list(self.excluded_goals),
        }


class SkillPlanner:
    def __init__(self, catalog: SkillCatalog, llm: Any | None = None):
        self.catalog = catalog
        self.llm = llm

    def plan(self, case: UserCase, normalized: NormalizedInput) -> TaskPlan:
        fallback = self._deterministic_plan(case, normalized)
        if not self.llm:
            return fallback
        system_prompt = (
            "You are the task planner of a leaf disease assistant. Select skills using "
            "only the supplied skill name and description metadata. Use image_case for "
            "leaf-image consultations and knowledge_qa for text-only questions. "
            "Understand negation: a phrase such as 'do not diagnose' must "
            "put diagnosis in excluded_goals rather than goals. Return JSON with exactly "
            "these planning fields: mode; goals (phenotype, diagnosis and/or management); "
            "excluded_goals; selected_skills; management_operation; entities; rationale. "
            "management_operation must be null unless disease_management is selected; "
            "otherwise choose exactly one of disease_management, pesticide_detail, "
            "pesticide_uses, or crop_disease_list. entities must contain crop, disease, "
            "and pesticide_name, using null when absent. Extract only entity names, never "
            "copy the complete user question into an entity field. Use disease_management "
            "for management of a named or diagnosed disease; pesticide_detail for the "
            "properties, ingredients, toxicity, precautions or registration of a named "
            "pesticide; pesticide_uses for crops or diseases controlled by a named "
            "pesticide; and crop_disease_list for diseases with pesticide records for a "
            "named crop. Do not use confidence scores."
        )
        payload = {
            "has_image": bool(case.image_path),
            "query": case.query,
            "normalized_intents": normalized.intents,
            "skills": self.catalog.metadata(),
        }
        try:
            generated = self.llm.complete_json(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                stage="task_planning",
            )
            selected = generated.get("selected_skills")
            goals = generated.get("goals")
            excluded_goals = generated.get("excluded_goals")
            entities = generated.get("entities")
            mode = str(generated.get("mode") or "")
            if not isinstance(selected, list) or not isinstance(goals, list):
                return fallback
            selected_names = [
                str(item).replace("-", "_")
                for item in selected
                if str(item).replace("-", "_") in self.catalog.names()
            ]
            goal_names = [str(item) for item in goals if str(item) in ALLOWED_GOALS]
            if not selected_names or mode not in {"image_case", "knowledge_qa"}:
                return fallback
            return self._enforce(
                case=case,
                normalized=normalized,
                mode=mode,
                goals=goal_names,
                selected=selected_names,
                rationale=str(generated.get("rationale") or "Selected by the center agent."),
                planner="agent_model",
                management_operation=generated.get("management_operation"),
                entities=entities if isinstance(entities, dict) else {},
                excluded_goals=(
                    excluded_goals if isinstance(excluded_goals, list) else []
                ),
            )
        except Exception:
            return fallback

    def _deterministic_plan(
        self,
        case: UserCase,
        normalized: NormalizedInput,
    ) -> TaskPlan:
        mode = "image_case" if case.image_path else "knowledge_qa"
        goals = [item for item in normalized.intents if item in ALLOWED_GOALS]
        if not goals:
            goals = ["diagnosis"]
        selected: list[str] = []
        if mode == "image_case":
            selected.append("phenotype_analysis")
        if "diagnosis" in goals:
            selected.append("disease_diagnosis")
        if "management" in goals:
            selected.append("disease_management")
        selected.append("report_generation")
        terminology = extract_disease_from_text(
            case.query,
            crop=normalized.crop,
        )
        return self._enforce(
            case=case,
            normalized=normalized,
            mode=mode,
            goals=goals,
            selected=selected,
            rationale="Rule-based fallback derived from input modality and normalized intent.",
            planner="deterministic",
            management_operation=(
                "disease_management" if "management" in goals else None
            ),
            entities={
                "crop": normalized.crop,
                "disease": terminology.canonical_id if terminology else None,
                "pesticide_name": None,
            },
        )

    def _enforce(
        self,
        *,
        case: UserCase,
        normalized: NormalizedInput,
        mode: str,
        goals: list[str],
        selected: list[str],
        rationale: str,
        planner: str,
        management_operation: Any = None,
        entities: dict[str, Any] | None = None,
        excluded_goals: list[Any] | None = None,
    ) -> TaskPlan:
        mode = "image_case" if case.image_path else "knowledge_qa"
        excluded = tuple(
            dict.fromkeys(
                str(goal)
                for goal in (excluded_goals or [])
                if str(goal) in ALLOWED_GOALS
            )
        )
        goals = [goal for goal in goals if goal not in excluded]
        names = list(dict.fromkeys(selected))
        if "diagnosis" in excluded:
            names = [name for name in names if name != "disease_diagnosis"]
        if "management" in excluded:
            names = [name for name in names if name != "disease_management"]
        if "phenotype" in excluded:
            names = [name for name in names if name != "phenotype_analysis"]
        if mode == "image_case":
            names.insert(0, "phenotype_analysis")
            if "management" in goals and "disease_diagnosis" not in names:
                names.append("disease_diagnosis")
        else:
            names = [name for name in names if name != "phenotype_analysis"]
        if "report_generation" not in names:
            names.append("report_generation")
        order = [
            "phenotype_analysis",
            "disease_diagnosis",
            "disease_management",
            "report_generation",
        ]
        names = [name for name in order if name in names and name in self.catalog.names()]
        final_goals = list(dict.fromkeys(goal for goal in goals if goal in ALLOWED_GOALS))
        raw_entities = entities or {}
        crop = _validated_entity_text(
            raw_entities.get("crop") or normalized.crop,
            case.query,
        )
        raw_disease = _validated_entity_text(
            raw_entities.get("disease"),
            case.query,
        )
        terminology = (
            resolve_disease(raw_disease, crop=crop)
            if raw_disease
            else extract_disease_from_text(case.query, crop=crop)
        )
        disease = terminology.canonical_id if terminology else raw_disease
        pesticide_name = _validated_entity_text(
            raw_entities.get("pesticide_name"),
            case.query,
        )
        operation = str(management_operation or "").strip() or None
        if operation not in ALLOWED_MANAGEMENT_OPERATIONS:
            operation = None
        if "disease_management" not in names:
            operation = None
        elif operation is None and disease:
            operation = "disease_management"
        if operation in {"pesticide_detail", "pesticide_uses"} and not pesticide_name:
            operation = "disease_management" if disease else None
        elif operation == "disease_management" and not disease:
            operation = None
        elif operation == "crop_disease_list" and not crop:
            operation = None
        return TaskPlan(
            mode=mode,
            goals=tuple(final_goals or ("diagnosis",)),
            selected_skills=tuple(names),
            rationale=rationale,
            planner=planner,
            management_operation=operation,
            entities={
                "crop": crop,
                "disease": disease,
                "pesticide_name": pesticide_name,
            },
            excluded_goals=excluded,
        )
