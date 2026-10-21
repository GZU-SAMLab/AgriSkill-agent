from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .config import AgentConfig
from .llm import AgentModelClient
from .models import (
    Diagnosis,
    Evidence,
    ManagementPlan,
    NormalizedInput,
    PhenotypeCard,
    SkillTrace,
)
from .skill_definitions import SkillDefinition
from .terminology import extract_disease_from_text, resolve_disease
from .tools import ToolRegistry


@dataclass(slots=True)
class SkillOutput:
    value: Any
    evidence: list[Evidence]
    trace: SkillTrace


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(
            text for item in value if (text := _text_value(item))
        )
    return str(value).strip()


def _unique_text_parts(*values: Any) -> list[str]:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text_value(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return parts


def _evidence_context(evidence: list[Evidence], max_chars: int = 1400) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "type": item.evidence_type.value,
            "source": item.source,
            "content": item.content[:max_chars],
            "knowledge_role": (
                "pesticide_registration"
                if item.evidence_type.value == "knowledge_graph"
                else "scientific_literature"
            ),
            **(
                {"content_type": item.metadata.get("content_type")}
                if item.metadata.get("content_type")
                else {}
            ),
            **(
                {"matched_disease": item.metadata.get("matched_disease")}
                if item.metadata.get("matched_disease")
                else {}
            ),
            **(
                {
                    "diagnostic_candidates": item.metadata.get(
                        "diagnostic_candidates"
                    )
                }
                if item.metadata.get("diagnostic_candidates")
                else {}
            ),
            **(
                {
                    "agent_retrieval_queries": item.metadata.get(
                        "agent_retrieval_queries"
                    )
                }
                if item.metadata.get("agent_retrieval_queries")
                else {}
            ),
        }
        for item in evidence
    ]


def _candidate_scope(item: dict[str, Any]) -> str:
    return _text_value(
        item.get("canonical_id")
        or item.get("raw_class_name")
        or item.get("label")
    )


def _candidate_crop(item: dict[str, Any]) -> str | None:
    direct = _canonical_crop(item.get("plant"))
    if direct:
        return direct
    scope = _candidate_scope(item)
    terminology = resolve_disease(scope) if scope else None
    return _canonical_crop(terminology.crop_en) if terminology else None


def _observed_phenotype_rationale(phenotype: PhenotypeCard) -> str:
    fields = (
        ("leaf color", phenotype.leaf_color),
        ("lesion color", phenotype.lesion_color),
        ("lesion shape", phenotype.lesion_shape),
        ("lesion distribution", phenotype.lesion_distribution),
        ("affected region", phenotype.affected_region),
        ("leaf curling", phenotype.leaf_curling),
        ("surface signs", phenotype.surface_signs),
    )
    details = [
        f"{name}={_text_value(value)}"
        for name, value in fields
        if _text_value(value)
    ]
    return "Observed phenotype card: " + "; ".join(details) + "."


def _inference_rationale(value: Any) -> list[str]:
    output: list[str] = []
    for item in _string_list(value):
        text = item.strip()
        if text.lower().startswith(("observed:", "image observation:")):
            continue
        if text.lower().startswith(("literature context:", "diagnostic inference:")):
            output.append(text)
        else:
            output.append(f"Diagnostic inference: {text}")
    return output


def _report_evidence_context(
    evidence: list[Evidence],
) -> list[dict[str, str]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "type": item.evidence_type.value,
            "source": item.source,
        }
        for item in evidence
    ]


def _canonical_crop(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace(" ", "_")
    if not text or text == "unknown":
        return None
    return {
        "corn": "maize",
        "maize": "maize",
        "bell_pepper": "pepper",
        "pepper,_bell": "pepper",
    }.get(text, text)


def _guidance_rule(definition: SkillDefinition | None) -> str:
    return "markdown_skill_loaded" if definition else "generic_stage_prompt"


class PhenotypeAnalysisSkill:
    name = "phenotype_analysis"

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig,
        definition: SkillDefinition | None = None,
    ):
        self.tools = tools
        self.config = config
        self.definition = definition

    def run(self, image_path: str | None, normalized: NormalizedInput) -> SkillOutput:
        steps = list(self.definition.steps) if self.definition else [
            "Verify the leaf region.",
            "Run visual tools.",
            "Compute deterministic severity.",
            "Build the phenotype card.",
        ]
        called: list[str] = []
        allowed_tools = set(self.definition.tools) if self.definition else set(self.tools.names())

        def call(name: str, **kwargs: Any) -> dict[str, Any]:
            if name not in allowed_tools:
                raise RuntimeError(f"Skill '{self.name}' is not allowed to call tool '{name}'")
            called.append(name)
            result = self.tools.invoke(name, **kwargs)
            return result.data if result.success else {}

        leaf_mask = call("leaf_segmenter", image_path=image_path)
        leaf_valid = bool(leaf_mask.get("mask_valid"))
        if normalized.organ and normalized.organ != "leaf":
            leaf_valid = False

        if not leaf_valid:
            card = PhenotypeCard(
                crop=normalized.crop,
                organ=normalized.organ,
                leaf_valid=False,
                unavailable_fields=[
                    "leaf_color",
                    "lesion_color",
                    "lesion_shape",
                    "lesion_distribution",
                    "affected_region",
                    "leaf_curling",
                    "severity",
                ],
            )
            return SkillOutput(
                value=card,
                evidence=[],
                trace=SkillTrace(
                    skill=self.name,
                    steps=steps[:2],
                    tool_calls=called,
                    rules_applied=[_guidance_rule(self.definition), "leaf_images_only"],
                ),
            )

        classifier = call("disease_classifier", image_path=image_path, crop=normalized.crop)
        top_k = classifier.get("top_k", [])
        inferred_crop = _canonical_crop(normalized.crop) or next(
            (
                _canonical_crop(item.get("plant"))
                for item in top_k
                if _canonical_crop(item.get("plant"))
            ),
            None,
        )
        lesion_mask = call("lesion_segmenter", image_path=image_path)
        description = call(
            "image_description",
            image_path=image_path,
            dimensions=[
                "leaf_color",
                "lesion_color",
                "lesion_shape",
                "lesion_distribution",
                "affected_region",
                "leaf_curling",
                "surface_signs",
            ],
        )

        ratio = None
        severity = None
        lesion_count = lesion_mask.get("connected_components")
        if leaf_mask.get("mask_valid") and lesion_mask.get("mask_valid"):
            leaf_area = float(leaf_mask.get("mask_area", 0))
            lesion_area = float(lesion_mask.get("mask_area", 0))
            service_ratio = lesion_mask.get("ratio")
            if isinstance(service_ratio, (int, float)):
                ratio = min(max(float(service_ratio), 0.0), 1.0)
            elif leaf_area > 0:
                ratio = min(max(lesion_area / leaf_area, 0.0), 1.0)
            if ratio is not None:
                severity_index = sum(ratio > boundary for boundary in self.config.severity_boundaries)
                severity = self.config.severity_labels[severity_index]

        unavailable = [
            field
            for field in (
                "leaf_color",
                "lesion_color",
                "lesion_shape",
                "lesion_distribution",
                "affected_region",
                "leaf_curling",
            )
            if not description.get(field)
        ]
        card = PhenotypeCard(
            crop=inferred_crop,
            organ="leaf",
            leaf_valid=True,
            leaf_color=description.get("leaf_color"),
            lesion_color=description.get("lesion_color"),
            lesion_shape=description.get("lesion_shape"),
            lesion_distribution=description.get("lesion_distribution"),
            affected_region=description.get("affected_region"),
            leaf_curling=description.get("leaf_curling"),
            surface_signs=description.get("surface_signs", []),
            lesion_count=(
                int(lesion_count)
                if lesion_count is not None
                else None
            ),
            lesion_area_ratio=round(ratio, 4) if ratio is not None else None,
            severity=severity,
            classifier_candidates=top_k,
            observations=description.get("observations", []),
            unavailable_fields=unavailable,
        )
        return SkillOutput(
            value=card,
            evidence=[],
            trace=SkillTrace(
                skill=self.name,
                steps=steps,
                tool_calls=called,
                rules_applied=[
                    _guidance_rule(self.definition),
                    "leaf_images_only",
                    "describe_visible_features_only",
                    "use_fixed_phenotype_dimensions",
                    "calculate_numeric_severity_with_code",
                    "mark_unavailable_attributes",
                ],
            ),
        )


class DiseaseDiagnosisSkill:
    name = "disease_diagnosis"

    def __init__(
        self,
        retriever: Any,
        llm: AgentModelClient | None = None,
        prompt_template: str | None = None,
        definition: SkillDefinition | None = None,
    ):
        self.retriever = retriever
        self.definition = definition
        if definition and "hybrid_retrieval" not in definition.tools:
            raise ValueError("disease_diagnosis must bind the hybrid_retrieval tool")
        self.llm = (
            llm
            if definition is None or "agent_model" in definition.tools
            else None
        )
        self.prompt_template = (
            definition.execution_prompt
            if definition
            else prompt_template
        ) or (
            "Diagnose the case using only the supplied phenotype, candidates and evidence."
        )

    def run(self, normalized: NormalizedInput, phenotype: PhenotypeCard) -> SkillOutput:
        if phenotype.leaf_valid is False:
            diagnosis = Diagnosis(
                primary_disease=None,
                differential_diagnoses=[],
                rationale=[],
                limitations=[
                    "The image was not processed because a valid leaf region was not detected."
                ],
                evidence_ids=[],
                status="insufficient_evidence",
            )
            return SkillOutput(
                value=diagnosis,
                evidence=[],
                trace=SkillTrace(
                    skill=self.name,
                    steps=["Validate leaf-only input.", "Stop image diagnosis."],
                    tool_calls=[],
                    rules_applied=[
                        _guidance_rule(self.definition),
                        "do_not_diagnose_non_leaf_images",
                    ],
                ),
            )
        target_crop = _canonical_crop(phenotype.crop or normalized.crop)
        candidate_pool = [
            item
            for item in phenotype.classifier_candidates
            if item.get("label")
            and str(item.get("condition") or "").strip().lower() != "healthy"
        ]
        crop_compatible = [
            item
            for item in candidate_pool
            if not target_crop
            or _candidate_crop(item) is None
            or _candidate_crop(item) == target_crop
        ]
        candidate_items = (crop_compatible or candidate_pool)[:3]
        candidates = [str(item["label"]).strip() for item in candidate_items]
        candidate_scopes = [_candidate_scope(item) for item in candidate_items]
        candidate_scope_by_label = {
            label: scope
            for label, scope in zip(candidates, candidate_scopes)
            if label and scope
        }
        primary = candidates[0] if candidates else None
        phenotype_payload = asdict(phenotype)
        phenotype_payload["classifier_candidates"] = candidate_items

        retrieval_query = ""
        query_planner_used = False
        query_planner_failed = False
        query_planner_error: str | None = None
        retrieval_enabled = not bool(getattr(self.retriever, "disabled", False))
        if retrieval_enabled and self.llm and candidates:
            query_system_prompt = (
                self.prompt_template
                + "\n\nExecute only the literature-query planning step of this "
                "activated skill. Write one natural-language query that explicitly "
                "names the inferred crop and every crop-compatible classifier "
                "candidate so retrieval can obtain comparative symptom evidence. "
                "Do not choose the winning disease in this step. Return JSON only "
                "as {\"query\": \"...\"}."
            )
            query_payload = {
                "normalized_input": asdict(normalized),
                "phenotype_card": phenotype_payload,
                "allowed_classifier_candidates": candidates,
                "candidate_retrieval_scope": candidate_scopes,
            }
            try:
                planned = self.llm.complete_json(
                    system_prompt=query_system_prompt,
                    user_prompt=json.dumps(query_payload, ensure_ascii=False),
                    stage="diagnosis_retrieval_planning",
                )
                retrieval_query = _text_value(planned.get("query"))
                if not retrieval_query:
                    raise ValueError("The agent model returned no valid retrieval query")
                query_planner_used = True
            except Exception as exc:
                query_planner_failed = True
                query_planner_error = str(exc)

        if not retrieval_query and primary:
            fallback_parts = _unique_text_parts(
                phenotype.crop or normalized.crop,
                candidates,
                phenotype.lesion_color,
                phenotype.lesion_shape,
                phenotype.lesion_distribution,
                phenotype.surface_signs,
                "leaf disease symptoms diagnosis",
            )
            retrieval_query = " ".join(fallback_parts)

        evidence = []
        if retrieval_enabled and retrieval_query:
            evidence = self.retriever.search(
                retrieval_query,
                crop=phenotype.crop or normalized.crop,
                section="diagnosis",
                top_k=3,
                allowed_diseases=candidate_scopes,
                content_types=["phenotype", "abstract"],
                balance_diseases=True,
            )

        rationale = [_observed_phenotype_rationale(phenotype)]
        if evidence:
            rationale.append(
                "The agent retrieved diagnostic literature for candidate comparison."
            )
        limitations = []
        if normalized.unknown_fields:
            limitations.append("Missing context: " + ", ".join(normalized.unknown_fields) + ".")
        if not primary:
            limitations.append("No disease candidate was returned by the visual classifier.")
        if query_planner_failed:
            limitations.append(
                "Agent retrieval-query planning failed; a deterministic Top-1 query "
                f"was used: {query_planner_error}"
            )
        if retrieval_enabled and retrieval_query and not evidence:
            limitations.append(
                "No literature chunks matched the crop, candidate-disease, and "
                "diagnostic content-type boundaries."
            )

        diagnosis = Diagnosis(
            primary_disease=primary,
            differential_diagnoses=candidates[1:3],
            rationale=rationale,
            limitations=limitations,
            evidence_ids=[item.evidence_id for item in evidence],
            status="diagnosed" if primary else "insufficient_evidence",
        )
        diagnosis_llm_used = False
        diagnosis_llm_failed = False
        if self.llm:
            system_prompt = (
                self.prompt_template
                + "\n\nExecute the activated skill with the supplied payload. "
                "Return JSON only with exactly these fields: "
                "primary_disease (string or null), differential_diagnoses (array of strings), "
                "rationale (array containing only literature context and diagnostic "
                "inference; never restate or add image observations), "
                "limitations (array of strings), status (diagnosed or insufficient_evidence)."
            )
            payload = {
                "normalized_input": asdict(normalized),
                "phenotype_card": phenotype_payload,
                "allowed_classifier_candidates": candidates,
                "agent_retrieval_query": retrieval_query,
                "evidence": _evidence_context(evidence),
            }
            try:
                generated = self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                    stage="diagnosis_synthesis",
                )
                generated_primary = generated.get("primary_disease")
                generated_status = str(generated.get("status") or "").strip()
                if generated_primary is not None:
                    generated_primary = str(generated_primary).strip() or None
                generated_differentials = _string_list(
                    generated.get("differential_diagnoses")
                )
                generated_rationale = _inference_rationale(generated.get("rationale"))
                generated_limitations = _string_list(generated.get("limitations"))
                if generated_primary and generated_primary not in candidates:
                    generated_limitations.append(
                        "The generated primary disease was outside the classifier Top-3; "
                        "the Top-1 fallback was retained."
                    )
                    generated_primary = primary
                if generated_primary and generated_primary != primary:
                    evidence_coverage = {
                        str(item.metadata.get("matched_disease") or "").strip()
                        for item in evidence
                        if item.metadata.get("matched_disease")
                    }
                    top1_scope = candidate_scope_by_label.get(primary or "", primary or "")
                    selected_scope = candidate_scope_by_label.get(
                        generated_primary,
                        generated_primary,
                    )
                    if not {top1_scope, selected_scope}.issubset(evidence_coverage):
                        generated_limitations.append(
                            "The alternative candidate lacked balanced comparative "
                            "literature coverage with Top-1; the Top-1 hypothesis was retained."
                        )
                        generated_primary = primary
                if generated_status not in {"diagnosed", "insufficient_evidence"}:
                    generated_status = (
                        "diagnosed" if generated_primary else "insufficient_evidence"
                    )
                if generated_primary:
                    generated_status = "diagnosed"
                diagnosis = Diagnosis(
                    primary_disease=generated_primary,
                    differential_diagnoses=[
                        item
                        for item in dict.fromkeys(generated_differentials)
                        if item in candidates and item != generated_primary
                    ][:3],
                    rationale=[
                        _observed_phenotype_rationale(phenotype),
                        *generated_rationale,
                    ],
                    limitations=generated_limitations,
                    evidence_ids=[item.evidence_id for item in evidence],
                    status=generated_status,
                )
                if diagnosis.status == "diagnosed" and not diagnosis.primary_disease:
                    diagnosis.status = "insufficient_evidence"
                diagnosis_llm_used = True
            except Exception as exc:
                diagnosis_llm_failed = True
                diagnosis.limitations.append(
                    f"Center-agent model synthesis failed; deterministic fallback used: {exc}"
                )
        return SkillOutput(
            value=diagnosis,
            evidence=evidence,
            trace=SkillTrace(
                skill=self.name,
                steps=list(self.definition.steps) if self.definition else [
                    "Initialize candidates.",
                    "Retrieve diagnostic knowledge.",
                    "Compare phenotype and knowledge.",
                    "Produce diagnosis and differentials.",
                ],
                tool_calls=[
                    "hybrid_retrieval",
                    *(
                        ["agent_model"]
                        if query_planner_used
                        or query_planner_failed
                        or diagnosis_llm_used
                        or diagnosis_llm_failed
                        else []
                    ),
                ],
                rules_applied=[
                    _guidance_rule(self.definition),
                    "top3_constrained_diagnosis",
                    "crop_compatible_candidates_only",
                    "single_agent_generated_retrieval_query",
                    *(
                        ["llm_generated_retrieval_query"]
                        if query_planner_used
                        else ["deterministic_query_fallback"]
                    ),
                    "link_diagnosis_to_visible_phenotypes",
                    "include_differential_diagnoses",
                    "state_missing_context",
                    *(
                        ["center_agent_evidence_synthesis"]
                        if diagnosis_llm_used
                        else ["deterministic_diagnosis_fallback"]
                    ),
                ],
            ),
        )

    def run_knowledge_query(self, normalized: NormalizedInput) -> SkillOutput:
        evidence = []
        retrieval_enabled = not bool(getattr(self.retriever, "disabled", False))
        retrieval_query = ""
        query_planner_used = False
        query_planner_failed = False
        terminology = extract_disease_from_text(
            normalized.original_query,
            crop=normalized.crop,
        )
        if retrieval_enabled and self.llm:
            try:
                planned = self.llm.complete_json(
                    system_prompt=(
                        self.prompt_template
                        + "\n\nRewrite the supplied agricultural question as one "
                        "high-precision scientific-literature query. Return JSON only "
                        "as {\"query\": \"...\"}."
                    ),
                    user_prompt=json.dumps(
                        {
                            "question": normalized.original_query,
                            "crop": normalized.crop,
                            "canonical_disease": (
                                terminology.canonical_id if terminology else None
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    stage="knowledge_retrieval_planning",
                )
                retrieval_query = _text_value(planned.get("query"))
                if not retrieval_query:
                    raise ValueError("The agent model returned no valid knowledge query")
                query_planner_used = True
            except Exception:
                query_planner_failed = True
        if not retrieval_query:
            retrieval_query = normalized.original_query
        if retrieval_enabled:
            evidence = self.retriever.search(
                retrieval_query,
                crop=normalized.crop,
                section="diagnosis",
                top_k=5,
                allowed_diseases=(
                    [terminology.canonical_id] if terminology else None
                ),
                content_types=["phenotype", "abstract"],
            )
        primary = next(
            (
                str(item.metadata.get("disease")).strip()
                for item in evidence
                if item.metadata.get("disease")
            ),
            None,
        )
        diagnosis = Diagnosis(
            primary_disease=primary,
            differential_diagnoses=[],
            rationale=[
                f"Relevant diagnostic knowledge was retrieved from [{item.evidence_id}]."
                for item in evidence[:3]
            ],
            limitations=[] if evidence else ["No relevant diagnostic document was retrieved."],
            evidence_ids=[item.evidence_id for item in evidence],
            status="diagnosed" if primary else "insufficient_evidence",
        )
        llm_used = False
        llm_failed = False
        if self.llm:
            system_prompt = (
                self.prompt_template
                + "\n\nExecute the activated skill for the supplied text-only payload. "
                "Return JSON with primary_disease (string or null), "
                "differential_diagnoses, rationale, limitations, and status. Use evidence IDs "
                "where applicable."
            )
            try:
                generated = self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(
                        {
                            "question": normalized.original_query,
                            "crop": normalized.crop,
                            "evidence": _evidence_context(
                                evidence,
                                max_chars=4000,
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    stage="knowledge_diagnosis_synthesis",
                )
                generated_primary = generated.get("primary_disease")
                generated_primary = (
                    str(generated_primary).strip() if generated_primary else None
                )
                diagnosis = Diagnosis(
                    primary_disease=generated_primary,
                    differential_diagnoses=_string_list(
                        generated.get("differential_diagnoses")
                    )[:3],
                    rationale=_string_list(generated.get("rationale")),
                    limitations=_string_list(generated.get("limitations")),
                    evidence_ids=[item.evidence_id for item in evidence],
                    status="diagnosed" if generated_primary else "insufficient_evidence",
                )
                llm_used = True
            except Exception as exc:
                diagnosis.limitations.append(
                    f"Center-agent synthesis failed; retrieval fallback used: {exc}"
                )
                llm_failed = True
        return SkillOutput(
            value=diagnosis,
            evidence=evidence,
            trace=SkillTrace(
                skill=self.name,
                steps=list(self.definition.steps) if self.definition else [
                    "Retrieve diagnostic knowledge.",
                    "Answer the disease knowledge question.",
                ],
                tool_calls=[
                    "hybrid_retrieval",
                    *(
                        ["agent_model"]
                        if query_planner_used
                        or query_planner_failed
                        or llm_used
                        or llm_failed
                        else []
                    ),
                ],
                rules_applied=[
                    _guidance_rule(self.definition),
                    "text_only_mode_skips_visual_tools",
                    *(
                        ["llm_generated_knowledge_retrieval_query"]
                        if query_planner_used
                        else ["original_question_query_fallback"]
                    ),
                    *(
                        ["center_agent_evidence_synthesis"]
                        if llm_used
                        else ["retrieval_diagnosis_fallback"]
                    ),
                ],
            ),
        )


class DiseaseManagementSkill:
    name = "disease_management"

    def __init__(
        self,
        retriever: Any,
        kg: Any,
        include_kg: bool = True,
        llm: AgentModelClient | None = None,
        prompt_template: str | None = None,
        definition: SkillDefinition | None = None,
    ):
        self.retriever = retriever
        self.kg = kg
        self.definition = definition
        if definition and "hybrid_retrieval" not in definition.tools:
            raise ValueError("disease_management must bind the hybrid_retrieval tool")
        kg_tools = {
            "get_management_options",
            "get_pesticide_detail",
            "get_pesticide_uses",
            "list_diseases_by_crop",
        }
        self.include_kg = (
            include_kg
            and not bool(getattr(kg, "disabled", False))
            and (definition is None or bool(kg_tools.intersection(definition.tools)))
        )
        self.llm = (
            llm
            if definition is None or "agent_model" in definition.tools
            else None
        )
        self.prompt_template = (
            definition.execution_prompt
            if definition
            else prompt_template
        ) or (
            "Create an integrated disease-management plan using only supplied evidence."
        )

    def run(self, phenotype: PhenotypeCard, diagnosis: Diagnosis) -> SkillOutput:
        if diagnosis.status != "diagnosed" or not diagnosis.primary_disease:
            plan = ManagementPlan(
                cultural_controls=["Collect a clearer image and confirm the diagnosis before treatment."],
                biological_controls=[],
                chemical_controls=[],
                monitoring=["Monitor symptom progression and record field distribution."],
                safety_notes=["Disease-specific pesticide advice is withheld because diagnosis is unavailable."],
                evidence_ids=[],
            )
            return SkillOutput(
                value=plan,
                evidence=[],
                trace=SkillTrace(
                    skill=self.name,
                    steps=["check_diagnosis", "withhold_disease_specific_chemical_advice"],
                    tool_calls=[],
                    rules_applied=[
                        _guidance_rule(self.definition),
                        "do_not_invent_treatment_without_a_diagnosis",
                    ],
                ),
            )

        disease = diagnosis.primary_disease
        evidence = []
        retrieval_enabled = not bool(getattr(self.retriever, "disabled", False))
        retrieval_query = ""
        query_planner_used = False
        query_planner_failed = False
        if retrieval_enabled and self.llm:
            query_system_prompt = (
                self.prompt_template
                + "\n\nExecute only the literature-query planning step. Generate "
                "one natural-language query for cultural control, biological control, "
                "epidemiology, monitoring, and general chemical-control context for "
                "the supplied crop and diagnosed disease. Return JSON only as "
                "{\"query\": \"...\"}."
            )
            try:
                planned = self.llm.complete_json(
                    system_prompt=query_system_prompt,
                    user_prompt=json.dumps(
                        {
                            "crop": phenotype.crop,
                            "diagnosed_disease": disease,
                            "severity": phenotype.severity,
                        },
                        ensure_ascii=False,
                    ),
                    stage="management_retrieval_planning",
                )
                retrieval_query = _text_value(planned.get("query"))
                if not retrieval_query:
                    raise ValueError("The agent model returned no valid management query")
                query_planner_used = True
            except Exception:
                query_planner_failed = True
        if not retrieval_query:
            retrieval_query = (
                f"{phenotype.crop} {disease} cultural biological management "
                "monitoring epidemiology"
            )
        if retrieval_enabled:
            evidence = self.retriever.search(
                retrieval_query,
                crop=phenotype.crop,
                disease=disease,
                section="management",
                top_k=3,
                content_types=["disease_management", "field_trial"],
            )
        if self.include_kg:
            evidence += self.kg.get_management_options(
                crop=phenotype.crop,
                disease=disease,
                limit=3,
            )

        cultural = [
            "Remove heavily affected residues where agronomically appropriate.",
            "Improve airflow and avoid prolonged leaf wetness.",
            "Use resistant cultivars and crop rotation when available.",
        ]
        biological = ["Use locally registered biological control products when supported by extension guidance."]
        chemical = [
            "If treatment is economically justified, select a locally registered fungicide for the diagnosed disease.",
            "Rotate products with different modes of action to reduce resistance risk.",
        ]
        monitoring = [
            f"Monitor lesion development; current deterministic severity is {phenotype.severity or 'unknown'}.",
            "Reassess after management and document weather and field distribution.",
        ]
        safety = [
            "Follow the local product label for crop registration, dose, timing, re-entry and pre-harvest intervals.",
            "Do not infer a legal application rate from this report.",
        ]
        plan = ManagementPlan(
            cultural_controls=cultural,
            biological_controls=biological,
            chemical_controls=chemical,
            monitoring=monitoring,
            safety_notes=safety,
            evidence_ids=[item.evidence_id for item in evidence],
        )
        llm_used = False
        llm_failed = False
        if self.llm:
            system_prompt = (
                self.prompt_template
                + "\n\nExecute the activated skill with the supplied payload. "
                "Use scientific-literature evidence for cultural, biological, "
                "epidemiological, and monitoring guidance. When pesticide-registration "
                "evidence is present, use it as the sole source for named products, "
                "registration numbers, labelled doses, application methods, and "
                "product-specific precautions. "
                "Return JSON only with exactly these array fields: cultural_controls, "
                "biological_controls, chemical_controls, monitoring, safety_notes."
            )
            payload = {
                "crop": phenotype.crop,
                "severity": phenotype.severity,
                "diagnosis": asdict(diagnosis),
                "evidence": _evidence_context(evidence),
            }
            try:
                generated = self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                    stage="management_synthesis",
                )
                candidate_plan = ManagementPlan(
                    cultural_controls=_string_list(generated.get("cultural_controls")),
                    biological_controls=_string_list(generated.get("biological_controls")),
                    chemical_controls=_string_list(generated.get("chemical_controls"))[:3],
                    monitoring=_string_list(generated.get("monitoring")),
                    safety_notes=_string_list(generated.get("safety_notes")),
                    evidence_ids=[item.evidence_id for item in evidence],
                )
                if any(
                    (
                        candidate_plan.cultural_controls,
                        candidate_plan.biological_controls,
                        candidate_plan.chemical_controls,
                        candidate_plan.monitoring,
                    )
                ):
                    plan = candidate_plan
                    llm_used = True
                else:
                    raise ValueError("Agent model returned an empty management plan")
            except Exception:
                llm_failed = True
        return SkillOutput(
            value=plan,
            evidence=evidence,
            trace=SkillTrace(
                skill=self.name,
                steps=list(self.definition.steps) if self.definition else [
                    "Read diagnosis and severity.",
                    "Retrieve management knowledge.",
                    "Query disease-management relations.",
                    "Compose integrated management.",
                    "Append safety constraints.",
                ],
                tool_calls=[
                    *(
                        ["hybrid_retrieval", "get_management_options"]
                        if self.include_kg
                        else ["hybrid_retrieval"]
                    ),
                    *(
                        ["agent_model"]
                        if query_planner_used
                        or query_planner_failed
                        or llm_used
                        or llm_failed
                        else []
                    ),
                ],
                rules_applied=[
                    _guidance_rule(self.definition),
                    "prioritize_integrated_management",
                    *(
                        ["llm_generated_management_retrieval_query"]
                        if query_planner_used
                        else ["deterministic_management_query_fallback"]
                    ),
                    "crop_disease_content_type_filter",
                    "literature_kg_role_separation",
                    "do_not_invent_dose_or_registration",
                    "recommend_mode_of_action_rotation",
                    "defer_to_local_labels",
                    *(
                        ["center_agent_management_synthesis"]
                        if llm_used
                        else ["deterministic_management_fallback"]
                    ),
                ],
            ),
        )

    def run_knowledge_query(
        self,
        normalized: NormalizedInput,
        diagnosis: Diagnosis | None = None,
        *,
        operation: str | None = None,
        entities: dict[str, str | None] | None = None,
    ) -> SkillOutput:
        structured_entities = entities or {}
        crop = structured_entities.get("crop") or normalized.crop
        disease = (
            diagnosis.primary_disease
            if diagnosis and diagnosis.primary_disease
            else structured_entities.get("disease")
        )
        pesticide_name = structured_entities.get("pesticide_name")
        evidence = []
        retrieval_enabled = not bool(getattr(self.retriever, "disabled", False))
        retrieval_query = ""
        query_planner_used = False
        query_planner_failed = False
        if retrieval_enabled and self.llm:
            try:
                planned = self.llm.complete_json(
                    system_prompt=(
                        self.prompt_template
                        + "\n\nRewrite the supplied agricultural management question "
                        "as one high-precision scientific-literature query. Include the "
                        "known crop, disease, and requested management aspect. Return "
                        "JSON only as {\"query\": \"...\"}."
                    ),
                    user_prompt=json.dumps(
                        {
                            "question": normalized.original_query,
                            "crop": crop,
                            "disease": disease,
                            "operation": operation,
                            "pesticide_name": pesticide_name,
                        },
                        ensure_ascii=False,
                    ),
                    stage="management_knowledge_retrieval_planning",
                )
                retrieval_query = _text_value(planned.get("query"))
                if not retrieval_query:
                    raise ValueError("The agent model returned no valid management query")
                query_planner_used = True
            except Exception:
                query_planner_failed = True
        if not retrieval_query:
            retrieval_query = normalized.original_query
        if retrieval_enabled:
            evidence = self.retriever.search(
                retrieval_query,
                crop=crop,
                disease=disease,
                section="management",
                top_k=3,
                content_types=["disease_management", "field_trial"],
            )
        kg_tool_calls: list[str] = []
        kg_call_withheld = False
        if self.include_kg:
            if operation == "disease_management" and disease:
                evidence += self.kg.get_management_options(
                    crop=crop,
                    disease=disease,
                    limit=5,
                )
                kg_tool_calls.append("get_management_options")
            elif operation == "pesticide_detail" and pesticide_name:
                evidence += self.kg.get_pesticide_detail(
                    pesticide_name=pesticide_name
                )
                kg_tool_calls.append("get_pesticide_detail")
            elif operation == "pesticide_uses" and pesticide_name:
                evidence += self.kg.get_pesticide_uses(
                    pesticide_name=pesticide_name
                )
                kg_tool_calls.append("get_pesticide_uses")
            elif operation == "crop_disease_list" and crop:
                evidence += self.kg.list_diseases_by_crop(crop=crop)
                kg_tool_calls.append("list_diseases_by_crop")
            elif operation:
                kg_call_withheld = True
        evidence = list({item.evidence_id: item for item in evidence}.values())
        plan = ManagementPlan(
            cultural_controls=[
                "Use crop sanitation, canopy and moisture management measures supported by the retrieved guidance."
            ] if evidence else [],
            biological_controls=[],
            chemical_controls=[
                "Use only locally registered products whose crop and disease indications match the current label."
            ] if evidence else [],
            monitoring=["Monitor symptom development and field distribution."],
            safety_notes=[
                "Do not infer a legal dose, interval, or registration from this answer; follow the current local label."
            ],
            evidence_ids=[item.evidence_id for item in evidence],
        )
        llm_used = False
        llm_failed = False
        if self.llm:
            system_prompt = (
                self.prompt_template
                + "\n\nExecute the activated skill for the supplied text-only payload. "
                "Use literature for cultural, biological, epidemiological, and "
                "monitoring guidance. Use pesticide-registration evidence as the sole "
                "source for named products, registrations, labelled doses, application "
                "methods, and product-specific precautions. "
                "Return JSON with exactly these array fields: "
                "cultural_controls, biological_controls, chemical_controls, monitoring, "
                "safety_notes."
            )
            try:
                generated = self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(
                        {
                            "question": normalized.original_query,
                            "crop": crop,
                            "disease_context": disease,
                            "management_operation": operation,
                            "pesticide_name": pesticide_name,
                            "evidence": _evidence_context(evidence),
                        },
                        ensure_ascii=False,
                    ),
                    stage="knowledge_management_synthesis",
                )
                candidate = ManagementPlan(
                    cultural_controls=_string_list(generated.get("cultural_controls")),
                    biological_controls=_string_list(generated.get("biological_controls")),
                    chemical_controls=_string_list(generated.get("chemical_controls"))[:3],
                    monitoring=_string_list(generated.get("monitoring")),
                    safety_notes=_string_list(generated.get("safety_notes")),
                    evidence_ids=[item.evidence_id for item in evidence],
                )
                if any(
                    (
                        candidate.cultural_controls,
                        candidate.biological_controls,
                        candidate.chemical_controls,
                        candidate.monitoring,
                    )
                ):
                    plan = candidate
                    llm_used = True
            except Exception:
                llm_failed = True
        return SkillOutput(
            value=plan,
            evidence=evidence,
            trace=SkillTrace(
                skill=self.name,
                steps=list(self.definition.steps) if self.definition else [
                    "Retrieve management knowledge.",
                    "Compose an integrated management answer.",
                ],
                tool_calls=[
                    "hybrid_retrieval",
                    *kg_tool_calls,
                    *(
                        ["agent_model"]
                        if query_planner_used
                        or query_planner_failed
                        or llm_used
                        or llm_failed
                        else []
                    ),
                ],
                rules_applied=[
                    _guidance_rule(self.definition),
                    "text_only_mode_skips_visual_tools",
                    *(
                        ["llm_generated_management_retrieval_query"]
                        if query_planner_used
                        else ["original_question_query_fallback"]
                    ),
                    "crop_disease_content_type_filter",
                    "literature_kg_role_separation",
                    "structured_management_operation",
                    "kg_tool_parameters_validated",
                    "do_not_invent_dose_or_registration",
                    *(
                        ["kg_call_withheld_missing_entity"]
                        if kg_call_withheld
                        else []
                    ),
                    *(
                        ["center_agent_management_synthesis"]
                        if llm_used
                        else ["retrieval_management_fallback"]
                    ),
                ],
            ),
        )


class ReportGenerationSkill:
    name = "report_generation"

    def __init__(
        self,
        llm: AgentModelClient | None = None,
        prompt_template: str | None = None,
        definition: SkillDefinition | None = None,
    ):
        self.definition = definition
        self.llm = (
            llm
            if definition is None or "agent_model" in definition.tools
            else None
        )
        self.prompt_template = (
            definition.execution_prompt
            if definition
            else prompt_template
        ) or (
            "Write a professional structured plant-disease report from supplied objects."
        )

    @staticmethod
    def _items(values: list[str]) -> str:
        return "\n".join(f"- {value}" for value in values) if values else "- None"

    @staticmethod
    def _authoritative_phenotype_section(
        phenotype: PhenotypeCard,
        classifier_line: str,
    ) -> str:
        fields = (
            ("Leaf color", phenotype.leaf_color),
            ("Lesion color", phenotype.lesion_color),
            ("Lesion shape", phenotype.lesion_shape),
            ("Lesion distribution", phenotype.lesion_distribution),
            ("Affected region", phenotype.affected_region),
            ("Leaf curling", phenotype.leaf_curling),
            (
                "Surface signs",
                ", ".join(phenotype.surface_signs)
                if phenotype.surface_signs
                else None,
            ),
        )
        lesion_area = (
            f"{phenotype.lesion_area_ratio * 100:.2f}%"
            if phenotype.lesion_area_ratio is not None
            else "unavailable"
        )
        lines = [
            "## Phenotype observations",
            "",
            classifier_line,
            *[
                f"- {label}: {value if value is not None else 'unavailable'}"
                for label, value in fields
            ],
            f"- Segmented lesion area ratio: {lesion_area} (segmentation-derived estimate)",
            (
                "- Connected lesion regions: "
                f"{phenotype.lesion_count if phenotype.lesion_count is not None else 'unavailable'} "
                "(segmentation-derived)"
            ),
            f"- Severity: {phenotype.severity or 'unavailable'} (deterministic from area ratio)",
        ]
        if phenotype.unavailable_fields:
            lines.append(
                "- Unavailable visual fields: "
                + ", ".join(phenotype.unavailable_fields)
            )
        return "\n".join(lines)

    @staticmethod
    def _enforce_phenotype_section(report: str, section: str) -> str:
        pattern = re.compile(
            r"(?ims)^##\s+(?:phenotype[^\n]*|visual\s+(?:observations|findings|characteristics)[^\n]*)\s*$"
            r".*?(?=^##\s+|\Z)"
        )
        if pattern.search(report):
            return pattern.sub(section + "\n\n", report, count=1).rstrip()

        diagnosis_heading = re.search(r"(?im)^##\s+diagnosis[^\n]*$", report)
        if diagnosis_heading:
            return (
                report[: diagnosis_heading.start()].rstrip()
                + "\n\n"
                + section
                + "\n\n"
                + report[diagnosis_heading.start() :].lstrip()
            )

        first_line, separator, remainder = report.partition("\n")
        if first_line.startswith("# Plant Disease Diagnostic Report"):
            return (
                first_line.rstrip()
                + "\n\n"
                + section
                + ("\n\n" + remainder.lstrip() if separator and remainder.strip() else "")
            )
        return section + "\n\n" + report.lstrip()

    def run(
        self,
        normalized: NormalizedInput,
        phenotype: PhenotypeCard,
        diagnosis: Diagnosis,
        management: ManagementPlan,
        evidence: list[Evidence],
        goals: list[str] | None = None,
    ) -> SkillOutput:
        goals = goals or ["phenotype", "diagnosis", "management"]
        leading_candidate = (
            phenotype.classifier_candidates[0]
            if phenotype.classifier_candidates
            else {}
        )
        candidate_name = (
            leading_candidate.get("display_name_en")
            or leading_candidate.get("label")
            or leading_candidate.get("raw_class_name")
        )
        candidate_probability = leading_candidate.get("probability")
        classifier_line = "- Classifier output: unavailable"
        if candidate_name:
            classifier_line = f"- Classifier output: {candidate_name}"
            if isinstance(candidate_probability, (int, float)):
                classifier_line += (
                    f" (model confidence: {candidate_probability * 100:.2f}%)"
                )
        phenotype_section = self._authoritative_phenotype_section(
            phenotype,
            classifier_line,
        )
        sections = [
            "# Plant Disease Diagnostic Report",
            "",
            "## Case",
            "",
            f"- Crop: {phenotype.crop or 'unknown'}",
            f"- Organ: {phenotype.organ or 'unknown'}",
            f"- User request: {normalized.original_query}",
            "",
            phenotype_section,
        ]
        if "diagnosis" in goals or "management" in goals:
            sections.extend(
                [
                    "",
                    "## Diagnosis",
                    "",
                    f"- Primary diagnosis: {diagnosis.primary_disease or 'insufficient evidence'}",
                    (
                        "- Differential diagnoses: "
                        f"{', '.join(diagnosis.differential_diagnoses) or 'none'}"
                    ),
                    "",
                    "### Diagnostic rationale",
                    "",
                    self._items(diagnosis.rationale),
                    "",
                    "### Limitations",
                    "",
                    self._items(diagnosis.limitations),
                ]
            )
        if "management" in goals:
            sections.extend(
                [
                    "",
                    "## Integrated disease management",
                    "",
                    "### Cultural controls",
                    "",
                    self._items(management.cultural_controls),
                    "",
                    "### Biological controls",
                    "",
                    self._items(management.biological_controls),
                    "",
                    "### Chemical controls",
                    "",
                    self._items(management.chemical_controls),
                    "",
                    "### Monitoring",
                    "",
                    self._items(management.monitoring),
                    "",
                    "### Safety and regulatory notes",
                    "",
                    self._items(management.safety_notes),
                ]
            )
        if evidence:
            sections.extend(
                [
                    "",
                    "## Evidence",
                    "",
                    self._items(
                        [
                            f"[{item.evidence_id}] {item.source}: {item.content}"
                            for item in evidence
                        ]
                    ),
                ]
            )
        fallback_report = "\n".join(sections)
        report = fallback_report
        llm_used = False
        llm_failed = False
        if self.llm:
            system_prompt = (
                self.prompt_template
                + "\n\nExecute the activated skill with the supplied image-case "
                "objects. Return Markdown only. The exact Markdown supplied as "
                "authoritative_phenotype_section must appear once and unchanged. "
                "Do not restate, reinterpret, infer, or add visible image findings "
                "in any other section. Unknown or unavailable findings must never "
                "be rewritten as absent."
            )
            payload = {
                "normalized_input": asdict(normalized),
                "phenotype_card": asdict(phenotype),
                "authoritative_phenotype_section": phenotype_section,
                "diagnosis": asdict(diagnosis),
                "management_plan": asdict(management),
                "evidence_references": _report_evidence_context(evidence),
                "goals": goals,
            }
            try:
                report = self.llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                    stage="report_generation",
                )
                if not report.startswith("# Plant Disease Diagnostic Report"):
                    report = "# Plant Disease Diagnostic Report\n\n" + report
                report = self._enforce_phenotype_section(
                    report,
                    phenotype_section,
                )
                llm_used = True
            except Exception:
                report = fallback_report
                llm_failed = True
        return SkillOutput(
            value=report,
            evidence=[],
            trace=SkillTrace(
                skill=self.name,
                steps=list(self.definition.steps) if self.definition else [
                    "Validate stage outputs.",
                    "Separate observations, inference, and knowledge.",
                    "Compose the structured report.",
                    "Append evidence and limitations.",
                ],
                tool_calls=["agent_model"] if llm_used or llm_failed else [],
                rules_applied=[
                    _guidance_rule(self.definition),
                    "do_not_add_unobserved_symptoms",
                    "keep_diagnosis_and_management_consistent",
                    "retain_evidence_identifiers",
                    "make_limitations_visible",
                    *(
                        ["center_agent_report_synthesis"]
                        if llm_used
                        else ["deterministic_report_fallback"]
                    ),
                ],
            ),
        )

    def run_knowledge_qa(
        self,
        normalized: NormalizedInput,
        diagnosis: Diagnosis,
        management: ManagementPlan,
        evidence: list[Evidence],
        goals: list[str],
    ) -> SkillOutput:
        evidence_lines = [
            f"- [{item.evidence_id}] {item.source}: {item.content}"
            for item in evidence
        ]
        sections = ["# Agricultural Disease Knowledge Answer", "", normalized.original_query]
        if "diagnosis" in goals:
            sections.extend(
                [
                    "",
                    "## Disease information",
                    "",
                    f"- Disease: {diagnosis.primary_disease or 'not determined'}",
                    *[f"- {item}" for item in diagnosis.rationale],
                    *[f"- Limitation: {item}" for item in diagnosis.limitations],
                ]
            )
        if "management" in goals:
            sections.extend(
                [
                    "",
                    "## Management",
                    "",
                    "### Cultural controls",
                    self._items(management.cultural_controls),
                    "",
                    "### Biological controls",
                    self._items(management.biological_controls),
                    "",
                    "### Chemical controls",
                    self._items(management.chemical_controls),
                    "",
                    "### Monitoring and safety",
                    self._items([*management.monitoring, *management.safety_notes]),
                ]
            )
        sections.extend(["", "## Evidence", "", *(evidence_lines or ["- No evidence retrieved."])])
        report = "\n".join(sections)
        llm_used = False
        llm_failed = False
        if self.llm:
            try:
                report = self.llm.complete(
                    system_prompt=(
                        self.prompt_template
                        + "\n\nExecute the activated skill with the supplied text-only "
                        "objects. Return Markdown only."
                    ),
                    user_prompt=json.dumps(
                        {
                            "question": normalized.original_query,
                            "goals": goals,
                            "diagnosis": asdict(diagnosis),
                            "management": asdict(management),
                            "evidence_references": _report_evidence_context(evidence),
                        },
                        ensure_ascii=False,
                    ),
                    stage="knowledge_report_generation",
                )
                if not report.startswith("# Agricultural Disease Knowledge Answer"):
                    report = "# Agricultural Disease Knowledge Answer\n\n" + report
                llm_used = True
            except Exception:
                llm_failed = True
        return SkillOutput(
            value=report,
            evidence=[],
            trace=SkillTrace(
                skill=self.name,
                steps=list(self.definition.steps) if self.definition else [
                    "Validate selected skill outputs.",
                    "Compose a direct knowledge answer.",
                ],
                tool_calls=["agent_model"] if llm_used or llm_failed else [],
                rules_applied=[
                    _guidance_rule(self.definition),
                    "omit_irrelevant_phenotype_sections",
                    "retain_evidence_identifiers",
                    *(
                        ["center_agent_report_synthesis"]
                        if llm_used
                        else ["deterministic_report_fallback"]
                    ),
                ],
            ),
        )
