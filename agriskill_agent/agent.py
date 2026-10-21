from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable

from .config import AgentConfig
from .models import (
    AgentResponse,
    Diagnosis,
    ManagementPlan,
    PhenotypeCard,
    UserCase,
)
from .modules import InputNormalizer
from .planning import SkillPlanner
from .skills import (
    DiseaseDiagnosisSkill,
    DiseaseManagementSkill,
    PhenotypeAnalysisSkill,
    ReportGenerationSkill,
)
from .skill_definitions import SkillCatalog
from .tools import ToolRegistry


class _ObservableRetriever:
    def __init__(self, target: Any, sink: Callable[..., None]):
        self._target = target
        self._sink = sink
        self.disabled = bool(getattr(target, "disabled", False))

    def search(self, query: str, **kwargs: Any) -> Any:
        params = {"query": query, **kwargs}
        self._sink("tool_start", name="hybrid_retrieval", params=params)
        started = time.perf_counter()
        try:
            result = self._target.search(query, **kwargs)
            self._sink(
                "tool_end",
                name="hybrid_retrieval",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                result_count=len(result),
                success=True,
                result=result,
            )
            return result
        except Exception as exc:
            self._sink(
                "tool_error",
                name="hybrid_retrieval",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(exc),
                result=None,
            )
            raise


class _ObservableKnowledgeGraph:
    def __init__(self, target: Any, sink: Callable[..., None]):
        self._target = target
        self._sink = sink
        self.disabled = bool(getattr(target, "disabled", False))

    def _call(self, tool_name: str, **kwargs: Any) -> Any:
        self._sink("tool_start", name=tool_name, params=kwargs)
        started = time.perf_counter()
        try:
            result = getattr(self._target, tool_name)(**kwargs)
            self._sink(
                "tool_end",
                name=tool_name,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                result_count=len(result),
                success=True,
                result=result,
            )
            return result
        except Exception as exc:
            self._sink(
                "tool_error",
                name=tool_name,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(exc),
                result=None,
            )
            raise

    def get_management_options(self, **kwargs: Any) -> Any:
        return self._call("get_management_options", **kwargs)

    def get_pesticide_uses(self, **kwargs: Any) -> Any:
        return self._call("get_pesticide_uses", **kwargs)

    def get_pesticide_detail(self, **kwargs: Any) -> Any:
        return self._call("get_pesticide_detail", **kwargs)

    def list_diseases_by_crop(self, **kwargs: Any) -> Any:
        return self._call("list_diseases_by_crop", **kwargs)


class PlantDiseaseAgent:
    def __init__(
        self,
        tools: ToolRegistry,
        retriever: Any,
        knowledge_graph: Any,
        llm: Any | None,
        config: AgentConfig | None = None,
        skill_catalog: SkillCatalog | None = None,
        use_skill_guidance: bool = True,
        event_sink: Callable[..., None] | None = None,
    ):
        self.config = config or AgentConfig()
        self.tools = tools
        self.normalizer = InputNormalizer()
        self.retriever = (
            _ObservableRetriever(retriever, event_sink)
            if event_sink
            else retriever
        )
        self.kg = (
            _ObservableKnowledgeGraph(knowledge_graph, event_sink)
            if event_sink
            else knowledge_graph
        )
        self.llm = llm
        self.event_sink = event_sink
        if self.llm is not None and hasattr(self.llm, "set_event_sink"):
            self.llm.set_event_sink(event_sink)
        self.tools.set_event_sink(event_sink)
        skills_path = os.getenv("AGRISKILL_SKILLS_DIR", "").strip()
        if skill_catalog is not None:
            self.skill_catalog = skill_catalog
        elif skills_path:
            candidate = Path(skills_path).expanduser()
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parents[1] / candidate
            self.skill_catalog = SkillCatalog.from_directory(candidate)
        else:
            self.skill_catalog = SkillCatalog.bundled()
        self.use_skill_guidance = use_skill_guidance
        self.skill_planner = SkillPlanner(self.skill_catalog, llm=self.llm)
        skill_definition = (
            self.skill_catalog.get if self.use_skill_guidance else lambda _name: None
        )
        self.phenotype_skill = PhenotypeAnalysisSkill(
            self.tools,
            self.config,
            definition=skill_definition("phenotype_analysis"),
        )
        self.diagnosis_skill = DiseaseDiagnosisSkill(
            self.retriever,
            llm=self.llm,
            definition=skill_definition("disease_diagnosis"),
        )
        self.management_skill = DiseaseManagementSkill(
            self.retriever,
            self.kg,
            self.config.include_kg,
            llm=self.llm,
            definition=skill_definition("disease_management"),
        )
        self.report_skill = ReportGenerationSkill(
            llm=self.llm,
            definition=skill_definition("report_generation"),
        )

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_sink:
            self.event_sink(event, **payload)

    def run(self, case: UserCase) -> AgentResponse:
        run_started = time.perf_counter()
        if self.llm is not None and hasattr(self.llm, "reset_usage"):
            self.llm.reset_usage()
        self._emit("input", query=case.query, image=case.image_path)
        normalized = self.normalizer.normalize(case)
        plan = self.skill_planner.plan(case, normalized)
        normalized.intents = list(plan.goals)
        if not normalized.crop and plan.entities.get("crop"):
            normalized.crop = plan.entities["crop"]
        self._emit(
            "intent",
            intents=normalized.intents,
            crop=normalized.crop,
            organ=normalized.organ,
            excluded_goals=list(plan.excluded_goals),
        )
        self._emit(
            "plan",
            mode=plan.mode,
            planner=plan.planner,
            skills=list(plan.selected_skills),
            management_operation=plan.management_operation,
            entities=plan.entities,
            excluded_goals=list(plan.excluded_goals),
        )
        activated = set(plan.selected_skills)
        traces = []
        phenotype = PhenotypeCard(
            crop=normalized.crop,
            organ=normalized.organ,
            leaf_valid=None,
        )
        diagnosis = Diagnosis(
            primary_disease=None,
            differential_diagnoses=[],
            rationale=[],
            limitations=[],
            evidence_ids=[],
            status="not_requested",
        )
        management = ManagementPlan(
            cultural_controls=[],
            biological_controls=[],
            chemical_controls=[],
            monitoring=[],
            safety_notes=[],
            evidence_ids=[],
        )
        diagnosis_evidence = []
        management_evidence = []

        if plan.mode == "image_case":
            self._emit("skill_start", name="phenotype_analysis")
            phenotype_output = self.phenotype_skill.run(case.image_path, normalized)
            phenotype = phenotype_output.value
            traces.append(phenotype_output.trace)
            self._emit(
                "skill_end",
                name="phenotype_analysis",
                summary=f"crop={phenotype.crop or 'unknown'}, severity={phenotype.severity or 'unknown'}",
            )
            if phenotype.crop and not normalized.crop:
                normalized.crop = phenotype.crop
            if phenotype.leaf_valid:
                normalized.organ = "leaf"
            if "disease_diagnosis" in activated:
                self._emit("skill_start", name="disease_diagnosis")
                diagnosis_output = self.diagnosis_skill.run(normalized, phenotype)
                diagnosis = diagnosis_output.value
                diagnosis_evidence = diagnosis_output.evidence
                traces.append(diagnosis_output.trace)
                self._emit(
                    "skill_end",
                    name="disease_diagnosis",
                    summary=f"primary={diagnosis.primary_disease or 'none'}",
                )
            if "disease_management" in activated:
                self._emit("skill_start", name="disease_management")
                management_output = self.management_skill.run(phenotype, diagnosis)
                management = management_output.value
                management_evidence = management_output.evidence
                traces.append(management_output.trace)
                self._emit("skill_end", name="disease_management")
        else:
            if "disease_diagnosis" in activated:
                self._emit("skill_start", name="disease_diagnosis")
                diagnosis_output = self.diagnosis_skill.run_knowledge_query(normalized)
                diagnosis = diagnosis_output.value
                diagnosis_evidence = diagnosis_output.evidence
                traces.append(diagnosis_output.trace)
                self._emit(
                    "skill_end",
                    name="disease_diagnosis",
                    summary=f"primary={diagnosis.primary_disease or 'none'}",
                )
            if "disease_management" in activated:
                self._emit("skill_start", name="disease_management")
                management_output = self.management_skill.run_knowledge_query(
                    normalized,
                    diagnosis if diagnosis.status != "not_requested" else None,
                    operation=plan.management_operation,
                    entities=plan.entities,
                )
                management = management_output.value
                management_evidence = management_output.evidence
                traces.append(management_output.trace)
                self._emit(
                    "skill_end",
                    name="disease_management",
                    summary=f"operation={plan.management_operation or 'rag_only'}",
                )

        evidence = [*diagnosis_evidence, *management_evidence]
        evidence = list({item.evidence_id: item for item in evidence}.values())
        if plan.mode == "knowledge_qa":
            self._emit("skill_start", name="report_generation")
            report_output = self.report_skill.run_knowledge_qa(
                normalized,
                diagnosis,
                management,
                evidence,
                list(plan.goals),
            )
        else:
            self._emit("skill_start", name="report_generation")
            report_output = self.report_skill.run(
                normalized,
                phenotype,
                diagnosis,
                management,
                evidence,
                list(plan.goals),
            )
        traces.append(report_output.trace)
        self._emit("skill_end", name="report_generation")
        self._emit("report", message="Structured diagnosis report generated.")
        response = AgentResponse(
            normalized_input=normalized,
            phenotype_card=phenotype,
            diagnosis=diagnosis,
            management=management,
            report=report_output.value,
            evidence=evidence,
            traces=traces,
            runtime={
                "center_agent_enabled": self.llm is not None,
                "center_agent_model": getattr(self.llm, "model", None),
                "visual_description_model": os.getenv("IMAGE_MODEL"),
                "execution_plan": plan.to_dict(),
                "activated_skills": list(plan.selected_skills),
                "skill_loading": (
                    "metadata_then_selected_markdown_body"
                    if self.use_skill_guidance
                    else "planner_metadata_with_generic_stage_prompts"
                ),
                "center_agent_stages": {
                    "diagnosis": self._stage_runtime(
                        traces,
                        activated,
                        "disease_diagnosis",
                        "center_agent_evidence_synthesis",
                    ),
                    "management": self._stage_runtime(
                        traces,
                        activated,
                        "disease_management",
                        "center_agent_management_synthesis",
                    ),
                    "report": self._stage_runtime(
                        traces,
                        activated,
                        "report_generation",
                        "center_agent_report_synthesis",
                    ),
                },
                "llm_usage": self._llm_usage(),
                "total_elapsed_ms": round(
                    (time.perf_counter() - run_started) * 1000,
                    3,
                ),
            },
        )
        return response

    def _llm_usage(self) -> dict[str, Any]:
        if self.llm is None:
            return {
                "available": False,
                "logical_calls": 0,
                "http_attempts": 0,
                "retry_count": 0,
                "calls": [],
            }
        if not hasattr(self.llm, "usage_summary"):
            return {
                "available": False,
                "reason": "Configured model client does not expose usage telemetry.",
            }
        return {
            "available": True,
            **self.llm.usage_summary(),
        }

    @staticmethod
    def _stage_runtime(
        traces: list[Any],
        activated: set[str],
        skill: str,
        llm_rule: str,
    ) -> str:
        if skill not in activated:
            return "not_activated"
        rules = next(
            (trace.rules_applied for trace in traces if trace.skill == skill),
            [],
        )
        return "llm" if llm_rule in rules else "fallback"
