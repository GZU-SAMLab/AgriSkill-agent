from __future__ import annotations

import base64
import json
import os
import pickle
import re
import threading
import time
import urllib.error
import urllib.request
from hashlib import sha1
from pathlib import Path
from typing import Any

from .models import Evidence, EvidenceType, ToolResult
from .terminology import (
    crop_query_terms,
    disease_query_terms,
    resolve_disease,
)
from .tools import Tool, ToolRegistry


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class PlantAPITool(Tool):
    def __init__(
        self,
        name: str,
        endpoint: str,
        normalizer: Any,
        timeout_seconds: float = 180.0,
    ):
        self.name = name
        self.description = f"Standalone plant visual service: {name}"
        self.endpoint = endpoint
        self.normalizer = normalizer
        self.timeout_seconds = timeout_seconds

    def invoke(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        image_path = kwargs.get("image_path")
        if not image_path:
            return ToolResult(tool=self.name, success=False, error="image_path is required")
        payload = {"image_path": str(Path(image_path).resolve())}
        if self.name == "disease_classifier":
            payload["topk"] = int(kwargs.get("topk", 5))
        try:
            raw = _post_json(self.endpoint, payload, self.timeout_seconds)
            if raw.get("code") != 200:
                raise RuntimeError(raw.get("error") or f"Visual API returned code {raw.get('code')}")
            return ToolResult(
                tool=self.name,
                success=True,
                data=self.normalizer(raw.get("data") or {}),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )


class _SegmentationCache:
    def __init__(self, endpoint: str, timeout_seconds: float = 300.0):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, image_path: str) -> dict[str, Any]:
        resolved = str(Path(image_path).resolve())
        with self._lock:
            if resolved in self._cache:
                return self._cache[resolved]
        raw = _post_json(
            self.endpoint,
            {"image_path": resolved, "save_masks": True},
            self.timeout_seconds,
        )
        if raw.get("code") != 200:
            raise RuntimeError(raw.get("error") or f"Visual API returned code {raw.get('code')}")
        data = raw.get("data") or {}
        with self._lock:
            self._cache[resolved] = data
        return data


class SegmentationViewTool(Tool):
    def __init__(self, name: str, cache: _SegmentationCache):
        self.name = name
        self.description = f"Standalone segmentation view: {name}"
        self.cache = cache

    def invoke(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        image_path = kwargs.get("image_path")
        if not image_path:
            return ToolResult(tool=self.name, success=False, error="image_path is required")
        try:
            data = self.cache.get(str(image_path))
            if self.name == "leaf_segmenter":
                normalized = {
                    "mask_area": data.get("leaf_area"),
                    "mask_valid": bool(data.get("leaf_area", 0)),
                    "mask_path": (data.get("mask_paths") or {}).get("leaf"),
                }
            else:
                normalized = {
                    "mask_area": data.get(
                        "lesion_on_leaf_area",
                        data.get("lesion_area"),
                    ),
                    "mask_valid": data.get("lesion_area") is not None,
                    "connected_components": data.get(
                        "connected_lesion_regions"
                    ),
                    "ratio": data.get("ratio"),
                    "severity_level": data.get("level"),
                    "min_region_area": data.get("min_lesion_region_area"),
                    "mask_path": (data.get("mask_paths") or {}).get("lesion"),
                    "overlay_path": (data.get("mask_paths") or {}).get("overlay"),
                }
            return ToolResult(
                tool=self.name,
                success=True,
                data=normalized,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )


def _extract_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(part.strip() for part in text.split("```") if "{" in part)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        candidate = candidate.removeprefix("json").strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {}


def normalize_image_description(description: str) -> dict[str, Any]:
    parsed = _extract_json_object(description)
    aliases = {
        "leaf_color": ("leaf_color", "leaf colour", "leaf color"),
        "lesion_color": ("lesion_color", "lesion colour", "lesion color"),
        "lesion_shape": ("lesion_shape", "lesion shape"),
        "lesion_distribution": ("lesion_distribution", "lesion distribution"),
        "affected_region": ("affected_region", "affected region", "location"),
        "leaf_curling": ("leaf_curling", "leaf curling", "curling"),
        "surface_signs": ("surface_signs", "surface signs"),
        "observations": ("observations",),
    }
    lowered = {str(key).lower(): value for key, value in parsed.items()}
    normalized: dict[str, Any] = {
        "raw_description": description.strip(),
        "observations": [],
        "surface_signs": [],
    }
    for target, source_names in aliases.items():
        value = next((lowered[name] for name in source_names if name in lowered), None)
        if value is not None:
            normalized[target] = value
    for field in ("observations", "surface_signs"):
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = [value]
        elif not isinstance(value, list):
            normalized[field] = []
    return normalized


class ImageDescriptionTool(Tool):
    name = "image_description"
    description = "Describe visible leaf phenotypes using a fixed JSON schema."

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 180.0,
    ):
        self.api_key = api_key or os.getenv("IMAGE_MODEL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("IMAGE_MODEL_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.model = model or os.getenv("IMAGE_MODEL", "qwen3-vl-plus")
        self.timeout_seconds = timeout_seconds

    def invoke(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        image_path = Path(str(kwargs.get("image_path") or "")).expanduser()
        if not image_path.is_file():
            return ToolResult(tool=self.name, success=False, error=f"Image not found: {image_path}")
        if not self.api_key:
            return ToolResult(
                tool=self.name,
                success=False,
                error="IMAGE_MODEL_API_KEY or DASHSCOPE_API_KEY is not configured",
            )
        dimensions = kwargs.get("dimensions") or [
            "leaf_color",
            "lesion_color",
            "lesion_shape",
            "lesion_distribution",
            "affected_region",
            "leaf_curling",
            "surface_signs",
        ]
        prompt = (
            "Inspect only visible leaf phenotypes. Do not diagnose the disease. "
            "Return one JSON object with these keys: "
            + ", ".join(dimensions)
            + ", observations. Use null for invisible attributes and concise factual observations."
        )
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            suffix = image_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            raw = _post_json(
                f"{self.base_url}/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "temperature": 0,
                },
                self.timeout_seconds,
                {"Authorization": f"Bearer {self.api_key}"},
            )
            description = raw["choices"][0]["message"]["content"]
            return ToolResult(
                tool=self.name,
                success=True,
                data=normalize_image_description(str(description)),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )


def build_visual_registry(
    plant_api_base: str = "http://127.0.0.1:10001",
) -> ToolRegistry:
    registry = ToolRegistry()

    def register(tool: Tool) -> None:
        registry.register(tool)

    def normalize_classifier(data: dict[str, Any]) -> dict[str, Any]:
        def normalize_condition(item: dict[str, Any]) -> str | None:
            raw = item.get("condition") or item.get("disease")
            if raw is None:
                return None
            return re.sub(r"_+", "_", str(raw).strip().lstrip("_")).lower()

        def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
            condition = normalize_condition(item)
            raw_class_name = item.get("disease")
            plant = item.get("plant")
            terminology = (
                resolve_disease(str(raw_class_name))
                if raw_class_name
                else None
            )
            normalized = {
                "label": terminology.canonical_id if terminology else condition,
                "probability": item.get("prob"),
                "plant": terminology.crop_en if terminology else plant,
                "condition": condition,
                "raw_class_name": raw_class_name,
                "class_index": item.get("label"),
            }
            if terminology:
                normalized.update(
                    {
                        "canonical_id": terminology.canonical_id,
                        "display_name_en": terminology.disease_name_en,
                        "disease_name_zh": terminology.disease_name_zh,
                        "crop_zh": terminology.crop_zh,
                        "kg_disease_name": terminology.kg_disease_name,
                        "terminology_mapping_type": terminology.mapping_type,
                        "management_supported": terminology.management_supported,
                    }
                )
            return normalized

        top_k = [
            normalize_item(item)
            for item in data.get("topk", [])
            if isinstance(item, dict)
        ]
        raw_pred = data.get("pred")
        return {
            "top_k": top_k,
            "pred": normalize_item(raw_pred) if isinstance(raw_pred, dict) else None,
        }

    base = plant_api_base.rstrip("/")
    register(
        PlantAPITool("disease_classifier", f"{base}/disease/classify", normalize_classifier)
    )
    segmentation = _SegmentationCache(f"{base}/segmentation/severity")
    register(SegmentationViewTool("leaf_segmenter", segmentation))
    register(SegmentationViewTool("lesion_segmenter", segmentation))
    register(ImageDescriptionTool())
    return registry


def _document_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    content = str(getattr(document, "page_content", ""))
    source = str(metadata.get("source") or metadata.get("title") or "")
    return sha1(f"{source}\0{content}".encode("utf-8")).hexdigest()


def _display_source_name(source: str) -> str:
    name = Path(str(source or "")).name
    name = re.sub(r"_origin\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    parts = name.split("-", 2)
    return parts[2].strip() if len(parts) == 3 and parts[2].strip() else name


def _strip_retrieval_header(content: str) -> str:
    first_line, separator, remainder = str(content or "").partition("\n")
    if not separator:
        return str(content or "")
    markers = ("作物：", "病害：", "别名：", "章节：", "证据类型：")
    if sum(marker in first_line for marker in markers) < 2:
        return str(content or "")
    return remainder.lstrip()


def _metadata_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("___", "_")
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def _metadata_crop_matches(metadata: dict[str, Any], crop: str | None) -> bool:
    if not crop:
        return True
    expected = {
        _metadata_text(item)
        for item in crop_query_terms(crop)
        if _metadata_text(item)
    }
    values = {
        _metadata_text(metadata.get(key))
        for key in ("crop_en", "crop_zh", "raw_crop_topic")
        if _metadata_text(metadata.get(key))
    }
    if not values:
        return False
    return any(
        value == term or value.startswith(f"{term} ") or term.startswith(f"{value} ")
        for value in values
        for term in expected
    )


def _matched_allowed_disease(
    metadata: dict[str, Any],
    allowed_diseases: list[str] | tuple[str, ...] | None,
    crop: str | None,
) -> str | None:
    if not allowed_diseases:
        return None
    metadata_values = [
        str(metadata.get(key) or "").strip()
        for key in (
            "disease_en",
            "disease_zh",
            "topic_label",
            "raw_disease_topic",
        )
        if str(metadata.get(key) or "").strip()
    ]
    resolved_metadata = {
        item.canonical_id
        for value in metadata_values
        if (item := resolve_disease(value, crop=crop)) is not None
    }
    normalized_metadata = {_metadata_text(value) for value in metadata_values}
    for candidate in allowed_diseases:
        terminology = resolve_disease(candidate, crop=crop)
        canonical = terminology.canonical_id if terminology else str(candidate)
        if terminology and canonical in resolved_metadata:
            return canonical
        terms = {
            _metadata_text(value)
            for value in disease_query_terms(candidate, crop=crop)
            if _metadata_text(value)
        }
        terms.add(_metadata_text(candidate))
        for value in normalized_metadata:
            for term in terms:
                if not value or not term:
                    continue
                if value == term or value.endswith(f" {term}") or term.endswith(f" {value}"):
                    return canonical
    return None


def _metadata_content_type_matches(
    metadata: dict[str, Any],
    content_types: list[str] | tuple[str, ...] | None,
) -> bool:
    if not content_types:
        return True
    actual = _metadata_text(metadata.get("content_type"))
    expected = {_metadata_text(value) for value in content_types}
    return bool(actual and actual in expected)


class LocalHybridRetriever:
    def __init__(
        self,
        vector_db_path: str | Path,
        embedding_model_path: str | Path,
        final_top_k: int = 5,
        semantic_top_k: int = 20,
        keyword_top_k: int = 20,
        candidate_top_k: int = 30,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        device: str | None = None,
        bm25_path: str | Path | None = None,
        reranker_model_path: str | Path | None = None,
        reranker_device: str | None = None,
        reranker_batch_size: int | None = None,
        reranker_max_length: int | None = None,
        max_chunks_per_source: int = 2,
        rrf_k: int = 60,
    ):
        self.vector_db_path = Path(vector_db_path).expanduser().resolve()
        self.embedding_model_path = str(Path(embedding_model_path).expanduser())
        self.final_top_k = final_top_k
        self.semantic_top_k = max(final_top_k, semantic_top_k)
        self.keyword_top_k = max(final_top_k, keyword_top_k)
        self.candidate_top_k = max(final_top_k, candidate_top_k)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cuda")
        self.bm25_path = (
            Path(bm25_path).expanduser().resolve()
            if bm25_path
            else self.vector_db_path / "bm25_retriever.pkl"
        )
        self.reranker_model_path = (
            str(Path(reranker_model_path).expanduser()) if reranker_model_path else None
        )
        self.reranker_device = reranker_device or self.device
        self.reranker_batch_size = max(
            1,
            int(
                reranker_batch_size
                if reranker_batch_size is not None
                else os.getenv("RERANKER_BATCH_SIZE", "4")
            ),
        )
        self.reranker_max_length = max(
            128,
            int(
                reranker_max_length
                if reranker_max_length is not None
                else os.getenv("RERANKER_MAX_LENGTH", "1024")
            ),
        )
        self.max_chunks_per_source = max(1, int(max_chunks_per_source))
        self.rrf_k = max(1, int(rrf_k))
        self._vector_store: Any = None
        self._bm25: Any = None
        self._reranker: Any = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        index = self.vector_db_path / "index.faiss"
        metadata = self.vector_db_path / "index.pkl"
        if not index.is_file() or not metadata.is_file():
            raise FileNotFoundError(
                f"FAISS database must contain index.faiss and index.pkl: {self.vector_db_path}"
            )
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "RAG dependencies are missing; install the project's knowledge extra"
            ) from exc
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store = FAISS.load_local(
            str(self.vector_db_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        if self.bm25_path.is_file():
            with self.bm25_path.open("rb") as stream:
                self._bm25 = pickle.load(stream)
        self._loaded = True

    def _load_reranker(self) -> Any | None:
        if not self.reranker_model_path:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Reranker is configured but sentence-transformers is not installed"
            ) from exc
        self._reranker = CrossEncoder(
            self.reranker_model_path,
            device=self.reranker_device,
            max_length=self.reranker_max_length,
        )
        return self._reranker

    def _semantic_documents(self, query: str, limit: int) -> list[tuple[Any, float]]:
        return list(self._vector_store.similarity_search_with_score(query, k=limit))

    def _keyword_documents(self, query: str, limit: int) -> list[Any]:
        if self._bm25 is None:
            return []
        if hasattr(self._bm25, "k"):
            self._bm25.k = limit
        if hasattr(self._bm25, "invoke"):
            return list(self._bm25.invoke(query))
        if hasattr(self._bm25, "get_relevant_documents"):
            return list(self._bm25.get_relevant_documents(query))
        raise TypeError("Unsupported BM25 retriever object")

    def search(
        self,
        query: str,
        crop: str | None = None,
        disease: str | None = None,
        section: str | None = None,
        top_k: int | None = None,
        allowed_diseases: list[str] | tuple[str, ...] | None = None,
        content_types: list[str] | tuple[str, ...] | None = None,
        balance_diseases: bool = False,
    ) -> list[Evidence]:
        self._load()
        result_limit = (
            self.final_top_k
            if top_k is None
            else max(1, min(int(top_k), self.final_top_k))
        )
        semantic_query = str(query or "").strip()
        if not semantic_query:
            return []
        scoped_diseases = list(allowed_diseases or ([disease] if disease else []))
        fetch_limit = max(
            self.semantic_top_k,
            self.keyword_top_k,
            self.candidate_top_k * 3,
        )
        semantic = self._semantic_documents(semantic_query, fetch_limit)
        keyword = self._keyword_documents(semantic_query, fetch_limit)
        records: dict[str, dict[str, Any]] = {}

        def scope(document: Any) -> tuple[bool, str | None]:
            metadata = dict(getattr(document, "metadata", {}) or {})
            if not _metadata_crop_matches(metadata, crop):
                return False, None
            matched_disease = _matched_allowed_disease(
                metadata,
                scoped_diseases,
                crop,
            )
            if scoped_diseases and not matched_disease:
                return False, None
            if not _metadata_content_type_matches(metadata, content_types):
                return False, None
            return True, matched_disease

        for rank, (document, distance) in enumerate(semantic, start=1):
            accepted, matched_disease = scope(document)
            if not accepted:
                continue
            key = _document_key(document)
            records.setdefault(
                key,
                {
                    "document": document,
                    "score": 0.0,
                    "channels": [],
                    "matched_disease": matched_disease,
                },
            )
            records[key]["score"] += self.semantic_weight / (self.rrf_k + rank)
            records[key]["channels"].append("semantic")
            records[key]["distance"] = float(distance)
        for rank, document in enumerate(keyword, start=1):
            accepted, matched_disease = scope(document)
            if not accepted:
                continue
            key = _document_key(document)
            records.setdefault(
                key,
                {
                    "document": document,
                    "score": 0.0,
                    "channels": [],
                    "matched_disease": matched_disease,
                },
            )
            records[key]["score"] += self.keyword_weight / (self.rrf_k + rank)
            records[key]["channels"].append("keyword")

        ranked = sorted(records.values(), key=lambda item: item["score"], reverse=True)[
            : self.candidate_top_k
        ]
        reranker = self._load_reranker()
        if reranker is not None and ranked:
            pairs = [
                [semantic_query, str(getattr(item["document"], "page_content", ""))]
                for item in ranked
            ]
            rerank_scores = reranker.predict(
                pairs,
                batch_size=self.reranker_batch_size,
                show_progress_bar=False,
            )
            for item, rerank_score in zip(ranked, rerank_scores):
                item["rerank_score"] = float(rerank_score)
            ranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        diversified: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}

        def add_item(item: dict[str, Any]) -> bool:
            document = item["document"]
            metadata = dict(getattr(document, "metadata", {}) or {})
            source_key = str(
                metadata.get("doc_id")
                or metadata.get("source")
                or metadata.get("title")
                or _document_key(document)
            )
            if source_counts.get(source_key, 0) >= self.max_chunks_per_source:
                return False
            diversified.append(item)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            return True

        if balance_diseases and scoped_diseases:
            canonical_order = []
            for value in scoped_diseases:
                terminology = resolve_disease(value, crop=crop)
                canonical = terminology.canonical_id if terminology else str(value)
                if canonical not in canonical_order:
                    canonical_order.append(canonical)
            for canonical in canonical_order:
                candidate_item = next(
                    (
                        item
                        for item in ranked
                        if item.get("matched_disease") == canonical
                        and item not in diversified
                    ),
                    None,
                )
                if candidate_item is not None:
                    add_item(candidate_item)
                if len(diversified) >= result_limit:
                    break

        for item in ranked:
            if item in diversified:
                continue
            add_item(item)
            if len(diversified) >= result_limit:
                break

        evidence: list[Evidence] = []
        for item in diversified:
            document = item["document"]
            content = _strip_retrieval_header(
                str(getattr(document, "page_content", ""))
            )
            metadata = dict(getattr(document, "metadata", {}) or {})
            metadata.pop("file_path", None)
            source_file = str(
                metadata.get("source") or metadata.get("title") or "local_literature"
            )
            source = _display_source_name(source_file)
            metadata["source"] = source
            key = _document_key(document)
            evidence.append(
                Evidence(
                    evidence_id=f"RAG-{key[:12]}",
                    evidence_type=EvidenceType.RETRIEVAL,
                    content=content,
                    source=source,
                    metadata={
                        **metadata,
                        "retrieval_channels": sorted(set(item["channels"])),
                        "rrf_score": round(float(item["score"]), 6),
                        **(
                            {"vector_distance": round(float(item["distance"]), 6)}
                            if "distance" in item
                            else {}
                        ),
                        **(
                            {"rerank_score": round(float(item["rerank_score"]), 6)}
                            if "rerank_score" in item
                            else {}
                        ),
                        "retrieval_stage": section,
                        "matched_disease": item.get("matched_disease"),
                        "content_type_filter": list(content_types or []),
                    },
                )
            )
        return evidence

_DEFAULT_ALIASES: dict[str, list[str]] = {
    "corn": ["corn", "maize", "玉米"],
    "maize": ["maize", "corn", "玉米"],
    "tomato": ["tomato", "番茄"],
    "potato": ["potato", "马铃薯"],
    "apple": ["apple", "苹果"],
    "grape": ["grape", "葡萄"],
    "wheat": ["wheat", "小麦"],
    "rice": ["rice", "水稻"],
    "soybean": ["soybean", "soya bean", "大豆"],
    "common rust": ["common rust", "rust", "锈病"],
    "southern rust": ["southern rust", "rust", "锈病"],
    "gray leaf spot": ["gray leaf spot", "leaf spot", "灰斑病"],
    "early blight": ["early blight", "早疫病"],
    "late blight": ["late blight", "晚疫病"],
    "powdery mildew": ["powdery mildew", "白粉病"],
    "downy mildew": ["downy mildew", "霜霉病"],
}


def _normalized_terms(value: str | None, aliases: dict[str, list[str]]) -> list[str]:
    if not value:
        return []
    raw = value.strip()
    humanized = raw.replace("___", " ").replace("__", " ").replace("_", " ")
    terms = [raw, humanized]
    if " " in humanized:
        terms.append(humanized.split(" ", 1)[1])
    lowered = humanized.lower()
    for key, values in aliases.items():
        if key in lowered or lowered in key:
            terms.extend(values)
    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


class Neo4jKnowledgeGraph:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        aliases_path: str | Path | None = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.aliases = dict(_DEFAULT_ALIASES)
        if aliases_path:
            path = Path(aliases_path).expanduser()
            if path.is_file():
                custom = json.loads(path.read_text(encoding="utf-8"))
                self.aliases.update(
                    {
                        str(key).lower(): (
                            [str(item) for item in value]
                            if isinstance(value, list)
                            else [str(value)]
                        )
                        for key, value in custom.items()
                    }
                )
        self._driver: Any = None

    def _get_driver(self) -> Any:
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError("neo4j is not installed") from exc
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self._driver.verify_connectivity()
        return self._driver

    def _run(self, cypher: str, **parameters: Any) -> list[dict[str, Any]]:
        driver = self._get_driver()
        kwargs = {"database": self.database} if self.database else {}
        with driver.session(**kwargs) as session:
            return session.run(cypher, **parameters).data()

    @staticmethod
    def _clean_ingredients(row: dict[str, Any]) -> dict[str, Any]:
        ingredients = row.get("active_ingredients")
        if isinstance(ingredients, list):
            row["active_ingredients"] = [
                item
                for item in ingredients
                if isinstance(item, dict) and item.get("name")
            ]
        return row

    def _as_evidence(
        self,
        operation: str,
        parameters: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> list[Evidence]:
        return [
            Evidence(
                evidence_id=f"KG-{sha1(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]}",
                evidence_type=EvidenceType.KNOWLEDGE_GRAPH,
                content=json.dumps(row, ensure_ascii=False),
                source="local_neo4j",
                metadata={
                    "operation": operation,
                    "query_parameters": parameters,
                    **row,
                },
            )
            for row in rows
        ]

    def get_management_options(
        self,
        crop: str | None,
        disease: str | None,
        limit: int = 5,
    ) -> list[Evidence]:
        terminology = resolve_disease(disease, crop=crop)
        if terminology and not terminology.kg_disease_name:
            return []
        crops = (
            [terminology.crop_zh]
            if terminology
            else crop_query_terms(crop)
        )
        diseases = (
            [terminology.kg_disease_name]
            if terminology and terminology.kg_disease_name
            else disease_query_terms(disease)
        )
        if not diseases:
            return []
        rows = self._run(
            """
            MATCH (p:Pesticide)-[:HAS_USE]->(u:PesticideUse)
                  -[:APPLIES_TO]->(c:Crop),
                  (u)-[:CONTROLS]->(d:Disease)
            WHERE (size($crops) = 0 OR any(term IN $crops WHERE
                       toLower(c.name) CONTAINS toLower(term)))
              AND any(term IN $diseases WHERE
                       toLower(d.name) CONTAINS toLower(term))
            OPTIONAL MATCH (p)-[:HELD_BY]->(h:Holder)
            OPTIONAL MATCH (p)-[ir:CONTAINS]->(i:ActiveIngredient)
            RETURN p.pesticide_id AS pesticide_id,
                   p.name AS product_name,
                   p.registration_number AS registration_number,
                   p.formulation AS formulation,
                   p.toxicity AS toxicity_class,
                   u.active_ingredients_raw AS active_ingredients_raw,
                   collect(DISTINCT {
                       name: i.name,
                       concentration: ir.concentration
                   }) AS active_ingredients,
                   c.name AS applicable_crop,
                   d.name AS disease_target,
                   u.dosage AS recommended_dosage,
                   u.application_method AS application_method,
                   u.precautions AS precautions,
                   u.emergency_measures AS emergency_measures,
                   head(collect(DISTINCT h.name)) AS holder
            ORDER BY p.name
            LIMIT $limit
            """,
            crops=crops,
            diseases=diseases,
            limit=max(1, min(int(limit), 50)),
        )
        rows = [self._clean_ingredients(row) for row in rows]
        parameters = {
            "crop": crop,
            "disease": disease,
            "crop_terms": crops,
            "disease_terms": diseases,
        }
        if terminology:
            parameters["terminology"] = terminology.to_dict()
        return self._as_evidence("get_management_options", parameters, rows)

    def get_pesticide_uses(
        self,
        pesticide_name: str,
        limit: int = 20,
    ) -> list[Evidence]:
        search_text = str(pesticide_name or "").strip()
        if not search_text:
            return []
        rows = self._run(
            """
            MATCH (p:Pesticide)-[:HAS_USE]->(u:PesticideUse)
                  -[:APPLIES_TO]->(c:Crop),
                  (u)-[:CONTROLS]->(d:Disease)
            WHERE toLower(p.name) CONTAINS toLower($search_text)
               OR (size(p.name) >= 2 AND
                   toLower($search_text) CONTAINS toLower(p.name))
               OR toLower(coalesce(p.registration_number, '')) =
                  toLower($search_text)
               OR ($search_text CONTAINS coalesce(p.registration_number, '') AND
                   coalesce(p.registration_number, '') <> '')
            RETURN p.pesticide_id AS pesticide_id,
                   p.name AS product_name,
                   p.registration_number AS registration_number,
                   c.name AS applicable_crop,
                   d.name AS disease_target,
                   u.dosage AS recommended_dosage,
                   u.application_method AS application_method
            ORDER BY p.name, c.name, d.name
            LIMIT $limit
            """,
            search_text=search_text,
            limit=max(1, min(int(limit), 50)),
        )
        return self._as_evidence(
            "get_pesticide_uses",
            {"pesticide_name": search_text},
            rows,
        )

    def get_pesticide_detail(
        self,
        pesticide_name: str,
        limit: int = 5,
    ) -> list[Evidence]:
        search_text = str(pesticide_name or "").strip()
        if not search_text:
            return []
        rows = self._run(
            """
            MATCH (p:Pesticide)-[:HAS_USE]->(u:PesticideUse)
            WHERE toLower(p.name) CONTAINS toLower($search_text)
               OR (size(p.name) >= 2 AND
                   toLower($search_text) CONTAINS toLower(p.name))
               OR toLower(coalesce(p.registration_number, '')) =
                  toLower($search_text)
               OR ($search_text CONTAINS coalesce(p.registration_number, '') AND
                   coalesce(p.registration_number, '') <> '')
            OPTIONAL MATCH (p)-[:HELD_BY]->(h:Holder)
            OPTIONAL MATCH (p)-[ir:CONTAINS]->(i:ActiveIngredient)
            RETURN p.pesticide_id AS pesticide_id,
                   p.name AS product_name,
                   p.registration_number AS registration_number,
                   p.formulation AS formulation,
                   p.toxicity AS toxicity_class,
                   u.active_ingredients_raw AS active_ingredients_raw,
                   collect(DISTINCT {
                       name: i.name,
                       concentration: ir.concentration
                   }) AS active_ingredients,
                   u.instructions AS usage_instructions,
                   u.precautions AS precautions,
                   u.emergency_measures AS emergency_measures,
                   u.storage_transport AS storage_transport,
                   head(collect(DISTINCT h.name)) AS holder
            LIMIT $limit
            """,
            search_text=search_text,
            limit=max(1, min(int(limit), 20)),
        )
        rows = [self._clean_ingredients(row) for row in rows]
        return self._as_evidence(
            "get_pesticide_detail",
            {"pesticide_name": search_text},
            rows,
        )

    def list_diseases_by_crop(
        self,
        crop: str,
        limit: int = 30,
    ) -> list[Evidence]:
        crops = crop_query_terms(crop)
        if not crops:
            return []
        rows = self._run(
            """
            MATCH (p:Pesticide)-[:HAS_USE]->(u:PesticideUse)
                  -[:APPLIES_TO]->(c:Crop),
                  (u)-[:CONTROLS]->(d:Disease)
            WHERE any(term IN $crops WHERE
                      toLower(c.name) CONTAINS toLower(term))
            RETURN c.name AS crop,
                   d.name AS disease_target,
                   count(DISTINCT p) AS pesticide_count
            ORDER BY pesticide_count DESC, d.name
            LIMIT $limit
            """,
            crops=crops,
            limit=max(1, min(int(limit), 100)),
        )
        return self._as_evidence(
            "list_diseases_by_crop",
            {"crop": crop, "crop_terms": crops},
            rows,
        )
