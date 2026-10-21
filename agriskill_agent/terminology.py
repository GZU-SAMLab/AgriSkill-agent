from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DiseaseTerminology:
    classifier_label: str
    canonical_id: str
    crop_en: str
    crop_zh: str
    disease_name_en: str
    disease_name_zh: str
    kg_disease_name: str | None
    mapping_type: str = "exact"

    @property
    def management_supported(self) -> bool:
        return self.kg_disease_name is not None

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "management_supported": self.management_supported,
        }


def _term(
    classifier_label: str,
    canonical_id: str,
    crop_en: str,
    crop_zh: str,
    disease_name_en: str,
    disease_name_zh: str,
    kg_disease_name: str | None,
    mapping_type: str = "exact",
) -> DiseaseTerminology:
    return DiseaseTerminology(
        classifier_label=classifier_label,
        canonical_id=canonical_id,
        crop_en=crop_en,
        crop_zh=crop_zh,
        disease_name_en=disease_name_en,
        disease_name_zh=disease_name_zh,
        kg_disease_name=kg_disease_name,
        mapping_type=mapping_type,
    )


DISEASE_MAPPING: dict[str, DiseaseTerminology] = {
    "apple_black_rot": _term(
        "apple_black_rot",
        "apple_black_rot",
        "apple",
        "苹果",
        "Apple black rot",
        "苹果黑腐病",
        "黑腐病",
    ),
    "apple_rust": _term(
        "apple_rust",
        "apple_cedar_apple_rust",
        "apple",
        "苹果",
        "Cedar-apple rust",
        "雪松苹果锈病",
        "锈病",
        "crop_scoped",
    ),
    "apple_powdery_mildew": _term(
        "apple_powdery_mildew",
        "apple_powdery_mildew",
        "apple",
        "苹果",
        "Apple powdery mildew",
        "苹果白粉病",
        "白粉病",
    ),
    "apple_scab": _term(
        "apple_scab",
        "apple_scab",
        "apple",
        "苹果",
        "Apple scab",
        "苹果黑星病",
        "黑星病",
    ),
    "banana_sigatoka": _term(
        "banana_sigatoka",
        "banana_black_sigatoka",
        "banana",
        "香蕉",
        "Banana Sigatoka leaf spot",
        "香蕉叶斑病",
        "叶斑病",
        "generalized",
    ),
    "coffee_rust": _term(
        "coffee_rust",
        "coffee_rust",
        "coffee",
        "咖啡",
        "Coffee leaf rust",
        "咖啡锈病",
        "锈病",
        "crop_scoped",
    ),
    "corn_common_rust": _term(
        "corn_common_rust",
        "corn_common_rust",
        "corn",
        "玉米",
        "Corn common rust",
        "玉米普通锈病",
        "锈病",
        "crop_scoped",
    ),
    "corn_northern_leaf_blight": _term(
        "corn_northern_leaf_blight",
        "corn_northern_leaf_blight",
        "corn",
        "玉米",
        "Northern corn leaf blight",
        "玉米大斑病",
        "大斑病",
    ),
    "grape_black_rot": _term(
        "grape_black_rot",
        "grape_black_rot",
        "grape",
        "葡萄",
        "Grape black rot",
        "葡萄黑腐病",
        None,
        "missing",
    ),
    "potato_early_blight": _term(
        "potato_early_blight",
        "potato_early_blight",
        "potato",
        "马铃薯",
        "Potato early blight",
        "马铃薯早疫病",
        "早疫病",
    ),
    "potato_late_blight": _term(
        "potato_late_blight",
        "potato_late_blight",
        "potato",
        "马铃薯",
        "Potato late blight",
        "马铃薯晚疫病",
        "晚疫病",
    ),
    "rice_brown_spot": _term(
        "rice_brown_spot",
        "rice_brown_spot",
        "rice",
        "水稻",
        "Rice brown spot",
        "水稻胡麻叶斑病",
        "胡麻叶斑病",
    ),
    "rice_leaf_blast": _term(
        "rice_leaf_blast",
        "rice_leaf_blast",
        "rice",
        "水稻",
        "Rice leaf blast",
        "水稻叶瘟",
        "稻瘟病",
    ),
    "soybean_downy_mildew": _term(
        "soybean_downy_mildew",
        "soybean_downy_mildew",
        "soybean",
        "大豆",
        "Soybean downy mildew",
        "大豆霜霉病",
        "霜霉病",
    ),
    "soybean_rust": _term(
        "soybean_rust",
        "soybean_rust",
        "soybean",
        "大豆",
        "Soybean rust",
        "大豆锈病",
        "锈病",
        "crop_scoped",
    ),
    "tomato_early_blight": _term(
        "tomato_early_blight",
        "tomato_early_blight",
        "tomato",
        "番茄",
        "Tomato early blight",
        "番茄早疫病",
        "早疫病",
    ),
    "tomato_late_blight": _term(
        "tomato_late_blight",
        "tomato_late_blight",
        "tomato",
        "番茄",
        "Tomato late blight",
        "番茄晚疫病",
        "晚疫病",
    ),
    "tomato_leaf_mold": _term(
        "tomato_leaf_mold",
        "tomato_leaf_mold",
        "tomato",
        "番茄",
        "Tomato leaf mold",
        "番茄叶霉病",
        "叶霉病",
    ),
    "tomato_septoria_leaf_spot": _term(
        "tomato_septoria_leaf_spot",
        "tomato_septoria_leaf_spot",
        "tomato",
        "番茄",
        "Tomato Septoria leaf spot",
        "番茄斑枯病",
        "叶斑病",
        "generalized",
    ),
    "wheat_brown_rust": _term(
        "wheat_brown_rust",
        "wheat_brown_rust",
        "wheat",
        "小麦",
        "Wheat leaf rust",
        "小麦叶锈病",
        "叶锈病",
    ),
}


GROUND_TRUTH_LABEL_ALIASES: dict[str, str] = {
    "apple_cedar_apple_rust": "apple_rust",
    "banana_black_sigatoka": "banana_sigatoka",
}

EXPERIMENT_DISEASES: tuple[DiseaseTerminology, ...] = tuple(
    DISEASE_MAPPING.values()
)

_CANONICAL_TO_CLASSIFIER = {
    entry.canonical_id: entry.classifier_label
    for entry in EXPERIMENT_DISEASES
}

_CROP_NAMES: dict[str, tuple[str, str]] = {
    "apple": ("apple", "苹果"),
    "banana": ("banana", "香蕉"),
    "coffee": ("coffee", "咖啡"),
    "corn": ("corn", "玉米"),
    "maize": ("corn", "玉米"),
    "grape": ("grape", "葡萄"),
    "potato": ("potato", "马铃薯"),
    "rice": ("rice", "水稻"),
    "soybean": ("soybean", "大豆"),
    "soyabean": ("soybean", "大豆"),
    "tomato": ("tomato", "番茄"),
    "wheat": ("wheat", "小麦"),
}


def _key(value: str | None) -> str:
    return str(value or "").strip().lower().replace("___", "_").replace("__", "_")


def _free_text_key(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def map_classifier_label(label: str | None) -> DiseaseTerminology | None:
    key = _key(label)
    key = GROUND_TRUTH_LABEL_ALIASES.get(key, key)
    key = _CANONICAL_TO_CLASSIFIER.get(key, key)
    return DISEASE_MAPPING.get(key)


def resolve_disease(
    label: str | None,
    crop: str | None = None,
) -> DiseaseTerminology | None:
    exact = map_classifier_label(label)
    if exact is not None:
        return exact

    query = _free_text_key(label)
    if not query:
        return None
    crop_key = _free_text_key(crop)
    matches: list[DiseaseTerminology] = []
    for entry in EXPERIMENT_DISEASES:
        if crop_key and crop_key not in {
            _free_text_key(entry.crop_en),
            _free_text_key(entry.crop_zh),
        }:
            continue
        aliases = {
            _free_text_key(entry.classifier_label),
            _free_text_key(entry.canonical_id),
            _free_text_key(entry.disease_name_en),
            _free_text_key(entry.disease_name_zh),
            _free_text_key(entry.kg_disease_name),
        }
        if crop_key:
            crop_aliases = {
                _free_text_key(entry.crop_en),
                _free_text_key(entry.crop_zh),
            }
            for alias in tuple(aliases):
                for crop_alias in crop_aliases:
                    prefix = f"{crop_alias} "
                    if crop_alias and alias.startswith(prefix):
                        aliases.add(alias[len(prefix) :])
        aliases.discard("")
        if query in aliases:
            matches.append(entry)

    unique = {entry.canonical_id: entry for entry in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def extract_disease_from_text(
    text: str | None,
    crop: str | None = None,
) -> DiseaseTerminology | None:
    query = _free_text_key(text)
    if not query:
        return None
    crop_key = _free_text_key(crop)
    matches: list[tuple[int, DiseaseTerminology]] = []
    padded_query = f" {query} "
    for entry in EXPERIMENT_DISEASES:
        if crop_key and crop_key not in {
            _free_text_key(entry.crop_en),
            _free_text_key(entry.crop_zh),
        }:
            continue
        aliases = {
            _free_text_key(entry.classifier_label),
            _free_text_key(entry.canonical_id),
            _free_text_key(entry.disease_name_en),
            _free_text_key(entry.disease_name_zh),
        }
        aliases.discard("")
        for alias in aliases:
            contains_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", alias))
            matched = alias in query if contains_cjk else f" {alias} " in padded_query
            if matched:
                matches.append((len(alias), entry))

    if not matches:
        return None
    longest = max(length for length, _ in matches)
    unique = {
        entry.canonical_id: entry
        for length, entry in matches
        if length == longest
    }
    return next(iter(unique.values())) if len(unique) == 1 else None


def crop_query_terms(value: str | None) -> list[str]:
    key = _key(value).replace(" ", "_")
    names = _CROP_NAMES.get(key)
    if not names:
        raw = str(value or "").strip()
        return [raw] if raw else []
    return list(dict.fromkeys(names))


def disease_query_terms(
    label: str | None,
    crop: str | None = None,
) -> list[str]:
    entry = resolve_disease(label, crop=crop)
    if not entry:
        raw = str(label or "").strip()
        return [raw, raw.replace("_", " ")] if raw else []
    return list(
        dict.fromkeys(
            term
            for term in (
                entry.kg_disease_name,
                entry.disease_name_zh,
                entry.disease_name_en,
                entry.canonical_id.replace("_", " "),
                entry.classifier_label,
            )
            if term
        )
    )


def experiment_mapping() -> list[dict[str, object]]:
    return [entry.to_dict() for entry in EXPERIMENT_DISEASES]


__all__ = [
    "DISEASE_MAPPING",
    "DiseaseTerminology",
    "EXPERIMENT_DISEASES",
    "GROUND_TRUTH_LABEL_ALIASES",
    "crop_query_terms",
    "disease_query_terms",
    "extract_disease_from_text",
    "experiment_mapping",
    "map_classifier_label",
    "resolve_disease",
]
