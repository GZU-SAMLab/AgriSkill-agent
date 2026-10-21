from __future__ import annotations

from .models import NormalizedInput, UserCase


class InputNormalizer:
    CROP_TERMS = {
        "玉米": "maize",
        "corn": "maize",
        "maize": "maize",
        "番茄": "tomato",
        "tomato": "tomato",
        "马铃薯": "potato",
        "土豆": "potato",
        "potato": "potato",
        "苹果": "apple",
        "apple": "apple",
        "葡萄": "grape",
        "grape": "grape",
        "小麦": "wheat",
        "wheat": "wheat",
        "水稻": "rice",
        "rice": "rice",
        "大豆": "soybean",
        "soybean": "soybean",
    }
    SYMPTOM_TERMS = {
        "黄点": "yellow spots",
        "黄斑": "yellow lesions",
        "褐斑": "brown lesions",
        "锈": "rust-like pustules",
        "卷曲": "leaf curling",
        "干枯": "necrosis",
        "霉": "mold-like growth",
        "yellow": "yellowing",
        "brown": "brown lesions",
        "curl": "leaf curling",
        "pustule": "pustules",
    }

    def normalize(self, case: UserCase) -> NormalizedInput:
        lowered = case.query.lower()
        crop = case.crop
        if not crop:
            crop = next((value for key, value in self.CROP_TERMS.items() if key in lowered), None)
        explicit_organ = next(
            (
                value
                for terms, value in (
                    (("叶片", "叶子", "leaf", "leaves"), "leaf"),
                    (("果实", "fruit"), "fruit"),
                    (("茎", "stem"), "stem"),
                    (("根", "root"), "root"),
                    (("花", "flower"), "flower"),
                )
                if any(term in lowered for term in terms)
            ),
            None,
        )
        organ = case.organ or explicit_organ
        symptoms = [value for key, value in self.SYMPTOM_TERMS.items() if key in lowered]
        symptoms = list(dict.fromkeys(symptoms))

        intents: list[str] = []
        if any(
            x in lowered
            for x in (
                "表型",
                "症状",
                "描述",
                "颜色",
                "病斑",
                "卷曲",
                "phenotype",
                "symptom",
                "describe",
                "lesion",
            )
        ):
            intents.append("phenotype")
        if any(
            x in lowered
            for x in (
                "什么病",
                "哪种病",
                "诊断",
                "鉴别",
                "病因",
                "disease",
                "diagnos",
                "identify",
                "cause",
            )
        ):
            intents.append("diagnosis")
        if any(
            x in lowered
            for x in (
                "用药",
                "打药",
                "防治",
                "治疗",
                "管理",
                "农药",
                "treat",
                "manage",
                "control",
                "spray",
                "pesticide",
                "注意事项",
                "毒性",
                "急救",
                "储存",
                "有效成分",
                "适用于",
            )
        ):
            intents.append("management")
        if not intents:
            intents = ["unspecified"]
        intents = list(dict.fromkeys(intents))

        unknown = []
        for name, value in (("location", case.location), ("growth_stage", case.growth_stage)):
            if not value:
                unknown.append(name)
        return NormalizedInput(
            original_query=case.query,
            crop=crop,
            organ=organ,
            symptoms=symptoms,
            intents=intents,
            location=case.location,
            growth_stage=case.growth_stage,
            unknown_fields=unknown,
        )
