from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable


def _frontmatter(text: str, source: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        raise ValueError(f"SKILL.md has no YAML frontmatter: {source}")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid SKILL.md frontmatter: {source}")
    metadata: dict[str, str] = {}
    active_key: str | None = None
    for raw_line in parts[1].splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line[:1].isspace() and active_key:
            metadata[active_key] = f"{metadata[active_key]} {line.strip()}".strip()
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid frontmatter line in {source}: {line}")
        active_key = key.strip()
        metadata[active_key] = value.strip().strip("\"'")
    for required in ("name", "description"):
        if not metadata.get(required):
            raise ValueError(f"SKILL.md is missing '{required}': {source}")
    return metadata, parts[2].strip()


def _sections(body: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", body, flags=re.MULTILINE))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        result[match.group(1).strip().lower()] = body[start:end].strip()
    return result


def _list_items(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.+?)\s*$", line)
        if match:
            values.append(match.group(1).strip().strip("`"))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    description: str
    body: str
    source: str
    role: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    steps: tuple[str, ...]
    rules: tuple[str, ...]
    tools: tuple[str, ...]
    prompt_template: str

    @property
    def execution_prompt(self) -> str:
        return f"Activated procedural definition: {self.name}\n\n{self.body}"

    def metadata(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


class SkillCatalog:
    def __init__(self, definitions: dict[str, SkillDefinition]):
        if not definitions:
            raise ValueError("At least one procedural definition is required")
        self._definitions = definitions

    @classmethod
    def bundled(cls) -> "SkillCatalog":
        root = files("agriskill_agent").joinpath("skills")
        definitions: dict[str, SkillDefinition] = {}
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            resource = folder.joinpath("SKILL.md")
            if not resource.is_file():
                continue
            definition = cls._parse(
                resource.read_text(encoding="utf-8"),
                str(resource),
            )
            if folder.name != definition.name.replace("_", "-"):
                raise ValueError(
                    f"Skill folder '{folder.name}' does not match skill name "
                    f"'{definition.name}'"
                )
            definitions[definition.name] = definition
        return cls(definitions)

    @classmethod
    def from_directory(cls, root: str | Path) -> "SkillCatalog":
        path = Path(root)
        definitions = {
            definition.name: definition
            for definition in (
                cls._parse(skill_file.read_text(encoding="utf-8"), str(skill_file))
                for skill_file in sorted(path.glob("*/SKILL.md"))
            )
        }
        return cls(definitions)

    @staticmethod
    def _parse(text: str, source: str) -> SkillDefinition:
        metadata, body = _frontmatter(text, source)
        sections = _sections(body)
        name = metadata["name"].replace("-", "_")
        role = sections.get("expert role", "").strip()
        purpose = sections.get("purpose", "").strip()
        prompt_template = sections.get("prompt template", "").strip()
        missing = [
            title
            for title, value in (
                ("Expert Role", role),
                ("Purpose", purpose),
                ("Workflow", sections.get("workflow")),
                ("Domain Rules", sections.get("domain rules")),
                ("Tool Bindings", sections.get("tool bindings")),
                ("Output Schema", sections.get("output schema")),
                ("Prompt Template", prompt_template),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"SKILL.md is missing required sections in {source}: {', '.join(missing)}"
            )
        return SkillDefinition(
            name=name,
            description=metadata["description"].lstrip("> ").strip(),
            body=body,
            source=source,
            role=role,
            purpose=purpose,
            inputs=_list_items(sections.get("input schema", "")),
            outputs=_list_items(sections["output schema"]),
            steps=_list_items(sections["workflow"]),
            rules=_list_items(sections["domain rules"]),
            tools=_list_items(sections["tool bindings"]),
            prompt_template=prompt_template,
        )

    def get(self, name: str) -> SkillDefinition:
        return self._definitions[name.replace("-", "_")]

    def names(self) -> list[str]:
        return list(self._definitions)

    def metadata(self) -> list[dict[str, str]]:
        return [self._definitions[name].metadata() for name in self.names()]

    def selected(self, names: Iterable[str]) -> list[SkillDefinition]:
        return [self.get(name) for name in names]
