from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .models import ToolResult


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def invoke(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._event_sink: Callable[..., None] | None = None

    def set_event_sink(self, sink: Callable[..., None] | None) -> None:
        self._event_sink = sink

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def replace(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            if self._event_sink:
                self._event_sink("tool_start", name=name, params=kwargs)
                self._event_sink(
                    "tool_error",
                    name=name,
                    elapsed_ms=0.0,
                    success=False,
                    error=f"Tool not registered: {name}",
                    result=None,
                )
            return ToolResult(tool=name, success=False, error=f"Tool not registered: {name}")
        if self._event_sink:
            self._event_sink("tool_start", name=name, params=kwargs)
        result = self._tools[name].invoke(**kwargs)
        if self._event_sink:
            self._event_sink(
                "tool_end" if result.success else "tool_error",
                name=name,
                elapsed_ms=result.elapsed_ms,
                success=result.success,
                error=result.error,
                result=result.data if result.success else None,
            )
        return result

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)
