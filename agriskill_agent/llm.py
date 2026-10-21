from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


class AgentModelError(RuntimeError):
    pass


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")


def _json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(part.strip() for part in text.split("```") if "{" in part)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        candidate = candidate.removeprefix("json").strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AgentModelError("Agent model did not return a valid JSON object")


def _usage_int(usage: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _token_usage(raw: dict[str, Any]) -> dict[str, int | None]:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


@dataclass(slots=True)
class AgentModelClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 180.0
    max_retries: int = 2
    temperature: float | None = None
    json_mode: bool = False
    event_sink: Callable[..., None] | None = field(default=None, repr=False)
    _calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _next_call_id: int = field(default=1, init=False, repr=False)

    @classmethod
    def from_environment(cls) -> "AgentModelClient":
        api_key = os.getenv("AGENT_MODEL_API_KEY", "").strip()
        base_url = os.getenv("AGENT_MODEL_BASE_URL", "").strip()
        model = os.getenv("AGENT_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("AGENT_MODEL_API_KEY", api_key),
                ("AGENT_MODEL_BASE_URL", base_url),
                ("AGENT_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Integrated mode requires the center-agent model variables: "
                + ", ".join(missing)
            )
        temperature_value = os.getenv("AGENT_MODEL_TEMPERATURE", "").strip()
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            timeout_seconds=float(os.getenv("AGENT_MODEL_TIMEOUT", "180")),
            max_retries=int(os.getenv("AGENT_MODEL_MAX_RETRIES", "2")),
            temperature=float(temperature_value) if temperature_value else None,
            json_mode=os.getenv("AGENT_MODEL_JSON_MODE", "").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def set_event_sink(self, sink: Callable[..., None] | None) -> None:
        self.event_sink = sink

    def reset_usage(self) -> None:
        self._calls.clear()
        self._next_call_id = 1

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_sink:
            self.event_sink(event, **payload)

    def usage_summary(self) -> dict[str, Any]:
        calls = [dict(item) for item in self._calls]

        def token_sum(name: str) -> int | None:
            values = [
                int(item[name])
                for item in calls
                if isinstance(item.get(name), int)
            ]
            return sum(values) if values else None

        return {
            "logical_calls": len(calls),
            "successful_calls": sum(bool(item.get("success")) for item in calls),
            "failed_calls": sum(not bool(item.get("success")) for item in calls),
            "http_attempts": sum(int(item.get("attempts") or 0) for item in calls),
            "retry_count": sum(max(0, int(item.get("attempts") or 0) - 1) for item in calls),
            "calls_with_token_usage": sum(
                isinstance(item.get("total_tokens"), int) for item in calls
            ),
            "prompt_tokens": token_sum("prompt_tokens"),
            "completion_tokens": token_sum("completion_tokens"),
            "total_tokens": token_sum("total_tokens"),
            "elapsed_ms": round(
                sum(float(item.get("elapsed_ms") or 0.0) for item in calls),
                3,
            ),
            "calls": calls,
        }

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        require_json: bool = False,
        stage: str = "unspecified",
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if require_json and self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        call_id = self._next_call_id
        self._next_call_id += 1
        started = time.perf_counter()
        attempts = 0
        prompt_chars = len(system_prompt) + len(user_prompt)
        self._emit(
            "llm_start",
            call_id=call_id,
            stage=stage,
            model=self.model,
            prompt_chars=prompt_chars,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            attempts += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                choices = raw.get("choices") or []
                if not choices:
                    raise AgentModelError(
                        f"Agent model response contains no choices: {json.dumps(raw)[:500]}"
                    )
                text = _content_text((choices[0].get("message") or {}).get("content"))
                if not text.strip():
                    raise AgentModelError("Agent model returned empty content")
                tokens = _token_usage(raw)
                elapsed_ms = (time.perf_counter() - started) * 1000
                record = {
                    "call_id": call_id,
                    "stage": stage,
                    "model": self.model,
                    "success": True,
                    "attempts": attempts,
                    "retry_count": attempts - 1,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "system_prompt_chars": len(system_prompt),
                    "user_prompt_chars": len(user_prompt),
                    "prompt_chars": prompt_chars,
                    "completion_chars": len(text),
                    **tokens,
                }
                self._calls.append(record)
                self._emit("llm_end", **record)
                return text.strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = AgentModelError(f"Agent model HTTP {exc.code}: {detail[:1000]}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError, AgentModelError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                self._emit(
                    "llm_retry",
                    call_id=call_id,
                    stage=stage,
                    model=self.model,
                    attempt=attempts,
                    next_attempt=attempts + 1,
                    error=str(last_error or "request failed")[:500],
                )
                time.sleep(min(2**attempt, 4))
        elapsed_ms = (time.perf_counter() - started) * 1000
        error = str(last_error or "Agent model request failed")
        record = {
            "call_id": call_id,
            "stage": stage,
            "model": self.model,
            "success": False,
            "attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "elapsed_ms": round(elapsed_ms, 3),
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            "prompt_chars": prompt_chars,
            "completion_chars": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "error": error[:1000],
        }
        self._calls.append(record)
        self._emit("llm_error", **record)
        raise AgentModelError(error)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str = "unspecified",
    ) -> dict[str, Any]:
        return _json_object(
            self.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                require_json=True,
                stage=stage,
            )
        )
