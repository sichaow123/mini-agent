from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    from langfuse import get_client
except ImportError:  # Keep tracing optional for minimal/offline environments.
    get_client = None


_FALSE_VALUES = {"0", "false", "no", "off"}

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|private[_-]?key|credential)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b|\b[A-Za-z0-9+/]{40,}={0,2}\b"
)


def _flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in _FALSE_VALUES


def include_prompts() -> bool:
    return _flag("LANGFUSE_TRACE_INCLUDE_PROMPTS")


def include_tool_output() -> bool:
    return _flag("LANGFUSE_TRACE_INCLUDE_TOOL_OUTPUT")


class _NoopObservation:
    def update(self, **kwargs) -> None:
        return None


class _SafeObservation:
    def __init__(self, observation):
        self._observation = observation

    def update(self, **kwargs) -> None:
        try:
            self._observation.update(**kwargs)
        except Exception as error:
            _report_trace_error("observation update", error)


def _report_trace_error(operation: str, error: Exception) -> None:
    print(f"[tracing] {operation} failed: {error}", file=sys.stderr)


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("[REDACTED]", value)
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def safe_json(value: Any, limit: int | None = None) -> Any:
    """Return a JSON-compatible, redacted and bounded value."""
    if limit is None:
        limit = int(os.environ.get("LANGFUSE_TRACE_MAX_CHARS", "20000"))
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            default=_json_default,
        )
        text = json.dumps(
            _redact(json.loads(text)),
            ensure_ascii=False,
        )
    except Exception:
        text = _SECRET_VALUE_RE.sub("[REDACTED]", str(value))

    if len(text) <= limit:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    return {
        "truncated": True,
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "preview": text[:limit],
    }


def value_hash(value: Any) -> str:
    """Return a stable identifier without exposing the original value."""
    return hashlib.sha256(str(value).encode()).hexdigest()


@dataclass
class TaskMetrics:
    """In-process metrics for one Agent task."""

    started_at: float = field(default_factory=time.perf_counter)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    llm_errors: int = 0
    context_compactions: int = 0
    reactive_compactions: int = 0
    llm_turns: int = 0
    tool_rounds: int = 0
    max_context_chars: int = 0
    retries: int = 0
    natural_termination: bool = False
    termination_reason: str = "unknown"

    def record_llm(self, usage: dict[str, int]) -> None:
        self.llm_requests += 1
        self.input_tokens += usage.get("input", 0)
        self.output_tokens += usage.get("output", 0)
        self.total_tokens += usage.get("total", 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "llm_errors": self.llm_errors,
            "context_compactions": self.context_compactions,
            "reactive_compactions": self.reactive_compactions,
            "llm_turns": self.llm_turns,
            "tool_rounds": self.tool_rounds,
            "total_interaction_rounds": self.llm_requests + self.tool_calls,
            "max_context_chars": self.max_context_chars,
            "retries": self.retries,
            "natural_termination": self.natural_termination,
            "termination_reason": self.termination_reason,
            "duration_seconds": round(time.perf_counter() - self.started_at, 3),
        }


def usage_details(usage) -> dict[str, int]:
    if usage is None:
        return {}

    result = {}
    for source_name, target_name in (
        ("input_tokens", "input"),
        ("output_tokens", "output"),
        ("total_tokens", "total"),
    ):
        value = getattr(usage, source_name, None)
        if value is not None:
            result[target_name] = value

    input_details = getattr(usage, "input_tokens_details", None)
    cached = getattr(input_details, "cached_tokens", None)
    if cached is not None:
        result["cache_read_input_tokens"] = cached

    output_details = getattr(usage, "output_tokens_details", None)
    reasoning = getattr(output_details, "reasoning_tokens", None)
    if reasoning is not None:
        result["reasoning_tokens"] = reasoning

    return result


def traced_llm_call(
    client,
    *,
    model: str,
    input: Any,
    name: str,
    instructions: str | None = None,
    tools: Any = None,
    metadata: dict[str, Any] | None = None,
    metrics: TaskMetrics | None = None,
):
    """Call an OpenAI-compatible client and record it as a generation.

    The LLM call remains the source of truth: tracing failures are absorbed by
    the observation wrapper and never change the caller's behavior.
    """
    generation_input: dict[str, Any] = {}
    if include_prompts():
        generation_input["messages"] = safe_json(input)
        if instructions is not None:
            generation_input["instructions"] = safe_json(instructions)
        if tools is not None:
            generation_input["tools"] = safe_json(tools)
    else:
        generation_input["prompt_logging"] = "disabled"

    with observation(
        as_type="generation",
        name=name,
        model=model,
        input=generation_input,
        metadata=metadata or {},
    ) as generation:
        try:
            response = client.responses.create(
                model=model,
                input=input,
                **({"instructions": instructions} if instructions is not None else {}),
                **({"tools": tools} if tools is not None else {}),
            )
        except Exception:
            if metrics is not None:
                metrics.llm_errors += 1
            raise
        usage = usage_details(getattr(response, "usage", None))
        if metrics is not None:
            metrics.record_llm(usage)
        generation.update(
            output=(
                safe_json(response.output)
                if include_prompts()
                else {"output_logging": "disabled"}
            ),
            usage_details=usage,
        )
        return response


def get_tracer():
    """Return the shared Langfuse client, or None when tracing is disabled."""
    if not _flag("LANGFUSE_TRACING_ENABLED") or get_client is None:
        return None
    try:
        return get_client()
    except Exception as error:
        _report_trace_error("client initialization", error)
        return None


@contextmanager
def observation(*args, **kwargs) -> Iterator[_SafeObservation | _NoopObservation]:
    """Create an observation without allowing tracing to break the Agent."""
    client = get_tracer()
    if client is None:
        yield _NoopObservation()
        return

    context = None
    try:
        context = client.start_as_current_observation(*args, **kwargs)
        current = context.__enter__()
    except Exception as error:
        _report_trace_error("observation start", error)
        yield _NoopObservation()
        return

    try:
        yield _SafeObservation(current)
    except BaseException:
        try:
            context.__exit__(*sys.exc_info())
        except Exception as error:
            _report_trace_error("observation close", error)
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception as error:
            _report_trace_error("observation close", error)


@contextmanager
def trace_attributes(*args, **kwargs):
    """Apply Langfuse attributes when available, otherwise be a no-op."""
    client = get_tracer()
    if client is None:
        yield
        return

    context = None
    try:
        from langfuse import propagate_attributes
        context = propagate_attributes(*args, **kwargs)
        context.__enter__()
    except Exception as error:
        _report_trace_error("trace attributes", error)
        yield
        return

    try:
        yield
    except BaseException:
        try:
            context.__exit__(*sys.exc_info())
        except Exception as error:
            _report_trace_error("trace attributes close", error)
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception as error:
            _report_trace_error("trace attributes close", error)


def flush() -> None:
    client = get_tracer()
    if client is None:
        return
    try:
        client.flush()
    except Exception as error:
        _report_trace_error("flush", error)
