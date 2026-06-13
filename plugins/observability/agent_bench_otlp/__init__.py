"""Agent Bench OpenTelemetry observer plugin for Hermes."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


HOOKS = (
    "on_session_start",
    "on_session_end",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "pre_approval_request",
    "post_approval_response",
    "subagent_start",
    "subagent_stop",
)


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)[:1024]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:8192]
    if isinstance(value, dict):
        return {str(key): _jsonable(inner, depth=depth + 1) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, depth=depth + 1) for item in list(value)[:200]]
    return str(value)[:1024]


def _attr(key: str, value: object) -> dict[str, object]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": "" if value is None else str(value)}}


def _jsonl_path() -> Path | None:
    raw = os.getenv("AGENT_BENCH_HERMES_OTEL_JSONL") or os.getenv("HERMES_AGENT_BENCH_OTEL_JSONL")
    return Path(raw) if raw else None


def _write_jsonl(event: dict[str, object]) -> None:
    path = _jsonl_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _post_otlp_log(event: dict[str, object]) -> None:
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip().rstrip("/")
    if not endpoint:
        return
    now_ns = str(time.time_ns())
    payload = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", os.getenv("OTEL_SERVICE_NAME") or "agent-bench-hermes"),
                        _attr("service.namespace", "agent-bench"),
                        _attr("agent_bench.run.id", os.getenv("AGENT_BENCH_RUN_ID", "")),
                        _attr("agent_bench.harness.kind", "hermes"),
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "hermes.agent_bench_otlp"},
                        "logRecords": [
                            {
                                "timeUnixNano": now_ns,
                                "severityText": "INFO",
                                "body": {"stringValue": json.dumps(event, sort_keys=True, ensure_ascii=False)[:8192]},
                                "attributes": [
                                    _attr("hermes.hook", event.get("hook")),
                                    _attr("hermes.session_id", event.get("session_id")),
                                    _attr("hermes.task_id", event.get("task_id")),
                                    _attr("hermes.model", event.get("model")),
                                    _attr("hermes.provider", event.get("provider")),
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    request = urllib.request.Request(
        endpoint + "/v1/logs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=0.5):
            return
    except Exception:
        return


def _handle_hook(hook: str, **kwargs: Any) -> None:
    event = {
        "hook": hook,
        "timestamp_unix_ns": time.time_ns(),
        "run_id": os.getenv("AGENT_BENCH_RUN_ID", ""),
        "session_id": kwargs.get("session_id") or os.getenv("AGENT_BENCH_HERMES_SESSION_ID", ""),
        "task_id": kwargs.get("task_id") or "",
        "model": kwargs.get("model") or "",
        "provider": kwargs.get("provider") or "",
        "payload": _jsonable(kwargs),
    }
    _write_jsonl(event)
    _post_otlp_log(event)


def register(ctx) -> None:
    for hook in HOOKS:
        ctx.register_hook(hook, lambda _hook=hook, **kwargs: _handle_hook(_hook, **kwargs))
