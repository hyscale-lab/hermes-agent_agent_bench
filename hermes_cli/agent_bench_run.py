"""Structured Agent Bench session runner for Hermes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hermes_cli.oneshot import (
    _create_session_db_for_oneshot,
    _normalize_toolsets,
    _validate_explicit_toolsets,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_instruction(args: argparse.Namespace) -> str:
    if args.instruction_file:
        if args.instruction_file == "-":
            return sys.stdin.read()
        return Path(args.instruction_file).read_text(encoding="utf-8")
    return args.instruction or ""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _message_tool_event_count(messages: object) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            count += 1
    return count


def _safe_result_status(result: dict[str, Any]) -> tuple[str, str | None]:
    failed = bool(result.get("failed"))
    interrupted = bool(result.get("interrupted"))
    partial = bool(result.get("partial"))
    final_response = str(result.get("final_response") or "").strip()
    completed = bool(result.get("completed"))
    turn_exit_reason = str(result.get("turn_exit_reason") or "").strip()
    if failed:
        return "failed", str(result.get("failure_reason") or turn_exit_reason or "hermes reported failure")
    if turn_exit_reason.startswith("max_iterations_reached"):
        return "incomplete", turn_exit_reason
    if interrupted or partial:
        return "incomplete", str(turn_exit_reason or "interrupted or partial")
    if final_response or completed:
        return "completed", None
    return "incomplete", "no final response"


def _clarify_callback(question: str, choices: object = None) -> str:
    return (
        "No interactive clarification channel is available in Agent Bench. "
        "Proceed with the safest reasonable assumption and state it briefly."
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _set_positive_env(name: str, value: int | None) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _request_overrides(args: argparse.Namespace) -> dict[str, object] | None:
    if args.llm_timeout is None:
        return None
    return {"timeout": args.llm_timeout}


def _apply_context_length(agent: object, args: argparse.Namespace) -> None:
    if args.context_length is None:
        return
    setattr(agent, "_config_context_length", args.context_length)
    compressor = getattr(agent, "context_compressor", None)
    update_model = getattr(compressor, "update_model", None)
    if callable(update_model):
        update_model(
            model=args.model,
            context_length=args.context_length,
            base_url=args.base_url,
            api_key=args.api_key,
            provider=args.provider,
            api_mode=args.api_mode,
        )


def run_agent_bench_session(args: argparse.Namespace) -> int:
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    os.environ["TERMINAL_ENV"] = "agent_bench"
    os.environ["TERMINAL_CWD"] = (
        os.getenv("AGENT_BENCH_SANDBOX_WORKDIR")
        or os.getenv("AGENT_BENCH_SANDBOX_WORKSPACE_DIR")
        or os.getenv("AGENT_BENCH_SANDBOX_AGENT_WORKSPACE_DIR")
        or "/workspace"
    )
    os.environ["AGENT_BENCH_HERMES_SESSION_ID"] = args.session_id
    _set_positive_env("AGENT_BENCH_TOOL_TIMEOUT_SECONDS", args.tool_timeout)
    _set_positive_env("AGENT_BENCH_FILE_TOOL_TIMEOUT_SECONDS", args.file_tool_timeout)

    instruction = _read_instruction(args).strip()
    artifacts_dir = Path(args.artifacts_dir).expanduser()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started_monotonic = time.monotonic()

    explicit_toolsets, toolsets_error = _validate_explicit_toolsets(args.toolsets)
    if toolsets_error:
        raise RuntimeError(toolsets_error.strip())
    toolsets_list = explicit_toolsets if explicit_toolsets is not None else _normalize_toolsets(args.toolsets)

    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins(force=True)
    except Exception as exc:  # noqa: BLE001 - plugin telemetry is best effort
        print(f"agent_bench_run: plugin discovery failed: {exc}", file=sys.stderr)

    from run_agent import AIAgent

    agent = AIAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        provider=args.provider,
        api_mode=args.api_mode,
        model=args.model,
        max_iterations=args.max_iterations,
        enabled_toolsets=toolsets_list,
        quiet_mode=True,
        platform="agent-bench",
        session_id=args.session_id,
        session_db=_create_session_db_for_oneshot(),
        clarify_callback=_clarify_callback,
        max_tokens=args.max_output_tokens,
        request_overrides=_request_overrides(args),
        skip_context_files=True,
        load_soul_identity=False,
        skip_memory=True,
    )
    _apply_context_length(agent, args)
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None

    result: dict[str, Any]
    try:
        raw_result = agent.run_conversation(instruction, task_id=args.session_id)
        result = raw_result if isinstance(raw_result, dict) else {"final_response": str(raw_result), "messages": []}
    except Exception as exc:  # noqa: BLE001 - runner must still emit artifacts
        result = {
            "final_response": None,
            "messages": [],
            "failed": True,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
        }

    status, failure_reason = _safe_result_status(result)
    finished_at = utc_now()
    duration_seconds = round(time.monotonic() - started_monotonic, 6)
    final_response = str(result.get("final_response") or "")

    final_response_path = artifacts_dir / "final_response.md"
    if final_response.strip():
        final_response_path.write_text(final_response.rstrip() + "\n", encoding="utf-8")
    elif final_response_path.exists():
        final_response_path.unlink()

    messages = result.get("messages") if isinstance(result.get("messages"), list) else []
    chat_history_all_path = artifacts_dir / "chat_history_all.json"
    _write_json(
        chat_history_all_path,
        {
            "sessions": {
                args.session_id: {
                    "messages": _jsonable(messages),
                    "status": status,
                }
            }
        },
    )
    _write_json(
        artifacts_dir / "session_tree.json",
        {"sessions": {args.session_id: {"parent": None, "status": status}}},
    )

    metrics = {
        "session_key": args.session_id,
        "external_session_id": args.session_id,
        "status": status,
        "failure_reason": failure_reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "api_calls": result.get("api_calls", 0),
        "tool_event_count": _message_tool_event_count(messages),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "model": result.get("model") or args.model,
        "provider": result.get("provider") or args.provider,
        "base_url": result.get("base_url") or args.base_url,
        "turn_exit_reason": result.get("turn_exit_reason"),
        "limits": {
            "max_iterations": args.max_iterations,
            "llm_timeout_seconds": args.llm_timeout,
            "tool_timeout_seconds": args.tool_timeout,
            "file_tool_timeout_seconds": args.file_tool_timeout,
            "context_length": args.context_length,
            "max_output_tokens": args.max_output_tokens,
        },
    }
    metrics_path = artifacts_dir / "metrics.json"
    response_path = artifacts_dir / "hermes_response.json"
    _write_json(metrics_path, metrics)
    _write_json(
        response_path,
        {
            "status": status,
            "failure_reason": failure_reason,
            "external_session_id": args.session_id,
            "final_response_present": bool(final_response.strip()),
            "result": _jsonable(result),
            "artifacts": {
                "metrics_path": str(metrics_path),
                "chat_history_all_path": str(chat_history_all_path),
                "session_tree_path": str(artifacts_dir / "session_tree.json"),
                "final_response_path": str(final_response_path) if final_response.strip() else None,
            },
        },
    )

    summary = {
        "status": status,
        "failure_reason": failure_reason,
        "response_json_path": str(response_path),
        "final_response_path": str(final_response_path) if final_response.strip() else None,
        "metrics_path": str(metrics_path),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if status == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Hermes Agent Bench session.")
    parser.add_argument("--instruction-file", help="Instruction file path, or '-' to read stdin.")
    parser.add_argument("--instruction", help="Instruction text when --instruction-file is not used.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory where session artifacts are written.")
    parser.add_argument("--session-id", required=True, help="External Agent Bench session id.")
    parser.add_argument("--model", required=True, help="Model name served by the Agent Bench model server.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="agent-bench-local-no-auth", help="API key passed to the serving endpoint.")
    parser.add_argument("--provider", default="custom", help="Hermes provider id.")
    parser.add_argument("--api-mode", default="chat_completions", help="Hermes API mode.")
    parser.add_argument("--toolsets", default="terminal,file", help="Comma-separated toolsets to enable.")
    parser.add_argument("--max-iterations", type=int, default=90, help="Maximum tool-calling iterations.")
    parser.add_argument("--llm-timeout", type=int, default=None, help="Per-request LLM timeout in seconds.")
    parser.add_argument("--tool-timeout", type=int, default=None, help="Default Agent Bench tool timeout in seconds.")
    parser.add_argument("--file-tool-timeout", type=int, default=None, help="Default Agent Bench file-tool timeout in seconds.")
    parser.add_argument("--context-length", type=int, default=None, help="Model context window hint in tokens.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Maximum model output tokens.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.instruction_file and not args.instruction:
        parser.error("one of --instruction-file or --instruction is required")
    for name in ("max_iterations", "llm_timeout", "tool_timeout", "file_tool_timeout", "context_length", "max_output_tokens"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    try:
        return run_agent_bench_session(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"agent_bench_run failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
