"""Agent Bench execution environment.

Routes Hermes terminal/file shell work through the Agent Bench HTTP exec bridge.
The benchmark harness owns the real sandbox container; Hermes only speaks the
bridge protocol and preserves its normal per-session cwd/snapshot behavior.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Any
from urllib.parse import urlparse, urlunparse

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle

logger = logging.getLogger(__name__)


class AgentBenchBridgeError(RuntimeError):
    """Raised when the Agent Bench exec bridge rejects or fails a request."""


def _bridge_endpoint() -> str:
    raw = os.getenv("AGENT_BENCH_TOOL_BRIDGE_ENDPOINT", "").strip()
    if not raw:
        raise AgentBenchBridgeError("AGENT_BENCH_TOOL_BRIDGE_ENDPOINT is required for TERMINAL_ENV=agent_bench")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AgentBenchBridgeError(f"invalid AGENT_BENCH_TOOL_BRIDGE_ENDPOINT: {raw!r}")
    path = parsed.path or "/v1/exec"
    if not path.endswith("/v1/exec"):
        path = raw.rstrip("/") + "/v1/exec"
        return path
    return urlunparse(parsed)


def _default_cwd() -> str:
    return (
        os.getenv("AGENT_BENCH_SANDBOX_WORKDIR")
        or os.getenv("AGENT_BENCH_SANDBOX_WORKSPACE_DIR")
        or os.getenv("AGENT_BENCH_SANDBOX_AGENT_WORKSPACE_DIR")
        or os.getenv("TERMINAL_CWD")
        or "/workspace"
    )


def _decode_b64(value: str | None) -> bytes:
    if not value:
        return b""
    try:
        return base64.b64decode(value.encode("ascii"))
    except Exception as exc:  # noqa: BLE001 - protocol error should be explicit
        raise AgentBenchBridgeError(f"invalid base64 in bridge response: {exc}") from exc


class AgentBenchEnvironment(BaseEnvironment):
    """Hermes environment backed by Agent Bench's /v1/exec protocol."""

    def __init__(self, cwd: str = "", timeout: int = 180, env: dict[str, str] | None = None):
        self.endpoint = _bridge_endpoint()
        self.bridge_session_id = os.getenv("AGENT_BENCH_HERMES_SESSION_ID") or uuid.uuid4().hex
        super().__init__(cwd=cwd or _default_cwd(), timeout=timeout, env=env or {})
        self.init_session()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        shell = os.getenv("AGENT_BENCH_SANDBOX_BASH", "bash")
        argv = [shell, "-lc" if login else "-c", cmd_string]
        timeout_ms = max(1, int(float(timeout) * 1000))

        def _exec() -> tuple[str, int]:
            return self._post_exec(argv=argv, timeout_ms=timeout_ms, stdin_data=stdin_data)

        return _ThreadedProcessHandle(_exec)

    def _bridge_env(self) -> dict[str, str]:
        env = {str(key): str(value) for key, value in self.env.items()}
        env.update(
            {
                "AGENT_BENCH_HERMES_SANDBOX_BACKEND": "agent_bench",
                "AGENT_BENCH_HERMES_SESSION_ID": self.bridge_session_id,
                "AGENT_BENCH_HERMES_TOOL_OPERATION": "exec",
            }
        )
        return env

    def _post_exec(
        self,
        *,
        argv: list[str],
        timeout_ms: int,
        stdin_data: str | None,
    ) -> tuple[str, int]:
        payload: dict[str, Any] = {
            "type": "exec",
            "id": f"hermes-{uuid.uuid4().hex}",
            "cwd": self.cwd or _default_cwd(),
            "argv": argv,
            "env": self._bridge_env(),
            "stdin_b64": base64.b64encode(stdin_data.encode("utf-8")).decode("ascii") if stdin_data is not None else None,
            "timeout_ms": timeout_ms,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, timeout_ms / 1000 + 5.0)) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AgentBenchBridgeError(f"Agent Bench bridge HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise AgentBenchBridgeError(f"Agent Bench bridge request failed: {exc}") from exc

        if body.get("type") != "exec_result":
            raise AgentBenchBridgeError(f"unexpected bridge response type: {body.get('type')!r}")
        stdout = _decode_b64(body.get("stdout_b64"))
        stderr = _decode_b64(body.get("stderr_b64"))
        exit_code = int(body.get("exit_code", 1))
        if bool(body.get("timed_out")) and exit_code == 0:
            exit_code = 124
        output = (stdout + stderr).decode("utf-8", errors="replace")
        return output, exit_code

    def cleanup(self) -> None:
        return None
