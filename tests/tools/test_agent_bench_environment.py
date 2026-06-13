from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from tools.environments.agent_bench import AgentBenchEnvironment


class FakeBridgeHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers.get("content-length") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(payload)
        body = {
            "type": "exec_result",
            "id": payload["id"],
            "exit_code": 0,
            "stdout_b64": base64.b64encode(b"bridge-ok\n").decode("ascii"),
            "stderr_b64": "",
            "timed_out": False,
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


class AgentBenchEnvironmentTest(unittest.TestCase):
    def setUp(self):
        FakeBridgeHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBridgeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1/exec"

    def test_execute_posts_agent_bench_exec_payload(self):
        with mock.patch.dict(
            os.environ,
            {
                "AGENT_BENCH_TOOL_BRIDGE_ENDPOINT": self.endpoint(),
                "AGENT_BENCH_SANDBOX_WORKDIR": "/app",
            },
            clear=False,
        ):
            env = AgentBenchEnvironment(cwd="/app", timeout=5)
            result = env.execute("printf hi", timeout=5)

        self.assertEqual(result["returncode"], 0)
        self.assertIn("bridge-ok", result["output"])
        self.assertGreaterEqual(len(FakeBridgeHandler.requests), 2)
        payload = FakeBridgeHandler.requests[-1]
        self.assertEqual(payload["type"], "exec")
        self.assertEqual(payload["cwd"], "/app")
        self.assertEqual(payload["argv"][:2], ["bash", "-c"])
        self.assertIn("printf hi", payload["argv"][2])
        self.assertEqual(payload["env"]["AGENT_BENCH_HERMES_SANDBOX_BACKEND"], "agent_bench")
        self.assertEqual(payload["env"]["AGENT_BENCH_HERMES_TOOL_OPERATION"], "exec")

    def test_agent_bench_runner_forces_agent_bench_terminal_backend(self):
        from hermes_cli.agent_bench_run import run_agent_bench_session

        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured["terminal_env"] = os.environ.get("TERMINAL_ENV")
                captured["terminal_cwd"] = os.environ.get("TERMINAL_CWD")
                captured["session_id"] = os.environ.get("AGENT_BENCH_HERMES_SESSION_ID")
                captured["kwargs"] = kwargs

            def run_conversation(self, instruction, task_id=None):
                captured["instruction"] = instruction
                captured["task_id"] = task_id
                return {"completed": True, "final_response": "ok", "messages": [], "api_calls": 1}

        fake_run_agent = types.SimpleNamespace(AIAgent=FakeAgent)
        fake_plugins = types.SimpleNamespace(discover_plugins=lambda force=False: None)

        with tempfile.TemporaryDirectory() as temp:
            args = types.SimpleNamespace(
                instruction="do the benchmark task",
                instruction_file=None,
                artifacts_dir=temp,
                toolsets=None,
                api_key="test-key",
                base_url="http://127.0.0.1:4000/v1",
                provider="custom",
                api_mode="chat_completions",
                model="bench-model",
                max_iterations=3,
                session_id="session-one",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "TERMINAL_ENV": "local",
                    "TERMINAL_CWD": "/wrong",
                    "AGENT_BENCH_SANDBOX_WORKDIR": "/app",
                    "AGENT_BENCH_HERMES_SESSION_ID": "old-session",
                },
                clear=False,
            ), mock.patch.dict(
                sys.modules,
                {"run_agent": fake_run_agent, "hermes_cli.plugins": fake_plugins},
            ):
                rc = run_agent_bench_session(args)

        self.assertEqual(rc, 0)
        self.assertEqual(captured["terminal_env"], "agent_bench")
        self.assertEqual(captured["terminal_cwd"], "/app")
        self.assertEqual(captured["session_id"], "session-one")
        self.assertEqual(captured["instruction"], "do the benchmark task")
        self.assertEqual(captured["task_id"], "session-one")

    def test_agent_bench_runner_treats_max_iterations_as_incomplete(self):
        from hermes_cli.agent_bench_run import _safe_result_status

        status, failure_reason = _safe_result_status(
            {
                "completed": False,
                "final_response": "Summary after max iterations.",
                "turn_exit_reason": "max_iterations_reached(90/90)",
            }
        )

        self.assertEqual(status, "incomplete")
        self.assertEqual(failure_reason, "max_iterations_reached(90/90)")
