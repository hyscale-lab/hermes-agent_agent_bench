from __future__ import annotations

import base64
import json
import os
import threading
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
