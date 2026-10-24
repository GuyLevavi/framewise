"""A tiny stand-in for Triton's HTTP API (KServe v2 protocol), backed by ONNX Runtime.

It speaks the same wire format as Triton for binary tensors, so TritonDetector is tested over
real HTTP. It is not Triton: no model repository, batching scheduler, metrics or DALI.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort

HEADER = "Inference-Header-Content-Length"


class FakeTriton:
    def __init__(self, model_path: Path, model_name: str = "yolo26n", fail_first: int = 0) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.model_name = model_name
        self.fail_first = fail_first
        self.requests = 0
        self.batch_sizes: list[int] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeTriton:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(200 if self.path.startswith("/v2/health") else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fake.requests += 1
                if self.path != f"/v2/models/{fake.model_name}/infer":
                    return self._reply(404, b'{"error":"unknown model"}')
                if fake.requests <= fake.fail_first:
                    return self._reply(503, b'{"error":"server busy"}')
                json_len = int(self.headers.get(HEADER, len(body)))
                request = json.loads(body[:json_len])
                spec = request["inputs"][0]
                tensor = np.frombuffer(body[json_len:], dtype=np.float32).reshape(spec["shape"])
                fake.batch_sizes.append(int(tensor.shape[0]))
                output = np.asarray(fake.session.run(None, {spec["name"]: tensor})[0], np.float32)
                raw = output.tobytes()
                header = json.dumps(
                    {
                        "model_name": fake.model_name,
                        "outputs": [
                            {
                                "name": request["outputs"][0]["name"],
                                "datatype": "FP32",
                                "shape": list(output.shape),
                                "parameters": {"binary_data_size": len(raw)},
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header(HEADER, str(len(header)))
                self.send_header("Content-Length", str(len(header) + len(raw)))
                self.end_headers()
                self.wfile.write(header + raw)
                return None

            def _reply(self, code: int, payload: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler
