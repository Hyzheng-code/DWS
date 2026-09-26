# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Stateless DWS arrival predictor, served over local TCP.

Launch with --prompt-backend minilm-dws-int8 and --prompt-backend-path pointing
to an external model package. All prediction uses prompt text and length.
"""
from __future__ import annotations
import argparse
import logging
import os
import socket
import socketserver
import threading
import torch
from sglang.srt.dllm.my_code.predictor_client import recv_msg, send_msg
from sglang.srt.dllm.my_code.predictor_backends import (
    BackendConfig, available_backends, build_backend,
)

logger = logging.getLogger("predictor_service")


class PredictorState:
    def __init__(self, backend_config):
        self.prompt_backend = build_backend(backend_config)
        self.prompt_lock = threading.Lock()

    def info(self):
        info = self.prompt_backend.info()
        return dict(ok=True, prompt_predict=True, stateful_predict=False,
                    prompt_backend=info["reported_backend"],
                    prompt_backend_name=info["name"], prompt_mode="arrival_only",
                    prompt_device=info["device"], device=info["device"],
                    prompt_supports_online_step=False)

    def prompt(self, items):
        with self.prompt_lock:
            return self.prompt_backend.predict(items)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                message = recv_msg(self.request)
            except Exception:
                return
            try:
                command = message.get("cmd")
                if command == "ping":
                    result = self.server.state.info()
                elif command == "prompt":
                    result = {"results": self.server.state.prompt(message["items"])}
                else:
                    result = {"ok": False, "error": f"unsupported command: {command!r}"}
            except Exception as exc:
                logger.exception("prediction failed")
                result = {"ok": False, "error": str(exc)}
            try:
                send_msg(self.request, result)
            except Exception:
                return


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31100)
    parser.add_argument("--prompt-backend", choices=available_backends(), required=True)
    parser.add_argument("--prompt-backend-path", required=True)
    parser.add_argument("--prompt-backend-device", default="cpu")
    parser.add_argument("--prompt-backend-batch-size", type=int, default=32)
    parser.add_argument("--prompt-block-size", type=int, default=32)
    parser.add_argument("--keep-chat-template", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=int(os.environ.get("PREDICTOR_TORCH_THREADS", "0")))
    parser.add_argument("--ready-file", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[predictor_service] %(asctime)s %(message)s")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.prompt_backend_device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("the predictor requests CUDA but CUDA is unavailable")
    state = PredictorState(BackendConfig(
        name=args.prompt_backend, path=args.prompt_backend_path,
        device=args.prompt_backend_device, batch_size=args.prompt_backend_batch_size,
        block_size=args.prompt_block_size, strip_chat_template=not args.keep_chat_template,
    ))
    try:
        state.prompt_backend.warmup()
        with _Server((args.host, args.port), _Handler) as server:
            server.state = state
            if args.ready_file:
                with open(args.ready_file, "w") as handle:
                    handle.write(f"{os.getpid()} {args.host}:{server.server_address[1]}\n")
            logger.info("listening on %s:%d backend=%s", *server.server_address, args.prompt_backend)
            server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        state.prompt_backend.close()


if __name__ == "__main__":
    serve()
