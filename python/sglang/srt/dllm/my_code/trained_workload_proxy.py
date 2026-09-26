# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""FP32 and INT8 adapters for arrival-only MiniLM-DWS prediction."""
from __future__ import annotations
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict
import torch
from sglang.srt.dllm.my_code.prompt_text import strip_single_turn_llada_chat_template
from sglang.srt.dllm.my_code.calibrated_surface import (
    attach_calibrated_surface, capture_calibrated_surface,
)
_DWS_IMPORT_NAMES = ("modeling_compact", "modeling_components")

def _load_package_module(package_dir: Path, entrypoint_name: str) -> ModuleType:
    """Load a deployment entrypoint while isolating its local imports."""

    entrypoint = package_dir / entrypoint_name
    if not entrypoint.is_file():
        raise FileNotFoundError(f"predictor entrypoint is missing: {entrypoint}")
    module_name = f"_sglang_trained_predictor_{abs(hash(entrypoint))}_{id(package_dir)}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import predictor entrypoint: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    previous_modules = {
        name: sys.modules.pop(name, None) for name in _DWS_IMPORT_NAMES
    }
    try:
        sys.path.insert(0, str(package_dir))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        for name in _DWS_IMPORT_NAMES:
            sys.modules.pop(name, None)
            previous = previous_modules[name]
            if previous is not None:
                sys.modules[name] = previous
    return module

class FinalDWSWorkloadProxy:
    """Adapter for FP32 and ONNX INT8 MiniLM-DWS bundles."""

    def __init__(
        self,
        package_dir: str,
        *,
        backend_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        strip_chat_template: bool = True,
        block_size: int = 32,
        quantized: bool = False,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        self.backend = str(backend_name)
        self.batch_size = max(int(batch_size), 1)
        self.strip_chat_template = bool(strip_chat_template)
        self.block_size = max(int(block_size), 1)
        self.quantized = bool(quantized)

        if self.quantized:
            if str(device) != "cpu":
                raise ValueError(
                    f"backend {self.backend!r} is an ONNX INT8 CPU-only predictor"
                )
            for required in ("model.onnx", "config.json", "predict_onnx.py"):
                path = self.package_dir / required
                if not path.is_file():
                    raise FileNotFoundError(
                        f"INT8 DWS predictor file is missing: {path}"
                    )
            module = _load_package_module(self.package_dir, "predict_onnx.py")
            predictor_cls = getattr(module, "QuantizedOurDWSPredictor", None)
            if predictor_cls is None:
                raise TypeError(
                    f"unsupported quantized DWS package: {self.package_dir}"
                )
            self.predictor = predictor_cls(
                self.package_dir,
                threads=max(int(torch.get_num_threads()), 1),
                provider="CPUExecutionProvider",
            )
            self.device = "cpu"
            self.parameters = 0
        else:
            for required in ("best.pt", "config.json", "predict.py"):
                path = self.package_dir / required
                if not path.is_file():
                    raise FileNotFoundError(f"DWS predictor file is missing: {path}")
            module = _load_package_module(self.package_dir, "predict.py")
            predictor_cls = getattr(module, "OurDWSPredictor", None)
            if predictor_cls is None:
                raise TypeError(f"unsupported DWS package: {self.package_dir}")
            self.predictor = predictor_cls(
                self.package_dir,
                device=str(device),
                cpu_threads=max(int(torch.get_num_threads()), 1),
            )
            self.device = str(self.predictor.device)
            model = getattr(self.predictor, "model", None)
            self.parameters = (
                int(sum(parameter.numel() for parameter in model.parameters()))
                if model is not None
                else 0
            )

    def _normalize(self, text: str) -> str:
        return (
            strip_single_turn_llada_chat_template(text)
            if self.strip_chat_template
            else text
        )

    def predict(self, items) -> Dict[str, dict]:
        if not items:
            return {}
        out: Dict[str, dict] = {}
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            texts = []
            prompt_lengths = []
            for item in chunk:
                text = item.get("text")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    raise ValueError("DWS prompt RPC requires a string 'text' field")
                texts.append(self._normalize(text))
                prompt_lengths.append(len(item.get("input_ids") or ()))

            started = time.perf_counter()
            kwargs = {
                "max_denoising_steps": self.block_size,
                "include_surface": False,
            }
            if not self.quantized:
                kwargs["batch_size"] = self.batch_size
            with capture_calibrated_surface(self.predictor) as surfaces:
                predictions = self.predictor.predict(texts, prompt_lengths, **kwargs)
            attach_calibrated_surface(chunk, predictions, surfaces, self.block_size)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            for item, text, prompt_tokens, prediction in zip(
                chunk, texts, prompt_lengths, predictions
            ):
                score = float(prediction["scheduler_score"])
                expected_blocks = float(prediction["expected_blocks"])
                total_len = expected_blocks * self.block_size
                prompt_blocks = (
                    math.ceil(prompt_tokens / self.block_size) if prompt_tokens else 0
                )
                workload_curve = list(prediction.get("workload_curve") or ())
                out[str(item["rid"])] = {
                    "total_steps": score,
                    "total_len": total_len,
                    "source": "prompt_final_dws_proxy",
                    "prompt_backend": self.backend,
                    "prompt_predict_ms": elapsed_ms,
                    "proxy_input_chars": len(text),
                    "score_mode": "surface",
                    "scheduler_score": score,
                    "scheduler_area": score,
                    "predicted_total_steps_32": float(
                        prediction.get("predicted_workload_32", score)
                    ),
                    "expected_n_blocks": expected_blocks,
                    "prompt_blocks": prompt_blocks,
                    "workload_curve": workload_curve,
                    **{key: prediction[key] for key in (
                        "dws_contract_version", "dws_max_new_tokens", "dws_surface", "dws_support_projection",
                    ) if key in prediction},
                    "quantized": self.quantized,
                }
        return out

    def warmup(self) -> None:
        self.predict(
            [{"rid": "__final_dws_warmup__", "text": "warmup", "input_ids": [0]}]
        )
