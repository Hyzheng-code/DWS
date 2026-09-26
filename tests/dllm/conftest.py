"""Exercise pure cost modules without importing the CUDA serving stack."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "python"
for name in ("sglang", "sglang.srt", "sglang.srt.dllm", "sglang.srt.dllm.my_code", "sglang.srt.dllm.mixin"):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules[name] = package
