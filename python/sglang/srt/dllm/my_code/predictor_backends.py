# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""The two deployment formats of the MiniLM-DWS prompt predictor."""
from dataclasses import dataclass
from sglang.srt.dllm.my_code.trained_workload_proxy import FinalDWSWorkloadProxy


@dataclass(frozen=True)
class BackendConfig:
    name: str
    path: str
    device: str = "cpu"
    batch_size: int = 32
    strip_chat_template: bool = True
    block_size: int = 32


def available_backends():
    return ("minilm-dws", "minilm-dws-int8")


class PromptBackend:
    """Stateless arrival prediction from an external model package."""
    mode = "arrival_only"

    def __init__(self, config):
        self.name = config.name
        self.proxy = FinalDWSWorkloadProxy(
            config.path, backend_name=config.name, device=config.device,
            batch_size=config.batch_size, strip_chat_template=config.strip_chat_template,
            block_size=config.block_size, quantized=config.name == "minilm-dws-int8",
        )
        self.device = self.proxy.device
        self.batch_size = self.proxy.batch_size

    def predict(self, items):
        return self.proxy.predict(items)

    def warmup(self):
        self.proxy.warmup()

    def close(self):
        pass

    def info(self):
        return dict(name=self.name, mode=self.mode, reported_backend=self.name,
                    device=str(self.device), batch_size=self.batch_size,
                    parameters=int(self.proxy.parameters))


def build_backend(config):
    if config.name not in available_backends():
        raise ValueError(f"unsupported DWS predictor backend: {config.name!r}")
    if not config.path:
        raise ValueError("a predictor package path is required")
    if config.batch_size <= 0 or config.block_size <= 0:
        raise ValueError("predictor batch size and block size must be positive")
    return PromptBackend(config)
