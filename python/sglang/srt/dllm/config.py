# DWS research fork: modified from the imported SGLang 0.5.10 source.
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        my_sjf: bool = False,
        my_sjf_dws: bool = False,
        my_sjf_cost_model: str = "auto",
        my_sjf_cost_profile: str = "",
        my_cost_probe: str = "auto",
        my_cost_probe_refresh: str = "off",
        my_cost_probe_timeout: float = 0.0,
        my_cost_probe_prompts: str = "",
        my_sjf_pool: int = 0,
        my_sjf_aging: float = 0.1,
        my_sjf_log: str = "",
        my_sjf_prefill_quota: int = 2,
        my_sjf_prefill_order: bool = False,
        my_sjf_ready_low: int = 0,
        my_sjf_ready_high: int = 0,
        my_sjf_prefill_pause_ratio: float = 1.0,
        my_sjf_promote_aging: float | None = None,
        my_occupancy_log: str = "",
        my_metrics_log: str = "",
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config or {}
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.my_sjf = my_sjf
        self.my_sjf_dws = my_sjf_dws
        self.my_sjf_cost_model_requested = str(my_sjf_cost_model).lower()
        if self.my_sjf_cost_model_requested not in {"auto", "dws-wsl"}:
            raise ValueError("my_sjf_cost_model must be auto or dws-wsl")
        self.my_sjf_cost_model = self.my_sjf_cost_model_requested
        if self.my_sjf_cost_model == "auto":
            self.my_sjf_cost_model = "dws-wsl" if my_sjf and my_sjf_dws else "unit"
        self.my_sjf_cost_profile = str(my_sjf_cost_profile or "")
        if self.my_sjf_cost_model == "dws-wsl":
            if not my_sjf or not my_sjf_dws:
                raise ValueError("DWS-WSL requires --my-sjf --my-sjf-dws")
            if not self.my_sjf_cost_profile:
                raise ValueError("DWS-WSL requires --my-sjf-cost-profile pointing to an external profile")
        self.my_cost_profile = None
        self.my_cost_probe = my_cost_probe
        self.my_cost_probe_refresh = my_cost_probe_refresh
        self.my_cost_probe_timeout = my_cost_probe_timeout
        self.my_cost_probe_prompts = my_cost_probe_prompts
        if (my_cost_probe not in {"auto", "force", "off"}
                or my_cost_probe_refresh not in {"off", "idle"}
                or not 0 <= my_cost_probe_timeout < float("inf")):
            raise ValueError("invalid DWS-WSL probe configuration")
        self.my_sjf_pool = my_sjf_pool if my_sjf_pool > 0 else 2 * max_running_requests
        self.my_sjf_aging = float(my_sjf_aging)
        self.my_sjf_promote_aging = (
            self.my_sjf_aging if my_sjf_promote_aging is None
            else float(my_sjf_promote_aging)
        )
        if any(
            not 0 <= rate < 1
            for rate in (self.my_sjf_aging, self.my_sjf_promote_aging)
        ):
            raise ValueError("SJF aging discounts must be fractions in [0, 1)")
        self.my_sjf_log = my_sjf_log
        # Closed prefill cohorts at low load, quota scheduling at high load.
        self.my_sjf_prefill_quota = my_sjf_prefill_quota
        self.my_sjf_prefill_order = my_sjf_prefill_order
        if my_sjf_ready_low < 0 or my_sjf_ready_high < 0:
            raise ValueError("my_sjf_ready_low/high must be non-negative")
        if my_sjf_prefill_pause_ratio <= 0:
            raise ValueError("my_sjf_prefill_pause_ratio must be positive")
        self.my_sjf_ready_low = my_sjf_ready_low
        self.my_sjf_ready_high = my_sjf_ready_high
        self.my_sjf_prefill_pause_ratio = my_sjf_prefill_pause_ratio
        self.my_occupancy_log = my_occupancy_log
        self.my_metrics_log = my_metrics_log

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)


        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            my_sjf=server_args.my_sjf,
            my_sjf_dws=server_args.my_sjf_dws,
            my_sjf_cost_model=server_args.my_sjf_cost_model,
            my_sjf_cost_profile=server_args.my_sjf_cost_profile,
            my_cost_probe=getattr(server_args, "my_cost_probe", "auto"),
            my_cost_probe_refresh=getattr(server_args, "my_cost_probe_refresh", "off"),
            my_cost_probe_timeout=getattr(server_args, "my_cost_probe_timeout", 0),
            my_cost_probe_prompts=getattr(server_args, "my_cost_probe_prompts", ""),
            my_sjf_pool=server_args.my_sjf_pool,
            my_sjf_aging=server_args.my_sjf_aging,
            my_sjf_log=server_args.my_sjf_log,
            my_sjf_prefill_quota=server_args.my_sjf_prefill_quota,
            my_sjf_prefill_order=server_args.my_sjf_prefill_order,
            my_sjf_ready_low=server_args.my_sjf_ready_low,
            my_sjf_ready_high=server_args.my_sjf_ready_high,
            my_sjf_prefill_pause_ratio=server_args.my_sjf_prefill_pause_ratio,
            my_sjf_promote_aging=server_args.my_sjf_promote_aging,
            my_occupancy_log=server_args.my_occupancy_log,
            my_metrics_log=server_args.my_metrics_log,
        )
