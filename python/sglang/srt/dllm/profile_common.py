# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""DWS profile publication and serving compatibility signatures."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def trajectory_keys(server_args, algorithm_config, block_size, max_output_blocks=33, max_new_tokens=1024):
    """Trajectory identity excludes deployment, paths and file timestamps.

    File layout is a cheap identity check, not a hash of tensor contents. Keep
    model_revision accurate when replacing weights with the same shard layout.
    """
    model = Path(server_args.model_path)
    shards = sorted((p.name, p.stat().st_size)
                    for pattern in ("*.safetensors", "*.bin", "*.pt")
                    for p in model.glob(pattern))
    return {
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "model_revision": getattr(server_args, "revision", None),
        "weights_layout_sha256": hashlib.sha256(json.dumps(shards).encode()).hexdigest(),
        "block_size": block_size, "max_output_blocks": max_output_blocks,
        "max_new_tokens": max_new_tokens,
        "threshold": float(algorithm_config.get("threshold", .95)),
        "algorithm_impl": "low_confidence_vectorized_v2" if os.environ.get(
            "SGLANG_DLLM_VECTORIZED", "1").lower() not in {"0", "false", "off"} else "low_confidence_legacy",
        "timing_version": "low_confidence_step_with_postprocess_v1",
    }


def cost_keys(server_args):
    import torch
    from sglang.version import __version__

    return {"gpu_model": torch.cuda.get_device_name(), "tp_size": server_args.tp_size,
            "dtype": str(server_args.dtype), "quantization": server_args.quantization,
            "attention_backend": server_args.attention_backend,
            "disable_cuda_graph": server_args.disable_cuda_graph,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "sglang_version": __version__}


def provenance(server_args):
    import socket
    import torch
    import subprocess

    model = Path(server_args.model_path).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         timeout=2, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"model_path": str(model), "hostname": socket.gethostname(),
            "gpu_index": torch.cuda.current_device(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "port": getattr(server_args, "port", None),
            "max_running_requests": server_args.max_running_requests,
            "sjf_pool": getattr(server_args, "my_sjf_pool", None), "source_commit": commit,
            "weights_mtime_ns": {p.name: p.stat().st_mtime_ns
                                 for pattern in ("*.safetensors", "*.bin", "*.pt")
                                 for p in model.glob(pattern)}}


def shape_scale_signature(server_args, algorithm_config, block_size, max_output_blocks=33, max_new_tokens=1024):
    return {"trajectory_keys": trajectory_keys(server_args, algorithm_config, block_size,
                                                max_output_blocks, max_new_tokens),
            "cost_keys": cost_keys(server_args), "provenance": provenance(server_args),
            "disable_radix_cache": bool(server_args.disable_radix_cache),
            "prefill_source": "cuda_event"}


def runtime_signature(server_args, algorithm_config):
    """Serving signature used when loading schema-4 DWS-WSL profiles."""
    import torch

    from sglang.version import __version__

    model_path = Path(server_args.model_path).resolve()
    config_path = model_path / "config.json"
    # File metadata also detects an in-place local weight replacement. It is
    # deliberately named a manifest hash, not a hash of tensor contents.
    shards = sorted(
        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
        for pattern in ("*.safetensors", "*.bin", "*.pt")
        for p in model_path.glob(pattern)
    )
    return {
        "model_path": str(model_path),
        "revision": getattr(server_args, "revision", None),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "weights_manifest_sha256": hashlib.sha256(json.dumps(shards).encode()).hexdigest(),
        "gpu_model": torch.cuda.get_device_name(),
        "tp_size": server_args.tp_size,
        "dtype": str(server_args.dtype),
        "quantization": server_args.quantization,
        "attention_backend": server_args.attention_backend,
        "disable_cuda_graph": server_args.disable_cuda_graph,
        "disable_radix_cache": server_args.disable_radix_cache,
        "threshold": float(algorithm_config.get("threshold", 0.95)),
        "vectorized": os.environ.get("SGLANG_DLLM_VECTORIZED", "1").lower(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sglang_version": __version__,
        "timing_version": "low_confidence_step_with_postprocess_v1",
    }


