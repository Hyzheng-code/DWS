# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Immutable, deployment-local DWS profiles. Dataset is provenance, not identity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

VERSION_PATTERN = re.compile(r"v[0-9]{3,}")


def model_name(value):
    name = Path(value).name
    return {"LLaDA2.0-mini": "mini", "LLaDA2.0-flash": "flash"}.get(name, name)


def hardware_name(value):
    return {
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": "PRO6000",
        "NVIDIA RTX A6000": "A6000",
    }.get(value, value.replace(" ", "_"))


@dataclass(frozen=True)
class ProfileKey:
    model: str
    hardware: str
    tp: int

    def __post_init__(self):
        for value in (self.model, self.hardware):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise ValueError("profile key must be a safe, nonempty path component")
        if type(self.tp) is not int or self.tp <= 0:
            raise ValueError("profile TP must be a positive integer")

    @classmethod
    def from_signature(cls, signature):
        return cls(model_name(signature["provenance"]["model_path"]),
                   hardware_name(signature["cost_keys"]["gpu_model"]),
                   signature["cost_keys"]["tp_size"])

    def to_dict(self):
        return {"model": self.model, "hardware": self.hardware, "tp": self.tp}


def validate_signature(expected, actual):
    from sglang.srt.dllm.dws_wsl import COST_FIELDS, TRAJECTORY_FIELDS

    for signature in (expected, actual):
        if not signature.get("disable_radix_cache") or signature.get("prefill_source") != "cuda_event":
            raise ValueError("managed profiles require disabled radix cache and CUDA-event prefill")
        if not TRAJECTORY_FIELDS.issubset(signature.get("trajectory_keys", {})) or not COST_FIELDS.issubset(signature.get("cost_keys", {})):
            raise ValueError("incomplete deployment signature")
    mismatches = []
    for section in ("trajectory_keys", "cost_keys"):
        left, right = expected[section], actual[section]
        mismatches.extend(f"{section}.{k}" for k in set(left) | set(right) if left.get(k) != right.get(k))
    if ProfileKey.from_signature(expected) != ProfileKey.from_signature(actual):
        mismatches.append("ProfileKey")
    if mismatches:
        raise ValueError("deployment mismatch (no cross-profile fallback): " + ", ".join(sorted(mismatches)))




def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_managed_profile(directory):
    """Load only meta/params; training statistics are never needed for serving."""
    import numpy as np
    from sglang.srt.dllm.dws_wsl import DWSWSLProfile

    directory = Path(directory)
    if (directory / "ACTIVE").exists():
        version = (directory / "ACTIVE").read_text().strip()
        if not VERSION_PATTERN.fullmatch(version):
            raise ValueError("invalid ACTIVE version")
        directory = directory / version
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("registry_schema") != 1 or meta.get("version") != directory.name:
        raise ValueError("unsupported or misplaced managed profile")
    if file_digest(directory / "params.npz") != meta["files"]["params.npz"]:
        raise ValueError("params.npz checksum mismatch")
    payload = dict(meta["runtime"])
    with np.load(directory / "params.npz", allow_pickle=False) as params:
        payload.update(denoising_cell_ms=params["W"].tolist(),
                       prefill_block_ms=float(params["alpha"]), refresh_block_ms=float(params["beta"]),
                       prompt_calibration={"delta_ms_per_step_per_prompt_block": float(params["delta"]),
                                           "mbar": float(params["m_bar"]), "block_rounding": "floor"},
                       fallback={"denoising_step_ms": params["fallback_depth_ms"].tolist()})
    profile = DWSWSLProfile.from_mapping(payload)
    key = ProfileKey.from_signature(payload["compatibility"])
    if (any(meta.get(k) != v for k, v in key.to_dict().items())
            or payload["managed_profile"]["key"] != key.to_dict()
            or meta.get("profile_id") != profile.profile_id
            or meta.get("block_size") != profile.block_size
            or meta.get("nmax") != profile.max_output_blocks
            or meta.get("dtype") != payload["compatibility"]["cost_keys"]["dtype"]
            or meta.get("threshold") != payload["compatibility"]["trajectory_keys"]["threshold"]):
        raise ValueError("inconsistent managed profile metadata")
    return profile


