# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""DWS-WSL profiles. Pure-Python runtime; immutable publications.

Schema 4 and legacy schema-5 profiles remain readable. Full WLS fits use the
schema-5 container with shape_origin=full_wls; g is just mean(W), and a is zero.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from sglang.srt.dllm.dws_profile_common import _surface, evaluate_reference_fallback
from sglang.srt.dllm.cost_model import DllmCostBreakdown
from sglang.srt.dllm.dws_wsl_legacy import (
    DWSWSLProfile as LegacyProfile, evaluate_dws_wsl as evaluate_legacy,
    profile_digest as legacy_digest,
)

FEATURE_VERSION = "dws_wsl_shape_scale_v1"
MODEL_TYPE = "dws-wsl"
TRAJECTORY_FIELDS = {"model_config_sha256", "model_revision", "weights_layout_sha256",
                     "block_size", "max_output_blocks", "max_new_tokens", "threshold",
                     "algorithm_impl", "timing_version"}
COST_FIELDS = {"gpu_model", "tp_size", "dtype", "quantization", "attention_backend",
               "disable_cuda_graph", "torch_version", "cuda_version", "sglang_version"}


def canonical(data):
    return json.dumps(data, sort_keys=True, allow_nan=False).encode()


def section_digest(section, name):
    return name + "-" + hashlib.sha256(canonical(
        {k: v for k, v in section.items() if k != name + "_id"})).hexdigest()[:16]


def profile_digest(payload):
    if payload.get("schema_version") != 5:
        return legacy_digest(payload)
    return "dws-wsl-" + hashlib.sha256((payload["shape"]["shape_id"] +
                                      payload["scale"]["scale_id"]).encode()).hexdigest()[:24]


def seal_profile(payload):
    data = json.loads(json.dumps(payload, allow_nan=False))
    for name in ("shape", "scale"):
        data[name][name + "_id"] = section_digest(data[name], name)
    data["profile_id"] = profile_digest(data)
    return data


def validate_suite(suite):
    if suite.get("version") != 1:
        raise ValueError("unsupported probe suite version")
    if "sidecar" in suite:
        if Path(suite["sidecar"]).name != suite["sidecar"] or not suite.get("sha256"):
            raise ValueError("probe sidecar must be a local filename with a digest")
        return
    seen = set()
    for row in suite.get("requests", []):
        ids = row.get("input_ids")
        if (not isinstance(ids, list) or not 0 < len(ids) <= 2048
                or any(type(v) is not int or v < 0 for v in ids)
                or row.get("kind") not in {"natural", "canvas"}
                or not row.get("probe_id") or row["probe_id"] in seen
                or type(row.get("max_new_tokens")) is not int or row["max_new_tokens"] <= 0
                or bool(row.get("ignore_eos")) != (row["kind"] == "canvas")):
            raise ValueError("invalid or duplicate probe suite request")
        seen.add(row["probe_id"])


@dataclass(frozen=True)
class DWSWSLProfile(LegacyProfile):
    shape_id: str = ""
    scale_id: str = ""

    @property
    def is_v5(self):
        return bool(self.shape_id)

    @property
    def is_full_fit(self):
        return self.payload.get("shape", {}).get("shape_origin") == "full_wls"

    @classmethod
    def from_mapping(cls, payload):
        if payload.get("schema_version") == 4:
            return super().from_mapping(payload)
        data = json.loads(json.dumps(payload, allow_nan=False))
        if (data.get("schema_version"), data.get("model_type"), data.get("feature_version")) != (
                5, MODEL_TYPE, FEATURE_VERSION):
            raise ValueError("expected DWS-WSL profile (schema 5)")
        shape, scale = data["shape"], data["scale"]
        if (data.get("status") != "ready" or any(section[name + "_id"] != section_digest(section, name)
                for name, section in (("shape", shape), ("scale", scale)))
                or data.get("profile_id") != profile_digest(data)):
            raise ValueError("DWS-WSL profile is incomplete or its digest changed")
        keys, support = shape["trajectory_keys"], shape["support"]
        if not TRAJECTORY_FIELDS.issubset(keys) or not COST_FIELDS.issubset(scale["cost_keys"]):
            raise ValueError("incomplete trajectory/cost keys")
        if type(keys["max_new_tokens"]) is not int or keys["max_new_tokens"] <= 0:
            raise ValueError("invalid generation limit")
        d, j = keys["block_size"], keys["max_output_blocks"]
        if type(d) is not int or type(j) is not int or not 0 < d <= 1024 or not 0 < j <= 65536:
            raise ValueError("invalid shape dimensions")
        w = shape["W_bar"]
        if len(w) != j or any(len(row) != d for row in w):
            raise ValueError("invalid shape dimensions")
        values = [float(v) for row in w for v in row]
        if not all(math.isfinite(v) and v >= 0 for v in values) or not math.isclose(
                math.fsum(values) / len(values), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("W_bar must be nonnegative with mean one")
        delta, mbar = float(shape["delta_bar"]), float(shape["m_bar"])
        lo, hi = support["m_lo"], support["m_hi"]
        if (type(lo) is not int or type(hi) is not int or not 0 <= lo <= mbar <= hi
                or not math.isfinite(delta) or not math.isfinite(mbar)
                or support["prompt_tokens_min"] // d != lo or support["prompt_tokens_max"] // d != hi
                or keys["max_new_tokens"] != support["max_new_tokens"]):
            raise ValueError("invalid shape support")
        origin = shape.get("shape_origin")
        if origin not in {"native", "converted_schema4", "full_wls"}:
            raise ValueError("invalid shape_origin")
        if origin == "native" and min(values) + min(delta*(lo-mbar), delta*(hi-mbar)) < -1e-12:
            raise ValueError("native shape has a negative endpoint kernel")
        g, a = float(scale["g"]), float(scale["a"])
        alpha, beta = float(scale["alpha_prefill_block_ms"]), float(scale["beta_refresh_block_ms"])
        if not all(math.isfinite(v) and v > 0 for v in (g, alpha, beta)) or not math.isfinite(a) or a < 0:
            raise ValueError("scale requires g/alpha/beta > 0 and a >= 0")
        if origin == "full_wls":
            fit = shape.get("fit", {})
            stats_id = fit.get("statistics_id", "")
            if (a != 0 or fit.get("objective") != "appendix_e_v1"
                    or len(stats_id) != 64 or any(c not in "0123456789abcdef" for c in stats_id)
                    or fit.get("requests", 0) < 3):
                raise ValueError("invalid full WLS fit metadata")
        if scale.get("probe_state") not in {"measured", "reused", "inherited"}:
            raise ValueError("invalid scale probe_state")
        validate_suite(shape.get("probe_suite", {}))
        depth = shape.get("fallback_depth_bar", [1.0]*d)
        if len(depth) != d or not all(math.isfinite(v) and v > 0 for v in depth):
            raise ValueError("positive normalized depth fallback required")
        obj = cls(d, j, tuple(tuple(g*float(v)+a for v in row) for row in w), alpha, beta,
                  tuple(g*v+a for v in depth), keys["max_new_tokens"], data["profile_id"], data,
                  g*delta, mbar, support["prompt_tokens_min"], support["prompt_tokens_max"],
                  shape["shape_id"], scale["scale_id"])
        if not math.isfinite((sum(map(sum, obj.denoising_cell_ms)) + abs(obj.delta)*(hi-lo)*d*j) / beta):
            raise ValueError("cost range overflows")
        return obj

    def validate_runtime(self, signature, block_size):
        if self.payload.get("managed_profile"):
            from sglang.srt.dllm.cost_profile_registry import validate_signature
            validate_signature(self.payload["compatibility"], signature)
            if block_size != self.block_size:
                raise ValueError("managed profile block_size mismatch")
            return
        if not self.is_v5:
            return super().validate_runtime(signature, block_size)
        if not signature.get("disable_radix_cache"):
            raise ValueError("DWS-WSL v2 requires disable_radix_cache=true")
        if (signature.get("prefill_source") != "cuda_event" or
                self.payload["shape"].get("timing", {}).get("prefill_source") != "cuda_event"):
            raise ValueError("deployment requires CUDA prefill; wall profiles are offline-only")
        expected = self.payload["shape"]["trajectory_keys"]
        actual = signature.get("trajectory_keys", {})
        mismatch = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        if block_size != self.block_size or mismatch:
            raise ValueError("trajectory mismatch: " + ", ".join(mismatch))

    def with_scale(self, scale):
        if not self.is_v5:
            raise ValueError("convert schema 4 before updating scale")
        return type(self).from_mapping(seal_profile({**self.payload, "scale": scale}))

    def prompt_correction(self, prompt_tokens):
        if not self.is_v5:
            return super().prompt_correction(prompt_tokens)
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("invalid prompt length")
        support = self.payload["shape"]["support"]
        m = prompt_tokens // self.block_size
        if not self.is_full_fit:
            m = min(max(m, support["m_lo"]), support["m_hi"])
        return self.delta * (m-self.mbar)

    def kernel(self, prompt_tokens):
        corr = self.prompt_correction(prompt_tokens)
        raw = tuple(tuple(w+corr for w in row) for row in self.denoising_cell_ms)
        if self.is_full_fit:
            # Eq. (15) constrains W, not K. Clipping K would change the model
            # optimized by the fitter, especially for signed delta.
            return raw, 0
        # A few ulps at an active native constraint are roundoff, not a clamp event.
        native = self.payload["shape"]["shape_origin"] == "native"
        count = 0 if native else sum(v < 0 for row in raw for v in row)
        return tuple(tuple(max(v, 0.0) for v in row) for row in raw), count


def load_dws_wsl_profile(path):
    path = Path(path)
    if path.is_dir():
        from sglang.srt.dllm.cost_profile_registry import load_managed_profile
        return load_managed_profile(path)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("managed_profile"):
        from sglang.srt.dllm.cost_profile_registry import load_managed_profile
        profile = load_managed_profile(path.parent)
        if data != profile.to_dict():
            raise ValueError("managed JSON export differs from meta.json/params.npz")
        return profile
    return DWSWSLProfile.from_mapping(data)


def read_probe_suite(profile, profile_path):
    suite = profile.payload["shape"]["probe_suite"]
    if "sidecar" not in suite:
        return suite.get("requests", [])
    path = Path(profile_path).parent / suite["sidecar"]
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != suite["sha256"]:
        raise ValueError("probe suite sidecar digest mismatch")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    validate_suite({"version": 1, "requests": rows})
    return rows


def copy_probe_sidecar(profile, profile_path, output_dir):
    """Keep a scale snapshot portable without changing its sealed shape ID."""
    suite = profile.payload["shape"]["probe_suite"]
    if "sidecar" not in suite:
        return None
    import os
    import tempfile
    source = Path(profile_path).parent/suite["sidecar"]
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != suite["sha256"]:
        raise ValueError("probe suite sidecar digest mismatch")
    target = Path(output_dir)/suite["sidecar"]
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError("probe sidecar filename collision; use a content-addressed filename")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name+".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def evaluate_dws_wsl(prediction, *, prompt_tokens, include_prefill, profile, max_new_tokens):
    if not profile.is_v5:
        return evaluate_legacy(prediction, prompt_tokens=prompt_tokens, include_prefill=include_prefill,
                               profile=profile, max_new_tokens=max_new_tokens)
    pred = prediction if isinstance(prediction, dict) else {}
    if (max_new_tokens != profile.max_new_tokens or
            ("dws_max_new_tokens" in pred and pred["dws_max_new_tokens"] != profile.max_new_tokens)):
        raise ValueError("max_new_tokens_mismatch")
    try:
        surface, h, blocks = _surface(pred, profile, prompt_tokens, max_new_tokens)
    except (ValueError, TypeError, OverflowError) as exc:
        reference = evaluate_reference_fallback({k: v for k, v in pred.items() if k != "dws_surface"},
            prompt_tokens=prompt_tokens, include_prefill=include_prefill, profile=profile, max_new_tokens=max_new_tokens)
        return replace(reference,
                       fallback_reason=("depth_curve_unavailable" if reference.cost_model.endswith("_length") else
                                        "surface_missing" if "dws_surface" not in pred else "surface_invalid:"+str(exc)),
                       shape_id=profile.shape_id, scale_id=profile.scale_id)
    kernel, count = profile.kernel(prompt_tokens)
    den = math.fsum(w*mass for row, probs in zip(kernel, surface) for w, mass in zip(row, probs))
    pre = prompt_tokens // profile.block_size * profile.prefill_block_ms if include_prefill else 0.0
    refresh = blocks * profile.refresh_block_ms
    total = pre + refresh + den
    return DllmCostBreakdown(pre, refresh, den, total, total/profile.refresh_block_ms, blocks, h,
                            profile.profile_id, MODEL_TYPE, None, profile.shape_id, profile.scale_id, count)
