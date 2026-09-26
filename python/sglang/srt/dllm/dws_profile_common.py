# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Shared DWS profile validation, surface contract, and reference-cost fallback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

from sglang.srt.dllm.cost_model import DllmCostBreakdown, DllmCostProfile, evaluate_dws_cost


@dataclass(frozen=True)
class DWSProfileBase:
    block_size: int
    max_output_blocks: int
    denoising_cell_ms: tuple[tuple[float, ...], ...]
    prefill_block_ms: float
    refresh_block_ms: float
    denoising_step_ms: tuple[float, ...]  # positive depth fallback; never W's column mean
    max_new_tokens: int
    profile_id: str
    payload: dict


    def validate(self):
        d, j = self.block_size, self.max_output_blocks
        if not (0 < d <= 1024 and 0 < j <= 65536 and self.max_new_tokens > 0):
            raise ValueError("invalid DWS dimensions/support")
        if len(self.denoising_cell_ms) != j or any(len(row) != d for row in self.denoising_cell_ms):
            raise ValueError("DWS matrix dimensions differ from profile")
        values = [v for row in self.denoising_cell_ms for v in row]
        if not all(math.isfinite(v) and v >= 0 for v in values) or not any(values):
            raise ValueError("W must be finite, nonnegative, and not identically zero")
        if len(self.denoising_step_ms) != d or not all(
            math.isfinite(v) and v > 0
            for v in (self.prefill_block_ms, self.refresh_block_ms, *self.denoising_step_ms)
        ):
            raise ValueError("phase/fallback costs must be finite and positive")
        if not math.isfinite((sum(values) + j * sum(self.denoising_step_ms)) / self.refresh_block_ms):
            raise ValueError("DWS cost range overflows")
        if not self.payload.get("compatibility"):
            raise ValueError("DWS profile requires serving compatibility metadata")

    def validate_runtime(self, signature, block_size):
        if self.block_size != block_size:
            raise ValueError("DWS block_size mismatch")
        if self.payload.get("timing", {}).get("prefill_source") != "cuda_event":
            raise ValueError("DWS deployment requires CUDA prefill; wall profiles are offline-only")
        expected = self.payload["compatibility"]
        mismatch = sorted(k for k in set(signature) | set(expected) if signature.get(k) != expected.get(k))
        if mismatch:
            raise ValueError("DWS runtime mismatch: " + ", ".join(mismatch))

    def to_dict(self):
        return json.loads(json.dumps(self.payload))

    def depth_profile(self):
        return DllmCostProfile(
            self.block_size, self.prefill_block_ms, self.refresh_block_ms,
            list(self.denoising_step_ms), self.profile_id,
        )


def _surface(prediction, profile, prompt_tokens, max_new_tokens):
    if prediction.get("dws_contract_version") != 2:
        raise ValueError("missing_surface_contract")
    if max_new_tokens != profile.max_new_tokens or prediction.get("dws_max_new_tokens") != profile.max_new_tokens:
        raise ValueError("max_new_tokens_mismatch")
    surface = prediction.get("dws_surface")
    d, j = profile.block_size, profile.max_output_blocks
    if not isinstance(surface, (list, tuple)) or len(surface) != j:
        raise ValueError("missing_or_invalid_dws_surface")
    rows = []
    for b, raw in enumerate(surface):
        if not isinstance(raw, (list, tuple)) or len(raw) != d:
            raise ValueError("invalid_surface_shape")
        row = [float(v) for v in raw]
        if any(not math.isfinite(v) or not 0 <= v <= 1 + 2e-4 for v in row):
            raise ValueError("invalid_surface_probability")
        if any(right > left + 2e-4 for left, right in zip(row, row[1:])):
            raise ValueError("invalid_surface_depth_survival")
        if b == 0 and any(v > 2e-4 for v in row[d - prompt_tokens % d:]):
            raise ValueError("invalid_partial_first_block_surface")
        rows.append(row)
    q = [row[0] for row in rows]
    if any(right > left + 2e-4 for left, right in zip(q, q[1:])):
        raise ValueError("invalid_surface_block_survival")
    h = [sum(row[s] for row in rows) for s in range(d)]
    expected = float(prediction.get("expected_n_blocks"))
    curve = prediction.get("workload_curve")
    if not isinstance(curve, (list, tuple)) or len(curve) != d:
        raise ValueError("missing_surface_workload_curve")
    cumulative = 0.0
    for mass, actual in zip(h, curve):
        cumulative += mass
        if not math.isclose(cumulative, float(actual), rel_tol=2e-4, abs_tol=2e-4):
            raise ValueError("inconsistent_surface_workload_curve")
    if not math.isclose(sum(q), expected, rel_tol=2e-4, abs_tol=2e-4):
        raise ValueError("inconsistent_surface_expected_blocks")
    return rows, h, sum(q)


def evaluate_reference_fallback(prediction, *, prompt_tokens, include_prefill, profile, max_new_tokens):
    """Use positive depth costs, or a prompt-length proxy in the same units."""
    prediction = prediction if isinstance(prediction, dict) else {}
    depth = evaluate_dws_cost(
        prediction, prompt_tokens=prompt_tokens, block_size=profile.block_size,
        include_prefill=include_prefill, profile=profile.depth_profile(),
    )
    if depth is not None and depth.denoising_ms > 0 and math.isfinite(depth.normalized_work):
        return replace(depth, cost_model="dws-wsl_fallback_depth")
    cap = max_new_tokens if isinstance(max_new_tokens, int) and max_new_tokens > 0 else profile.max_new_tokens
    length = min(max(prompt_tokens, 1), cap)
    blocks = math.ceil((prompt_tokens % profile.block_size + length) / profile.block_size)
    denoising = length * sum(profile.denoising_step_ms) / profile.block_size
    prefill = prompt_tokens // profile.block_size * profile.prefill_block_ms if include_prefill else 0.0
    refresh = blocks * profile.refresh_block_ms
    total = prefill + refresh + denoising
    return DllmCostBreakdown(prefill, refresh, denoising, total, total / profile.refresh_block_ms,
                            blocks, [], profile.profile_id, "dws-wsl_fallback_length")
