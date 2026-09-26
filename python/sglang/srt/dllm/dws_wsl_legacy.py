# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""DWS-WSL: request WLS with first-difference smoothing and prompt calibration.

C_den = <W, A> + delta * (floor(P / d) - mbar) * sum(A).
W is nonnegative; delta is signed. Runtime scoring has no numpy/torch dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

from sglang.srt.dllm.dws_profile_common import (
    DWSProfileBase,
    _surface,
    evaluate_reference_fallback,
)
from sglang.srt.dllm.cost_model import DllmCostBreakdown

FEATURE_VERSION = "dws_wsl_floor_prompt_v1"
MODEL_TYPE = "dws-wsl"


def profile_digest(payload):
    canonical = {k: v for k, v in payload.items() if k != "profile_id"}
    return "dws-wsl-" + hashlib.sha256(
        json.dumps(canonical, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:24]


@dataclass(frozen=True)
class DWSWSLProfile(DWSProfileBase):
    delta: float
    mbar: float
    prompt_tokens_min: int
    prompt_tokens_max: int

    @classmethod
    def from_mapping(cls, payload):
        data = json.loads(json.dumps(payload, allow_nan=False))
        if (data.get("schema_version"), data.get("model_type"), data.get("feature_version")) != (
            4, MODEL_TYPE, FEATURE_VERSION
        ):
            raise ValueError("expected DWS-WSL profile (schema 4)")
        if data.get("status") != "ready" or data.get("profile_id") != profile_digest(data):
            raise ValueError("DWS-WSL profile is incomplete or its digest changed")
        support, prompt = data["support"], data["prompt_calibration"]
        if prompt.get("block_rounding") != "floor":
            raise ValueError("DWS-WSL requires floor prompt blocks")
        result = cls(
            block_size=int(data["block_size"]), max_output_blocks=int(data["max_output_blocks"]),
            denoising_cell_ms=tuple(tuple(float(v) for v in row) for row in data["denoising_cell_ms"]),
            prefill_block_ms=float(data["prefill_block_ms"]), refresh_block_ms=float(data["refresh_block_ms"]),
            denoising_step_ms=tuple(map(float, data["fallback"]["denoising_step_ms"])),
            max_new_tokens=int(support["max_new_tokens"]), profile_id=data["profile_id"], payload=data,
            delta=float(prompt["delta_ms_per_step_per_prompt_block"]), mbar=float(prompt["mbar"]),
            prompt_tokens_min=int(support["prompt_tokens_min"]),
            prompt_tokens_max=int(support["prompt_tokens_max"]),
        )
        result.validate()
        return result

    def validate(self):
        super().validate()
        if not (math.isfinite(self.delta) and math.isfinite(self.mbar) and self.mbar >= 0):
            raise ValueError("DWS-WSL requires finite delta and nonnegative mbar")
        if not 0 <= self.prompt_tokens_min <= self.prompt_tokens_max:
            raise ValueError("invalid DWS-WSL prompt support")
        low, high = (p // self.block_size for p in (self.prompt_tokens_min, self.prompt_tokens_max))
        if not low <= self.mbar <= high:
            raise ValueError("DWS-WSL mbar must lie within the profiling prompt range")
        if self.payload.get("timing", {}).get("denoising_target") != "forward_plus_postprocess":
            raise ValueError("DWS-WSL requires denoising forward-plus-postprocess timings")
        correction = max(abs(self.delta * (m - self.mbar)) for m in (low, high))
        bound = (sum(sum(row) for row in self.denoising_cell_ms)
                 + correction * self.block_size * self.max_output_blocks
                 + self.prefill_block_ms * high + self.refresh_block_ms * self.max_output_blocks)
        if not math.isfinite(bound) or not math.isfinite(bound / self.refresh_block_ms):
            raise ValueError("DWS-WSL cost range overflows")

    def prompt_correction(self, prompt_tokens):
        if type(prompt_tokens) is not int or not self.prompt_tokens_min <= prompt_tokens <= self.prompt_tokens_max:
            raise ValueError("prompt_outside_profile_support")
        return self.delta * (prompt_tokens // self.block_size - self.mbar)


def evaluate_dws_wsl(prediction, *, prompt_tokens, include_prefill, profile, max_new_tokens):
    prediction = prediction if isinstance(prediction, dict) else {}
    try:
        correction = profile.prompt_correction(prompt_tokens)
        surface, h, blocks = _surface(prediction, profile, prompt_tokens, max_new_tokens)
        # A nonnegative W alone does not guarantee nonnegative K. Check feasible
        # cells even if their predicted mass is zero to keep the cost kernel valid.
        first_depth = profile.block_size - prompt_tokens % profile.block_size
        if any(w + correction < 0
               for b, weights in enumerate(profile.denoising_cell_ms)
               for w in weights[:first_depth if b == 0 else profile.block_size]):
            raise ValueError("negative_kernel_cost")
        denoising = math.fsum(w * a for weights, row in zip(profile.denoising_cell_ms, surface)
                              for w, a in zip(weights, row)) + correction * math.fsum(h)
        if not math.isfinite(denoising) or denoising <= 0:
            raise ValueError("nonpositive_predicted_denoising_cost")
        prefill = prompt_tokens // profile.block_size * profile.prefill_block_ms if include_prefill else 0.0
        refresh = blocks * profile.refresh_block_ms
        total = prefill + refresh + denoising
        return DllmCostBreakdown(prefill, refresh, denoising, total, total / profile.refresh_block_ms,
                                 blocks, h, profile.profile_id, MODEL_TYPE, None)
    except (ValueError, TypeError, OverflowError) as exc:
        # Reuse the positive reference-cost fallback, never the uncorrected W.
        reference_prediction = {k: v for k, v in prediction.items() if k != "dws_surface"}
        reference = evaluate_reference_fallback(
            reference_prediction, prompt_tokens=prompt_tokens, include_prefill=include_prefill,
            profile=profile, max_new_tokens=max_new_tokens,
        )
        return replace(reference, fallback_reason=str(exc))
