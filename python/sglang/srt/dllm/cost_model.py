# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""DWS phase costs and depth-based fallback, independent of model execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class DllmCostBreakdown:
    prefill_ms: float
    refresh_ms: float
    denoising_ms: float
    total_ms: float
    normalized_work: float
    expected_output_blocks: float
    expected_cells_by_depth: List[float]
    profile_id: str
    cost_model: str = "dws-wsl_fallback_depth"
    fallback_reason: Optional[str] = None
    shape_id: str = ""
    scale_id: str = ""
    kernel_clamp_count: int = 0


@dataclass(frozen=True)
class DllmCostProfile:
    block_size: int
    prefill_block_ms: float
    refresh_block_ms: float
    denoising_step_ms: List[float]
    profile_id: str


    def validate(self) -> None:
        if self.block_size <= 0 or len(self.denoising_step_ms) != self.block_size:
            raise ValueError("cost profile denoising vector has the wrong length")
        values = [
            self.prefill_block_ms,
            self.refresh_block_ms,
            *self.denoising_step_ms,
        ]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("cost profile coefficients must be finite and positive")


def _finite_nonnegative(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def evaluate_dws_cost(
    prediction: Mapping[str, Any],
    *,
    prompt_tokens: int,
    block_size: int,
    include_prefill: bool,
    profile: DllmCostProfile,
) -> Optional[DllmCostBreakdown]:
    """Convert a DWS cumulative workload curve into calibrated GPU work.

    ``workload_curve[s - 1]`` is the expected number of cells in depths
    ``1..s``.  Its adjacent difference is therefore the expected number of
    output blocks that execute denoising depth ``s``.
    """

    if profile.block_size != int(block_size):
        return None

    expected_blocks = _finite_nonnegative(prediction.get("expected_n_blocks"))
    curve = prediction.get("workload_curve")
    if expected_blocks is None or not isinstance(curve, (list, tuple)):
        return None
    if len(curve) < block_size:
        return None

    cumulative: List[float] = []
    for raw in curve[:block_size]:
        value = _finite_nonnegative(raw)
        if value is None:
            return None
        cumulative.append(value)

    cells: List[float] = []
    previous = 0.0
    tolerance = 1e-5 * max(1.0, cumulative[-1] if cumulative else 1.0)
    for value in cumulative:
        delta = value - previous
        if delta < -tolerance:
            return None
        cells.append(max(delta, 0.0))
        previous = value

    # A valid survival curve has non-increasing depth mass.  Do not rewrite a
    # predictor's distribution here, but reject a materially malformed curve.
    for left, right in zip(cells, cells[1:]):
        if right > left + tolerance:
            return None

    # In the finalized DWS model P(T>=1 | block exists)=1, hence H_1=E[N].
    # Permit ordinary floating-point drift while rejecting mixed/old contracts.
    if cells and abs(cells[0] - expected_blocks) > max(
        1e-3, 1e-3 * max(expected_blocks, 1.0)
    ):
        return None

    denoising_ms = sum(
        mass * cost for mass, cost in zip(cells, profile.denoising_step_ms)
    )
    refresh_ms = expected_blocks * profile.refresh_block_ms
    prefill_rounds = max(int(prompt_tokens), 0) // int(block_size)
    prefill_ms = (
        prefill_rounds * profile.prefill_block_ms if include_prefill else 0.0
    )
    total_ms = prefill_ms + refresh_ms + denoising_ms
    normalized = total_ms / profile.refresh_block_ms
    return DllmCostBreakdown(
        prefill_ms=prefill_ms,
        refresh_ms=refresh_ms,
        denoising_ms=denoising_ms,
        total_ms=total_ms,
        normalized_work=normalized,
        expected_output_blocks=expected_blocks,
        expected_cells_by_depth=cells,
        profile_id=profile.profile_id,
    )


class CostProbeRecorder:
    """Bounded in-memory CUDA timings for deployment scale probes only."""

    def __init__(self, enabled: bool = False):
        self.probe_enabled = bool(enabled)
        self.probe_rids = set()
        self.probe_records = {}

    def is_probe(self, rid: Any) -> bool:
        return self.probe_enabled and str(rid) in self.probe_rids

    def should_time(self, rid: Any) -> bool:
        return self.is_probe(rid)

    def _append(self, rid: str, row: Dict[str, Any]) -> None:
        records = self.probe_records.setdefault(rid, [])
        if len(records) >= 8192:
            raise ValueError("probe measurement buffer limit exceeded")
        records.append({"rid": rid, "calibration": False, **row})

    def record_prefill(self, rid: Any, elapsed_ms: float) -> None:
        rid, elapsed_ms = str(rid), float(elapsed_ms)
        if self.is_probe(rid) and math.isfinite(elapsed_ms) and elapsed_ms > 0:
            self._append(rid, {"kind": "prefill", "elapsed_ms": elapsed_ms})

    def record_decode_block(self, rid: Any, step_ms: List[float], refresh_ms: float) -> None:
        rid = str(rid)
        if self.is_probe(rid):
            records = self.probe_records.get(rid, [])
            self._append(rid, {
                "kind": "decode_block",
                "block_index": sum(row["kind"] == "decode_block" for row in records),
                "step_ms": [float(value) for value in step_ms],
                "refresh_ms": float(refresh_ms),
            })
