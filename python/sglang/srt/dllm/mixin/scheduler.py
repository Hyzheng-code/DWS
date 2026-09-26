# DWS research fork: modified from the imported SGLang 0.5.10 source.
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.cost_probe import CostProbeMixin
from sglang.srt.dllm.profile_common import shape_scale_signature
from sglang.srt.dllm.cost_model import DllmCostBreakdown
from sglang.srt.dllm.profile_common import runtime_signature
from sglang.srt.dllm.dws_wsl import DWSWSLProfile, evaluate_dws_wsl, load_dws_wsl_profile
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


def _my_phase_work(req: Req, include_prefill: bool) -> float:
    """Return the phase-appropriate predicted block work for SJF.

    The outer promotion queue has not paid the prompt cost, while the
    decode-ready queue has. Predictors without these component fields retain
    the historical ``my_sjf_score`` behavior.
    """
    calibrated = _my_calibrated_cost(req, include_prefill=include_prefill)
    if calibrated is not None:
        return calibrated.normalized_work
    pred = getattr(req, "my_pred", None)
    if isinstance(pred, dict):
        cfg = getattr(req, "dllm_config", None)
        decode_work = pred.get("expected_decode_work")
        if decode_work is None:
            expected_blocks = pred.get("expected_n_blocks")
            denoising_work = pred.get("denoising_work")
            # The finalized DWS adapters intentionally expose their probability-
            # surface integral as scheduler_area while the historical SJF baseline
            # sorts by that scalar alone. sjf-dws opts into the complete cost
            # K(x) = prompt_blocks + E[N] + E[sum_b T_b], so reinterpret the same
            # surface as its denoising component and add the expected KV-refresh
            # block count. Predictors without both DWS fields retain fail-soft
            # fallback to my_sjf_score below.
            if denoising_work is None and getattr(cfg, "my_sjf_dws", False):
                denoising_work = pred.get("scheduler_area")
            if expected_blocks is not None and denoising_work is not None:
                try:
                    decode_work = float(expected_blocks) + float(denoising_work)
                except (TypeError, ValueError):
                    decode_work = None

        try:
            decode_work = float(decode_work)
        except (TypeError, ValueError):
            decode_work = None
        if decode_work is not None and math.isfinite(decode_work):
            decode_work = max(decode_work, 0.0)
            if not include_prefill:
                return decode_work

            prompt_work = pred.get("prompt_blocks")
            if prompt_work is None:
                cfg = getattr(req, "dllm_config", None)
                block_size = max(int(getattr(cfg, "block_size", 1) or 1), 1)
                prompt_work = math.ceil(len(req.origin_input_ids) / block_size)
            try:
                prompt_work = float(prompt_work)
            except (TypeError, ValueError):
                prompt_work = None
            if prompt_work is not None and math.isfinite(prompt_work):
                return max(prompt_work, 0.0) + decode_work

    score = getattr(req, "my_sjf_score", None)
    if score is not None:
        return float(score)
    return float(len(req.origin_input_ids))


def _my_calibrated_cost(
    req: Req, *, include_prefill: bool
) -> Optional[DllmCostBreakdown]:
    cfg = getattr(req, "dllm_config", None)
    if getattr(req, "my_cost_probe", False) or getattr(req, "my_cost_health_check", False):
        return None
    if (
        cfg is None
        or not getattr(cfg, "my_sjf_dws", False)
        or getattr(cfg, "my_sjf_cost_model", "unit") != "dws-wsl"
    ):
        return None
    profile = getattr(cfg, "my_cost_profile", None)
    pred = getattr(req, "my_pred", None)
    prediction_identity = pred
    if profile is None:
        raise RuntimeError("DWS-WSL profile must be installed before scheduling")
    if not isinstance(pred, dict):
        pred = {}

    prompt_tokens = len(getattr(req, "origin_input_ids", ()))
    block_size = max(int(getattr(cfg, "block_size", 1) or 1), 1)
    curve = pred.get("workload_curve")
    expected_blocks = pred.get("expected_n_blocks")
    max_new = getattr(getattr(req, "sampling_params", None), "max_new_tokens", None)
    cache = getattr(req, "my_calibrated_cost_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("prediction") is prediction_identity
        and cache.get("curve") is curve
        and cache.get("profile") is profile
        and cache.get("shape_id") == getattr(profile, "shape_id", "")
        and cache.get("scale_id") == getattr(profile, "scale_id", "")
        and cache.get("expected_n_blocks") == expected_blocks
        and cache.get("prompt_tokens") == prompt_tokens
        and cache.get("block_size") == block_size
        and cache.get("contract") == (pred.get("dws_contract_version"), pred.get("dws_max_new_tokens"))
        and cache.get("max_new_tokens") == max_new
        and cache.get("surface") is pred.get("dws_surface")
    ):
        return cache["arrival"] if include_prefill else cache["decode"]

    # Reuse the arrival cost after subtracting already-completed prefill.
    arrival = evaluate_dws_wsl(
        pred, prompt_tokens=prompt_tokens, include_prefill=True,
        profile=profile, max_new_tokens=max_new,
    )
    decode = None
    if arrival is not None:
        decode_total_ms = arrival.refresh_ms + arrival.denoising_ms
        decode = replace(
            arrival,
            prefill_ms=0.0,
            refresh_ms=arrival.refresh_ms,
            denoising_ms=arrival.denoising_ms,
            total_ms=decode_total_ms,
            normalized_work=decode_total_ms / profile.refresh_block_ms,
            expected_output_blocks=arrival.expected_output_blocks,
            expected_cells_by_depth=arrival.expected_cells_by_depth,
            profile_id=arrival.profile_id,
        )
    req.my_calibrated_cost_cache = {
        # Keep strong references so Python cannot reuse an object's id and
        # accidentally turn a replacement into a false cache hit.
        "prediction": prediction_identity,
        "curve": curve,
        "profile": profile,
        "shape_id": getattr(profile, "shape_id", ""),
        "scale_id": getattr(profile, "scale_id", ""),
        "expected_n_blocks": expected_blocks,
        "prompt_tokens": prompt_tokens,
        "block_size": block_size,
        "contract": (pred.get("dws_contract_version"), pred.get("dws_max_new_tokens")),
        "max_new_tokens": max_new,
        "surface": pred.get("dws_surface"),
        # Cache malformed predictions as well; their fallback remains stable
        # until the predictor or active profile changes.
        "arrival": arrival,
        "decode": decode,
    }
    return arrival if include_prefill else decode


def _my_sjf_enabled(cfg) -> bool:
    return bool(cfg is not None and getattr(cfg, "my_sjf", False))


def _my_sjf_promote_aging(cfg) -> float:
    """Promotion inherits the decode discount unless explicitly overridden."""
    rate = getattr(cfg, "my_sjf_promote_aging", None)
    return getattr(cfg, "my_sjf_aging", 0.1) if rate is None else rate


def _my_sjf_wait_periods(req: Req, now: float) -> int:
    """Completed 30-second periods since entering the scheduler waiting queue."""
    entered = getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", 0.0)
    return int(max(now - entered, 0.0) // 30.0) if entered else 0


def _my_sjf_aged_work(req: Req, work: float, rate: float, now: float) -> float:
    """Paper §6.4: discount the original score once per full waiting period.

    TP0 supplies the period count for each scheduling round. Counts stop
    changing on decode admission; neither prediction updates nor prefill
    progress reset the waiting age. Repeated sorting never compounds discounts.
    """
    if rate == 0:
        return work
    periods = getattr(req, "my_sjf_aging_periods", None)
    if periods is None:
        periods = _my_sjf_wait_periods(req, now)
    return work * (1.0 - rate) ** periods


class _PromptPredictWorker:
    """TP0-owned background prompt prediction worker.

    By default, requests remain in the scheduler's outer queue while their early
    score is pending.  The experimental non-blocking mode instead installs a
    placeholder score and permits immediate admission. Completed predictions
    replace these placeholders at the main-thread synchronization point.

    Only TP rank 0 owns the predictor clients and background thread.  The thread
    never mutates scheduler-visible request fields: it queues completed results,
    which the scheduler broadcasts and applies on every TP rank at the same
    main-thread synchronization point. Failure is fail-soft: all ranks install
    the same deterministic prompt-length proxy together.
    """

    def __init__(self, cfg: DllmConfig, *, owner: bool = True):
        ac = cfg.algorithm_config or {}
        self.owner = bool(owner)
        self.batch_size = max(int(ac.get("prompt_predict_batch", 4)), 1)
        self.timeout = max(float(ac.get("prompt_predict_timeout", 120.0)), 1.0)
        self.block_admission = bool(
            ac.get("prompt_predict_block_admission", True)
        )
        self.initial_score = max(
            float(ac.get("prompt_predict_initial_score", 1024.0)), 0.0
        )
        self.rid_prefix = str(ac.get("predict_namespace", "") or "")
        self._q: queue.Queue = queue.Queue()
        self._updates: queue.Queue = queue.Queue()
        self._reqs: Dict[str, Req] = {}
        self._log_path = str(ac.get("predict_log", "") or "")
        self._log_fh = None
        self.client = None

        # Followers deliberately do not import/create a predictor client. Their
        # state is initialized optimistically and replaced by TP0's state in the
        # first main-thread broadcast before admission.
        self.enabled = True
        self._rpc_operational = False
        if not self.owner:
            return

        from sglang.srt.dllm.my_code.predictor_client import get_client

        host = str(ac.get("predict_host", "127.0.0.1"))
        port = int(ac.get("predict_port", 31100))
        self.client = get_client(
            host,
            port,
            channel="prompt",
            timeout=self.timeout,
            rid_prefix=self.rid_prefix,
        )
        info = self.client.ping() or {}
        self._rpc_operational = bool(info.get("prompt_predict", False))
        self.enabled = self._rpc_operational
        if self.enabled:
            logger.warning(
                "[my_prompt_predict] TP0 prompt scorer ready at %s:%d "
                "device=%s batch=%d block_admission=%s initial_score=%.1f",
                host,
                port,
                info.get("prompt_device"),
                self.batch_size,
                self.block_admission,
                self.initial_score,
            )
            self._thread = threading.Thread(
                target=self._run,
                name="dllm-prompt-predict",
                daemon=True,
            )
            self._thread.start()
        else:
            logger.error(
                "[my_prompt_predict] TP0 service at %s:%d has no prompt model; "
                "falling back to the synchronized prompt-length proxy",
                host,
                port,
            )

    def evict(self, rids: List[str]) -> None:
        for rid in rids:
            self._reqs.pop(str(rid), None)

    def submit(self, reqs: List[Req]) -> None:
        for req in reqs:
            if (
                getattr(req, "my_scored", False)
                or getattr(req, "my_prompt_predict_submitted", False)
            ):
                continue
            rid = str(req.rid)
            self._reqs[rid] = req
            req.my_prompt_predict_submitted = True
            req.my_prompt_predict_pending = True
            req.my_prompt_predict_submit_time = time.perf_counter()
            if not self.block_admission:
                # A conservative placeholder permits prefill to overlap CPU
                # prediction; the completed result replaces it in _store().
                score = self.initial_score
                req.my_pred = {
                    "total_steps": score,
                    "source": "prompt_initial_score",
                }
                req.my_sjf_score = score
                req.my_scored = True
            if self.owner:
                if self._rpc_operational:
                    self._q.put(req)
                else:
                    self._updates.put({"rid": rid, "ok": False})

    def make_sync_payload(self) -> dict:
        """Drain TP0 results for one scheduler-thread broadcast."""
        updates = []
        while True:
            try:
                updates.append(self._updates.get_nowait())
            except queue.Empty:
                break
        return {
            "enabled": bool(self._rpc_operational),
            "updates": updates,
        }

    def apply_sync_payload(self, payload: dict) -> None:
        """Apply one TP0 payload to this rank's local request objects."""
        self.enabled = bool(payload.get("enabled", False))
        for update in payload.get("updates", ()):
            req = self._reqs.get(str(update.get("rid", "")))
            if req is None:
                # A request may finish while its asynchronous prediction is in
                # flight; eviction removes it from this local lookup table.
                continue
            if update.get("ok", False):
                entry = update.get("entry")
                if not isinstance(entry, dict):
                    continue
                self._store(req, entry)
                if self.owner:
                    self._log(
                        req,
                        entry,
                        float(update.get("submitted", time.perf_counter())),
                        int(update.get("rpc_batch_size", 1)),
                    )
            else:
                # Every TP rank installs the same deterministic fallback.
                score = float(len(req.origin_input_ids))
                self._store(
                    req,
                    {
                        "total_steps": score,
                        "source": "prompt_predict_failed_length_proxy",
                        "predict_error": "prompt_predict_failed",
                    },
                )

    @staticmethod
    def _store(req: Req, entry: dict) -> None:
        req.my_pred = entry
        req.my_sjf_score = float(entry.get("total_steps", 0.0))
        req.my_scored = True
        req.my_prompt_scored = True
        req.my_prompt_predict_pending = False

    def _log(
        self, req: Req, entry: dict, submitted: float, rpc_batch_size: int
    ) -> None:
        if not self._log_path:
            return
        import json
        import os

        if self._log_fh is None:
            try:
                os.makedirs(
                    os.path.dirname(os.path.abspath(self._log_path)) or ".",
                    exist_ok=True,
                )
                self._log_fh = open(self._log_path, "a")
            except Exception:  # noqa: BLE001
                self._log_fh = False
        if self._log_fh is False:
            return
        row = {
            "ts": time.time(),
            "rid": req.rid,
            "phase": "arrival",
            "prompt_tokens": len(req.origin_input_ids),
            "async_wait_ms": max((time.perf_counter() - submitted) * 1000.0, 0.0),
            "queue_wait_ms": max(
                (
                    submitted
                    - getattr(req, "my_prompt_predict_submit_time", submitted)
                )
                * 1000.0,
                0.0,
            ),
            "prompt_rpc_batch_size": int(rpc_batch_size),
            "block_admission": self.block_admission,
            "initial_score": self.initial_score if not self.block_admission else None,
        }
        row.update(entry)
        try:
            self._log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._log_fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        while True:
            first = self._q.get()
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            submitted = time.perf_counter()
            items = [
                {
                    "rid": str(req.rid),
                    "input_ids": list(req.origin_input_ids),
                    "max_new_tokens": getattr(getattr(req, "sampling_params", None), "max_new_tokens", None),
                    "dws_return_surface": getattr(getattr(req, "dllm_config", None), "my_sjf_cost_model", "unit") == "dws-wsl",
                    "dws_cost_model": getattr(getattr(req, "dllm_config", None), "my_sjf_cost_model", "unit"),
                    # The text predictor uses its own tokenizer. input_ids
                    # supplies the served model's prompt length for block bounds.
                    "text": getattr(req, "origin_input_text", ""),
                }
                for req in batch
            ]
            try:
                results = self.client.prompt(items)
                if results is None:
                    self._rpc_operational = False
                    results = {}
                for req in batch:
                    entry = results.get(str(req.rid))
                    if entry is None:
                        self._updates.put({"rid": str(req.rid), "ok": False})
                        continue
                    self._updates.put(
                        {
                            "rid": str(req.rid),
                            "ok": True,
                            "entry": entry,
                            "submitted": submitted,
                            "rpc_batch_size": len(batch),
                        }
                    )
            except Exception:  # noqa: BLE001
                self._rpc_operational = False
                logger.exception(
                    "[my_prompt_predict] prompt scoring failed; using synchronized "
                    "prompt-length fallback"
                )
                for req in batch:
                    self._updates.put({"rid": str(req.rid), "ok": False})
            finally:
                for _ in batch:
                    self._q.task_done()


class SchedulerDllmMixin(CostProbeMixin):
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)
        # Scheduler-local state for adaptive prefill/decode arbitration.
        self._dllm_prefill_cohort: Set[str] = set()
        self._dllm_prefill_burst_started = 0.0
        self._dllm_decode_rounds_since_prefill = 0
        self._dllm_high_load = False
        self._dllm_decode_round_ewma = 1.0
        self._dllm_prefill_round_ewma = 0.02
        self._dllm_forward_started = 0.0
        self._dllm_forward_mode = ""
        self._my_cost_profile_synced = False
        self._my_cost_runtime_signature = None
        self._my_cost_shape_signature = None
        if self.dllm_config is not None and self.dllm_config.my_sjf_cost_model == "dws-wsl":
            self._my_cost_runtime_signature = runtime_signature(
                self.server_args, self.dllm_config.algorithm_config
            )
            self._sync_dws_wsl_profile()
        self._my_prompt_predict_worker = None
        if self.dllm_config is not None and bool(
            (self.dllm_config.algorithm_config or {}).get("prompt_predict", False)
        ):
            self._my_prompt_predict_worker = _PromptPredictWorker(
                self.dllm_config,
                owner=int(getattr(self.tp_group, "rank_in_group", 0) or 0) == 0,
            )

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_priority_preemption:
            self.running_batch.batch_is_full = False

        if (getattr(self, "_my_cost_probe_refresh_disabled", False)
                and getattr(self, "_my_cost_probe_window", None)
                and self._my_cost_probe_window["idle"]):
            self._cost_probe_cancel()
        # Early exit if batch is full or no requests available
        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        # The dLLM request uses the same DLLM_EXTEND forward mode for its
        # prompt prefill and subsequent denoising rounds.  Stamp the first
        # round in which the request is no longer a prefill request as the
        # actual decode start.  This is intentionally separate from
        # ``my_first_token_time`` (first committed/demasked token).
        if getattr(self.dllm_config, "my_metrics_log", ""):
            decode_start = time.perf_counter()
            for req in can_run_list:
                ts = getattr(req, "time_stats", None)
                if (
                    ts is not None
                    and not req.is_dllm_prefill()
                    and getattr(ts, "my_decode_start_time", 0.0) == 0.0
                ):
                    ts.my_decode_start_time = decode_start
        self._update_state_for_batch(can_run_list, adder, running_bs)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        self._record_dllm_forward_start(can_run_list)
        self._maybe_log_sjf(can_run_list)
        self._maybe_log_occupancy(can_run_list)
        return new_batch

    def _maybe_log_occupancy(self: Scheduler, can_run_list: List[Req]) -> None:
        """Append one JSONL line per forwarded round recording batch occupancy.

        Diagnostic for batch utilization: works for FCFS and SJF (gated
        only by --my-occupancy-log, independent of my_sjf). Records whether the
        round is prefill/decode/mixed, how many requests were forwarded (vs the
        per-round block budget B), the held active-set / staging / cold-backlog
        sizes, and KV-pool usage — so a drop in decode occupancy (slots left
        idle) or a high prefill-round fraction can be attributed directly.
        """
        cfg = self.dllm_config
        if cfg is None or not getattr(cfg, "my_occupancy_log", ""):
            return
        import json
        import time

        fh = getattr(self, "_my_occ_fh", None)
        if fh is None:
            try:
                import os

                os.makedirs(
                    os.path.dirname(os.path.abspath(cfg.my_occupancy_log)) or ".",
                    exist_ok=True,
                )
                fh = open(cfg.my_occupancy_log, "a")
                self._my_occ_fh = fh
                self._my_occ_round = 0
            except Exception:  # noqa: BLE001
                self._my_occ_fh = False
                return
        if fh is False:
            return

        n_prefill = sum(1 for r in can_run_list if r.is_dllm_prefill())
        n_fwd = len(can_run_list)
        n_decode = n_fwd - n_prefill
        mode = (
            "mixed"
            if n_prefill and n_decode
            else ("prefill" if n_prefill else "decode")
        )
        budget = max(getattr(cfg, "max_running_requests", 1), 1)

        kv_used = None
        try:
            alloc = self.token_to_kv_pool_allocator
            total = float(getattr(self, "max_total_num_tokens", 0) or 0)
            avail = float(alloc.available_size())
            if total > 0:
                kv_used = max(0.0, min(1.0, 1.0 - avail / total))
        except Exception:  # noqa: BLE001
            kv_used = None

        self._my_occ_round += 1
        try:
            fh.write(
                json.dumps(
                    {
                        "t": time.perf_counter(),
                        "round": self._my_occ_round,
                        "mode": mode,
                        "n_fwd": n_fwd,
                        "cost_probe": any(getattr(r, "my_cost_probe", False) for r in can_run_list),
                        "n_prefill": n_prefill,
                        "n_decode": n_decode,
                        "budget": budget,
                        "decode_occ": (n_decode / budget) if mode != "prefill" else None,
                        "active_set": len(self.dllm_manager.waiting_queue),
                        "staging": len(self.dllm_manager.staging_queue),
                        "backlog": len(self.waiting_queue),
                        "kv_used": kv_used,
                        "prefill_policy": "adaptive",
                        "prefill_cohort": len(
                            getattr(self, "_dllm_prefill_cohort", ())
                        ),
                        "adaptive_high_load": bool(
                            getattr(self, "_dllm_high_load", False)
                        ),
                        "dynamic_prompt_gate": bool(
                            getattr(self, "_dllm_dynamic_prompt_gate", False)
                        ),
                        "decode_round_ewma_s": getattr(
                            self, "_dllm_decode_round_ewma", None
                        ),
                        "prefill_round_ewma_s": getattr(
                            self, "_dllm_prefill_round_ewma", None
                        ),
                    }
                )
                + "\n"
            )
            if self._my_occ_round % 50 == 0:
                fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def _record_dllm_forward_start(
        self: Scheduler, can_run_list: List[Req]
    ) -> None:
        """Start cheap wall-clock timing for adaptive round-cost EWMAs."""
        if not _my_sjf_enabled(self.dllm_config) or not can_run_list:
            return
        n_prefill = sum(1 for req in can_run_list if req.is_dllm_prefill())
        mode = "prefill" if n_prefill == len(can_run_list) else "decode"
        self._dllm_forward_started = time.perf_counter()
        self._dllm_forward_mode = mode
        if mode == "prefill":
            self._dllm_decode_rounds_since_prefill = 0
        else:
            self._dllm_decode_rounds_since_prefill += 1

    def _finish_dllm_forward_timing(self: Scheduler) -> None:
        """Update the corresponding round EWMA after its result is available."""
        if not _my_sjf_enabled(self.dllm_config):
            return
        started = getattr(self, "_dllm_forward_started", 0.0)
        mode = getattr(self, "_dllm_forward_mode", "")
        self._dllm_forward_started = 0.0
        self._dllm_forward_mode = ""
        if not started or mode not in {"prefill", "decode"}:
            return
        elapsed = time.perf_counter() - started
        # Ignore pathological process stalls; they are not GPU service cost and
        # would make the prefill pause bound unusably large for many later rounds.
        if elapsed <= 0.0 or elapsed > 30.0:
            return
        attr = (
            "_dllm_prefill_round_ewma"
            if mode == "prefill"
            else "_dllm_decode_round_ewma"
        )
        previous = float(getattr(self, attr, elapsed))
        setattr(self, attr, 0.8 * previous + 0.2 * elapsed)

    def _maybe_log_sjf(self: Scheduler, can_run_list: List[Req]) -> None:
        """Append one JSONL line per admitted decode request (rid / predicted
        score / sticky flag / queue wait). Gated by --my-sjf-log; default off."""
        cfg = self.dllm_config
        if not _my_sjf_enabled(cfg) or not cfg.my_sjf_log:
            return
        import json
        import time

        fh = getattr(self, "_my_sjf_fh", None)
        if fh is None:
            try:
                import os

                os.makedirs(os.path.dirname(os.path.abspath(cfg.my_sjf_log)) or ".",
                            exist_ok=True)
                fh = open(cfg.my_sjf_log, "a")
                self._my_sjf_fh = fh
            except Exception:  # noqa: BLE001
                self._my_sjf_fh = False
                return
        if fh is False:
            return
        now = time.perf_counter()
        for req in can_run_list:
            if req.is_dllm_prefill() or getattr(req, "my_cost_probe", False):
                continue
            ts = getattr(req, "time_stats", None)
            wt = getattr(ts, "wait_queue_entry_time", 0.0) or 0.0
            arrival_cost = _my_calibrated_cost(req, include_prefill=True)
            decode_cost = _my_calibrated_cost(req, include_prefill=False)
            try:
                fh.write(json.dumps({
                    "ts": time.time(),
                    "rid": req.rid,
                    "phase": str(getattr(req, "dllm_phase", "")),
                    "score": getattr(req, "my_sjf_score", None),
                    "score_mode": (
                        decode_cost.cost_model
                        if decode_cost is not None
                        else "dws_k"
                        if getattr(cfg, "my_sjf_dws", False)
                        else "legacy"
                    ),
                    "arrival_work": _my_phase_work(req, include_prefill=True),
                    "decode_work": _my_phase_work(req, include_prefill=False),
                    "score_source": (
                        (getattr(req, "my_pred", None) or {}).get("source")
                        if isinstance(getattr(req, "my_pred", None), dict)
                        else None
                    )
                    or (
                        "prompt_predictor"
                        if getattr(req, "my_prompt_scored", False)
                        else "prompt_length_proxy"
                    ),
                    "scored": getattr(req, "my_scored", False),
                    "decoding": getattr(req, "my_decoding", False),
                    "wait_s": max(now - wt, 0.0) if wt else None,
                    "prompt_tokens": len(req.origin_input_ids),
                    "out_tokens": len(req.output_ids),
                    "cost_profile_id": (
                        decode_cost.profile_id if decode_cost is not None else None
                    ),
                    "cost_model_requested": getattr(cfg, "my_sjf_cost_model_requested", getattr(cfg, "my_sjf_cost_model", "unit")),
                    "cost_model_used": decode_cost.cost_model if decode_cost else "unit",
                    "shape_id": decode_cost.shape_id if decode_cost else "",
                    "scale_id": decode_cost.scale_id if decode_cost else "",
                    "kernel_clamp_count": decode_cost.kernel_clamp_count if decode_cost else 0,
                    "cost_fallback_reason": decode_cost.fallback_reason if decode_cost else None,
                    "dws_support_projection": (
                        req.my_pred.get("dws_support_projection")
                        if isinstance(getattr(req, "my_pred", None), dict) else None
                    ),
                    "predicted_prefill_ms": (
                        arrival_cost.prefill_ms if arrival_cost is not None else None
                    ),
                    "predicted_refresh_ms": (
                        decode_cost.refresh_ms if decode_cost is not None else None
                    ),
                    "predicted_denoising_ms": (
                        decode_cost.denoising_ms if decode_cost is not None else None
                    ),
                    "predicted_decode_ms": (
                        decode_cost.total_ms if decode_cost is not None else None
                    ),
                    "predicted_total_ms": (
                        arrival_cost.total_ms if arrival_cost is not None else None
                    ),
                    "predicted_output_blocks": (
                        decode_cost.expected_output_blocks
                        if decode_cost is not None
                        else None
                    ),
                }, ensure_ascii=False) + "\n")
            except Exception:  # noqa: BLE001
                pass
        try:
            fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()
        self._cost_probe_forward_end(batch.reqs)

        # ``copy_done`` is the first point at which the scheduled forward is
        # known complete in both overlap and non-overlap modes. Its elapsed time
        # feeds the adaptive prefill pause bound; no CUDA timing is added to the
        # hot path.
        self._finish_dllm_forward_timing()

        if result.next_token_ids:
            self.token_to_kv_pool_allocator.free_group_begin()
            finished_rids = []

            now = time.perf_counter()

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                # dLLM has a custom result-processing loop, so mirror the
                # standard scheduler's customized-info collection explicitly.
                # The dedicated SGLang prompt Engine uses this channel to return
                # the training-identical pooled prompt embedding in memory.
                if (
                    result.logits_output is not None
                    and result.logits_output.customized_info is not None
                ):
                    self.maybe_collect_customized_info(
                        idx, req, result.logits_output
                    )

                next_token_ids = result.next_token_ids[idx].tolist()
                new_tokens = len(next_token_ids)
                if new_tokens == 0:
                    continue

                req.fill_ids[-new_tokens:] = next_token_ids[:]
                self.num_generated_tokens += new_tokens

                req.output_ids.extend(next_token_ids)
                # Server-side TTFT (dLLM): stamp the first-demask moment (first
                # committed token) on the scheduler time_stats, which is shipped to
                # the tokenizer with the output. The metrics log then reports a true
                # time-to-first-token (arrival -> first block demasked) instead of
                # the non-streaming completion time.
                ts = getattr(req, "time_stats", None)
                if ts is not None and getattr(ts, "my_first_token_time", 0.0) == 0.0:
                    ts.my_first_token_time = now
                req.check_finished(new_accepted_len=new_tokens)

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()
                    finished_rids.append(str(req.rid))

            self.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()
            prompt_worker = getattr(self, "_my_prompt_predict_worker", None)
            if finished_rids and prompt_worker is not None:
                prompt_worker.evict(finished_rids)

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _sync_sjf_aging(self: Scheduler) -> None:
        """Use the same discrete aging periods on every tensor-parallel rank.

        Computing floor(wait/30) separately can reverse the order on just one
        rank at a 30-second boundary. Freeze one TP0 snapshot per round before
        promotion, prefill ordering and decode admission use the scores.
        """
        cfg = self.dllm_config
        if not _my_sjf_enabled(cfg) or not (
            cfg.my_sjf_aging > 0 or _my_sjf_promote_aging(cfg) > 0
        ):
            return
        reqs = [
            req for req in self.waiting_queue + self.dllm_manager.waiting_queue
            if not getattr(req, "my_decoding", False)
        ]
        group = getattr(self, "tp_group", None)
        world = int(getattr(group, "world_size", 1) or 1)
        owner = int(getattr(group, "rank_in_group", 0) or 0) == 0
        periods = None
        if owner:
            now = time.perf_counter()
            periods = {str(req.rid): _my_sjf_wait_periods(req, now) for req in reqs}
        if world > 1:
            objects = [periods]
            torch.distributed.broadcast_object_list(
                objects, src=group.first_rank, group=self.tp_cpu_group
            )
            periods = objects[0]
        for req in reqs:
            req.my_sjf_aging_periods = periods[str(req.rid)]

    def _fetch_waiting_reqs(self: Scheduler):
        self._sync_cost_profile()
        if getattr(self, "_my_cost_probe_window", None):
            if not self.dllm_manager.waiting_queue:
                for i, req in enumerate(self.waiting_queue):
                    if getattr(req, "my_cost_probe", False):
                        self.dllm_manager.add_waiting_reqs([self.waiting_queue.pop(i)])
                        break
            return
        if getattr(self, "_my_cost_probe_state", "") in {"pending", "failed"}:
            return
        prompt_worker = getattr(self, "_my_prompt_predict_worker", None)
        if prompt_worker is not None and self.waiting_queue:
            # Submission is asynchronous. Blocking mode holds pending requests in
            # this outer queue; non-blocking mode installs a placeholder score and
            # allows them to move into the dLLM manager immediately.
            prompt_worker.submit(self.waiting_queue)

        # TP0 is the only predictor RPC owner. Its completed results and health
        # state become visible to scheduling only through this main-thread CPU
        # group broadcast, so every TP rank applies identical scores before it
        # constructs the next model batch.
        if prompt_worker is not None:
            tp_group = self.tp_group
            if int(getattr(tp_group, "world_size", 1) or 1) > 1:
                obj_list = [
                    prompt_worker.make_sync_payload()
                    if int(getattr(tp_group, "rank_in_group", 0) or 0) == 0
                    else None
                ]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=tp_group.first_rank,
                    group=self.tp_cpu_group,
                )
                prompt_payload = obj_list[0]
            else:
                prompt_payload = prompt_worker.make_sync_payload()
            prompt_worker.apply_sync_payload(prompt_payload)

        self._sync_sjf_aging()

        early_prompt_enabled = bool(
            prompt_worker is not None and getattr(prompt_worker, "enabled", False)
        )

        block_prompt_admission = bool(
            prompt_worker is not None
            and getattr(prompt_worker, "block_admission", True)
        )
        # Adaptive two-stage admission: low load keeps the configured non-blocking
        # placeholder path (eliminates predictor TTFT). Once the scheduler's 3B/4
        # high-load latch is active, pending placeholder scores stay in the OUTER
        # queue until the real CPU prediction lands. Promotion can then rank the
        # complete outer backlog by a real score instead of filling the bounded 2B
        # GPU pool approximately FIFO with identical initial_score=1024 entries.
        # This restores global SJF visibility without paying prediction latency in
        # the low-load regime where ordering has no benefit.
        dynamic_prompt_gate = bool(
            prompt_worker is not None
            and _my_sjf_enabled(self.dllm_config)
            and getattr(self, "_dllm_high_load", False)
        )
        block_prompt_admission = block_prompt_admission or dynamic_prompt_gate
        self._dllm_dynamic_prompt_gate = dynamic_prompt_gate

        # Calculate how many requests can be added to DLLM manager.
        # Under SJF we prefill a larger "scored pool" (my_sjf_pool) ahead of the
        # decode set so the shortest-job-first decode admission has candidates.
        if not _my_sjf_enabled(self.dllm_config):
            cap = self.dllm_config.max_running_requests
        elif _my_sjf_enabled(self.dllm_config) and not getattr(
            self, "_dllm_high_load", False
        ):
            # The dedicated CPU prompt predictor supplies the global arrival
            # score before main-GPU prefill, so a very deep GPU-resident scored
            # pool buys no additional SJF visibility. Bound it to one active
            # batch plus one lookahead batch to limit KV residency and make
            # demand feedback responsive. Sustained high load deliberately falls
            # through to the configured deep global SJF pool below.
            cap = 2 * self.dllm_config.max_running_requests
        else:
            cap = self.dllm_config.my_sjf_pool
        max_dllm_capacity = cap - len(self.dllm_manager.waiting_queue)
        if (
            prompt_worker is not None
            and block_prompt_admission
            and not _my_sjf_enabled(self.dllm_config)
        ):
            # FCFS+init must remain strict FCFS: do not let a later prompt whose
            # prompt score returned first pass an earlier pending request.
            eligible = []
            for i, req in enumerate(self.waiting_queue):
                if getattr(req, "my_prompt_predict_pending", False):
                    break
                eligible.append(i)
        else:
            # SJF intentionally permits ready, predicted-short requests to
            # bypass slower pending prompt forwards.
            eligible = [
                i
                for i, req in enumerate(self.waiting_queue)
                if (
                    not block_prompt_admission
                    or not getattr(req, "my_prompt_predict_pending", False)
                )
            ]
        num_requests_to_add = min(max_dllm_capacity, len(eligible))

        if num_requests_to_add > 0:
            use_order = _my_sjf_enabled(self.dllm_config) and (
                early_prompt_enabled
                or getattr(self.dllm_config, "my_sjf_prefill_order", False)
            )
            if use_order:
                # Rank the outer backlog by predicted work; aging also lets
                # long-waiting requests enter the bounded active pool.
                promote_aging = _my_sjf_promote_aging(self.dllm_config)
                now = time.perf_counter()

                def _promotion_work(req: Req) -> float:
                    if early_prompt_enabled and getattr(req, "my_scored", False):
                        return _my_phase_work(req, include_prefill=True)
                    return float(len(req.origin_input_ids))

                def _promotion_key(i: int) -> float:
                    req = self.waiting_queue[i]
                    return _my_sjf_aged_work(
                        req, _promotion_work(req), promote_aging, now
                    )

                order = sorted(eligible, key=_promotion_key)
                picked = set(order[:num_requests_to_add])
                requests_to_add = [self.waiting_queue[i] for i in order[:num_requests_to_add]]
                self.waiting_queue = [
                    req for i, req in enumerate(self.waiting_queue) if i not in picked
                ]
            else:
                picked = set(eligible[:num_requests_to_add])
                requests_to_add = [
                    self.waiting_queue[i] for i in eligible[:num_requests_to_add]
                ]
                self.waiting_queue = [
                    req for i, req in enumerate(self.waiting_queue) if i not in picked
                ]
            self.dllm_manager.add_waiting_reqs(requests_to_add)


    def _sync_dws_wsl_profile(self: Scheduler) -> None:
        """Load a request-level cost profile, including DWS-WSL, before admission."""
        cfg, group = self.dllm_config, self.tp_group
        world = int(getattr(group, "world_size", 1) or 1)
        owner = int(getattr(group, "rank_in_group", 0) or 0) == 0
        payload = None
        if owner:
            try:
                payload = {"profile": load_dws_wsl_profile(cfg.my_sjf_cost_profile).to_dict()}
            except Exception as exc:
                payload = {"error": str(exc)}
        if world > 1:
            objects = [payload]
            torch.distributed.broadcast_object_list(objects, src=group.first_rank, group=self.tp_cpu_group)
            payload = objects[0]
        error = payload.get("error")
        if not error:
            try:
                profile = DWSWSLProfile.from_mapping(payload["profile"])
                if isinstance(profile, DWSWSLProfile) and (profile.is_v5 or profile.payload.get("managed_profile")):
                    self._my_cost_shape_signature = shape_scale_signature(self.server_args, cfg.algorithm_config,
                        cfg.block_size, profile.max_output_blocks, profile.max_new_tokens)
                    profile.validate_runtime(self._my_cost_shape_signature, cfg.block_size)
                    if profile.is_v5 and (getattr(self.server_args, "dp_size", 1) != 1 or getattr(self.server_args, "tokenizer_worker_num", 1) != 1):
                        raise ValueError("DWS-WSL v2 probe coordinator requires dp_size=1 and tokenizer_worker_num=1 (TP is supported)")
                else:
                    profile.validate_runtime(self._my_cost_runtime_signature, cfg.block_size)
            except Exception as exc:
                error = str(exc)
        errors = [error]
        if world > 1:
            errors = [None] * world
            torch.distributed.all_gather_object(errors, error, group=self.tp_cpu_group)
        if any(errors):
            raise ValueError(f"Cannot activate {cfg.my_sjf_cost_model} profile: {errors}")
        cfg.my_cost_profile = profile
        if isinstance(profile, DWSWSLProfile) and profile.is_v5:
            self._cost_probe_init()
            # Matching/off/cached profiles also work through the in-process
            # Engine API. A new measurement is driven by HTTP startup below.
            from sglang.srt.dllm.cost_fit_cache import choose_profile
            selected = choose_profile(profile, self._my_cost_shape_signature["cost_keys"], cfg.my_cost_probe) if owner else None
            objects = [selected.to_dict() if selected else None]
            if world > 1:
                torch.distributed.broadcast_object_list(objects, src=group.first_rank, group=self.tp_cpu_group)
            if objects[0] is not None:
                profile = DWSWSLProfile.from_mapping(objects[0])
                cfg.my_cost_profile = profile
                if not getattr(self.server_args, "_dllm_cost_probe_http_startup", False):
                    self._my_cost_probe_state = profile.payload["scale"]["probe_state"]
            elif not getattr(self.server_args, "_dllm_cost_probe_http_startup", False):
                raise ValueError("This DWS-WSL profile requires full cost fitting: start an HTTP server to probe, then reuse its snapshot with Engine (or explicitly set my_cost_probe=off for a legacy profile)")
        self._my_cost_profile_synced = True
        logger.warning("[dllm-cost] activated %s profile %s", cfg.my_sjf_cost_model, profile.profile_id)

    def _sync_cost_profile(self: Scheduler) -> None:
        """Activate the same validated external DWS-WSL profile on every TP rank."""
        cfg = self.dllm_config
        if (cfg is not None and cfg.my_sjf_cost_model == "dws-wsl"
                and not getattr(self, "_my_cost_profile_synced", False)):
            self._sync_dws_wsl_profile()

    def _should_skip_prefill(self: Scheduler) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        cfg = self.dllm_config
        if getattr(self, "_my_cost_probe_window", None):
            prefill = self.dllm_manager.get_prefill_requests()
            reqs = prefill or self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(adder, reqs[:1],
                DllmReqPhase.STAGING_PREFILL if prefill else DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_PREFILL if prefill else DllmReqPhase.INCOMING_DECODE,
                preserve_order=True)
            return forward_mode
        if _my_sjf_enabled(cfg):
            # Keep each round homogeneous. Adaptive arbitration uses a closed
            # prefill cohort at low load and quota cadence at high load.
            prefill_reqs = self.dllm_manager.get_prefill_requests()
            decode_reqs = self.dllm_manager.get_decode_requests()
            should_prefill = self._dllm_should_prefill(
                cfg, prefill_reqs, decode_reqs
            )
            if should_prefill:
                if _my_sjf_enabled(cfg) and not getattr(
                    self, "_dllm_high_load", False
                ):
                    prefill_reqs = self._dllm_adaptive_prefill_batch(prefill_reqs)
                self._process_batch_by_phase(
                    adder,
                    prefill_reqs,
                    DllmReqPhase.STAGING_PREFILL,
                    DllmReqPhase.INCOMING_PREFILL,
                    preserve_order=True,
                )
            else:
                self._process_batch_by_phase(
                    adder,
                    decode_reqs,
                    DllmReqPhase.STAGING_DECODE,
                    DllmReqPhase.INCOMING_DECODE,
                    preserve_order=True,
                )
            return forward_mode

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        return forward_mode

    def _dllm_should_prefill(self: Scheduler, cfg, prefill_reqs: List[Req], decode_reqs: List[Req]) -> bool:
        """Bound continuous prefill pauses while keeping enough decode candidates."""
        if not prefill_reqs:
            self._dllm_prefill_cohort = set()
            self._dllm_prefill_burst_started = 0.0
            budget = max(int(getattr(cfg, "max_running_requests", 1)), 1)
            visible = len(decode_reqs) + len(getattr(self, "waiting_queue", ()))
            if visible < max(budget // 2, 1):
                self._dllm_high_load = False
            return False
        return self._dllm_should_prefill_adaptive(cfg, prefill_reqs, decode_reqs)

    @staticmethod
    def _dllm_req_key(req: Req) -> str:
        return str(req.rid)

    def _dllm_start_prefill_cohort(
        self: Scheduler, cfg, prefill_reqs: List[Req], decode_reqs: List[Req]
    ) -> None:
        """Freeze the next prompt cohort until every selected prompt is ready.

        A closed cohort is the key difference from the legacy quota: arrivals
        cannot extend a burst indefinitely, while a long prompt's consecutive
        blocks are no longer separated by second-long decode rounds.
        """
        budget = max(int(getattr(cfg, "max_running_requests", 1)), 1)
        running = sum(
            1 for req in decode_reqs if getattr(req, "my_decoding", False)
        )
        ready = max(len(decode_reqs) - running, 0)
        ready_low = int(getattr(cfg, "my_sjf_ready_low", 0) or max(budget // 4, 1))
        ready_high = int(
            getattr(cfg, "my_sjf_ready_high", 0)
            or max(budget // 2, ready_low)
        )
        ready_high = max(ready_high, ready_low)

        # Fill an actual decode shortage first; otherwise replenish the ready
        # lookahead to its high watermark. One cohort never exceeds B requests,
        # so even all-max-length prompts have a bounded amount of work.
        cohort_size = max(
            budget - len(decode_reqs),
            ready_high - ready,
            1,
        )
        cohort_size = min(cohort_size, budget, len(prefill_reqs))
        selected = prefill_reqs[:cohort_size]
        self._dllm_prefill_cohort = {
            self._dllm_req_key(req) for req in selected
        }
        self._dllm_prefill_burst_started = time.perf_counter()

    def _dllm_should_prefill_adaptive(
        self: Scheduler, cfg, prefill_reqs: List[Req], decode_reqs: List[Req]
    ) -> bool:
        """Switch between low-load burst and legacy-compatible overload service.

        Below 3B/4 decodable requests, use the bounded closed cohort that removes
        the multi-second quota gap. At sustained overload, preserve the previously
        validated deep-pool + q/B behavior; changing both axes at once destroyed
        the global SJF throughput benefit. A B/2 low watermark releases the latch.
        """
        now = time.perf_counter()
        budget = max(int(getattr(cfg, "max_running_requests", 1)), 1)
        ready_low = int(getattr(cfg, "my_sjf_ready_low", 0) or max(budget // 4, 1))
        ready_high = int(
            getattr(cfg, "my_sjf_ready_high", 0)
            or max(budget // 2, ready_low)
        )
        high_load_threshold = max((3 * budget) // 4, ready_high)
        total_visible = (
            len(decode_reqs)
            + len(prefill_reqs)
            + len(getattr(self, "waiting_queue", ()))
        )
        high_load = bool(getattr(self, "_dllm_high_load", False))
        if len(decode_reqs) >= high_load_threshold:
            high_load = True
        elif total_visible < ready_high:
            high_load = False
        self._dllm_high_load = high_load

        if high_load:
            self._dllm_prefill_cohort = set()
            self._dllm_prefill_burst_started = 0.0
            if not decode_reqs:
                return True
            if getattr(cfg, "my_sjf_prefill_quota", 0) > 0:
                return self._dllm_prefill_round_due(cfg)
            return True

        available = {self._dllm_req_key(req) for req in prefill_reqs}
        active = set(getattr(self, "_dllm_prefill_cohort", ())) & available
        self._dllm_prefill_cohort = active
        if active:
            started = float(getattr(self, "_dllm_prefill_burst_started", 0.0))
            if started <= 0.0:
                self._dllm_prefill_burst_started = now
                return True
            decode_cost = max(
                float(getattr(self, "_dllm_decode_round_ewma", 1.0)), 0.001
            )
            prefill_cost = max(
                float(getattr(self, "_dllm_prefill_round_ewma", 0.02)), 0.001
            )
            pause_budget = max(
                decode_cost
                * float(getattr(cfg, "my_sjf_prefill_pause_ratio", 1.0)),
                2.0 * prefill_cost,
                0.05,
            )
            if decode_reqs and now - started >= pause_budget:
                self._dllm_prefill_burst_started = 0.0
                return False
            return True

        self._dllm_prefill_burst_started = 0.0
        if not decode_reqs:
            self._dllm_start_prefill_cohort(cfg, prefill_reqs, decode_reqs)
            return True

        running = sum(
            1 for req in decode_reqs if getattr(req, "my_decoding", False)
        )
        ready = max(len(decode_reqs) - running, 0)
        decode_rounds = int(
            getattr(self, "_dllm_decode_rounds_since_prefill", 0)
        )
        critical_shortage = len(decode_reqs) < ready_high
        low_lookahead = ready < ready_low and decode_rounds >= 1

        oldest_wait = 0.0
        for req in prefill_reqs:
            ts = getattr(req, "time_stats", None)
            entered = getattr(ts, "wait_queue_entry_time", 0.0) or now
            oldest_wait = max(oldest_wait, now - entered)
        starved = oldest_wait >= max(
            2.0 * float(getattr(self, "_dllm_decode_round_ewma", 1.0)),
            1.0,
        )
        if critical_shortage or low_lookahead or starved:
            self._dllm_start_prefill_cohort(cfg, prefill_reqs, decode_reqs)
            return True
        return False

    def _dllm_adaptive_prefill_batch(
        self: Scheduler, prefill_reqs: List[Req]
    ) -> List[Req]:
        """Return only members of the currently frozen prefill cohort."""
        cohort = set(getattr(self, "_dllm_prefill_cohort", ()))
        return [
            req for req in prefill_reqs if self._dllm_req_key(req) in cohort
        ]

    def _dllm_prefill_round_due(self: Scheduler, cfg) -> bool:
        """Accrue quota credit to share contested rounds with decode.

        One prefill round costs the per-round block budget, so its frequency is
        approximately quota / max_running_requests.
        """
        quota = cfg.my_sjf_prefill_quota
        budget = max(cfg.max_running_requests, 1)
        credit = getattr(self, "_dllm_prefill_credit", 0.0) + quota
        if credit >= budget:
            self._dllm_prefill_credit = credit - budget
            return True
        self._dllm_prefill_credit = credit
        return False

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        *,
        preserve_order: bool = False,
    ) -> None:
        """Process a phase-homogeneous round without changing candidate order.

        An SJF ``batch`` may interleave staging and incoming requests after sorting.
        In that mode, dispatch contiguous runs to their phase-specific admission
        helpers instead of stable-partitioning all staging requests ahead of all
        incoming requests. The latter silently changed the global SJF order and
        could starve initial short-prompt ``INCOMING_DECODE`` requests behind
        post-prefill ``STAGING_DECODE`` requests regardless of predicted work.

        Non-SJF callers retain the historical staging-first behavior, which helps
        finish already-started chunks and is not preceded by a global SJF sort.
        """
        if not preserve_order:
            staging_reqs = [
                req for req in batch if req.dllm_phase == staging_phase
            ]
            if staging_reqs:
                result = self.process_dllm_staging_reqs(adder, staging_reqs)
                if result != AddReqResult.CONTINUE:
                    return

            incoming_reqs = [
                req for req in batch if req.dllm_phase == incoming_phase
            ]
            if incoming_reqs:
                self.process_dllm_incoming_reqs(adder, incoming_reqs)
            return

        begin = 0
        while begin < len(batch):
            phase = batch[begin].dllm_phase
            end = begin + 1
            while end < len(batch) and batch[end].dllm_phase == phase:
                end += 1
            phase_reqs = batch[begin:end]

            if phase == staging_phase:
                result = self.process_dllm_staging_reqs(adder, phase_reqs)
            elif phase == incoming_phase:
                result = self.process_dllm_incoming_reqs(adder, phase_reqs)
            else:
                begin = end
                continue

            if result != AddReqResult.CONTINUE:
                return
            begin = end

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats.from_adder(
            self.adder, self.running_batch.reqs, self.enable_priority_scheduling
        )

        self._cost_probe_forward_start()
        return new_batch

    def _mark_sjf_decode_admission(self: Scheduler, req: Req) -> None:
        """Make a request sticky only after PrefillAdder admits its decode block."""
        if _my_sjf_enabled(self.dllm_config) and not req.is_dllm_prefill():
            if not hasattr(req, "my_sjf_aging_periods"):
                req.my_sjf_aging_periods = _my_sjf_wait_periods(req, time.perf_counter())
            req.my_decoding = True

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            # Try preemption if batch is full
            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    res = AddReqResult.OTHER
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            before_len = len(adder.can_run_list)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            admitted = len(adder.can_run_list) > before_len
            # SJF: any admitted decode request is now decoding -> mark sticky so
            # it is never bumped on later rounds. Check the phase after
            # init_next_round_input(), because short prompts start as
            # INCOMING_DECODE but become STAGING_DECODE before admission.
            if admitted:
                self._mark_sjf_decode_admission(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            before_len = len(adder.can_run_list)
            res = adder.add_dllm_staging_req(req)
            admitted = len(adder.can_run_list) > before_len
            # SJF: any admitted decode request is now decoding -> mark sticky
            # (non-preemptive). Prefill staging reqs are left unmarked. The adder
            # can return NO_TOKEN after appending the last budget-fitting request,
            # so mark based on whether the request was actually admitted.
            if admitted:
                self._mark_sjf_decode_admission(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue.

        Under SJF prefill-order (B), finish already-started prefill chunks first,
        then serve shorter incoming prompts first. This keeps partially-prefilled
        requests from occupying pool slots indefinitely while still scoring likely
        short jobs early.
        """
        reqs = [req for req in self.waiting_queue if req.is_dllm_prefill()]
        if self.dllm_config is not None and _my_sjf_enabled(
            self.dllm_config
        ) and getattr(self.dllm_config, "my_sjf_prefill_order", False):
            # Apply the promotion discount to the prompt-length ordering key;
            # keep already-started prefill chunks first as before.
            promote_aging = _my_sjf_promote_aging(self.dllm_config)
            now = time.perf_counter()
            reqs.sort(
                key=lambda r: (
                    0 if r.dllm_phase == DllmReqPhase.STAGING_PREFILL else 1,
                    _my_sjf_aged_work(
                        r, float(len(r.origin_input_ids)), promote_aging, now
                    ),
                )
            )
        return reqs

    def get_decode_requests(self) -> List[Req]:
        """Keep running requests first, then admit the shortest predicted work."""
        reqs = [req for req in self.waiting_queue if not req.is_dllm_prefill()]
        cfg = self.dllm_config
        if not _my_sjf_enabled(cfg):
            return reqs
        now = time.perf_counter()
        aging = cfg.my_sjf_aging

        def key(req: Req):
            # A short prompt enters decode directly and has not paid its prompt cost.
            include_prefill = getattr(req, "dllm_phase", None) == DllmReqPhase.INCOMING_DECODE
            score = _my_phase_work(req, include_prefill=include_prefill)
            score = _my_sjf_aged_work(req, score, aging, now)
            return (0 if getattr(req, "my_decoding", False) else 1, score)

        return sorted(reqs, key=key)

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_chunked_count(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
