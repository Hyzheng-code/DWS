# DWS research fork: modified from the imported SGLang 0.5.10 source.
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

import os as _os

# The stock implementation processes every request in a Python loop and calls
# ``.item()`` several times per request and denoising step.  At serving batch
# sizes this serializes the CPU with the GPU hundreds or thousands of times per
# round.  Keep an escape hatch for exact A/B comparisons, but use the batched
# implementation by default.
_VECTORIZED = _os.environ.get("SGLANG_DLLM_VECTORIZED", "1").lower() not in {
    "0",
    "false",
    "off",
}


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.track_actual_denoise_steps = bool(
            getattr(config, "my_metrics_log", "")
        )

    @staticmethod
    def _record_actual_denoise_steps(
        forward_batch: ForwardBatch, steps_per_request: List[int]
    ) -> None:
        """Accumulate logical denoising work, Σs_b, for each request."""
        reqs = getattr(forward_batch, "reqs", None) or ()
        for req, steps in zip(reqs, steps_per_request):
            ts = getattr(req, "time_stats", None)
            if (
                ts is None
                or not getattr(ts, "my_enable_first_token_time_export", False)
            ):
                continue
            ts.my_actual_denoise_steps = int(
                getattr(ts, "my_actual_denoise_steps", 0)
            ) + int(steps)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if not _VECTORIZED:
            return self._run_legacy(model_runner, forward_batch)
        return self._run_vectorized(model_runner, forward_batch)

    def _run_vectorized(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        """Run LowConfidence with one set of GPU ops for the whole batch.

        This is algorithmically equivalent to the per-request implementation:
        every masked position above ``threshold`` is committed and, when fewer
        than the configured schedule pass, the most-confident masked positions
        fill that step's transfer quota.  The important difference is that
        request-level reductions stay on the GPU; only loop termination and the
        final small metadata lists synchronize with the CPU.
        """
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.reshape(batch_size, self.block_size)
        mask_index = input_ids == self.mask_id

        rids = list(getattr(forward_batch, "rids", None) or ())
        if len(rids) != batch_size:
            reqs = getattr(forward_batch, "reqs", None) or ()
            rids = [str(getattr(req, "rid", "")) for req in reqs]
        calibrator = getattr(self, "cost_calibrator", None)
        cost_rid = str(rids[0]) if batch_size == 1 and rids else ""
        cost_owner = int(getattr(model_runner, "tp_rank", 0)) == 0
        cost_time = bool(
            calibrator is not None
            and cost_owner
            and cost_rid
            and calibrator.should_time(cost_rid)
            and input_ids.is_cuda
        )
        # Fast path: prompt prefill / KV population contains no mask token.
        if not torch.any(mask_index).item():
            if cost_time:
                prefill_start_event = torch.cuda.Event(enable_timing=True)
                prefill_end_event = torch.cuda.Event(enable_timing=True)
                prefill_start_event.record()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if cost_time:
                prefill_end_event.record()
                prefill_end_event.synchronize()
                calibrator.record_prefill(
                    cost_rid,
                    prefill_start_event.elapsed_time(prefill_end_event),
                )
            return out.logits_output, [], out.can_run_graph

        # One D2H synchronization for the whole batch instead of one per request.
        start_list = (self.block_size - mask_index.sum(dim=1)).tolist()

        cell_event_records = [] if cost_time else None

        actual_steps = None
        if self.track_actual_denoise_steps:
            actual_steps = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=forward_batch.input_ids.device,
            )

        for _ in range(self.block_size):
            mask_index = input_ids == self.mask_id
            active_blocks = mask_index.any(dim=1)
            if not torch.any(active_blocks).item():
                break

            if cost_time:
                step_start_event = torch.cuda.Event(enable_timing=True)
                step_end_event = torch.cuda.Event(enable_timing=True)
                step_start_event.record()

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            block_logits = logits_output.full_logits.reshape(
                batch_size, self.block_size, -1
            )

            # Exact confidence formulation used by the legacy implementation,
            # applied to the whole batch in one GPU operation.
            confidence, chosen_ids = F.softmax(block_logits, dim=-1).max(dim=-1)

            confidence = confidence.masked_fill(~mask_index, -torch.inf)
            transfer_index = confidence > self.threshold

            # Preserve the stock acceptance rule: when no position clears the
            # threshold, commit exactly the single most-confident masked token.
            needs_floor = active_blocks & (transfer_index.sum(dim=1) == 0)
            topk_index = confidence.topk(k=1, dim=1).indices
            floor_mask = torch.zeros_like(transfer_index).scatter_(
                1, topk_index, True
            )
            floor_mask &= mask_index
            transfer_index = torch.where(
                needs_floor.unsqueeze(1), floor_mask, transfer_index
            )
            if actual_steps is not None:
                actual_steps.add_(active_blocks)

            input_ids[transfer_index] = chosen_ids[transfer_index]

            if cost_time:
                step_end_event.record()
                cell_event_records.append(
                    {
                        "step_start": step_start_event,
                        "step_end": step_end_event,
                    }
                )

        if actual_steps is not None:
            self._record_actual_denoise_steps(
                forward_batch, actual_steps.tolist()
            )

        if cost_time:
            refresh_start_event = torch.cuda.Event(enable_timing=True)
            refresh_end_event = torch.cuda.Event(enable_timing=True)
            refresh_start_event.record()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if cost_time:
            refresh_end_event.record()
            # One synchronization per block round, after all timed work.  Event
            # durations therefore exclude CPU waiting.
            refresh_end_event.synchronize()
            refresh_ms = refresh_start_event.elapsed_time(refresh_end_event)
            if cost_time:
                calibrator.record_decode_block(
                    cost_rid,
                    [
                        record["step_start"].elapsed_time(record["step_end"])
                        for record in cell_event_records
                    ],
                    refresh_ms,
                )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        next_token_ids_list = [
            input_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        return logits_output, next_token_ids_list, can_run_cuda_graph

    def _run_legacy(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        actual_steps = (
            [0] * batch_size if self.track_actual_denoise_steps else None
        )

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                if actual_steps is not None:
                    actual_steps[batch_id] += 1
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

        if actual_steps is not None:
            self._record_actual_denoise_steps(forward_batch, actual_steps)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
