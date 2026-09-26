# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Scheduler-owned probe barrier and immutable TP cost-profile publication."""
import logging
import time
from pathlib import Path

from sglang.srt.dllm.dws_wsl import DWSWSLProfile, copy_probe_sidecar
from sglang.srt.dllm.cost_fit_cache import copy_statistics_sidecar
from sglang.srt.dllm.profile_common import atomic_json

logger = logging.getLogger(__name__)


class CostProbeMixin:
    def _cost_probe_init(self):
        self._my_cost_probe_window = None
        self._my_cost_probe_cancelled = False
        self._my_cost_probe_expected = ''
        self._my_cost_probe_records = []
        self._my_cost_probe_control_error = None
        self._my_cost_probe_last_activity = time.monotonic()
        self._my_cost_serving_gpu_s = 0.0
        self._my_cost_idle_probe_gpu_s = 0.0
        self._my_cost_probe_refresh_disabled = False
        self._my_cost_probe_state = 'pending'
        self._my_cost_gpu_event = None

    def _cost_probe_requests(self, active_only=False):
        collections = [([] if active_only else self.waiting_queue), self.dllm_manager.waiting_queue, self.dllm_manager.staging_queue,
                       getattr(getattr(self, 'running_batch', None), 'reqs', []),
                       getattr(getattr(self, 'cur_batch', None), 'reqs', [])]
        return list({id(r): r for rows in collections for r in rows if not r.finished()}.values())

    def _cost_probe_credit(self):
        return self._my_cost_serving_gpu_s*.005/.995-self._my_cost_idle_probe_gpu_s

    def _cost_probe_cancel(self, rid=None):
        # dLLM explicitly disables overlap scheduling: previous result/KV writes
        # have completed before incoming RPCs/requests reach this method.
        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        from sglang.srt.mem_cache.common import release_kv_cache
        for req in self._cost_probe_requests():
            if getattr(req, 'my_cost_probe', False) and (rid is None or req.rid == rid):
                req.finished_reason = FINISH_ABORT()
                if req.req_pool_idx is not None:
                    release_kv_cache(req, self.tree_cache)
                self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
        self.waiting_queue = [r for r in self.waiting_queue if not r.finished()]
        self.dllm_manager.filter_finished_reqs()
        if rid is None:
            self._my_cost_probe_cancelled = True
            self._my_cost_probe_window = None
            self._my_cost_probe_expected = ''
            calibrator = self.tp_worker.dllm_algorithm.cost_calibrator
            calibrator.probe_rids.clear()
            calibrator.probe_records.clear()
            self._my_cost_probe_records = []

    def _cost_probe_accept_request(self, req):
        """Mark the single registered probe; real traffic cancels idle refresh."""
        if not hasattr(self, '_my_cost_probe_state'):
            return
        profile = getattr(self.dllm_config, 'my_cost_profile', None)
        if not isinstance(profile, DWSWSLProfile) or not profile.is_v5:
            return
        req.my_cost_probe = bool(self._my_cost_probe_window and req.rid == self._my_cost_probe_expected)
        if req.my_cost_probe:
            req.my_scored = True
            req.my_sjf_score = 0.0
        else:
            self._my_cost_probe_last_activity = time.monotonic()
            if self._my_cost_probe_window and self._my_cost_probe_window['idle']:
                self._cost_probe_cancel()

    def _cost_probe_status(self):
        if not hasattr(self, '_my_cost_probe_state'):
            return {}
        busy = bool(self._cost_probe_requests())
        return {'dllm_cost_probe_state': self._my_cost_probe_state,
                'dllm_cost_probe_control_error': self._my_cost_probe_control_error,
                'dllm_cost_probe_cancelled': self._my_cost_probe_cancelled,
                'dllm_cost_probe_records': self._my_cost_probe_records,
                'dllm_cost_idle_seconds': 0 if busy else time.monotonic()-self._my_cost_probe_last_activity,
                'dllm_cost_probe_gpu_credit': self._cost_probe_credit(),
                'dllm_cost_probe_refresh_disabled': self._my_cost_probe_refresh_disabled}

    def _cost_probe_publish(self, payload):
        import torch
        group = self.tp_group
        error, profile = None, None
        try:
            profile = DWSWSLProfile.from_mapping(payload)
            profile.validate_runtime(self._my_cost_shape_signature, self.dllm_config.block_size)
            old = self.dllm_config.my_cost_profile
            if old.payload['shape']['trajectory_keys'] != profile.payload['shape']['trajectory_keys']:
                raise ValueError('cost publication cannot change trajectory configuration')
            if (profile.is_full_fit and profile.payload['scale']['cost_keys'] !=
                    self._my_cost_shape_signature['cost_keys'] and
                    getattr(self.server_args, 'my_cost_probe', 'auto') != 'off'):
                raise ValueError('cost publication deployment mismatch')
            if any(not getattr(r, 'my_cost_probe', False) for r in self._cost_probe_requests(active_only=True)):
                raise ValueError('cannot publish cost profile while real requests are active')
        except Exception as exc:
            error = str(exc)
        errors = [error]
        world = int(getattr(group, 'world_size', 1) or 1)
        if world > 1:
            errors = [None]*world
            torch.distributed.all_gather_object(errors, error, group=self.tp_cpu_group)
        if any(errors):
            raise ValueError(f'cost profile publication validation failed: {errors}')
        owner = int(getattr(group, 'rank_in_group', 0) or 0) == 0
        if owner:
            try:
                snapshot_dir = Path(f'/tmp/dllm_cost_profiles_{self.server_args.port}')
                source = getattr(self.server_args, 'my_sjf_cost_profile', '')
                copy_probe_sidecar(profile, source, snapshot_dir)
                copy_statistics_sidecar(profile, source, snapshot_dir)
                atomic_json(snapshot_dir / 'profile.json', profile.to_dict())
            except Exception as exc:
                error = str(exc)
        if world > 1:
            errors = [None]*world
            torch.distributed.all_gather_object(errors, error, group=self.tp_cpu_group)
        else:
            errors = [error]
        if any(errors):
            raise ValueError(f'cost snapshot failed; old profile retained: {errors}')
        # All ranks switch only after validation and successful archival.
        self.dllm_config.my_cost_profile = profile
        self._my_cost_probe_state = profile.payload['scale']['probe_state']
        if owner:
            scale = profile.payload['scale']
            logger.warning('[dllm-cost] shape_id=%s scale_id=%s probe_state=%s parameters=%s stats=%s',
                           profile.shape_id, profile.scale_id, scale['probe_state'],
                           {'delta': profile.delta, 'mbar': profile.mbar,
                            'alpha': profile.prefill_block_ms, 'beta': profile.refresh_block_ms}, scale.get('probe_stats', {}))

    def _cost_probe_control(self, command):
        if not hasattr(self, '_my_cost_probe_state'):
            raise ValueError('DWS-WSL v2 probing is not enabled')
        action = command['action']
        window = self._my_cost_probe_window
        calibrator = self.tp_worker.dllm_algorithm.cost_calibrator
        if action == 'begin':
            if window or self._cost_probe_requests(active_only=not command.get('idle')):
                raise ValueError('probe barrier requires empty queues')
            if command.get('idle') and (self._my_cost_probe_refresh_disabled or
                    time.monotonic()-self._my_cost_probe_last_activity < 60 or self._cost_probe_credit() <= 0):
                raise ValueError('idle probe is not eligible')
            self._my_cost_probe_window = {'token': command['token'], 'idle': bool(command.get('idle'))}
            self._my_cost_probe_cancelled = False
            self._my_cost_probe_records = []
            calibrator.probe_records.clear()
            calibrator.probe_rids.clear()
            return
        if action == 'reuse':
            self._cost_probe_publish(command['profile'])
            return
        if window is None or command.get('token') != window['token']:
            if action in {'abort', 'cancel', 'end'} and self._my_cost_probe_cancelled:
                return
            raise ValueError('probe window cancelled or token mismatch')
        if action == 'expect':
            if self._cost_probe_requests(active_only=True):
                raise ValueError('a probe is still active')
            rid = command['rid']
            if not rid.startswith('__dllm_cost_probe__'+window['token']+'_'):
                raise ValueError('invalid probe RID')
            self._my_cost_probe_expected = rid
            calibrator.probe_rids.clear()
            if command.get('timed'):
                calibrator.probe_rids.add(rid)
        elif action == 'collect':
            self._my_cost_probe_records = calibrator.probe_records.pop(command['rid'], [])
        elif action == 'abort':
            self._cost_probe_cancel(command['rid'])
            calibrator.probe_records.pop(command['rid'], None)
        elif action == 'publish':
            self._cost_probe_publish(command['profile'])
        elif action in {'end', 'cancel', 'fail'}:
            self._cost_probe_cancel()
            calibrator.probe_rids.clear()
            calibrator.probe_records.clear()
            self._my_cost_probe_records = []
            self._my_cost_probe_last_activity = time.monotonic()
            if action == 'fail':
                self._my_cost_probe_state = 'failed'
        else:
            raise ValueError('unknown probe control action')

    def _cost_probe_forward_start(self):
        if not hasattr(self, '_my_cost_probe_state') or getattr(self.server_args, 'my_cost_probe_refresh', 'off') != 'idle':
            return
        import torch
        self._my_cost_gpu_event = torch.cuda.Event(enable_timing=True)
        self._my_cost_gpu_event.record()

    def _cost_probe_forward_end(self, reqs):
        start = getattr(self, '_my_cost_gpu_event', None)
        if start is None:
            return
        import torch
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)/1000
        if int(getattr(self.tp_group, 'world_size', 1) or 1) > 1:
            # A local budget decision could cancel on only one TP rank and
            # deadlock the next forward. Use one conservative duration on all ranks.
            duration = torch.tensor(elapsed, dtype=torch.float64, device='cpu')
            torch.distributed.all_reduce(duration, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group)
            elapsed = duration.item()
        self._my_cost_gpu_event = None
        window = self._my_cost_probe_window
        if reqs and all(getattr(r, 'my_cost_probe', False) for r in reqs):
            if window and window['idle']:
                self._my_cost_idle_probe_gpu_s += elapsed
                if self._cost_probe_credit() < 0:
                    self._my_cost_probe_refresh_disabled = True
                    logger.warning('[dllm-scale] idle GPU budget exceeded; disabling refresh')
        else:
            self._my_cost_serving_gpu_s += elapsed
            self._my_cost_probe_last_activity = time.monotonic()
