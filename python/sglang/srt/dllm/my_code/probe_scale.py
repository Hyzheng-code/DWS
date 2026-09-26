# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Full cost-profile fitting and serving micro-probe driver.

Only the HTTP coordinator performs I/O. Scheduler controls admission and TP
publication through the existing internal-state RPC, at forward boundaries.
"""
from __future__ import annotations
import json
import logging
import math
import threading
import time
import uuid
from pathlib import Path

from sglang.srt.dllm.dws_wsl import DWSWSLProfile, read_probe_suite, seal_profile, validate_suite
from sglang.srt.dllm.cost_fit_cache import (
    choose_profile, newest_profile, read_statistics, save_statistics, write_profile_cache,
)

logger = logging.getLogger(__name__)


def summarize_probe(records, probe, rid, d):
    records = [r for r in records if r['rid'] == rid]
    pre = [r for r in records if r['kind'] == 'prefill']
    blocks = [r for r in records if r['kind'] == 'decode_block']
    if len(pre) != len(probe['input_ids'])//d or not blocks or [r['block_index'] for r in blocks] != list(range(len(blocks))):
        raise ValueError('probe timing coverage mismatch')
    values = [r['elapsed_ms'] for r in pre]+[v for r in blocks for v in r['step_ms']]+[r['refresh_ms'] for r in blocks]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('nonpositive or nonfinite probe timing')
    depths = [len(r['step_ms']) for r in blocks]
    if any(not 0 < t <= d for t in depths) or depths[0] > d-len(probe['input_ids']) % d:
        raise ValueError('invalid probe trajectory')
    return {**probe, 'rid': rid, 'status': 'success', 'prompt_tokens': len(probe['input_ids']),
            'prefill_round_count': len(pre), 'prefill_gpu_ms': sum(r['elapsed_ms'] for r in pre),
            'output_block_steps': [len(r['step_ms']) for r in blocks], 'output_block_count': len(blocks),
            'denoising_gpu_ms': sum(sum(r['step_ms']) for r in blocks),
            'refresh_gpu_ms': sum(r['refresh_ms'] for r in blocks)}


class ProbeClient:
    def __init__(self, server_args):
        import requests
        self.args = server_args
        self.base = server_args.url().rstrip('/')
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.verify = server_args.ssl_verify()

    def call(self, path, data=None, timeout=10):
        key = (getattr(self.args, 'admin_api_key', None) or self.args.api_key) if path in {'/server_info', '/set_internal_state'} else self.args.api_key
        headers = {'Authorization': 'Bearer '+key} if key else {}
        response = self.session.get(self.base+path, timeout=timeout, headers=headers) if data is None else self.session.post(self.base+path, json=data, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.json()

    def states(self):
        return self.call('/server_info')['internal_states']

    def control(self, action, **values):
        self.call('/set_internal_state', {'server_args': {'dllm_cost_probe_control': {'action': action, **values}}})
        # The RPC's detailed result is exposed identically on each scheduler rank.
        states = self.states()
        if any(s.get('dllm_cost_probe_control_error') for s in states):
            raise RuntimeError(str([s.get('dllm_cost_probe_control_error') for s in states]))
        return states

    def warmup(self, profile, suite):
        """Retain an unmeasured B=1 warmup when reusing a cost profile."""
        token = uuid.uuid4().hex
        self.control('begin', token=token, idle=False)
        ids = suite[0]['input_ids'] if suite else [10, 11, 12]
        try:
            for attempt in range(2):
                rid = '__dllm_cost_probe__'+token+'_warmup_'+str(attempt)
                self.control('expect', token=token, rid=rid, timed=False)
                try:
                    result = self.call('/generate', {'rid': rid, 'input_ids': ids,
                        'sampling_params': {'temperature': 0, 'max_new_tokens': profile.block_size},
                        'stream': False, 'log_metrics': False}, timeout=self.args.my_cost_probe_timeout or None)
                    if result.get('meta_info', {}).get('finish_reason', {}).get('type') not in {'stop', 'length'}:
                        raise ValueError('warmup did not complete')
                    break
                except Exception:
                    self.control('abort', token=token, rid=rid)
                    if attempt:
                        raise
        finally:
            self.control('end', token=token)

    def run(self, profile, suite, *, idle=False):
        selected = suite
        if idle:
            # Both kinds, both short and long contexts; deterministic six-row subset.
            selected = [suite[i] for i in (0, 2, 4, 6, 12, 14) if i < len(suite)]
        if not selected:
            raise ValueError('probe suite is empty; provide --my-cost-probe-prompts or retain a matching full cost profile')
        token = uuid.uuid4().hex
        self.control('begin', token=token, idle=idle)
        rows, failures = [], 0
        started = time.monotonic()
        try:
            warm = {**selected[0], 'max_new_tokens': profile.block_size, 'ignore_eos': False}
            jobs = [('warmup', warm)]+[(r['probe_id'], r) for r in selected]
            for name, probe in jobs:
                success = False
                for attempt in range(2):
                    rid = '__dllm_cost_probe__'+token+'_'+name+'_'+str(attempt)
                    try:
                        self.control('expect', token=token, rid=rid, timed=name != 'warmup')
                        result = self.call('/generate', {'rid': rid, 'input_ids': probe['input_ids'],
                            'sampling_params': {'temperature': 0, 'max_new_tokens': probe['max_new_tokens'], 'ignore_eos': probe['ignore_eos']},
                            'stream': False, 'log_metrics': False}, timeout=self.args.my_cost_probe_timeout or None)
                        finish = result.get('meta_info', {}).get('finish_reason', {}).get('type')
                        if finish not in {'stop', 'length'} or (probe['ignore_eos'] and finish != 'length'):
                            raise ValueError('probe did not complete')
                        states = self.control('collect', token=token, rid=rid)
                        if name != 'warmup':
                            records = next((s['dllm_cost_probe_records'] for s in states if s.get('dllm_cost_probe_records')), [])
                            rows.append(summarize_probe(records, probe, rid, profile.block_size))
                        success = True
                        break
                    except Exception:
                        self.control('abort', token=token, rid=rid)
                        if any(s.get('dllm_cost_probe_cancelled') for s in self.states()):
                            raise RuntimeError('idle probe yielded to traffic or exhausted GPU budget')
                if not success:
                    failures += 1
                    if name == 'warmup':
                        raise RuntimeError('probe warmup failed twice')
            if failures > len(selected)/4:
                raise RuntimeError('more than one quarter of probes failed')
            states = self.states()
            from sglang.srt.dllm.cost_fit import fit_cost_profile
            cost = states[0]['dllm_cost_shape_signature']['cost_keys']
            base = newest_profile(profile, cost)
            state = (read_statistics(base, profile_path=self.args.my_sjf_cost_profile)
                     if base.is_full_fit and base.payload['scale']['cost_keys'] == cost else None)
            # Retain the actual suite (including a CLI override) with the fit.
            # Inline token IDs make cached profiles and snapshots portable.
            base = DWSWSLProfile.from_mapping(seal_profile({**base.payload, 'shape': {
                **base.payload['shape'], 'probe_suite': {'version': 1, 'requests': suite}}}))
            updated, state = fit_cost_profile(base, rows, cost, state)
            scale = updated.payload['scale']
            updated = updated.with_scale({**scale, 'probe_stats': {
                **scale['probe_stats'], 'failed_requests': failures,
                'wall_seconds': time.monotonic()-started}})
            # Persist immutable statistics first. Cancellation can leave an
            # unreferenced sidecar, but cannot advance the active profile/cache.
            save_statistics(updated, state)
            self.control('publish', token=token, profile=updated.to_dict())
            try:
                write_profile_cache(updated, state, profile_path=self.args.my_sjf_cost_profile)
            except (OSError, ValueError) as exc:
                logger.warning('Cost profile published; cache could not be saved: %s', exc)
            self.control('end', token=token)
            return updated
        except Exception:
            try:
                self.control('cancel' if idle else 'fail', token=token)
            except Exception:
                logger.exception('Could not close failed probe window')
            raise


def start_serving_probes(server_args):
    """Called from the existing HTTP warmup thread, once per single-DP server."""
    from sglang.srt.dllm.dws_wsl import load_dws_wsl_profile
    profile = load_dws_wsl_profile(server_args.my_sjf_cost_profile)
    if not profile.is_v5:
        return False
    client = ProbeClient(server_args)
    for _ in range(120):
        try:
            states = client.states()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('HTTP server did not start for cost profiling')
    if not states or any(s['dllm_cost_shape_signature']['cost_keys'] != states[0]['dllm_cost_shape_signature']['cost_keys'] for s in states):
        raise ValueError('DWS-WSL v2 requires homogeneous cost keys across TP ranks')
    selected = choose_profile(profile, states[0]['dllm_cost_shape_signature']['cost_keys'], server_args.my_cost_probe)
    suite_profile = selected or newest_profile(profile, states[0]['dllm_cost_shape_signature']['cost_keys'])
    suite = read_probe_suite(suite_profile, server_args.my_sjf_cost_profile)
    if server_args.my_cost_probe_prompts:
        suite = [json.loads(line) for line in Path(server_args.my_cost_probe_prompts).read_text().splitlines() if line.strip()]
        validate_suite({'version': 1, 'requests': suite})
    if any(row['max_new_tokens'] != profile.max_new_tokens for row in suite):
        raise ValueError('probe suite max_new_tokens differs from trajectory contract')
    if selected is None:
        profile = client.run(profile, suite)
    else:
        if not server_args.skip_server_warmup:
            client.warmup(selected, suite)
        # Reuse is also an atomic publication; metadata/probe_state changes have IDs.
        client.control('reuse', profile=selected.to_dict())
        profile = selected
    if server_args.my_cost_probe_refresh == 'idle':
        def refresh():
            nonlocal profile
            while True:
                time.sleep(5)
                try:
                    states = client.states()
                    if any(s.get('dllm_cost_probe_refresh_disabled') for s in states):
                        return
                    if all(s.get('dllm_cost_idle_seconds', 0) >= 60 for s in states):
                        # Require enough accumulated GPU credit to cover the last
                        # measured suite, conservatively accounting for warmup.
                        estimate = profile.payload['scale'].get('probe_stats', {}).get('gpu_seconds', 60)
                        if all(s.get('dllm_cost_probe_gpu_credit', 0) >= estimate for s in states):
                            profile = client.run(profile, suite, idle=True)
                except Exception as exc:
                    logger.warning('DWS-WSL idle refresh discarded: %s', exc)
        threading.Thread(target=refresh, name='dllm-cost-refresh', daemon=True).start()
    return True
