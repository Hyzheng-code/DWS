# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Per-request server-side latency metrics (TTFT / latency / per-token) for dLLM
serving experiments.

When `--my-metrics-log <path>` is set, the TokenizerManager writes one JSON line
per finished request with its end-to-end timings (measured at the server, same
`perf_counter` clock as `ReqTimeStats`).

Definitions (seconds):
  ttft          = first_token_time - created_time        (time to first token)
                  For dLLM, first_token_time is the scheduler-stamped *first-demask*
                  moment (first committed token of the first decode block), shipped
                  via SchedulerReqTimeStats.my_first_token_time — so TTFT is a true
                  arrival -> first-token time (incl. queue/admission delay) even for
                  non-streaming requests, instead of the completion time. Falls back
                  to the tokenizer first-output time if unset.
  e2e_latency   = finished_time   - created_time         (total latency)
  decode_time   = finished_time   - first_token_time
  tpot          = decode_time / max(completion_tokens-1, 1)   (per-output-token)
  per_token     = e2e_latency / max(completion_tokens, 1)     (amortized)
  actual_denoise_steps = sum of the request's real per-block s_b values

Default off; writes only when explicitly enabled.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

# ---- writer (used inside TokenizerManager) ---------------------------------
_LOCK = threading.Lock()
_FH = {}  # path -> file handle (process-wide, append mode)


def _handle(path: str):
    fh = _FH.get(path)
    if fh is None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fh = open(path, "a")
        _FH[path] = fh
    return fh


def log_request(
    path: str,
    rid: str,
    created_time: float,
    first_token_time: float,
    finished_time: float,
    completion_tokens: int,
    prompt_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one request's metrics as a JSON line. Fail-soft (never raises)."""
    try:
        ct = int(completion_tokens or 0)
        has_first_token = ct > 0 and first_token_time > 0
        ttft = first_token_time - created_time if has_first_token else None
        e2e = finished_time - created_time if finished_time > 0 else None
        decode_time = (
            finished_time - first_token_time
            if (finished_time > 0 and has_first_token)
            else None
        )
        tpot = (
            decode_time / max(ct - 1, 1)
            if (decode_time is not None and ct > 0)
            else None
        )
        per_token = e2e / ct if (e2e is not None and ct > 0) else None
        row = {
            "rid": rid,
            "created_time": created_time,
            "completion_tokens": ct,
            "prompt_tokens": prompt_tokens,
            "ttft": ttft,
            "e2e_latency": e2e,
            "decode_time": decode_time,
            "tpot": tpot,
            "per_token": per_token,
            "finish_reason": finish_reason,
        }
        if extra:
            row.update(extra)
        line = json.dumps(row, ensure_ascii=False)
        with _LOCK:
            fh = _handle(path)
            fh.write(line + "\n")
            fh.flush()
    except Exception:  # noqa: BLE001 — metrics must never break serving
        pass
