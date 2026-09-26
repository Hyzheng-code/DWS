# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""LowConfidence entrypoint for DWS prompt-based scheduling.

The scheduler's prompt worker obtains and broadcasts arrival scores before
admission. Denoising uses the normal vectorized LowConfidence implementation,
including when prompt prediction fails and scheduling falls back to length.
"""
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence


class LowConfidencePredict(LowConfidence):
    """Keep the deployment algorithm name without capturing hidden states."""
