# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Registration shim for the LowConfidencePredict algorithm.

The implementation lives in ``my_code`` (user-local research code). dLLM
algorithm discovery only scans this package, so we re-export the class here so
``--dllm-algorithm LowConfidencePredict`` resolves. See
``sglang.srt.dllm.my_code.predict_algorithm`` for the actual logic.
"""

from sglang.srt.dllm.my_code.predict_algorithm import LowConfidencePredict

Algorithm = LowConfidencePredict
