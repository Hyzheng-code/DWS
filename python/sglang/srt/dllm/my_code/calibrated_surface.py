# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Capture the predictor probability surface required for DWS-WSL scoring."""

from contextlib import contextmanager

import numpy as np

from sglang.srt.dllm.my_code.dws_marginals import surface_marginals


@contextmanager
def capture_calibrated_surface(predictor):
    """Retain the forward's surface for one later projection/reduction pass."""
    rows = []

    def capture(surface):
        if hasattr(surface, "detach"):
            surface = surface.detach()
        else:
            surface = np.asarray(surface)
        if surface.ndim != 3:
            raise ValueError("calibrated DWS surface requires [batch, block, depth]")
        rows.extend(surface)

    original = getattr(predictor, "session", None)
    hook = None
    if original is not None:
        class Session:
            def __getattr__(self, name):
                return getattr(original, name)

            def run(self, names, inputs, *args, **kwargs):
                values = original.run(names, inputs, *args, **kwargs)
                outputs = names or [v.name for v in original.get_outputs()]
                if "surface" in outputs:
                    capture(values[outputs.index("surface")])
                return values

        predictor.session = Session()
    else:
        model = getattr(predictor, "model", None)
        if model is not None and hasattr(model, "register_forward_hook"):
            def on_forward(_module, _inputs, output):
                if isinstance(output, dict) and "surface" in output:
                    capture(output["surface"])
            hook = model.register_forward_hook(on_forward)
    try:
        yield rows
    finally:
        if original is not None:
            predictor.session = original
        if hook is not None:
            hook.remove()


def attach_calibrated_surface(items, predictions, captured, block_size):
    """Project requested full surfaces, or reduce compact marginals once."""
    if len(items) != len(predictions):
        raise ValueError("prediction count differs from request count")
    if captured and len(captured) != len(predictions):
        raise ValueError("calibrated surface count differs from prediction count")
    for i, (item, prediction) in enumerate(zip(items, predictions)):
        surface = captured[i] if captured else prediction.get("surface")
        if surface is None:
            continue
        if not hasattr(surface, "detach"):
            surface = np.asarray(surface)
        if item.get("dws_return_surface"):
            array = surface.float().cpu().numpy() if hasattr(surface, "detach") else surface
            prediction["dws_surface"] = array.tolist()
            if apply_dws_wsl_support([item], [prediction], block_size):
                continue  # Projection already produced the final marginals.
        prediction.update(surface_marginals(surface[None])[0])


def apply_dws_wsl_support(items, predictions, block_size):
    """Apply known execution bounds to DWS-WSL only; keep all RPC totals coherent.

Some deployed heads predict 32 depths for the partial first output block.
There can be at most d - P%d steps there because every step commits a token.
Zeroing the unreachable survival tail represents min(T, capacity), without
renormalizing any reachable probability or modifying the predictor weights.
"""
    projected = set()
    for index, (item, prediction) in enumerate(zip(items, predictions)):
        if item.get("dws_cost_model") != "dws-wsl" or "dws_surface" not in prediction:
            continue
        try:
            surface = np.asarray(prediction["dws_surface"], dtype=np.float64)
        except (ValueError, TypeError):
            continue  # Runtime reports malformed predictions through its fallback.
        if (surface.ndim != 2 or surface.shape[1] != block_size or not len(surface)
                or not np.isfinite(surface).all() or np.any(surface < 0)
                or np.any(surface > 1 + 2e-4)):
            continue
        prompt_tokens = len(item.get("input_ids") or ())
        cap = item.get("max_new_tokens", prediction.get("dws_max_new_tokens"))
        if type(cap) is not int or cap <= 0:
            continue
        first_depth = block_size - prompt_tokens % block_size
        max_blocks = (prompt_tokens % block_size + cap + block_size - 1) // block_size
        surface = surface.copy()
        original_mass = float(surface.sum())
        surface[0, first_depth:] = 0
        surface[max_blocks:] = 0
        marginals = surface_marginals(surface[None])[0]
        prediction.update(marginals)
        prediction["dws_surface"] = surface.tolist()
        prediction["expected_blocks"] = marginals["expected_n_blocks"]
        prediction["scheduler_score"] = marginals["workload_curve"][-1]
        prediction["predicted_workload_32"] = marginals["workload_curve"][-1]
        prediction["dws_support_projection"] = {
            "first_block_max_depth": first_depth, "max_output_blocks": max_blocks,
            "removed_step_mass": original_mass - float(surface.sum()),
        }
        projected.add(index)
    return projected
