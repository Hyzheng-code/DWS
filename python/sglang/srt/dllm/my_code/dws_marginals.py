# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Extract compact DWS marginals without modifying deployed predictor weights."""

import numpy as np


def surface_marginals(surface):
    """Accept a batched torch tensor or ndarray; reduce before host transfer."""
    if hasattr(surface, "detach"):
        surface = surface.detach().float()
        h = surface.sum(dim=1).cpu().numpy()
        q = surface[:, :, 0].cpu().numpy()
    else:
        surface = np.asarray(surface, dtype=np.float64)
        if surface.ndim != 3:
            raise ValueError("DWS surface must be [requests, blocks, depths]")
        h, q = surface.sum(axis=1), surface[:, :, 0]
    return [
        {"workload_curve": np.cumsum(depth, dtype=np.float64).tolist(),
         "expected_n_blocks": float(prob.sum()),
         "dws_contract_version": 2, "dws_max_new_tokens": 1024}
        for depth, prob in zip(h, q)
    ]
