"""Appendix E, Eqs. (14)--(17): full request-level cost fitting.

The retained state contains additive statistics, never historical requests.
Training/validation statistics are separate so regularization can be selected
on held-out traces before refitting on their union.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields

import numpy as np

from sglang.srt.dllm.dws_wsl import canonical

DEFAULT_FIT_CONFIG = {
    "lambda_s_candidates": [1e-6, 1e-4, 1e-2],
    "lambda_b_candidates": [1e-6, 1e-4, 1e-2],
    "validation_every": 5,
}


def fit_config(value=None):
    config = {**DEFAULT_FIT_CONFIG, **(value or {})}
    if set(config) != set(DEFAULT_FIT_CONFIG):
        raise ValueError("unknown WLS fit configuration")
    for key in ("lambda_s_candidates", "lambda_b_candidates"):
        values = config[key]
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("regularization candidates must be finite and nonnegative")
        config[key] = sorted(set(map(float, values)))
    if type(config["validation_every"]) is not int or config["validation_every"] < 2:
        raise ValueError("validation_every must be an integer >= 2")
    return config


def deployment_identity(profile, cost):
    return {"trajectory_keys": profile.payload["shape"]["trajectory_keys"], "cost_keys": cost}


def actual_probe_steps(rows, block_size, max_blocks):
    steps = np.zeros((len(rows), max_blocks), dtype=np.int64)
    seen = set()
    for i, row in enumerate(rows):
        raw = np.asarray(row["output_block_steps"], dtype=float)
        if (row["rid"] in seen or row.get("status") != "success"
                or row.get("forced_single_commit", False) or row.get("calibration", False)):
            raise ValueError("requires unique completed natural requests; forced calibration is invalid")
        seen.add(row["rid"])
        if (raw.ndim != 1 or not 0 < len(raw) <= max_blocks or not np.isfinite(raw).all()
                or np.any(raw != np.floor(raw)) or np.any((raw < 1) | (raw > block_size))):
            raise ValueError("invalid natural block depths")
        p = row["prompt_tokens"]
        if type(p) is not int or p < 0 or raw[0] > block_size - p % block_size:
            raise ValueError("invalid prompt length or partial first block depth")
        if row["prefill_round_count"] != p // block_size:
            raise ValueError("prefill count does not match prompt (cache hit or incomplete measurement)")
        if row.get("output_block_count", len(raw)) != len(raw):
            raise ValueError("incomplete output block coverage")
        times = [float(row[k]) for k in ("prefill_gpu_ms", "refresh_gpu_ms", "denoising_gpu_ms")]
        if (not all(math.isfinite(v) for v in times) or min(times[1:]) <= 0
                or (times[0] <= 0 if p // block_size else times[0] != 0)):
            raise ValueError("invalid probe timing")
        steps[i, :len(raw)] = raw
    return steps


@dataclass
class SufficientStats:
    G: np.ndarray
    r: np.ndarray
    q: np.ndarray
    count: int = 0
    z02: float = 0.0
    z0y: float = 0.0
    y2: float = 0.0
    sum_t: float = 0.0
    sum_mt: float = 0.0
    denoising_ms: float = 0.0
    prefill_ms: float = 0.0
    prefill_blocks: float = 0.0
    refresh_ms: float = 0.0
    output_blocks: float = 0.0
    prompt_min: int = -1
    prompt_max: int = -1

    @classmethod
    def empty(cls, cells):
        return cls(np.zeros((cells, cells)), np.zeros(cells), np.zeros(cells))

    @classmethod
    def from_rows(cls, rows, block_size, max_blocks):
        steps = actual_probe_steps(rows, block_size, max_blocks)
        if not rows:
            return cls.empty(block_size * max_blocks)
        a = (np.arange(block_size)[None, None, :] < steps[:, :, None]).reshape(len(rows), -1).astype(float)
        t = a.sum(axis=1)
        prompts = np.array([r["prompt_tokens"] for r in rows])
        m = prompts // block_size
        z0 = m * t
        y = np.array([r["denoising_gpu_ms"] for r in rows], dtype=float)
        w = (t + 10) ** -2
        return cls(
            G=a.T @ (w[:, None] * a), r=a.T @ (w * y), q=a.T @ (w * z0),
            count=len(rows), z02=float(w @ z0**2), z0y=float(w @ (z0 * y)),
            y2=float(w @ y**2), sum_t=float(t.sum()), sum_mt=float(z0.sum()),
            denoising_ms=float(y.sum()), prefill_ms=sum(r["prefill_gpu_ms"] for r in rows),
            prefill_blocks=float(m.sum()), refresh_ms=sum(r["refresh_gpu_ms"] for r in rows),
            output_blocks=float((steps > 0).sum()),
            prompt_min=int(prompts.min()), prompt_max=int(prompts.max()),
        )

    def merge(self, other):
        if self.G.shape != other.G.shape:
            raise ValueError("cannot merge statistics with different dimensions")
        values = {f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        lows = [s.prompt_min for s in (self, other) if s.count]
        values["prompt_min"] = min(lows, default=-1)
        values["prompt_max"] = max(self.prompt_max, other.prompt_max)
        return type(self)(**values)

    @property
    def mbar(self):
        if self.sum_t <= 0:
            raise ValueError("empty profiling statistics")
        return self.sum_mt / self.sum_t

    def normal_equations(self, mbar=None):
        """Center z0=m*T using T=1^T*a, without replaying any trace."""
        mbar = self.mbar if mbar is None else mbar
        gt = self.G.sum(axis=1)
        q = self.q - mbar * gt
        zz = self.z02 - 2 * mbar * self.q.sum() + mbar**2 * gt.sum()
        h = np.empty((len(self.r) + 1, len(self.r) + 1))
        h[:-1, :-1], h[:-1, -1], h[-1, :-1] = self.G, q, q
        # Cancellation may produce a few negative ulps when all m are equal.
        h[-1, -1] = max(float(zz), 0.0)
        rhs = np.r_[self.r, self.z0y - mbar * self.r.sum()]
        return h / self.count, rhs / self.count


def smoothness_matrix(blocks, steps, lambda_s, lambda_b):
    """Laplacian of first differences along the two separate axes."""
    cells = blocks * steps
    result = np.zeros((cells + 1, cells + 1))
    grid = np.arange(cells).reshape(blocks, steps)
    for left, right, strength in ((grid[:, :-1], grid[:, 1:], lambda_s),
                                  (grid[:-1, :], grid[1:, :], lambda_b)):
        for a, b in zip(left.flat, right.flat):
            result[a, a] += strength
            result[b, b] += strength
            result[a, b] -= strength
            result[b, a] -= strength
    return result


def solve_wls(stats, blocks, steps, lambda_s, lambda_b):
    """Solve Eq. (15); every W cell is free subject only to W >= 0."""
    from scipy.optimize import minimize

    h, rhs = stats.normal_equations()
    h += smoothness_matrix(blocks, steps, lambda_s, lambda_b)
    # Diagonal scaling improves conditioning of the signed prompt coefficient.
    diagonal = np.diag(h)
    scale = 1 / np.sqrt(np.maximum(diagonal, 1e-12))
    scaled_h = h * scale[:, None] * scale[None, :]
    scaled_rhs = rhs * scale
    initial = np.r_[np.full(blocks * steps, stats.denoising_ms / stats.sum_t), 0.0] / scale
    bounds = [(0, None)] * (blocks * steps) + [(None, None)]
    if diagonal[-1] <= 1e-12:
        # Constant prompt lengths do not identify delta; choose zero explicitly.
        bounds[-1] = (0, 0)

    def objective(x):
        residual = scaled_h @ x - scaled_rhs
        return float(x @ (residual - scaled_rhs)), 2 * residual

    result = minimize(objective, initial, jac=True, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-9, "maxls": 50})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"full WLS fit failed: {result.message}")
    solution = result.x * scale
    return solution[:-1].reshape(blocks, steps), float(solution[-1])


def validation_loss(stats, w, delta, mbar):
    h, rhs = stats.normal_equations(mbar)
    x = np.r_[w.ravel(), delta]
    return max(0.0, float(x @ h @ x - 2 * rhs @ x + stats.y2 / stats.count))


@dataclass
class FitState:
    identity: dict
    config: dict
    train: SufficientStats
    validation: SufficientStats

    @classmethod
    def empty(cls, identity, config=None):
        keys = identity["trajectory_keys"]
        cells = keys["block_size"] * keys["max_output_blocks"]
        return cls(json.loads(canonical(identity)), fit_config(config),
                   SufficientStats.empty(cells), SufficientStats.empty(cells))

    @property
    def total(self):
        return self.train.merge(self.validation)

    def updated(self, rows):
        keys = self.identity["trajectory_keys"]
        d, j = keys["block_size"], keys["max_output_blocks"]
        actual_probe_steps(rows, d, j)  # Also detect duplicates across the split.
        offset = self.train.count + self.validation.count
        period = self.config["validation_every"]
        training, validation = [], []
        for i, row in enumerate(rows):
            (validation if (offset + i + 1) % period == 0 else training).append(row)
        return type(self)(self.identity, self.config,
                          self.train.merge(SufficientStats.from_rows(training, d, j)),
                          self.validation.merge(SufficientStats.from_rows(validation, d, j)))

    def arrays(self):
        meta = {"version": 1, "identity": self.identity, "config": self.config}
        arrays = {}
        for name in ("train", "validation"):
            stats = getattr(self, name)
            meta[name] = {}
            for f in fields(stats):
                value = getattr(stats, f.name)
                if isinstance(value, np.ndarray):
                    arrays[name + "_" + f.name] = value
                else:
                    meta[name][f.name] = value
        arrays["metadata"] = np.frombuffer(canonical(meta), dtype=np.uint8)
        return arrays

    @property
    def digest(self):
        digest = hashlib.sha256()
        for name, array in sorted(self.arrays().items()):
            digest.update(name.encode())
            digest.update(np.asarray(array, dtype="u1" if name == "metadata" else "<f8").tobytes())
        return digest.hexdigest()

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].tobytes())
            if meta.pop("version") != 1:
                raise ValueError("unsupported WLS statistics version")
            stats = {}
            cells = meta["identity"]["trajectory_keys"]["block_size"] * meta["identity"]["trajectory_keys"]["max_output_blocks"]
            for name in ("train", "validation"):
                arrays = {key: archive[name + "_" + key].copy() for key in ("G", "r", "q")}
                if (arrays["G"].shape != (cells, cells) or arrays["r"].shape != (cells,)
                        or arrays["q"].shape != (cells,) or any(not np.isfinite(a).all() for a in arrays.values())):
                    raise ValueError("invalid WLS statistics arrays")
                stats[name] = SufficientStats(**arrays, **meta[name])
            return cls(meta["identity"], fit_config(meta["config"]), **stats)


def fit_cost_profile(profile, rows, cost, state=None):
    """Merge new traces, select smoothing on held-out data, refit all data.

    Schema-5 normalization is only a storage convention: g=mean(W), a=0.
    The optimizer never uses the old W_bar or restricts W to an affine family.
    """
    import time
    from sglang.srt.dllm.dws_wsl import DWSWSLProfile, seal_profile

    if not rows:
        raise ValueError("empty profiling batch")
    identity = deployment_identity(profile, cost)
    config = fit_config(profile.payload["shape"].get("fit_config"))
    changed_deployment = state is not None and state.identity != identity
    if changed_deployment:
        state = None  # A changed deployment starts fresh (Appendix E).
    if state is None:
        if profile.is_full_fit and profile.payload["scale"]["cost_keys"] == cost and not changed_deployment:
            raise ValueError("missing sufficient statistics for this profile; retain its .npz sidecar")
        state = FitState.empty(identity, config)
    elif state.config != config:
        raise ValueError("cannot change validation split/candidates while retaining profiling state")
    elif profile.is_full_fit and state.digest != profile.payload["shape"]["fit"]["statistics_id"]:
        raise ValueError("profile and sufficient statistics do not match")
    state = state.updated(rows)
    if state.train.count < 2 or state.validation.count < 1:
        raise ValueError("insufficient profiling traces: need training and held-out requests (default: >= 5 total)")
    d, j = profile.block_size, profile.max_output_blocks
    trials = []
    for ls in config["lambda_s_candidates"]:
        for lb in config["lambda_b_candidates"]:
            w, delta = solve_wls(state.train, j, d, ls, lb)
            trials.append({"lambda_s": ls, "lambda_b": lb,
                           "validation_mse": validation_loss(state.validation, w, delta, state.train.mbar)})
    selected = min(trials, key=lambda row: row["validation_mse"])
    total = state.total
    if total.prefill_blocks <= 0:
        raise ValueError("profiling set has no prefill measurements")
    w, delta = solve_wls(total, j, d, selected["lambda_s"], selected["lambda_b"])
    gain = float(w.mean())
    if gain <= 0:
        raise ValueError("full WLS fit has no positive baseline cost")
    shape = {**profile.payload["shape"], "shape_origin": "full_wls",
             "W_bar": (w / gain).tolist(), "delta_bar": delta / gain, "m_bar": total.mbar,
             # The emergency depth-only fallback uses observed mean step cost;
             # column means of W would mix the request and block dimensions.
             "fallback_depth_bar": [total.denoising_ms / total.sum_t / gain] * d,
             "support": {"m_lo": total.prompt_min // d, "m_hi": total.prompt_max // d,
                         "prompt_tokens_min": total.prompt_min, "prompt_tokens_max": total.prompt_max,
                         "max_new_tokens": profile.max_new_tokens},
             "fit_config": config,
             "fit": {"objective": "appendix_e_v1", "statistics_id": state.digest,
                     "requests": total.count, "training_requests": state.train.count,
                     "validation_requests": state.validation.count, **selected, "trials": trials}}
    scale = {"cost_keys": cost, "g": gain, "a": 0.0,
             "alpha_prefill_block_ms": total.prefill_ms / total.prefill_blocks,
             "beta_refresh_block_ms": total.refresh_ms / total.output_blocks,
             "probe_state": "measured", "measured_at": time.time(),
             "probe_stats": {"requests": len(rows), "accumulated_requests": total.count,
                             "weighted_mse": validation_loss(total, w, delta, total.mbar),
                             "gpu_seconds": sum(r["prefill_gpu_ms"] + r["denoising_gpu_ms"] + r["refresh_gpu_ms"] for r in rows) / 1000}}
    updated = DWSWSLProfile.from_mapping(seal_profile({**profile.payload, "shape": shape, "scale": scale}))
    return updated, state


def main(argv=None):
    """Offline equivalent of the serving fitter, using request-summary JSONL."""
    import argparse
    from pathlib import Path
    from sglang.srt.dllm.cost_fit_cache import read_statistics, save_statistics
    from sglang.srt.dllm.dws_wsl import copy_probe_sidecar, load_dws_wsl_profile
    from sglang.srt.dllm.profile_common import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Schema-5 template or previous full WLS profile")
    parser.add_argument("--traces", required=True, help="JSONL request summaries, measured individually")
    parser.add_argument("--output", required=True, help="Output profile JSON; statistics are saved alongside it")
    parser.add_argument("--cost-keys", help="JSON deployment cost keys; a changed signature starts fresh")
    args = parser.parse_args(argv)
    profile = load_dws_wsl_profile(args.profile)
    if not profile.is_v5:
        parser.error("online/offline refresh requires a schema-5 template; schema 4 is read-only")
    cost = json.loads(Path(args.cost_keys).read_text()) if args.cost_keys else profile.payload["scale"]["cost_keys"]
    rows = [json.loads(line) for line in Path(args.traces).read_text().splitlines() if line.strip()]
    state = (read_statistics(profile, profile_path=args.profile)
             if profile.is_full_fit and profile.payload["scale"]["cost_keys"] == cost else None)
    fitted, state = fit_cost_profile(profile, rows, cost, state)
    output = Path(args.output)
    save_statistics(fitted, state, output.parent)
    copy_probe_sidecar(fitted, args.profile, output.parent)
    atomic_json(output, fitted.to_dict())
    print(f"Saved {fitted.profile_id}: {state.total.count} accumulated requests to {output}")


if __name__ == "__main__":
    main()
