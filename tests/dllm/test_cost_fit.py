import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import lsq_linear

from sglang.srt.dllm.cost_fit import (
    SufficientStats, fit_cost_profile,
    smoothness_matrix, solve_wls, validation_loss,
)
from sglang.srt.dllm.cost_fit_cache import (
    cache_path, choose_profile, read_profile_cache, read_statistics,
    save_statistics, statistics_filename, write_profile_cache,
)
from sglang.srt.dllm.dws_wsl import (
    COST_FIELDS, TRAJECTORY_FIELDS, DWSWSLProfile, evaluate_dws_wsl, seal_profile,
)


@pytest.fixture
def profile():
    keys = {key: "test" for key in TRAJECTORY_FIELDS}
    keys.update(block_size=3, max_output_blocks=2, max_new_tokens=6, threshold=.95)
    cost = {key: "test" for key in COST_FIELDS}
    cost.update(tp_size=1, disable_cuda_graph=True)
    return DWSWSLProfile.from_mapping(seal_profile({
        "schema_version": 5, "model_type": "dws-wsl", "feature_version": "dws_wsl_shape_scale_v1",
        "status": "ready", "shape": {
            "trajectory_keys": keys, "W_bar": [[1.] * 3] * 2, "delta_bar": 0., "m_bar": 2.,
            "shape_origin": "native", "fallback_depth_bar": [1.] * 3,
            "support": {"m_lo": 1, "m_hi": 5, "prompt_tokens_min": 3,
                        "prompt_tokens_max": 15, "max_new_tokens": 6},
            "probe_suite": {"version": 1, "requests": []},
            "timing": {"prefill_source": "cuda_event", "denoising_target": "forward_plus_postprocess"},
            "fit_config": {"lambda_s_candidates": [0.], "lambda_b_candidates": [0.], "validation_every": 5}},
        "scale": {"cost_keys": cost, "g": 10., "a": 0., "alpha_prefill_block_ms": 10.,
                  "beta_refresh_block_ms": 10., "probe_state": "inherited"}}))


@pytest.fixture
def rows():
    rng = np.random.default_rng(531)
    true_w = np.array([[.8, 1.4, 2.0], [2.8, 1., 3.5]])
    result = []
    for i in range(100):
        m = int(rng.integers(1, 6))
        depths = rng.integers(1, 4, size=int(rng.integers(1, 3))).tolist()
        result.append({"rid": str(i), "status": "success", "prompt_tokens": 3*m,
                       "prefill_round_count": m, "prefill_gpu_ms": 2.5*m,
                       "output_block_steps": depths, "output_block_count": len(depths),
                       "refresh_gpu_ms": 4.*len(depths)})
    center = sum(r["prefill_round_count"] * sum(r["output_block_steps"]) for r in result) / sum(sum(r["output_block_steps"]) for r in result)
    for row in result:
        row["denoising_gpu_ms"] = float(sum(true_w[b, :s].sum() for b, s in enumerate(row["output_block_steps"]))
                                        + .2*(row["prefill_round_count"]-center)*sum(row["output_block_steps"]))
    return result


def explicit_design(rows, center):
    a = np.zeros((len(rows), 2, 3))
    for i, row in enumerate(rows):
        for b, depth in enumerate(row["output_block_steps"]):
            a[i, b, :depth] = 1
    a = a.reshape(len(rows), -1)
    t = a.sum(1)
    m = np.array([r["prompt_tokens"] // 3 for r in rows])
    return np.column_stack([a, (m-center)*t]), (t+10)**-2, np.array([r["denoising_gpu_ms"] for r in rows])


def test_statistics_centering_and_merging_match_raw_requests(rows):
    full = SufficientStats.from_rows(rows, 3, 2)
    merged = SufficientStats.from_rows(rows[:23], 3, 2).merge(SufficientStats.from_rows(rows[23:], 3, 2))
    for name in vars(full):
        assert_allclose(getattr(merged, name), getattr(full, name), atol=1e-11)
    for center in (full.mbar, .3, 12.):
        x, weights, y = explicit_design(rows, center)
        h, rhs = merged.normal_equations(center)
        assert_allclose(h, x.T @ (weights[:, None]*x)/len(rows), atol=1e-12)
        assert_allclose(rhs, x.T @ (weights*y)/len(rows), atol=1e-12)


def test_two_axis_smoothness():
    w = np.array([[1., 2., 4.], [7., 3., 2.]])
    x = np.r_[w.ravel(), -123.]
    penalty = smoothness_matrix(2, 3, .7, 1.3)
    assert_allclose(x @ penalty @ x, .7*(np.diff(w, axis=1)**2).sum() + 1.3*(np.diff(w, axis=0)**2).sum())


@pytest.mark.parametrize("lambda_s,lambda_b", [(0., 0.), (.002, .01), (.1, .2)])
def test_solver_matches_explicit_bounded_least_squares(rows, lambda_s, lambda_b):
    stats = SufficientStats.from_rows(rows, 3, 2)
    x, weight, y = explicit_design(rows, stats.mbar)
    difference_rows = []
    grid = np.arange(6).reshape(2, 3)
    for left, right, strength in ((grid[:, :-1], grid[:, 1:], lambda_s), (grid[:-1], grid[1:], lambda_b)):
        for a, b in zip(left.flat, right.flat):
            row = np.zeros(7)
            row[a], row[b] = np.sqrt(strength), -np.sqrt(strength)
            difference_rows.append(row)
    design = np.vstack([x*np.sqrt(weight[:, None]/len(rows)), difference_rows])
    target = np.r_[y*np.sqrt(weight/len(rows)), np.zeros(len(difference_rows))]
    reference = lsq_linear(design, target, bounds=(np.r_[np.zeros(6), -np.inf], np.full(7, np.inf)), tol=1e-12)
    w, delta = solve_wls(stats, 2, 3, lambda_s, lambda_b)
    assert_allclose(np.r_[w.ravel(), delta], reference.x, atol=2e-6, rtol=2e-6)


def test_full_fit_learns_non_affine_matrix(profile, rows):
    fitted, state = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    assert fitted.is_full_fit
    assert fitted.shape_id != profile.shape_id
    # An affine transform of the all-ones template could only be constant.
    assert_allclose(fitted.denoising_cell_ms, [[.8, 1.4, 2.], [2.8, 1., 3.5]], atol=2e-6)
    assert_allclose(fitted.delta, .2, atol=2e-6)
    assert fitted.prefill_block_ms == 2.5
    assert fitted.refresh_block_ms == 4.
    assert fitted.payload["scale"]["a"] == 0.
    assert state.total.count == len(rows)


def test_incremental_refresh_matches_full_refit(profile, rows):
    cost = profile.payload["scale"]["cost_keys"]
    first, state = fit_cost_profile(profile, rows[:37], cost)
    updated, state = fit_cost_profile(first, rows[37:], cost, state)
    full, full_state = fit_cost_profile(profile, rows, cost)
    assert_allclose(updated.denoising_cell_ms, full.denoising_cell_ms, atol=2e-6)
    assert_allclose([updated.delta, updated.mbar, updated.prefill_block_ms, updated.refresh_block_ms],
                    [full.delta, full.mbar, full.prefill_block_ms, full.refresh_block_ms], atol=2e-6)
    assert state.train.count == full_state.train.count == 80
    assert state.validation.count == 20
    assert updated.mbar != first.mbar


def test_regularization_selected_on_holdout_then_refit_all(profile, rows):
    payload = profile.to_dict()
    payload["shape"]["fit_config"].update(lambda_s_candidates=[0., .002], lambda_b_candidates=[0., .003])
    profile = DWSWSLProfile.from_mapping(seal_profile(payload))
    fitted, state = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    trials = fitted.payload["shape"]["fit"]["trials"]
    for trial in trials:
        w, delta = solve_wls(state.train, 2, 3, trial["lambda_s"], trial["lambda_b"])
        assert_allclose(trial["validation_mse"], validation_loss(state.validation, w, delta, state.train.mbar))
    selected = min(trials, key=lambda r: r["validation_mse"])
    w, delta = solve_wls(state.total, 2, 3, selected["lambda_s"], selected["lambda_b"])
    assert_allclose(fitted.denoising_cell_ms, w)
    assert fitted.delta == pytest.approx(delta)


def test_signed_delta_and_nonnegative_w(profile, rows):
    # Add a strongly negative cell to the generating model, keeping durations
    # positive; the constrained optimum must put some baseline cells at zero.
    modified = []
    for row in rows:
        depth = row["output_block_steps"]
        y = 8. + sum(depth)*1.5 - (6. if depth[0] >= 2 else 0.) - .1*row["prefill_round_count"]*sum(depth)
        modified.append({**row, "denoising_gpu_ms": y})
    fitted, _ = fit_cost_profile(profile, modified, profile.payload["scale"]["cost_keys"])
    assert np.min(fitted.denoising_cell_ms) == pytest.approx(0., abs=1e-9)
    assert fitted.delta < 0


def test_constant_prompt_length_is_identifiable_without_delta(profile, rows):
    rows = [{**r, "prompt_tokens": 6, "prefill_round_count": 2, "prefill_gpu_ms": 5.} for r in rows]
    fitted, _ = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    assert fitted.delta == 0.


def test_runtime_uses_unclipped_paper_kernel(profile, rows):
    fitted, _ = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    payload = fitted.to_dict()
    payload["shape"]["delta_bar"] = -10. / payload["scale"]["g"]
    fitted = DWSWSLProfile.from_mapping(seal_profile(payload))
    # Both prompt extrapolation and negative individual K cells remain linear.
    kernel, count = fitted.kernel(300)
    assert_allclose(kernel, np.array(fitted.denoising_cell_ms) + fitted.delta*(100-fitted.mbar))
    assert np.min(kernel) < 0
    assert count == 0


def test_runtime_cost_matches_fitted_request_equation(profile, rows):
    fitted, _ = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    for row in rows[::11]:
        surface = np.zeros((2, 3))
        for b, depth in enumerate(row["output_block_steps"]):
            surface[b, :depth] = 1.
        prediction = {"dws_contract_version": 2, "dws_max_new_tokens": 6,
                      "dws_surface": surface.tolist(), "expected_n_blocks": float(surface[:, 0].sum()),
                      "workload_curve": surface.sum(0).cumsum().tolist()}
        result = evaluate_dws_wsl(prediction, prompt_tokens=row["prompt_tokens"], include_prefill=True,
                                  profile=fitted, max_new_tokens=6)
        assert result.fallback_reason is None
        assert result.denoising_ms == pytest.approx(row["denoising_gpu_ms"], abs=1e-5)
        assert result.total_ms == pytest.approx(row["denoising_gpu_ms"] + row["prefill_gpu_ms"] + row["refresh_gpu_ms"], abs=1e-5)


@pytest.mark.parametrize("field", ["gpu_model", "tp_size"])
def test_deployment_change_starts_fresh_statistics(profile, rows, field):
    cost = profile.payload["scale"]["cost_keys"]
    fitted, state = fit_cost_profile(profile, rows, cost)
    new_cost = {**cost, field: "changed"}
    updated, new_state = fit_cost_profile(fitted, rows[:20], new_cost, state)
    assert new_state.total.count == 20
    assert updated.payload["scale"]["cost_keys"] == new_cost


def test_model_change_starts_fresh_statistics(profile, rows):
    cost = profile.payload["scale"]["cost_keys"]
    fitted, state = fit_cost_profile(profile, rows, cost)
    payload = fitted.to_dict()
    payload["shape"]["trajectory_keys"]["model_revision"] = "new-revision"
    changed = DWSWSLProfile.from_mapping(seal_profile(payload))
    _, state = fit_cost_profile(changed, rows[:20], cost, state)
    assert state.total.count == 20


def test_statistics_and_cache_roundtrip_and_tamper_detection(profile, rows, tmp_path):
    cost = profile.payload["scale"]["cost_keys"]
    fitted, state = fit_cost_profile(profile, rows, cost)
    write_profile_cache(fitted, state, root=tmp_path)
    assert cache_path(profile, cost, tmp_path).is_file()
    restored = read_statistics(fitted, root=tmp_path)
    assert restored.digest == state.digest
    assert read_profile_cache(profile, cost, tmp_path).profile_id == fitted.profile_id
    selected = choose_profile(profile, cost, cache_root=tmp_path)
    assert selected.shape_id == fitted.shape_id
    assert selected.payload["scale"]["probe_state"] == "reused"
    assert choose_profile(profile, {**cost, "gpu_model": "another"}, cache_root=tmp_path) is None
    assert choose_profile(profile, cost, "force", tmp_path) is None
    assert choose_profile(profile, cost, "off", tmp_path) is profile
    restored.train.r[0] += 1
    with (tmp_path / statistics_filename(fitted)).open("wb") as handle:
        np.savez_compressed(handle, **restored.arrays())
    with pytest.raises(ValueError, match="digest"):
        read_statistics(fitted, root=tmp_path)
    assert read_profile_cache(profile, cost, tmp_path) is None


def test_missing_statistics_cannot_silently_discard_history(profile, rows, tmp_path):
    cost = profile.payload["scale"]["cost_keys"]
    fitted, _ = fit_cost_profile(profile, rows, cost)
    with pytest.raises(ValueError, match="missing sufficient statistics"):
        fit_cost_profile(fitted, rows, cost)
    with pytest.raises(ValueError, match="missing WLS statistics"):
        read_statistics(fitted, root=tmp_path)


@pytest.mark.parametrize("changes", [
    {"prefill_round_count": 1000}, {"denoising_gpu_ms": float("nan")},
    {"refresh_gpu_ms": 0.}, {"forced_single_commit": True},
    {"output_block_steps": [4]}, {"status": "aborted"},
])
def test_incomplete_or_invalid_traces_rejected(rows, changes):
    with pytest.raises(ValueError):
        SufficientStats.from_rows([{**rows[0], **changes}], 3, 2)


def test_insufficient_holdout_and_duplicate_requests_rejected(profile, rows):
    cost = profile.payload["scale"]["cost_keys"]
    with pytest.raises(ValueError, match="held-out"):
        fit_cost_profile(profile, rows[:4], cost)
    with pytest.raises(ValueError, match="unique"):
        fit_cost_profile(profile, [rows[0]]*5, cost)


def test_publication_accepts_changed_matrix_and_rejects_wrong_deployment(profile, rows, monkeypatch, tmp_path):
    from sglang.srt.dllm.mixin import cost_probe
    fitted, state = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    scheduler = cost_probe.CostProbeMixin()
    scheduler.tp_group = SimpleNamespace(world_size=1)
    scheduler.dllm_config = SimpleNamespace(block_size=3, my_cost_profile=profile)
    scheduler.server_args = SimpleNamespace(my_sjf_cost_profile="", port=9876)
    scheduler._my_cost_shape_signature = {
        "trajectory_keys": profile.payload["shape"]["trajectory_keys"],
        "cost_keys": profile.payload["scale"]["cost_keys"],
        "disable_radix_cache": True, "prefill_source": "cuda_event"}
    scheduler._cost_probe_requests = lambda **kwargs: []
    monkeypatch.setattr(cost_probe, "copy_statistics_sidecar", lambda *args: save_statistics(fitted, state, tmp_path))
    monkeypatch.setattr(cost_probe, "atomic_json", lambda path, payload: (tmp_path / "snapshot.json").write_text(json.dumps(payload)))
    scheduler._cost_probe_publish(fitted.to_dict())
    assert scheduler.dllm_config.my_cost_profile.profile_id == fitted.profile_id
    changed = fitted.to_dict()
    changed["scale"]["cost_keys"]["gpu_model"] = "wrong GPU"
    with pytest.raises(ValueError, match="deployment mismatch"):
        scheduler._cost_probe_publish(seal_profile(changed))
    assert scheduler.dllm_config.my_cost_profile.profile_id == fitted.profile_id
    scheduler._cost_probe_requests = lambda **kwargs: [SimpleNamespace(my_cost_probe=False)]
    with pytest.raises(ValueError, match="real requests"):
        scheduler._cost_probe_publish(fitted.to_dict())
    # probe=off is the documented explicit escape hatch for stale profiles.
    scheduler._cost_probe_requests = lambda **kwargs: []
    scheduler.server_args.my_cost_probe = "off"
    scheduler._my_cost_shape_signature["cost_keys"] = {**profile.payload["scale"]["cost_keys"], "gpu_model": "another GPU"}
    scheduler._cost_probe_publish(fitted.to_dict())
    assert scheduler.dllm_config.my_cost_profile.profile_id == fitted.profile_id


def test_failed_snapshot_preserves_active_profile(profile, rows, monkeypatch):
    from sglang.srt.dllm.mixin import cost_probe
    fitted, _ = fit_cost_profile(profile, rows, profile.payload["scale"]["cost_keys"])
    scheduler = cost_probe.CostProbeMixin()
    scheduler.tp_group = SimpleNamespace(world_size=1)
    scheduler.dllm_config = SimpleNamespace(block_size=3, my_cost_profile=profile)
    scheduler.server_args = SimpleNamespace(my_sjf_cost_profile="", port=9876)
    scheduler._my_cost_shape_signature = {
        "trajectory_keys": profile.payload["shape"]["trajectory_keys"],
        "cost_keys": profile.payload["scale"]["cost_keys"],
        "disable_radix_cache": True, "prefill_source": "cuda_event"}
    scheduler._cost_probe_requests = lambda **kwargs: []
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(cost_probe, "copy_statistics_sidecar", fail)
    with pytest.raises(ValueError, match="old profile retained"):
        scheduler._cost_probe_publish(fitted.to_dict())
    assert scheduler.dllm_config.my_cost_profile is profile


def test_offline_command_refreshes_saved_statistics(profile, rows, tmp_path):
    from sglang.srt.dllm.cost_fit import main
    template, traces, output = (tmp_path / name for name in ("template.json", "traces.jsonl", "profile.json"))
    (tmp_path / "README.md").write_text("Existing dataset documentation.\n")
    template.write_text(json.dumps(profile.to_dict()))
    traces.write_text("\n".join(json.dumps(r) for r in rows[:50]))
    main(["--profile", str(template), "--traces", str(traces), "--output", str(output)])
    first = DWSWSLProfile.from_mapping(json.loads(output.read_text()))
    assert read_statistics(first, root=tmp_path).total.count == 50
    traces.write_text("\n".join(json.dumps(r) for r in rows[50:]))
    main(["--profile", str(output), "--traces", str(traces), "--output", str(output)])
    fitted = DWSWSLProfile.from_mapping(json.loads(output.read_text()))
    assert read_statistics(fitted, root=tmp_path).total.count == 100
    assert_allclose(fitted.denoising_cell_ms, [[.8, 1.4, 2.], [2.8, 1., 3.5]], atol=2e-6)
    readme = (tmp_path / "README.md").read_text()
    assert readme.startswith("Existing dataset documentation.\n")
    assert readme.count("<!-- dws-cost-profiles:start -->") == 1


@pytest.mark.parametrize("cancel_publish", [False, True])
def test_probe_coordinator_full_fit_and_cancelled_publication(profile, rows, monkeypatch, tmp_path, cancel_publish):
    from sglang.srt.dllm.my_code import probe_scale
    from sglang.srt.dllm import cost_fit_cache

    suite = [{"probe_id": str(i), "kind": "natural", "input_ids": [10]*row["prompt_tokens"],
              "max_new_tokens": 6, "ignore_eos": False} for i, row in enumerate(rows[:20])]
    client = object.__new__(probe_scale.ProbeClient)
    client.args = SimpleNamespace(my_sjf_cost_profile=str(tmp_path / "template.json"), my_cost_probe_timeout=0)
    cost = profile.payload["scale"]["cost_keys"]
    states = [{"dllm_cost_shape_signature": {"cost_keys": cost}, "dllm_cost_probe_records": [{}]}]
    commands = []
    published = []
    def control(action, **values):
        commands.append(action)
        if action == "publish":
            if cancel_publish:
                raise RuntimeError("probe window cancelled or token mismatch")
            published.append(DWSWSLProfile.from_mapping(values["profile"]))
        return states
    client.control = control
    client.states = lambda: states
    client.call = lambda *args, **kwargs: {"meta_info": {"finish_reason": {"type": "length"}}}
    monkeypatch.setattr(probe_scale, "summarize_probe", lambda records, probe, rid, d: {**rows[int(probe["probe_id"])], "rid": rid})
    monkeypatch.setattr(probe_scale, "newest_profile", lambda p, c: cost_fit_cache.newest_profile(p, c, tmp_path))
    monkeypatch.setattr(probe_scale, "save_statistics", lambda p, s: save_statistics(p, s, tmp_path))
    monkeypatch.setattr(probe_scale, "read_statistics", lambda p, **kwargs: read_statistics(p, root=tmp_path))
    monkeypatch.setattr(probe_scale, "write_profile_cache", lambda p, s, **kwargs: write_profile_cache(p, s, root=tmp_path))
    if cancel_publish:
        with pytest.raises(RuntimeError, match="cancelled"):
            client.run(profile, suite, idle=True)
        assert not published
        assert not cache_path(profile, cost, tmp_path).exists()
        assert commands[-1] == "cancel"
    else:
        first = client.run(profile, suite)
        assert first.is_full_fit
        assert read_statistics(first, root=tmp_path).total.count == 20
        refreshed = client.run(first, suite, idle=True)
        assert refreshed.shape_id != first.shape_id
        assert read_statistics(refreshed, root=tmp_path).total.count == 26
        assert read_profile_cache(profile, cost, tmp_path).profile_id == refreshed.profile_id
        assert refreshed.payload["shape"]["probe_suite"]["requests"] == suite
        assert commands[-1] == "end"
