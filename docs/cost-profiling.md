# Cost profiling and refresh (Appendix E)

The serving coordinator in `my_code/probe_scale.py` now fits every entry of `W`
and the signed coefficient `delta`. It no longer fits a gain/offset against a
fixed cost matrix, and idle refresh no longer blends old and new parameters.
The DWS predictor is unchanged.

## Objective and retained state

Each successfully completed probe is measured individually with CUDA events.
The existing probe barrier isolates it from serving batches. Its natural block
depths define `A[b,s] = 1{s < depth[b]}` (zero-based indexing), `T = sum(A)`,
and `m = floor(prompt_tokens / block_size)`.

The fitter implements:

```text
alpha = sum(prefill_gpu_ms) / sum(m)
beta  = sum(refresh_gpu_ms) / sum(output_block_count)
mbar  = sum(m*T) / sum(T)

minimize over W >= 0 and unrestricted delta:
    mean((T + 10)^(-2) * (<W,A> + delta*(m-mbar)*T - denoising_gpu_ms)^2)
    + lambda_s * sum(diff(W, axis=step)^2)
    + lambda_b * sum(diff(W, axis=block)^2)
```

`cost_fit.py` retains `G`, `r`, `q`, `sum(weight*z0^2)`,
`sum(weight*z0*y)`, `sum(weight*y^2)`, phase sums, prompt support, and request/step
counts, with `z0=m*T`. It reconstructs the centered normal equations using
`T = ones.T @ vec(A)`. Thus changing `mbar` after a merge requires no historical
trace replay. Statistics consume `O((blocks*steps)^2)` space; historical request
content and token IDs are not retained in the statistics file.

Separate training and validation statistics support held-out selection of
`lambda_s` and `lambda_b`; the selected pair is then used to fit the union.
By default every fifth successful trace enters validation, and each smoothing
parameter is selected from `[1e-6, 1e-4, 1e-2]`. These are implementation defaults,
not values specified by the paper. Override them in `shape.fit_config` before
the first fit:

```json
{
  "lambda_s_candidates": [0.000001, 0.0001, 0.01],
  "lambda_b_candidates": [0.000001, 0.0001, 0.01],
  "validation_every": 5
}
```

Changing a sealed profile requires recomputing its IDs with
`dws_wsl.seal_profile`. A retained statistics state must keep its original split
and candidate configuration. The initial default fit requires at least five
successful probes, including at least one request with prefill work. This is a
minimum for the split, not a guarantee of sufficient coverage: use diverse
prompt lengths, output-block counts, and denoising depths. Sparse coverage is
handled by smoothing, and validation scores are recorded in `shape.fit`.

## Online use and migration

Use the existing serving arguments:

```text
--my-cost-probe auto --my-cost-probe-refresh idle
```

- `auto` reuses a matching full WLS profile/cache; a legacy schema-5 profile
  triggers a new full fit using its probe suite. A legacy matrix is only a
  container for configuration and probe inputs; its relative cell costs are
  not used as a fitting constraint or initialization.
- `force` collects another probe batch and refits. Matching full-fit statistics
  are merged rather than discarded.
- `off` explicitly preserves the old behavior of using the supplied profile,
  including a legacy or stale profile. Such a profile is not a new Appendix E fit.
- `--my-cost-probe-refresh idle` merges further probe statistics and refits all
  coefficients. Existing idle/traffic checks and the 0.5% probe GPU budget remain
  in effect. A cancelled probe window does not advance the active profile.
- `--my-cost-probe-prompts suite.jsonl` supplies an alternative probe suite;
  the actual suite is embedded in the resulting online profile for reuse.

Schema-4 profiles remain readable for historical runs, but do not participate
in this online coordinator. Use a schema-5 template with the matching trajectory
keys, CUDA timing convention, and probe suite for the full-fit path. The serving
coordinator currently requires one data-parallel worker and one tokenizer worker;
tensor parallelism is supported by the existing publication barrier.

For compatibility with the existing schema-5 container, full fits carry
`shape_origin: "full_wls"`, store `g = mean(W)`, `W_bar = W/g`,
`delta_bar = delta/g`, and `a = 0`. This normalization is a serialization
convention. `W_bar` and `delta_bar` are regenerated on every fit, and `shape_id`
can change. The new runtime uses the fitted linear kernel
`K(m) = W + delta*(m-mbar)` directly: it does not clamp the prompt length to the
observed support or clip individual kernel cells. Equation (15) constrains `W`
alone, so it does not guarantee nonnegative `K` for every prompt length.
Historical `native`/`converted_schema4` profiles keep their previous evaluation
rules until they are refitted.

Publication validates the deployment and trajectory, archives the new profile
and statistics, then switches all TP ranks. Request cost caches bind the full
profile identity, so a changed matrix invalidates old scores. Publication
failure leaves the active profile unchanged.

## Persistence and deployment changes

Set `DWS_PROFILE_CACHE` to a persistent directory before starting the server.
Use separate directories for independent server instances; this local cache is
not a distributed statistics aggregator. The default is `/tmp/dllm_wls_cache`.
It contains:

- `profile-<deployment-hash>.json`: latest published full profile for the exact
  trajectory and hardware/software cost signature;
- `wls-stats-<statistics-digest>.npz`: immutable NumPy statistics with a digest
  bound into the profile;
- `README.md`: artifact format and provenance notes.

The scheduler also writes a portable snapshot to
`/tmp/dllm_cost_profiles_<port>/profile.json`, with its statistics sidecar.
Preserve both JSON and NPZ when copying a full profile. A missing or mismatched
sidecar fails a refresh instead of silently forgetting historical measurements.
Any probe-suite sidecar referenced by an offline profile must also be retained.
No raw historical profiling traces are required to resume.

Profiles are keyed by both trajectory and deployment keys, including model
identity, decoding configuration, GPU type, TP size, dtype, attention backend,
and software versions. A new identity starts fresh statistics. The old
`/tmp/dllm_scale_cache` is not used by the new full fitter. Cached full profiles
have no automatic expiry; use `force` or idle refresh to update timings.

## Offline fit or refresh

In the environment where SGLang is installed, run:

```bash
/path/to/conda/env/bin/python -m sglang.srt.dllm.cost_fit \
  --profile /path/to/template-or-previous-profile.json \
  --traces /path/to/new-request-summaries.jsonl \
  --output /path/to/output/profile.json
```

The command writes the fitted profile, its statistics, and an artifact README.
For a new hardware/software deployment, supply `--cost-keys /path/to/cost-keys.json`
with the actual target signature. The trace measurements must come from that
deployment. Each JSONL row contains:

```json
{"rid":"probe-1","status":"success","prompt_tokens":64,"prefill_round_count":2,"prefill_gpu_ms":3.2,"output_block_steps":[5,3],"output_block_count":2,"denoising_gpu_ms":12.8,"refresh_gpu_ms":2.1}
```

This example uses block size 32; a real initial fit needs multiple distinct
completed requests. Prefill must be zero when there are no full prompt blocks;
all measured denoising/refresh durations must be positive. Failed requests,
incomplete phase coverage, forced calibration traces, and duplicate request IDs
within an update are rejected. Submit each successful batch once.

## Validation

The CPU tests in `tests/dllm` compare the fitter with an explicit request-design
bounded least-squares solve, verify both smoothness axes, recover a non-affine
cost matrix, compare incremental and complete refits, and exercise cache and
publication failure paths. They import the pure modules without initializing
the CUDA serving stack. They do not replace a real GPU/TP serving validation.
