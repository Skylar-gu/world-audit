# worldaudit — invariants

One work package per session; finish each by running its acceptance test; commit only green.
Human review required on `oracle/`, `contrast/`, `metrics/` diffs.

1. **Oracle null test (sacred):** snapshot mid-trajectory (after ≥100 steps, so warm-start is
   non-trivial), restore, re-step N=1000 — `np.array_equal` (bitwise, float64) on qpos/qvel vs.
   the original continuation. Runs in CI on CPU. A second variant cross-checks `mujoco.rollout`
   against raw `mj_step` for the same seed/state.
2. **Contrast null test:** zero-magnitude poke ⇒ truth contrast exactly zero; model contrast
   exactly zero under CRN. Bitwise, both.
3. All randomness flows through explicit seeds in `schema`; no bare `np.random` / global torch
   seeding anywhere.
4. Artifacts are immutable and manifest-stamped (git SHA, lib versions, scene hash, config hash);
   regeneration goes to a new content-addressed path.
5. Unit tests never load GPUs or take >2 min; anything heavier lives behind `make integ`.
6. Rendering must never introduce state: `render.py` consumes `RolloutPair`s, it never steps
   physics.
7. No claim strings in site copy beyond the honesty perimeter in the v2 plan; copy lives in one
   `site/copy.md` file for single-point review.

## State handling (verified against MuJoCo 3.10.0, pinned in requirements.lock)

- All snapshots use `mj_stateSize / mj_getState / mj_setState` with `mjtState.mjSTATE_INTEGRATION`.
  Never snapshot qpos/qvel alone. Warm-start restoration is required for bitwise reproducibility.
- `mujoco.rollout` takes `initial_state` as `mjSTATE_FULLPHYSICS`, warm-start via
  `initial_warmstart`, and per-step user inputs via `control` + `control_spec` (bits within
  `mjSTATE_USER`). Pokes are injected through `control_spec = mjSTATE_CTRL | mjSTATE_XFRC_APPLIED`,
  never baked into the init snapshot.
- Audit scene XML: `solver="Newton" tolerance="0"` (fixed iteration count), fixed timestep,
  no stochastic elements.

## Environment

- Use `.venv/bin/python` (Python 3.11). `make test` = CPU unit tests; `make integ` = GPU/long.
