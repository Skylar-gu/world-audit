# Implementation Spec — `worldaudit` (Physics Auditor Demo)

Audience: Claude Code. This is the build contract for the demo in `yc_demo_build_plan_v2_physics.md`. One work package per session; finish each by running its acceptance test; commit only green. Human review required on `oracle/`, `contrast/`, `metrics/` diffs.

## 0. Verified facts and pinned decisions (do not re-litigate; re-verify only if the pinned version changes)

- **State snapshot/restore (verified against MuJoCo docs):** the complete input to forward dynamics is `mjSTATE_INTEGRATION` (union of time, qpos, qvel, act, history, **warmstart**, ctrl, qfrc_applied, xfrc_applied, eq_active, mocap, userdata, plugin). Docs are explicit that warm-start restoration is required for perfect numerical reproducibility when loading a non-initial state, and that warm-start differences amplify exponentially. All snapshots use `mj_stateSize / mj_getState / mj_setState` with `mjtState.mjSTATE_INTEGRATION`. Never snapshot qpos/qvel alone.
- **Batch rollouts (verified):** use the official `mujoco.rollout` module (threaded C, open-loop rollouts from arbitrary initial states, explicit `initial_warmstart` support, per-step state + sensordata out). The oracle is a thin wrapper over it; do not hand-roll the stepping loop except in the single-trajectory null test, which deliberately uses raw `mj_step` to cross-check `rollout`.
- **Determinism aids (verified in docs):** solver `tolerance = 0` forces a fixed iteration count, removing early-termination numerical variation; set it in scene XML for the audit configuration. Pin the MuJoCo version in `requirements.lock` and record it in every artifact manifest (changelog shows warm-start semantics have moved between versions).
- **TD-MPC2 (verified):** 324 official checkpoints on HuggingFace (`nicklashansen/tdmpc2`); single-task default `model_size=5`; evaluable on an 8 GB GPU. **Caveat that shapes the design:** TD-MPC2 is an implicit, *decoder-free* latent world model — auditing it in state space requires a trained decoder probe (module `namebrand/`, gated, §7). Its checkpoints are per-task (DMControl etc.), so the name-brand audit runs on a DMControl task, not our custom scenes.
- **Unverified (Day-1 gate, generous cut candidate):** MuJoCo-in-browser WASM builds. Do not architect anything that depends on it; the site works entirely from precomputed assets.

## 1. Repository layout

```
worldaudit/
  CLAUDE.md                 # invariants below, verbatim
  Makefile                  # test / integ / grid / render / site / demo
  requirements.lock         # mujoco==<pin>, torch==<pin>, numpy==<pin>
  wa/
    schema.py               # dataclasses + JSON (de)serialization, the only source of types
    scenes/                 # billiards.xml, blocks.xml, arm.xml + scene registry
    oracle.py               # snapshot/restore + rollout wrapper + poke application
    data.py                 # training-data generation (random pokes + passive rollouts)
    models.py               # ensemble dynamics models + training loop
    contrast.py             # paired counterfactual computation (truth + model), CRN
    metrics.py              # sub-test scores; pure functions of contrast outputs
    render.py               # EGL offscreen rendering, side-by-side compositing, encode
    grid.py                 # orchestrates scene × model × poke × magnitude → artifacts
    namebrand/              # (gated) TD-MPC2 checkpoint loading + latent→state decoder probe
  site/                     # Vite+React static viewer; reads manifest.json only
  tests/                    # unit (CPU, <2 min) ; tests/integ (marked, GPU/long)
  experiments/manifests/    # one JSON per generated artifact batch
```

## 2. Core schemas (`wa/schema.py`)

```python
@dataclass
class Poke:                     # one intervention
    site: str                   # named body/site in the scene registry
    direction: np.ndarray       # unit vector, world frame, shape (3,)
    magnitude: float            # Newtons (force) — grid of 8 log-spaced values per scene
    t_start: int                # step index at which force turns on
    duration: int               # steps force is held (via xfrc_applied on the body)

@dataclass
class RolloutPair:              # the atomic audit object
    scene: str; model_id: str   # model_id == "oracle" for ground truth
    poke: Poke | None           # None == null rollout
    states: np.ndarray          # (T, nq+nv) tracked full state trajectory
    tracked: dict[str, np.ndarray]  # object-centric series: positions, velocities, momenta
    seed: int; manifest: dict   # git SHA, mujoco/torch versions, scene hash, config hash

# Contrast(s) = poked − null, computed per source (truth or a specific model), same seed set.
```

Site manifest (`site/public/manifest.json`): `scenes[] → models[] → subtests[] → entries[]`, each entry `{poke_id, magnitude, clip_url, curves: {x, y_truth, y_model, ci}, score, light: "green|yellow|red"}`. The site renders the manifest and nothing else — no computation client-side beyond plotting.

## 3. Oracle (`wa/oracle.py`) — the load-bearing module

- `snapshot(model, data) -> np.ndarray` / `restore(model, data, s)`: `mjSTATE_INTEGRATION`, exactly.
- `run(scene, init_state, ctrl_seq, poke) -> RolloutPair`: applies the poke by writing the world-frame force into `xfrc_applied[body_id, :3]` for `t ∈ [t_start, t_start+duration)`, zero otherwise; rolls via `mujoco.rollout` from `init_state` (which includes warm-start). Passive scenes use zero `ctrl_seq`.
- **Contract:** truth-side counterfactual pairs share the *identical* `init_state` byte-for-byte; the only difference between poked and null runs is `xfrc_applied` during the poke window. Note the subtlety: because `xfrc_applied` is itself part of `mjSTATE_INTEGRATION`, the init snapshot must be taken with the poke *not yet applied* and the poke injected through the control/rollout interface, never baked into the snapshot.
- Scene XML requirements: `option solver="Newton" tolerance="0"`, fixed `timestep`, no stochastic elements, named tracked bodies; scene registry maps scene → tracked bodies, poke sites, allowed magnitude range (calibrated so max magnitude is dramatic but non-degenerate — no bodies ejected from the arena).

## 4. Models (`wa/models.py`) — the audit subjects

State-space probabilistic ensembles (PETS-style), the minimum machinery that gives a quality spectrum plus principled stochasticity:
- Input `(s_t, a_t)` with `s = (qpos, qvel)` flattened (passive scenes: `a` omitted); target `Δs = s_{t+1} − s_t`; heads output Gaussian mean and diagonal log-variance; loss = NLL; inputs/targets normalized by training-set statistics (store the normalizer in the checkpoint).
- Ensemble of E=5 MLPs; spectrum per scene = capacities {2×64, 3×256, 4×512} × data budgets {5k, 100k transitions} → 6 models/scene. Training data from `wa/data.py`: random pokes across the grid range + passive rollouts, generated by the oracle (seeded).
- `rollout_model(model, init_state, ctrl_seq, poke, K, seed) -> list[RolloutPair]`: iterated one-step prediction (compounding error is the phenomenon under audit, not a bug); the poke enters as the model's action/force input channel — models are trained with the applied-force vector as an input so that "the same poke" is well-defined on the model side. K samples per rollout, each sample = one ensemble member drawn per trajectory + head-noise sequence from a `torch.Generator` seeded per (pair, k).
- **CRN contract:** the poked and null model rollouts for sample k use the *same* member assignment and the *same* head-noise sequence. This makes the k-th contrast a common-random-numbers estimate; the null contrast is then exactly zero by construction, which is the model-side null test.

## 5. Metrics (`wa/metrics.py`) — pure functions, formal definitions

Let $\Delta^{\text{truth}}(m)$ and $\Delta^{(k)}_{\text{model}}(m)$ denote poked-minus-null tracked-state contrasts at poke magnitude $m$ (truth is deterministic; model has K CRN samples).
- **Response curve / impulse response:** target-object displacement at horizon H, $r(m) = \lVert \Delta x_{\text{target}}(H; m) \rVert$. Score = normalized area between truth and model mean curves over the magnitude grid, $\frac{\int |r_M - r_T|\,dm}{\int r_T\,dm}$; report the K-sample band.
- **Momentum check (billiards):** total linear momentum $p(t) = \sum_i m_i v_i(t)$ over a window bracketing the first collision. Score compares the model's *change* in total momentum across the window to the truth's change — never to zero, since MuJoCo contacts with friction/solver softness are not exactly conservative; the truth trajectory is the reference, and the on-page copy must say "vs. simulated ground truth," not "vs. conservation law."
- **Propagation (chain in blocks/arm scene):** arrival time $t_i = \min\{t : \lVert \Delta x_i(t)\rVert > \varepsilon\}$ per link $i$; compare arrival-time profiles (slope = propagation speed). $\varepsilon$ fixed per scene in the registry, chosen ≥ 10× the null-test numerical floor.
- **Stability threshold (blocks):** topple indicator vs. magnitude; threshold $m^*$ by bisection on the grid; score $= |m^*_M - m^*_T| / m^*_T$, plus the disagreement region shaded on the curve.
- Traffic lights: green/yellow/red at configurable thresholds (defaults 0.1/0.3), calibrated once on the model spectrum in D5 and then frozen; thresholds live in one config file and are printed on the site's method page.

## 6. Rendering (`wa/render.py`) and grid (`wa/grid.py`)

- Offscreen EGL (`MUJOCO_GL=egl`), 720p, fixed camera per scene. Model-predicted trajectories are rendered by writing predicted `qpos` into a fresh `mjData` and calling `mj_forward` per frame *for rendering only* (kinematics, no stepping); label baked into the frame: "model-predicted states, simulator renderer."
- Composite side-by-side (truth left, model right), synchronized clocks, poke moment flashed; encode WebM+MP4, ~6 s clips, target <1.5 MB each. Budget check: 3 scenes × 7 models × ~6 pokes × 8 magnitudes ≈ 1000 clips ≈ 1–1.5 GB — fine for static hosting; grid.py writes the manifest incrementally so partial runs produce a servable site.
- `grid.py` is the only entry point that touches everything: `make grid SCENE=billiards` → oracle runs, model runs, metrics, clips, manifest rows, experiment manifest. Idempotent by content hash; re-running skips existing artifacts.

## 7. `namebrand/` (gated module — build only after core demo is green)

Load an official single-task TD-MPC2 checkpoint (`model_size=5`, a DMControl task with clean visuals, e.g., walker or cheetah). Because the model is decoder-free: (1) collect episodes with the checkpoint agent in the true env, logging $(o_t, z_t = \text{enc}(o_t))$; (2) train a ridge/MLP decoder $z \to (qpos, qvel)$, report held-out $R^2$ per coordinate on the method page — the decoder's own error is the audit's noise floor and must be displayed, not hidden; (3) run the poke protocol where the "action" perturbation is applied through the task's actuators (this scene family pokes via `ctrl`, not `xfrc`, since the latent model only knows the action channel); (4) same metrics, extra caveat row. Kill criterion: decoder held-out $R^2 < 0.9$ on positions → publish the negative honestly on the method page and drop the report card for this model.

## 8. Tests and invariants (CLAUDE.md, verbatim)

1. **Oracle null test (sacred):** snapshot mid-trajectory (after ≥100 steps, so warm-start is non-trivial), restore, re-step N=1000 — `np.array_equal` (bitwise, float64) on qpos/qvel vs. the original continuation. Runs in CI on CPU. A second variant cross-checks `mujoco.rollout` against raw `mj_step` for the same seed/state.
2. **Contrast null test:** zero-magnitude poke ⇒ truth contrast exactly zero; model contrast exactly zero under CRN. Bitwise, both.
3. All randomness flows through explicit seeds in `schema`; no bare `np.random` / global torch seeding anywhere.
4. Artifacts are immutable and manifest-stamped (git SHA, lib versions, scene hash, config hash); regeneration goes to a new content-addressed path.
5. Unit tests never load GPUs or take >2 min; anything heavier lives behind `make integ`.
6. Rendering must never introduce state: `render.py` consumes `RolloutPair`s, it never steps physics.
7. No claim strings in site copy beyond the honesty perimeter in the v2 plan; copy lives in one `site/copy.md` file for single-point review.

## 9. Makefile targets → work packages (one Claude Code session each)

- `make test` — WP1: schema + oracle + null tests (CPU). Acceptance: tests 1–2 green.
- `make data && make train SCENE=…` — WP2: data gen + ensemble training (GPU, ~hours). Acceptance: spectrum shows monotone-ish held-out NLL across capacity/data cells; if the spectrum is flat, widen the data-budget axis before proceeding.
- `make grid SCENE=…` — WP3: contrasts + metrics + manifest (GPU+CPU). Acceptance: contrast null test green across all models; response curves render for one scene.
- `make render` — WP4: clips + composites. Acceptance: hero-candidate clips exist; spot-check labels burned in.
- `make site` — WP5: viewer (grid, slider over 8 magnitudes, report cards, scatter, method page). Acceptance: static build serves locally from manifest alone, loads <2 s on throttled connection.
- `make namebrand` — WP6 (gated): §7. Acceptance: decoder R² reported; report card or published kill-criterion note.
- `make demo` — end-to-end regeneration of one scene from scratch; this is the reproducibility claim in the public README, so it must actually work on a single L40S box.
