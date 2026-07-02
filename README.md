# worldaudit — behavioral physics audits for learned world models

We poke a simulated world, poke a learned model of it the same way, and compare
what happens next. Every score is a comparison against **simulated ground truth**
from a pinned MuJoCo build — never against an idealized law.

The output is a static site of report cards: per scene, per model, per sub-test
(impulse response, momentum transfer, disturbance propagation, stability
threshold, time-to-divergence), each with paired counterfactual curves and
side-by-side truth|model clips.

## Why behavioral audits

Every model here has excellent one-step training metrics; the report cards
still range from green to catastrophically red. One-step likelihood measures
what a model was trained on; the audit measures whether the physics holds
together over seconds of iterated prediction — which is what you actually use
a world model for.

## Method guarantees (enforced in CI, bitwise)

- **Oracle null test:** restore a mid-trajectory snapshot (`mjSTATE_INTEGRATION`,
  warm-start included) and re-step 1000 steps — reproduces the original
  continuation bit-for-bit. A second variant cross-checks `mujoco.rollout`
  against raw `mj_step`.
- **Contrast null test:** a zero-magnitude poke yields an *exactly* zero
  contrast — truth side, and model side under common random numbers (each
  sample's ensemble member and head-noise sequence are shared between the
  poked and unpoked rollouts).
- All randomness flows through explicit seeds; artifacts are immutable and
  stamped (git SHA, library versions, scene hash, config hash); regeneration
  goes to a new content-addressed path.
- Rendering never steps physics: clips are kinematic playback of stored
  trajectories, labeled "model-predicted states, simulator renderer."

## Reproduce

Requires Linux, an NVIDIA GPU (developed on a single L40S), Python 3.11,
Node ≥ 20. Versions are pinned in `requirements.lock`.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.lock
make test                  # CPU unit tests incl. the bitwise null tests (<1 min)
make demo SCENE=billiards  # data → train → grid → render → site, from scratch
cd site && npx vite preview
```

`make demo` regenerates one scene end-to-end (training data, the 6-model
ensemble spectrum, contrasts + metrics + manifest, clips, static site) on a
single GPU box. Individual stages: `make data / train / grid / render / site`.

The name-brand audit (`make namebrand`) runs an official TD-MPC2 checkpoint
through the same protocol via a decoder probe whose held-out R² is reported as
the audit's noise floor (pre-registered kill criterion: R² < 0.9 on observable
positions ⇒ publish the negative, drop the report card). It additionally needs
`third_party/tdmpc2` (cloned from the official repo) and downloads the
checkpoint from HuggingFace.

## Honesty perimeter

- Momentum is scored against the simulator's momentum change (MuJoCo contacts
  with friction and solver softness are not exactly conservative) — the copy
  says "vs. simulated ground truth," never "vs. conservation law."
- Diverged model rollouts are flagged and capped, not smoothed over.
- Traffic-light thresholds are calibrated once, frozen, and printed on the
  site's method page from the same config the pipeline used.
- All user-facing claim strings live in `site/copy.md` for single-point review.
