# worldaudit

We intervene on a simulated world, give a learned model of that world the
exact same intervention, and compare what happens next. Every score compares the model against
**simulated ground truth** from a pinned MuJoCo build — never against an
idealized textbook law.

The output is a static website of report cards: for each scene and each model,
five behavioral tests (impulse response, momentum transfer, disturbance
propagation, stability threshold, time-to-divergence), each with paired
curves and side-by-side truth-vs-model video clips.

## Why test behavior instead of training loss

Every model here looks great on its one-step training metrics — yet the report
cards range from green to badly red. One-step accuracy measures what the model
was trained on; the audit measures whether its physics stays coherent over
seconds of predictions fed back into themselves, which is how world models are
actually used.

## Built-in correctness checks (run in CI, exact to the last bit)

- **Oracle null test:** save the simulator mid-trajectory, restore it, and run
  1000 more steps — the continuation must match the original bit-for-bit.
  A second check confirms two independent simulation code paths agree the
  same way.
- **Contrast null test:** an intervention with zero strength must produce a
  measured effect of exactly zero — for the simulator, and for the model too
  (the intervened and unintervened runs share the same random numbers, so
  noise cancels perfectly).
- Every random number comes from an explicit seed. Every output file is
  immutable and stamped with the exact code version, library versions, and
  settings that produced it; regenerating anything writes to a new path
  instead of overwriting.
- Video rendering never runs physics: clips are pure playback of stored
  trajectories, labeled "model-predicted states, simulator renderer."

## Reproduce

Requires Linux, an NVIDIA GPU (developed on a single L40S), Python 3.11,
Node ≥ 20. Exact versions are pinned in `requirements.lock`.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.lock
make test                  # CPU unit tests incl. the bitwise null tests (<1 min)
make demo SCENE=billiards  # data → train → grid → render → site, from scratch
cd site && npx vite preview
```

`make demo` rebuilds one scene end-to-end (training data, six models of varying
quality — each an ensemble of 5 networks — contrasts + metrics, clips, static
site) on a single GPU box.
Individual stages: `make data / train / grid / render / site`.

The name-brand audit (`make namebrand`) runs an official TD-MPC2 checkpoint
through the same protocol. TD-MPC2 predicts in a latent space, so a trained
probe translates its state back to physical coordinates; the probe's held-out
accuracy (R²) is reported as the audit's noise floor, with a pre-registered
kill rule — if the probe can't reach R² ≥ 0.9 on observable positions, the
report card is dropped and the negative result published instead. This stage
also needs `third_party/tdmpc2` (cloned from the official repo) and downloads
the checkpoint from HuggingFace.

## Honesty rules

- Momentum is scored against the simulator's momentum change (MuJoCo contacts
  with friction are not perfectly conservative) — the site says "vs. simulated
  ground truth," never "vs. conservation law."
- Model rollouts that blow up are flagged and capped, not smoothed over.
- Traffic-light thresholds were calibrated once, then frozen; the site prints
  them from the same config file the pipeline used.
- Every user-facing claim lives in one file, `site/copy.md`, so it can be
  reviewed in one place.
