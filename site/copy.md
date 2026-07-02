# worldaudit site copy

All user-facing claim strings live in this one file (invariant 7). Review here,
nowhere else. Keys are `##` headers; body text below each header is the copy.

## title
worldaudit — does your world model know physics?

## tagline
We poke a simulated world, poke a learned model of it the same way, and compare what happens next. Every score below is a comparison against simulated ground truth from a pinned MuJoCo build — never against an idealized law.

## report-intro
Each row is one learned dynamics model (ensemble of 5, trained on logged interactions). Each column is one behavioral sub-test built from paired counterfactuals: the same initial state, with and without a calibrated poke. Click a cell to see the curve and the side-by-side rollout.

## clip-caption
Left: ground truth (MuJoCo). Right: model-predicted states, simulator renderer. Clocks are synchronized; the red flash marks the poke window.

## divergence-note
"Diverged" means the model's predicted state left the plausible regime (non-finite or |state| > 10³) at the marked step. Scores for diverged rollouts are capped and shown red; the raw curves keep their gaps rather than being smoothed over.

## momentum-note
The momentum check compares the model's change in total linear momentum across the first-collision window to the simulated ground truth's change — not to an exact conservation law. MuJoCo contacts with friction and solver softness are not exactly conservative, so the simulator itself is the reference.

## method-title
Method

## method-body
Ground truth is MuJoCo with a pinned version, Newton solver, tolerance 0 (fixed iteration count), fixed timestep, snapshot/restore via mjSTATE_INTEGRATION including warm-start. Counterfactual pairs share a byte-identical initial state; the only difference is the applied force during the poke window. The oracle's null test — restore a mid-trajectory snapshot and re-step 1000 steps — must reproduce the original continuation bit-for-bit, and runs in CI. Model contrasts use common random numbers: for each of the K samples, the poked and unpoked rollouts share the same ensemble-member assignment and the same head-noise sequence, so a zero-magnitude poke yields an exactly-zero contrast on both the truth side and the model side. Model rollouts are iterated one-step predictions; compounding error is the phenomenon under audit, not an artifact we correct. Quaternion blocks are renormalized to keep predicted states interpretable as states; the dynamics error itself is never corrected. Every artifact is stamped with git SHA, library versions, scene hash, and config hash; regeneration goes to a new content-addressed path.

## method-thresholds
Traffic lights: score ≤ {green} is green, ≤ {yellow} is yellow, otherwise red. Thresholds were calibrated once on the model spectrum and then frozen; they are identical for every model and printed here from the same config the pipeline used. K = {k} contrast samples per model per magnitude.

## footer
All assets on this page are precomputed by the audit pipeline; the page renders a manifest and plays clips — nothing is simulated in your browser.
