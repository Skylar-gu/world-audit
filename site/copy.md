# worldaudit site copy

All user-facing claim strings live in this one file (invariant 7). Review here,
nowhere else. Keys are `##` headers; body text below each header is the copy.

## title
WorldAudit: Physics Sanity Check for World Models

## tagline
We intervene on a simulated world, give a learned model of it the exact same intervention, and compare what happens next. Every score below compares the model to simulated ground truth from a pinned MuJoCo build.

## report-intro
Each row is one learned dynamics model: six in total, spanning small to large networks and short to long training, each an ensemble of 5 networks trained on logged interactions. Each column is one behavioral test built from a matched pair of runs: the same starting state, with and without a calibrated intervention. Click a cell to see the curves and the side-by-side clip.

## legend-green
matches ground truth

## legend-yellow
noticeable error

## legend-red
wrong or diverged

## clip-caption
Left: ground truth (MuJoCo). Right: the model's predicted states, drawn by the same renderer. Clocks are synchronized; the red flash marks when the intervention is applied.

## divergence-note
"Diverged" means the model's prediction blew up — values became non-finite or larger than 10³ — at the marked step. Diverged runs are capped at the worst score and shown red; the raw curves keep their gaps instead of being smoothed over.

## momentum-note
The momentum check compares the model's change in total momentum around the first collision to the simulator's change — not to an exact conservation law. MuJoCo's contacts (friction, solver softness) are not perfectly conservative, so the simulator itself is the reference.

## method-title
Method

## method-body
Ground truth is MuJoCo with a pinned version, a fixed timestep, and fixed solver settings, so runs are exactly repeatable. Snapshots save the simulator's full internal state: restoring one and stepping 1000 more steps must reproduce the original continuation bit-for-bit, and this is tested in CI. Each counterfactual pair starts from a byte-identical state; the only difference is the force applied during the intervention window. Model runs use common random numbers — the intervened and unintervened run share the same noise — so a zero-strength intervention gives an exactly-zero difference, on the truth side and the model side alike. Model rollouts are one-step predictions fed back into themselves; compounding error is the thing being measured, not a flaw we correct. Rotations are renormalized so predicted states stay valid states, but the prediction error itself is never corrected. Every output file is stamped with the exact code version, library versions, and settings that produced it, and is never overwritten.

## method-thresholds
Traffic lights: a score of {green} or less is green, {yellow} or less is yellow, anything higher is red. Thresholds were calibrated once, then frozen; they are the same for every model and are printed here from the same config file the pipeline used. K = {k} samples per model per intervention strength.

## namebrand-caveat
This row audits an official pre-trained TD-MPC2 checkpoint. TD-MPC2 predicts in a latent space with no built-in decoder, so we trained a probe to translate its latent state back into physical coordinates; the probe's held-out accuracy (R² per coordinate, reported below) is the audit's noise floor. Interventions are applied through the task's actuators — the only input channel the model has. Root x position is excluded because the model's observations are translation-invariant, so forward position is unrecoverable by construction, not by choice. Joint angles that can wind past 360° are decoded as (cos, sin) pairs, because winding count likewise isn't observable: the audit compares postures, not turn counts.

## namebrand-clip-caption
Left: ground truth (dm_control). Right: the model's latent prediction decoded through the probe — the probe's own error (the noise floor above) is baked into every frame, and root x is frozen at its starting value because it is unobservable: the right panel shows posture and gait, not forward travel. The red flash marks the perturbation window.

## namebrand-killed
Kill criterion triggered: the decoder probe failed to reach R² ≥ 0.9 on observable position coordinates, so scores would mostly measure the probe's error rather than the model's. As pre-registered, no report card is shown for this model; the decoder accuracy table below is the honest negative result.

## footer
Everything on this page was precomputed by the audit pipeline; the page just displays results and plays clips — nothing is simulated in your browser.
