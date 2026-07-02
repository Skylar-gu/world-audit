"""Paired counterfactual computation: Contrast = poked − null, per source.

Truth contrasts are deterministic (K=1). Model contrasts carry K CRN samples:
sample k of the poked and null runs shares its ensemble-member assignment and
head-noise sequence (wa.models.rollout_model), so the null contrast is exactly
zero by construction — enforced bitwise by `assert_null_contrast_zero`, which
grid.py runs for truth and every model before computing anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wa import config
from wa.models import EnsembleDynamics, rollout_model
from wa.oracle import Oracle
from wa.schema import Poke, RolloutPair


@dataclass
class ContrastSet:
    """Poked−null tracked contrasts for one (source, poke, magnitude)."""

    scene: str
    model_id: str
    poke: Poke
    seed: int
    contrast: dict[str, np.ndarray]   # key -> (K, T, d), poked − null
    poked: dict[str, np.ndarray]      # key -> (K, T, d), absolute poked series
    poked_states_k0: np.ndarray       # (T, nq+nv) representative sample, for render
    null_states_k0: np.ndarray
    diverged_at: int | None = None    # first step any sample is non-finite / huge
    K: int = field(init=False)

    def __post_init__(self):
        self.K = next(iter(self.contrast.values())).shape[0]


def _stack(pairs: list[RolloutPair]) -> dict[str, np.ndarray]:
    keys = pairs[0].tracked.keys()
    return {k: np.stack([p.tracked[k] for p in pairs]) for k in keys}


def _diverged_at(states: list[np.ndarray]) -> int | None:
    """First step at which any sample leaves the finite/plausible regime."""
    first = None
    for st in states:
        bad = ~np.isfinite(st) | (np.abs(st) > config.DIVERGENCE_LIMIT)
        rows = np.flatnonzero(bad.any(axis=1))
        if rows.size:
            first = int(rows[0]) if first is None else min(first, int(rows[0]))
    return first


def truth_contrast(oracle: Oracle, init_state: np.ndarray, poke: Poke,
                   horizon: int | None = None,
                   null: RolloutPair | None = None) -> ContrastSet:
    """Deterministic truth contrast; pass a precomputed null to share it."""
    horizon = horizon or oracle.scene.horizon
    if null is None:
        null = oracle.run(init_state, None, horizon=horizon)
    poked = oracle.run(init_state, poke, horizon=horizon)
    poked_t, null_t = _stack([poked]), _stack([null])
    return ContrastSet(
        scene=oracle.scene.name, model_id="oracle", poke=poke, seed=0,
        contrast={k: poked_t[k] - null_t[k] for k in poked_t},
        poked=poked_t,
        poked_states_k0=poked.states, null_states_k0=null.states,
        diverged_at=None,
    )


def model_contrast(net: EnsembleDynamics, oracle: Oracle, init_state: np.ndarray,
                   poke: Poke, K: int, seed: int, model_id: str,
                   horizon: int | None = None) -> ContrastSet:
    """K-sample CRN contrast: poked and null share member + head-noise per k."""
    poked = rollout_model(net, oracle, init_state, poke, K=K, seed=seed,
                          horizon=horizon, model_id=model_id)
    null = rollout_model(net, oracle, init_state, None, K=K, seed=seed,
                         horizon=horizon, model_id=model_id)
    poked_t, null_t = _stack(poked), _stack(null)
    return ContrastSet(
        scene=oracle.scene.name, model_id=model_id, poke=poke, seed=seed,
        contrast={k: poked_t[k] - null_t[k] for k in poked_t},
        poked=poked_t,
        poked_states_k0=poked[0].states, null_states_k0=null[0].states,
        diverged_at=_diverged_at([p.states for p in poked + null]),
    )


def assert_null_contrast_zero(cs: ContrastSet) -> None:
    """Invariant 2 enforcement: a zero-magnitude contrast must be bitwise zero."""
    assert cs.poke.magnitude == 0.0, "null test requires a zero-magnitude poke"
    for key, arr in cs.contrast.items():
        if not np.all(arr == 0.0):
            raise AssertionError(
                f"contrast null test FAILED for {cs.model_id} on {cs.scene}:{key}")
    # tobytes: true bitwise identity (np.array_equal treats NaN != NaN, but a
    # diverged-to-NaN pair is still a valid exactly-equal null contrast)
    if cs.poked_states_k0.tobytes() != cs.null_states_k0.tobytes():
        raise AssertionError(
            f"contrast null test FAILED for {cs.model_id} on {cs.scene}: states differ")
