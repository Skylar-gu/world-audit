"""Invariants 1–2 (CLAUDE.md): the sacred oracle null tests. Bitwise, CPU."""

import mujoco
import numpy as np
import pytest

from wa.oracle import Oracle, restore, snapshot
from wa.scenes import REGISTRY
from wa.schema import Poke

SCENES = list(REGISTRY)


def _active_midtrajectory_state(oracle: Oracle, steps: int = 150) -> np.ndarray:
    """Snapshot mid-trajectory (≥100 steps in, bodies in motion, warm-start non-trivial)."""
    mujoco.mj_resetData(oracle.model, oracle.data)
    body = oracle.scene.poke_sites[0]
    body_id = mujoco.mj_name2id(oracle.model, mujoco.mjtObj.mjOBJ_BODY, body)
    direction = np.array(oracle.scene.poke_directions[body])
    for t in range(steps):
        oracle.data.xfrc_applied[:] = 0.0
        if 20 <= t < 40:  # kick so the mid-trajectory state is dynamic, not settled
            oracle.data.xfrc_applied[body_id, :3] = 5.0 * direction
        mujoco.mj_step(oracle.model, oracle.data)
    oracle.data.xfrc_applied[:] = 0.0
    assert np.any(oracle.data.qacc_warmstart != 0.0), "warm-start is trivial; test is vacuous"
    return snapshot(oracle.model, oracle.data)


@pytest.mark.parametrize("scene", SCENES)
def test_snapshot_restore_bitwise(scene):
    """Invariant 1: restore + re-step N=1000 reproduces the continuation bitwise."""
    N = 1000
    oracle = Oracle(scene)
    s = _active_midtrajectory_state(oracle)

    cont_qpos = np.empty((N, oracle.nq))
    cont_qvel = np.empty((N, oracle.nv))
    for t in range(N):
        mujoco.mj_step(oracle.model, oracle.data)
        cont_qpos[t] = oracle.data.qpos
        cont_qvel[t] = oracle.data.qvel

    # restore into a *fresh* MjData: nothing may leak outside the snapshot
    data2 = mujoco.MjData(oracle.model)
    restore(oracle.model, data2, s)
    for t in range(N):
        mujoco.mj_step(oracle.model, data2)
        assert np.array_equal(cont_qpos[t], data2.qpos), f"qpos diverged at step {t}"
        assert np.array_equal(cont_qvel[t], data2.qvel), f"qvel diverged at step {t}"


@pytest.mark.parametrize("scene", SCENES)
def test_rollout_vs_mjstep(scene):
    """Invariant 1 variant: mujoco.rollout cross-checked against raw mj_step, bitwise."""
    oracle = Oracle(scene)
    s0 = _active_midtrajectory_state(oracle)
    body = oracle.scene.poke_sites[0]
    poke = Poke(site=body, direction=np.array(oracle.scene.poke_directions[body]),
                magnitude=5.0, t_start=10, duration=10)
    a = oracle.run(s0, poke, horizon=500)
    b = oracle.run_raw(s0, poke, horizon=500)
    assert np.array_equal(a.states, b.states)
    for k in a.tracked:
        assert np.array_equal(a.tracked[k], b.tracked[k]), k


@pytest.mark.parametrize("scene", SCENES)
def test_contrast_null(scene):
    """Invariant 2: zero-magnitude poke ⇒ truth contrast exactly zero, bitwise."""
    oracle = Oracle(scene)
    s0 = oracle.init_state()
    body = oracle.scene.poke_sites[0]
    zero_poke = Poke(site=body, direction=np.array(oracle.scene.poke_directions[body]),
                     magnitude=0.0, t_start=10, duration=10)
    null = oracle.run(s0, None, horizon=500)
    poked = oracle.run(s0, zero_poke, horizon=500)
    assert np.array_equal(null.states, poked.states)
    for k in null.tracked:
        assert np.array_equal(null.tracked[k], poked.tracked[k]), k


@pytest.mark.parametrize("scene", SCENES)
def test_nonzero_poke_has_effect(scene):
    """Sanity guard: the poke path is live (a zero contrast must mean zero cause)."""
    oracle = Oracle(scene)
    s0 = oracle.init_state()
    body = oracle.scene.poke_sites[0]
    poke = Poke(site=body, direction=np.array(oracle.scene.poke_directions[body]),
                magnitude=oracle.scene.magnitude_range[1], t_start=10, duration=10)
    null = oracle.run(s0, None, horizon=500)
    poked = oracle.run(s0, poke, horizon=500)
    assert not np.array_equal(null.states, poked.states)
