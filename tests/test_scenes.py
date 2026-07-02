"""Scene registry guards: audit configuration must stay deterministic-friendly."""

import mujoco
import numpy as np
import pytest

from wa.oracle import Oracle
from wa.scenes import REGISTRY
from wa.schema import Poke


@pytest.mark.parametrize("scene", list(REGISTRY))
def test_audit_options_pinned(scene):
    spec = REGISTRY[scene]
    model = mujoco.MjModel.from_xml_path(str(spec.xml_path))
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.tolerance == 0.0, "tolerance=0 forces a fixed solver iteration count"
    assert model.opt.timestep == 0.002


@pytest.mark.parametrize("scene", list(REGISTRY))
def test_registry_names_resolve(scene):
    spec = REGISTRY[scene]
    model = mujoco.MjModel.from_xml_path(str(spec.xml_path))
    for body in spec.tracked_bodies + spec.poke_sites:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, body
    for body in spec.tracked_bodies:
        for kind in ("pos", "vel"):
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, f"{body}.{kind}") >= 0
    assert spec.settle_steps >= 100, "init snapshot must have non-trivial warm-start"
    lo, hi = spec.magnitude_range
    assert 0 < lo < hi and len(spec.magnitudes()) == spec.n_magnitudes


@pytest.mark.parametrize("scene", list(REGISTRY))
def test_max_magnitude_non_degenerate(scene):
    """Max poke is dramatic but keeps every tracked body in the arena."""
    spec = REGISTRY[scene]
    oracle = Oracle(scene)
    s0 = oracle.init_state()
    body = spec.poke_sites[0]
    poke = Poke(site=body, direction=np.array(spec.poke_directions[body]),
                magnitude=spec.magnitude_range[1],
                t_start=spec.poke_t_start, duration=spec.poke_duration)
    out = oracle.run(s0, poke, horizon=spec.horizon)
    for tracked_body in spec.tracked_bodies:
        pos = out.tracked[f"{tracked_body}.pos"]
        assert np.all(np.abs(pos[:, :2]) < 2.0), f"{tracked_body} left the arena"
        assert np.all(pos[:, 2] > -0.05), f"{tracked_body} fell through the floor"
