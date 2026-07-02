"""Metrics are pure functions: test them on synthetic contrast outputs."""

import numpy as np
import pytest

from wa import config
from wa.contrast import ContrastSet
from wa.metrics import (momentum_check, propagation, response_curve, stability,
                        _san)
from wa.scenes import get_scene
from wa.schema import Poke

T = 60


def _poke(spec, magnitude):
    site = spec.poke_sites[0]
    return Poke(site=site, direction=np.array(spec.poke_directions[site]),
                magnitude=float(magnitude), t_start=5, duration=5)


def _cs(spec, poke, contrast, poked=None, K=1, model_id="m"):
    zeros = np.zeros((T, 2))
    return ContrastSet(scene=spec.name, model_id=model_id, poke=poke, seed=0,
                       contrast=contrast, poked=poked or {},
                       poked_states_k0=zeros, null_states_k0=zeros)


def _linear_contrast(spec, scale, K=1):
    """Displacement growing linearly in t and magnitude for every tracked body."""
    sets = []
    for m in spec.magnitudes():
        c = {}
        for b in spec.tracked_bodies:
            arr = np.zeros((K, T, 3))
            arr[:, :, 0] = scale * m * np.linspace(0, 1, T)
            c[f"{b}.pos"] = arr
            c[f"{b}.vel"] = np.gradient(arr, axis=1)
            c[f"{b}.mom"] = 0.17 * c[f"{b}.vel"]
        sets.append(_cs(spec, _poke(spec, m), c, K=K))
    return sets


def test_response_identical_is_green_zero():
    spec = get_scene("billiards")
    truth = _linear_contrast(spec, scale=0.01)
    model = _linear_contrast(spec, scale=0.01, K=4)
    r = response_curve(spec, truth, model)
    assert r.score == pytest.approx(0.0, abs=1e-12)
    assert r.light == "green"
    assert len(r.x) == spec.n_magnitudes


def test_response_doubled_scores_one():
    spec = get_scene("billiards")
    truth = _linear_contrast(spec, scale=0.01)
    model = _linear_contrast(spec, scale=0.02, K=4)
    r = response_curve(spec, truth, model)
    assert r.score == pytest.approx(1.0, rel=1e-9)   # ∫|2r−r| / ∫r == 1
    assert r.light == "red"


def test_momentum_identical_zero():
    spec = get_scene("billiards")
    truth = _linear_contrast(spec, scale=0.01)
    model = _linear_contrast(spec, scale=0.01, K=4)
    r = momentum_check(spec, truth, model)
    assert r.score == pytest.approx(0.0, abs=1e-9)
    assert r.details["reference"] == "simulated ground truth"


def _arrival_sets(spec, offsets, K=1):
    """Body i's contrast crosses eps exactly at step arrivals[i]."""
    sets = []
    for m in spec.magnitudes():
        c = {}
        for b in spec.tracked_bodies:
            c[f"{b}.pos"] = np.zeros((K, T, 3))
            c[f"{b}.vel"] = np.zeros((K, T, 3))
            c[f"{b}.mom"] = np.zeros((K, T, 3))
        for i, b in enumerate(spec.chain):
            c[f"{b}.pos"][:, offsets[i]:, 0] = 10 * spec.arrival_eps
        sets.append(_cs(spec, _poke(spec, m), c, K=K))
    return sets


def test_propagation_exact_arrivals():
    spec = get_scene("blocks")
    truth = _arrival_sets(spec, offsets=[5, 10, 15, 20])
    model_same = _arrival_sets(spec, offsets=[5, 10, 15, 20], K=3)
    r = propagation(spec, truth, model_same)
    assert r.score == pytest.approx(0.0)
    assert r.y_truth == [5, 10, 15, 20]
    model_late = _arrival_sets(spec, offsets=[10, 20, 30, 40], K=3)
    r2 = propagation(spec, truth, model_late)
    # mean |Δt| = 12.5, mean t_T = 12.5 -> score 1.0
    assert r2.score == pytest.approx(1.0)
    assert r2.light == "red"


def _stability_sets(spec, topple_from, K=1):
    """Top block ends below topple_z for magnitudes >= grid index topple_from."""
    sets = []
    for mi, m in enumerate(spec.magnitudes()):
        z_end = 0.05 if mi >= topple_from else 0.28
        poked = {f"{spec.topple_body}.pos": np.full((K, T, 3), 0.28)}
        poked[f"{spec.topple_body}.pos"][:, -1, 2] = z_end
        c = {f"{b}.pos": np.zeros((K, T, 3)) for b in spec.tracked_bodies}
        sets.append(_cs(spec, _poke(spec, m), c, poked=poked, K=K))
    return sets


def test_stability_threshold_and_score():
    spec = get_scene("blocks")
    truth = _stability_sets(spec, topple_from=4)
    same = _stability_sets(spec, topple_from=4, K=3)
    r = stability(spec, truth, same)
    assert r.score == pytest.approx(0.0)
    mags = spec.magnitudes()
    assert r.details["m_star_truth"] == pytest.approx(np.sqrt(mags[3] * mags[4]))
    shifted = _stability_sets(spec, topple_from=6, K=3)
    r2 = stability(spec, truth, shifted)
    expected = abs(np.sqrt(mags[5] * mags[6]) - np.sqrt(mags[3] * mags[4])) \
        / np.sqrt(mags[3] * mags[4])
    assert r2.score == pytest.approx(expected)
    assert list(r2.details["disagreement_mags"]) == pytest.approx(list(mags[4:6]))


def test_diverged_model_is_red_and_json_safe():
    spec = get_scene("billiards")
    truth = _linear_contrast(spec, scale=0.01)
    model = _linear_contrast(spec, scale=0.01, K=4)
    for cs in model:
        for key in cs.contrast:
            cs.contrast[key][:] = np.inf
        cs.diverged_at = 3
    r = response_curve(spec, truth, model)
    assert r.light == "red" and r.diverged
    j = r.to_json()
    assert j["score"] == config.SCORE_CAP
    assert all(v is None for v in j["curves"]["y_model"])


def test_san_and_lights():
    assert _san([1.0, np.inf, np.nan, -np.inf]) == [1.0, None, None, None]
    assert config.light(0.05) == "green"
    assert config.light(0.10) == "green"
    assert config.light(0.30) == "yellow"
    assert config.light(0.31) == "red"
    assert config.light(float("nan")) == "red"
