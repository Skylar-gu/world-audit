"""Grid smoke test: injected tiny model, tmp dirs, full pipeline shape checks."""

import json

import numpy as np
import pytest

from wa import grid as wa_grid
from wa.contrast import assert_null_contrast_zero, model_contrast, truth_contrast
from wa.models import EnsembleDynamics
from wa.oracle import Oracle
from wa.schema import Poke


def _patch_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(wa_grid, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(wa_grid, "RESULTS_DIR", tmp_path / "artifacts/results")
    monkeypatch.setattr(wa_grid, "ROLLOUTS_DIR", tmp_path / "artifacts/rollouts")
    monkeypatch.setattr(wa_grid, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(wa_grid, "SITE_MANIFEST", tmp_path / "site/manifest.json")


def test_contrastset_null_assertion_catches_violation():
    oracle = Oracle("billiards")
    s0 = oracle.init_state()
    net = EnsembleDynamics(oracle.nq + oracle.nv, 3, hidden=(8,), seed=1, e=2)
    zero = Poke(site="cue", direction=np.array([1.0, 0.0, 0.0]), magnitude=0.0,
                t_start=5, duration=5)
    cs = model_contrast(net, oracle, s0, zero, K=2, seed=3, model_id="tiny", horizon=60)
    assert_null_contrast_zero(cs)  # must pass
    cs.contrast["cue.pos"][0, 10, 0] = 1e-9
    with pytest.raises(AssertionError, match="null test FAILED"):
        assert_null_contrast_zero(cs)
    cs2 = truth_contrast(oracle, s0, zero, horizon=60)
    assert_null_contrast_zero(cs2)


def test_grid_smoke_billiards(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    oracle = Oracle("billiards")
    net = EnsembleDynamics(oracle.nq + oracle.nv, 3, hidden=(8,), seed=1, e=2)
    paths = wa_grid.run_grid("billiards", seed=0, K=2, horizon=120,
                             models={"tiny": net})
    assert len(paths) == 1 and paths[0].exists()
    r = json.loads(paths[0].read_text())
    assert r["null_test"] == "pass"
    assert [s["name"] for s in r["subtests"]] == ["response", "momentum"]
    assert len(r["magnitudes"]) == 8

    m = json.loads((tmp_path / "site/manifest.json").read_text())
    assert m["config"]["momentum_reference"] == "simulated ground truth"
    scene = m["scenes"][0]
    assert scene["name"] == "billiards"
    sub = scene["models"][0]["subtests"][0]
    assert len(sub["entries"]) == 8
    entry = sub["entries"][0]
    for field in ("poke_id", "magnitude", "clip_url", "curves", "score", "light"):
        assert field in entry
    assert len(entry["curves"]["x"]) == 8
    assert (tmp_path / "artifacts/rollouts/billiards/oracle__cue__m0.npz").exists()
    assert (tmp_path / "artifacts/rollouts/billiards/tiny__cue__m7.npz").exists()

    # idempotency: second run skips (same content key), manifest still rebuilds
    paths2 = wa_grid.run_grid("billiards", seed=0, K=2, horizon=120,
                              models={"tiny": net})
    assert paths2 == paths
