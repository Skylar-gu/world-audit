"""WP2 unit tests: CPU only, tiny problems (invariant 5)."""

import numpy as np
import pytest
import torch

from wa import data as wa_data
from wa.models import EnsembleDynamics, rollout_model
from wa.oracle import Oracle
from wa.schema import Poke, derive_seed


def _tiny_net(ds=4, da=2, seed=0):
    return EnsembleDynamics(ds, da, hidden=(16, 16), seed=seed, e=3)


def _synthetic(n=2048, ds=4, da=2, seed=1):
    g = np.random.default_rng(seed)
    obs = g.normal(size=(n, ds))
    act = g.normal(size=(n, da))
    A = g.normal(size=(ds, ds)) * 0.1
    B = g.normal(size=(da, ds)) * 0.1
    delta = obs @ A + act @ B + 0.01 * g.normal(size=(n, ds))
    return obs, act, delta


def test_nll_decreases_with_training():
    obs, act, delta = _synthetic()
    x = torch.from_numpy(np.concatenate([obs, act], axis=1)).float()
    tgt = torch.from_numpy(delta).float()
    net = _tiny_net()
    net.fit_normalizer(x, tgt)
    xe = x.unsqueeze(0).expand(net.e, -1, -1)
    with torch.no_grad():
        before = float(net.nll(xe, tgt))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(derive_seed(0, "batches"))
    for _ in range(300):
        bi = torch.stack([torch.randint(x.shape[0], (128,), generator=gen)
                          for _ in range(net.e)])
        loss = net.nll(x[bi], tgt[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = float(net.nll(xe, tgt))
    assert after < before - 0.5, (before, after)


def test_normalizer_in_state_dict():
    net = _tiny_net()
    x = torch.randn(64, 6, generator=torch.Generator().manual_seed(3))
    tgt = torch.randn(64, 4, generator=torch.Generator().manual_seed(4))
    net.fit_normalizer(x, tgt)
    sd = net.state_dict()
    for key in ("in_mean", "in_std", "tgt_mean", "tgt_std"):
        assert key in sd
    net2 = _tiny_net()
    net2.load_state_dict(sd)
    assert torch.equal(net2.in_std, net.in_std)


def test_crn_model_null_bitwise():
    """Invariant 2, model side: zero-magnitude poke ⇒ contrast exactly zero under CRN."""
    oracle = Oracle("billiards")
    s0 = oracle.init_state()
    net = EnsembleDynamics(oracle.nq + oracle.nv, 3, hidden=(16,), seed=5, e=3)
    zero_poke = Poke(site="cue", direction=np.array([1.0, 0.0, 0.0]), magnitude=0.0,
                     t_start=5, duration=5)
    K, H = 4, 50
    null = rollout_model(net, oracle, s0, None, K=K, seed=9, horizon=H, with_tracked=False)
    poked = rollout_model(net, oracle, s0, zero_poke, K=K, seed=9, horizon=H,
                          with_tracked=False)
    for a, b in zip(null, poked):
        assert np.array_equal(a.states, b.states)


def test_crn_same_seed_reproducible_and_seeds_distinct():
    oracle = Oracle("billiards")
    s0 = oracle.init_state()
    net = EnsembleDynamics(oracle.nq + oracle.nv, 3, hidden=(16,), seed=5, e=3)
    a = rollout_model(net, oracle, s0, None, K=2, seed=9, horizon=20, with_tracked=False)
    b = rollout_model(net, oracle, s0, None, K=2, seed=9, horizon=20, with_tracked=False)
    c = rollout_model(net, oracle, s0, None, K=2, seed=10, horizon=20, with_tracked=False)
    assert np.array_equal(a[0].states, b[0].states)
    assert not np.array_equal(a[0].states, c[0].states)
    assert not np.array_equal(a[0].states, a[1].states), "samples must differ"


def test_datagen_reproducible_bitwise(tmp_path, monkeypatch):
    arrays = []
    for sub in ("a", "b"):
        monkeypatch.setattr(wa_data, "DATA_DIR", tmp_path / sub / "data")
        monkeypatch.setattr(wa_data, "MANIFEST_DIR", tmp_path / sub / "manifests")
        path = wa_data.generate("billiards", seed=0, n_total=600)
        z = np.load(path)
        arrays.append((z["obs"], z["act"], z["next_obs"]))
    for x, y in zip(*arrays):
        assert np.array_equal(x, y)


def test_data_load_nested_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(wa_data, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(wa_data, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(wa_data, "HOLDOUT", 200)
    monkeypatch.setattr(wa_data, "BUDGETS", {"tiny": 100, "small": 300})
    wa_data.generate("billiards", seed=0, n_total=500)
    small = wa_data.load("billiards", "small")
    tiny = wa_data.load("billiards", "tiny")
    assert small["obs"].shape[0] == 300 and tiny["obs"].shape[0] == 100
    assert np.array_equal(small["obs"][:100], tiny["obs"]), "budgets must be nested"
    assert np.array_equal(small["holdout_obs"], tiny["holdout_obs"])
    assert small["holdout_obs"].shape[0] == 200


def test_poke_enters_model_input():
    """The action channel is live: a strong poke changes model rollouts."""
    oracle = Oracle("billiards")
    s0 = oracle.init_state()
    net = EnsembleDynamics(oracle.nq + oracle.nv, 3, hidden=(16,), seed=5, e=3)
    poke = Poke(site="cue", direction=np.array([1.0, 0.0, 0.0]), magnitude=10.0,
                t_start=5, duration=5)
    null = rollout_model(net, oracle, s0, None, K=1, seed=9, horizon=20, with_tracked=False)
    poked = rollout_model(net, oracle, s0, poke, K=1, seed=9, horizon=20, with_tracked=False)
    assert not np.array_equal(null[0].states, poked[0].states)
