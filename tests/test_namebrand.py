"""Namebrand pure-function tests: no dm_control, no checkpoint, CPU only."""

import numpy as np
import pytest

from wa.namebrand.audit import (MLPDecoder, apply_ridge, fit_ridge,
                                perturb_actions, r2_per_coord)


def test_ridge_recovers_linear_map():
    g = np.random.default_rng(0)
    Z = g.normal(size=(2000, 16))
    W_true = g.normal(size=(16, 4))
    Y = Z @ W_true + 0.5
    W = fit_ridge(Z, Y, lam=1e-6)
    r2 = r2_per_coord(Y, apply_ridge(W, Z))
    assert (r2 > 0.999).all()


def test_r2_detects_unobservable_coord():
    g = np.random.default_rng(1)
    Z = g.normal(size=(1000, 8))
    Y = np.concatenate([Z[:, :2], g.normal(size=(1000, 1))], axis=1)  # coord 2 = noise
    W = fit_ridge(Z[:800], Y[:800])
    r2 = r2_per_coord(Y[800:], apply_ridge(W, Z[800:]))
    assert r2[0] > 0.99 and r2[1] > 0.99
    assert r2[2] < 0.5


def test_perturb_actions_window_and_null():
    g = np.random.default_rng(2)
    nominal = g.uniform(-0.5, 0.5, size=(20, 6))
    u = np.ones(6)
    zero = perturb_actions(nominal, 0.0, u, t0=3, duration=5)
    assert zero.tobytes() == nominal.tobytes(), "m=0 must be bitwise identity"
    poked = perturb_actions(nominal, 0.3, u, t0=3, duration=5)
    assert np.array_equal(poked[:3], nominal[:3])
    assert np.array_equal(poked[8:], nominal[8:])
    assert np.allclose(poked[3:8], nominal[3:8] + 0.3)
    big = perturb_actions(nominal, 10.0, u, t0=0, duration=20)
    assert big.max() <= 1.0 and (big == 1.0).all(), "must clip to action bounds"


def test_mlp_decoder_fits_nonlinear_map():
    g = np.random.default_rng(3)
    Z = g.normal(size=(3000, 8))
    Y = np.stack([np.sin(Z[:, 0] * 2), Z[:, 1] ** 2], axis=1)
    dec = MLPDecoder(8, 2, seed=0, hidden=64).fit(Z[:2500], Y[:2500], epochs=120)
    r2 = r2_per_coord(Y[2500:], dec.predict(Z[2500:]))
    assert (r2 > 0.85).all(), r2
    # ridge cannot fit these; the MLP upgrade must beat it
    W = fit_ridge(Z[:2500], Y[:2500])
    r2_lin = r2_per_coord(Y[2500:], apply_ridge(W, Z[2500:]))
    assert (r2 > r2_lin + 0.2).all()
