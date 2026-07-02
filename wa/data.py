"""Training-data generation: random pokes + passive rollouts, all via the oracle.

One dataset per scene holds budget_max + holdout transitions; budget cells are
nested prefixes of a seeded shuffle, so every model trains on the same
distribution and is evaluated on the same held-out split.

Artifacts are immutable and content-addressed (invariant 4):
  experiments/data/<scene>-<hash>.npz + experiments/manifests/data-<scene>-<hash>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wa.oracle import Oracle, manifest_stamp
from wa.schema import Poke, content_hash, derive_seed, rng
from wa.scenes import get_scene

DATA_DIR = Path("experiments/data")
MANIFEST_DIR = Path("experiments/manifests")

BUDGETS = {"5k": 5_000, "100k": 100_000}
HOLDOUT = 10_000
ROLLOUT_HORIZON = 500       # shorter than audit horizon: more inits per budget
PASSIVE_FRAC = 0.3          # fraction of rollouts with no poke
INIT_JITTER = 0.005


def _random_poke(oracle: Oracle, seed: int, horizon: int) -> Poke:
    g = rng(seed)
    spec = oracle.scene
    site = spec.poke_sites[int(g.integers(len(spec.poke_sites)))]
    theta = g.uniform(0.0, 2.0 * np.pi)
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    lo, hi = spec.magnitude_range
    magnitude = float(np.exp(g.uniform(np.log(lo), np.log(hi))))
    t_start = int(g.integers(0, horizon // 2))
    duration = int(g.integers(5, 21))
    if t_start + duration > horizon:
        t_start = horizon - duration
    return Poke(site=site, direction=direction, magnitude=magnitude,
                t_start=t_start, duration=duration)


def generate(scene: str, seed: int = 0, n_total: int | None = None) -> Path:
    """Generate (obs, act, next_obs) transitions; returns the npz path."""
    spec = get_scene(scene)
    oracle = Oracle(spec)
    n_total = n_total or (max(BUDGETS.values()) + HOLDOUT)

    obs, act, nxt = [], [], []
    n = 0
    i = 0
    while n < n_total:
        rollout_seed = derive_seed(seed, "rollout", i)
        g = rng(derive_seed(rollout_seed, "kind"))
        poke = None
        if g.uniform() >= PASSIVE_FRAC:
            poke = _random_poke(oracle, derive_seed(rollout_seed, "poke"), ROLLOUT_HORIZON)
        s0 = oracle.init_state(seed=derive_seed(rollout_seed, "init"), jitter=INIT_JITTER)
        pair = oracle.run(s0, poke, horizon=ROLLOUT_HORIZON, seed=rollout_seed)
        s_seq = np.vstack([oracle.qpos_qvel_from_snapshot(s0)[None], pair.states])
        a_seq = oracle.force_channels(poke, ROLLOUT_HORIZON)
        obs.append(s_seq[:-1])
        act.append(a_seq)
        nxt.append(s_seq[1:])
        n += ROLLOUT_HORIZON
        i += 1

    obs = np.concatenate(obs)[:n_total]
    act = np.concatenate(act)[:n_total]
    nxt = np.concatenate(nxt)[:n_total]
    perm = rng(derive_seed(seed, "shuffle")).permutation(n_total)
    obs, act, nxt = obs[perm], act[perm], nxt[perm]

    manifest = {
        **manifest_stamp(spec),
        "kind": "data",
        "seed": seed,
        "n_total": n_total,
        "holdout": HOLDOUT,
        "budgets": BUDGETS,
        "rollout_horizon": ROLLOUT_HORIZON,
        "passive_frac": PASSIVE_FRAC,
        "init_jitter": INIT_JITTER,
    }
    h = content_hash(manifest)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{scene}-{h}.npz"
    if path.exists():
        return path  # immutable: same content hash == same artifact
    np.savez_compressed(path, obs=obs, act=act, next_obs=nxt)
    (MANIFEST_DIR / f"data-{scene}-{h}.json").write_text(json.dumps(manifest, indent=2))
    return path


def load(scene: str, budget: str | None = None) -> dict[str, np.ndarray]:
    """Load the newest dataset for a scene; slice a nested budget cell + holdout."""
    paths = sorted(DATA_DIR.glob(f"{scene}-*.npz"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"no dataset for scene {scene!r}; run `make data SCENE={scene}`")
    z = np.load(paths[-1])
    obs, act, nxt = z["obs"], z["act"], z["next_obs"]
    n_train_max = obs.shape[0] - HOLDOUT
    n = BUDGETS[budget] if budget else n_train_max
    assert n <= n_train_max
    return {
        "obs": obs[:n], "act": act[:n], "next_obs": nxt[:n],
        "holdout_obs": obs[n_train_max:], "holdout_act": act[n_train_max:],
        "holdout_next_obs": nxt[n_train_max:],
        "path": str(paths[-1]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = generate(args.scene, seed=args.seed)
    print(out)
