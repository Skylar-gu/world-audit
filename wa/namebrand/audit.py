"""TD-MPC2 name-brand audit: decoder probe + ctrl-poke protocol (spec §7).

Pipeline: load official checkpoint with the official code (third_party/tdmpc2)
→ collect episodes with the planning agent in the true env, logging
(obs, z=enc(obs), qpos, qvel) → ridge decoder z→(qpos,qvel) with held-out R²
per coordinate → poke protocol where the perturbation is applied through the
task's actuators (clip(a + m·u)); the identical perturbed action sequence is
fed open-loop to the true env (from a byte-identical restored state) and to
the latent dynamics — the contrast machinery and response score are the same
as the core audit, with the decoder R² caveat attached to every output.

The latent rollout is deterministic (SimNorm keeps z on a simplex), so K=1
and the CRN null test reduces to: zero-magnitude perturbation ⇒ bitwise
identical trajectories, truth side and model side. Enforced before scoring.

Root x is translation-invariant in the walker observation and therefore
fundamentally undecodable; it is excluded from the audited coordinate set and
reported as such (this is disclosure, not cherry-picking: every other
coordinate is audited).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TDMPC2_DIR = REPO_ROOT / "third_party/tdmpc2/tdmpc2"
NB_DIR = Path("experiments/namebrand")
RESULTS_DIR = Path("experiments/artifacts/results")

from wa import config
from wa.schema import content_hash, derive_seed, rng

TASK = "walker-walk"
CKPT = NB_DIR / "ckpts/dmcontrol/walker-walk-1.pt"
N_EPISODES = 20
HOLDOUT_EPISODES = 4
WARMUP_STEPS = 100          # agent steps before the audit snapshot
HORIZON = 100               # agent steps (= 5 s simulated)
POKE_T0, POKE_DUR = 5, 25   # agent steps
MAGNITUDES = np.geomspace(0.05, 1.0, 8)   # ctrl perturbation, actions in [-1,1]
RIDGE_LAMBDA = 1e-3
R2_KILL = 0.9


# ---------------------------------------------------------------- environment

class WalkerEnv:
    """Minimal faithful port of the tdmpc2 DMControl wrapper:
    action_scale to [-1,1], action repeat 2, concatenated float32 obs."""

    def __init__(self, seed: int = 1):
        from dm_control import suite
        from dm_control.suite.wrappers import action_scale
        domain, task = TASK.split("-", 1)
        env = suite.load(domain, task, task_kwargs={"random": seed},
                         visualize_reward=False)
        self.env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
        self.action_dim = self.env.action_spec().shape[0]

    @property
    def physics(self):
        return self.env.physics

    def _obs(self, ts) -> np.ndarray:
        return np.concatenate([np.atleast_1d(v).ravel()
                               for v in ts.observation.values()]).astype(np.float32)

    def reset(self) -> np.ndarray:
        return self._obs(self.env.reset())

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        reward = 0.0
        for _ in range(2):  # action repeat, as in the official wrapper
            ts = self.env.step(action)
            reward += ts.reward or 0.0
        return self._obs(ts), reward

    # ---- exact state snapshot/restore (mujoco warm-start included)

    def snapshot(self) -> dict[str, np.ndarray]:
        d = self.physics.data
        return {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(),
                "warmstart": d.qacc_warmstart.copy(), "time": np.array([d.time])}

    def restore(self, s: dict[str, np.ndarray]) -> np.ndarray:
        d = self.physics.data
        with self.physics.reset_context():
            d.qpos[:] = s["qpos"]
            d.qvel[:] = s["qvel"]
            d.act[:] = s["act"]
            d.time = float(s["time"][0])
        d.qacc_warmstart[:] = s["warmstart"]
        return self._obs_now()

    def _obs_now(self) -> np.ndarray:
        obs = self.env.task.get_observation(self.physics)
        return np.concatenate([np.atleast_1d(v).ravel()
                               for v in obs.values()]).astype(np.float32)

    def state_vec(self) -> np.ndarray:
        d = self.physics.data
        return np.concatenate([d.qpos, d.qvel])


# ---------------------------------------------------------------- agent

def load_agent(device: str | None = None):
    sys.path.insert(0, str(TDMPC2_DIR))
    from omegaconf import OmegaConf
    from tdmpc2 import TDMPC2
    from common.layers import api_model_conversion

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(TDMPC2_DIR / "config.yaml")
    cfg.task = TASK
    cfg.obs = "state"
    cfg.model_size = 5
    cfg.multitask = False
    cfg.task_dim = 0
    cfg.compile = False
    cfg.obs_shape = {"state": [24]}
    cfg.action_dim = 6
    cfg.episode_length = 500
    cfg.num_envs = 1
    cfg.seed_steps = 0
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.device = device

    agent = TDMPC2(cfg)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)  # tensors only
    target = dict(agent.model.state_dict())
    shims = [p + "params." + k for p in ("_Qs.", "_detach_Qs_", "_target_Qs_")
             for k in ("__batch_size", "__device")]
    for s in shims:
        target.setdefault(s, torch.empty(0))
    sd = api_model_conversion(target, dict(ckpt["model"]))
    for s in shims:
        sd.pop(s, None)
    missing, unexpected = agent.model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if "__batch_size" not in m and "__device" not in m]
    assert not real_missing and not unexpected, (real_missing, unexpected)
    agent.model.eval()
    return agent, cfg, device


@torch.no_grad()
def encode(agent, obs: np.ndarray, device) -> torch.Tensor:
    o = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    return agent.model.encode(o, None)


@torch.no_grad()
def latent_rollout(agent, z0: torch.Tensor, actions: np.ndarray, device) -> np.ndarray:
    """Deterministic latent rollout; returns latents (T, latent_dim)."""
    z = z0
    out = []
    for a in actions:
        z = agent.model.next(z, torch.from_numpy(a).float().to(device).unsqueeze(0), None)
        out.append(z.squeeze(0).cpu().numpy())
    return np.array(out)


# ---------------------------------------------------------------- decoder

def perturb_actions(nominal: np.ndarray, m: float, u: np.ndarray,
                    t0: int, duration: int) -> np.ndarray:
    """The ctrl-channel poke: clip(a + m·u) inside the window, a elsewhere.

    Truth env and latent model receive the *identical* perturbed sequence, so
    "the same poke" is well-defined on both sides; m=0 returns `nominal`
    bit-for-bit (clip of in-bounds actions is the identity)."""
    out = nominal.copy()
    out[t0:t0 + duration] = np.clip(out[t0:t0 + duration] + m * u, -1.0, 1.0)
    return out


def fit_ridge(Z: np.ndarray, Y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """Closed-form ridge with bias feature; returns W of shape (dz+1, dy)."""
    X = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def apply_ridge(W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    return X @ W


def r2_per_coord(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    sse = ((Y - Yhat) ** 2).sum(axis=0)
    sst = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


class StateEmbedding:
    """Physical-observable representation of (qpos, qvel).

    Unbounded hinge angles (no joint limits) are embedded as (cos, sin): the
    winding number of a revolute angle is not a physical observable — two
    states differing by 2π are the same posture — and the task observation
    (which encodes orientations as cos/sin pairs) cannot and should not
    recover it. Slide joints and limited hinges stay raw; all velocities are
    raw (angular velocity is winding-independent). Root x translation is
    marked unobservable (translation-invariant observation).
    """

    def __init__(self, physics):
        import mujoco
        m = physics.model
        self.nq, self.nv = m.nq, m.nv
        self._raw: list[tuple[str, int]] = []     # (name, qpos index)
        self._wrap: list[tuple[str, int]] = []    # (name, qpos index) -> cos,sin
        vel_names = []
        for j in range(m.njnt):
            name = physics.model.id2name(j, "joint")
            adr = int(m.jnt_qposadr[j])
            unlimited_hinge = (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                               and not m.jnt_limited[j])
            if unlimited_hinge:
                self._wrap.append((name, adr))
            else:
                self._raw.append((name, adr))
            vel_names.append((name, int(m.jnt_dofadr[j])))
        self.names = ([f"qpos.{n}" for n, _ in self._raw]
                      + [x for n, _ in self._wrap for x in (f"qpos.{n}.cos", f"qpos.{n}.sin")]
                      + [f"qvel.{n}" for n, _ in vel_names])
        self._vel = vel_names
        self.pos_cols = np.array([c.startswith("qpos.") for c in self.names])
        self.observable = np.array([c != "qpos.rootx" for c in self.names])
        self.excluded = ["qpos.rootx (translation-invariant observation)"]
        self.wrapped = [n for n, _ in self._wrap]

    def transform(self, states_raw: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(states_raw)
        cols = [s[:, [i]] for _, i in self._raw]
        for _, i in self._wrap:
            cols += [np.cos(s[:, [i]]), np.sin(s[:, [i]])]
        cols += [s[:, [self.nq + d]] for _, d in self._vel]
        return np.concatenate(cols, axis=1)


class MLPDecoder:
    """Small MLP probe z -> (qpos, qvel); the ridge upgrade §7 allows."""

    def __init__(self, dz: int, dy: int, seed: int, hidden: int = 512):
        gen = torch.Generator().manual_seed(derive_seed(seed, "decoder"))
        def lin(i, o):
            l = torch.nn.Linear(i, o)
            with torch.no_grad():
                l.weight.copy_(torch.randn(o, i, generator=gen) / np.sqrt(i))
                l.bias.zero_()
            return l
        self.net = torch.nn.Sequential(lin(dz, hidden), torch.nn.SiLU(),
                                       lin(hidden, hidden), torch.nn.SiLU(),
                                       lin(hidden, dy))
        self.y_mean = torch.zeros(dy)
        self.y_std = torch.ones(dy)
        self._gen = gen

    def fit(self, Z: np.ndarray, Y: np.ndarray, epochs: int = 200, batch: int = 512,
            lr: float = 1e-3, device: str = "cpu"):
        Zt = torch.from_numpy(Z).float().to(device)
        Yt = torch.from_numpy(Y).float().to(device)
        self.y_mean = Yt.mean(0)
        self.y_std = Yt.std(0).clamp_min(1e-8)
        Yn = (Yt - self.y_mean) / self.y_std
        self.net.to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = Zt.shape[0]
        for _ in range(epochs):
            idx = torch.randperm(n, generator=self._gen).to(device)
            for b0 in range(0, n, batch):
                bi = idx[b0:b0 + batch]
                loss = torch.nn.functional.mse_loss(self.net(Zt[bi]), Yn[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, Z: np.ndarray) -> np.ndarray:
        device = next(self.net.parameters()).device
        out = self.net(torch.from_numpy(Z).float().to(device))
        return (out * self.y_std + self.y_mean).cpu().numpy().astype(np.float64)


def collect_episodes(agent, env: WalkerEnv, device, n_episodes: int = N_EPISODES):
    obs_l, z_l, state_l, ep_id, returns = [], [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        ret = 0.0
        for t in range(500):
            a = agent.act(torch.from_numpy(obs).to(device), t0=(t == 0), eval_mode=True)
            obs_l.append(obs)
            z_l.append(encode(agent, obs, device).squeeze(0).cpu().numpy())
            state_l.append(env.state_vec())
            ep_id.append(ep)
            obs, r = env.step(a.cpu().numpy())
            ret += r
        returns.append(ret)
        print(f"  episode {ep}: return {ret:.1f}")
    return (np.array(obs_l), np.array(z_l), np.array(state_l),
            np.array(ep_id), returns)


# ---------------------------------------------------------------- audit

def run(seed: int = 0) -> Path:
    t_start = time.time()
    agent, cfg, device = load_agent()
    env = WalkerEnv(seed=1)
    nq = env.physics.model.nq

    cache = NB_DIR / f"episodes-{TASK}-n{N_EPISODES}.npz"
    if cache.exists():
        print(f"[namebrand] loading cached episodes: {cache}")
        z = np.load(cache)
        obs, Z, states, ep_id = z["obs"], z["Z"], z["states"], z["ep_id"]
        returns = list(z["returns"])
    else:
        print("[namebrand] collecting episodes with the checkpoint agent...")
        obs, Z, states, ep_id, returns = collect_episodes(agent, env, device)
        np.savez_compressed(cache, obs=obs, Z=Z, states=states, ep_id=ep_id,
                            returns=np.array(returns))

    # ---- decoder probe (train/holdout split by episode; no leakage)
    # targets in the physical-observable embedding (see StateEmbedding)
    emb = StateEmbedding(env.physics)
    Y = emb.transform(states)
    coord_names = emb.names
    hold = ep_id >= (N_EPISODES - HOLDOUT_EPISODES)
    W = fit_ridge(Z[~hold], Y[~hold])
    r2_ridge = r2_per_coord(Y[hold], apply_ridge(W, Z[hold]))
    mlp = MLPDecoder(Z.shape[1], Y.shape[1], seed=seed).fit(
        Z[~hold], Y[~hold], device=device)
    pred_hold = mlp.predict(Z[hold])
    r2 = r2_per_coord(Y[hold], pred_hold)
    rmse = np.sqrt(((Y[hold] - pred_hold) ** 2).mean(axis=0))
    hold_std = Y[hold].std(axis=0)
    pos_r2 = r2[emb.pos_cols & emb.observable]
    kill = bool(pos_r2.min() < R2_KILL)
    print(f"[namebrand] decoder holdout R²: min(observable pos)={pos_r2.min():.4f} "
          f"median={np.median(r2):.4f}  kill={kill}")

    result = {
        "kind": "namebrand-results",
        "task": TASK, "checkpoint": str(CKPT), "model_size": "5M",
        "agent_returns": [round(float(r), 1) for r in returns],
        "decoder": {
            "type": "mlp(512,512)", "ridge_lambda_baseline": RIDGE_LAMBDA,
            "n_train": int((~hold).sum()), "n_holdout": int(hold.sum()),
            "representation": {
                "wrapped_angles_as_cos_sin": emb.wrapped,
                "why": "winding number of an unlimited revolute angle is not "
                       "a physical observable; posture is audited, not winding",
            },
            "r2_per_coord": {n: round(float(v), 5) for n, v in zip(coord_names, r2)},
            "r2_per_coord_ridge": {n: round(float(v), 5)
                                   for n, v in zip(coord_names, r2_ridge)},
            # context for low-variance coords: R² can fail while absolute
            # error is small; both are reported, the criterion stays R²
            "rmse_per_coord": {n: round(float(v), 5) for n, v in zip(coord_names, rmse)},
            "holdout_std_per_coord": {n: round(float(v), 5)
                                      for n, v in zip(coord_names, hold_std)},
            "excluded_coords": emb.excluded,
            "min_observable_pos_r2": float(pos_r2.min()),
            "kill_criterion": f"min observable-position R2 < {R2_KILL}",
            "killed": kill,
        },
        "seed": seed,
    }

    if not kill:
        result.update(_poke_audit(agent, env, mlp, device, seed, emb))

    result["wall_time_s"] = round(time.time() - t_start, 1)
    out_dir = RESULTS_DIR / "namebrand"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tdmpc2-{TASK}__{content_hash({'r': result['decoder'], 's': seed})}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[namebrand] wrote {out}")
    return out


def _poke_audit(agent, env: WalkerEnv, decoder, device, seed: int,
                emb: "StateEmbedding") -> dict:
    """Ctrl-poke protocol: same contrast logic, decoder R² as noise floor."""
    # warm up with the agent to a mid-episode state, then snapshot
    obs = env.reset()
    for t in range(WARMUP_STEPS):
        a = agent.act(torch.from_numpy(obs).to(device), t0=(t == 0), eval_mode=True)
        obs, _ = env.step(a.cpu().numpy())
    snap = env.snapshot()
    obs_star = obs.copy()

    # nominal action sequence: keep planning from the snapshot
    nominal = []
    o = obs
    for t in range(HORIZON):
        a = agent.act(torch.from_numpy(o).to(device), t0=False, eval_mode=True)
        nominal.append(a.cpu().numpy())
        o, _ = env.step(nominal[-1])
    nominal = np.array(nominal)
    u = np.ones(env.action_dim)  # uniform actuator bias: a physical "shove"

    def perturbed(m: float) -> np.ndarray:
        return perturb_actions(nominal, m, u, POKE_T0, POKE_DUR)

    def truth_traj(actions: np.ndarray) -> np.ndarray:
        env.restore(snap)
        out = np.empty((HORIZON, env.physics.model.nq + env.physics.model.nv))
        for t, a in enumerate(actions):
            env.step(a)
            out[t] = env.state_vec()
        return emb.transform(out)

    def model_traj(actions: np.ndarray) -> np.ndarray:
        z0 = encode(agent, obs_star, device)
        lat = latent_rollout(agent, z0, actions.astype(np.float32), device)
        return decoder.predict(lat)

    # ---- null test (sacred): zero perturbation ⇒ bitwise identical, both sides
    tn1, tn2 = truth_traj(nominal), truth_traj(perturbed(0.0))
    assert tn1.tobytes() == tn2.tobytes(), "truth null test FAILED (ctrl channel)"
    mn1, mn2 = model_traj(nominal), model_traj(perturbed(0.0))
    assert mn1.tobytes() == mn2.tobytes(), "model null test FAILED (ctrl channel)"
    print("[namebrand] ctrl-channel null test: PASS (truth + model)")

    truth_null, model_null = tn1, mn1
    obs_idx = emb.observable
    r_t, r_m = [], []
    for m in MAGNITUDES:
        acts = perturbed(float(m))
        dt = truth_traj(acts) - truth_null
        dm = model_traj(acts) - model_null
        r_t.append(float(np.linalg.norm(dt[-1][obs_idx])))
        r_m.append(float(np.linalg.norm(dm[-1][obs_idx])))
        print(f"  m={m:5.2f}  truth response {r_t[-1]:8.3f}   model response {r_m[-1]:8.3f}")

    r_t, r_m = np.array(r_t), np.array(r_m)
    score = float(np.trapezoid(np.abs(r_m - r_t), MAGNITUDES)
                  / max(np.trapezoid(r_t, MAGNITUDES), 1e-12))
    return {
        "poke": {"channel": "ctrl", "pattern": "uniform +u on all actuators",
                 "t0": POKE_T0, "duration": POKE_DUR, "horizon": HORIZON,
                 "magnitudes": [round(float(m), 4) for m in MAGNITUDES]},
        "null_test": "pass",
        "subtests": [{
            "name": "response",
            "poke_id": "ctrl",
            "score": round(score, 6),
            "light": config.light(score),
            "curves": {
                "x": [round(float(m), 4) for m in MAGNITUDES],
                "x_label": "ctrl perturbation magnitude",
                "y_truth": [round(v, 6) for v in r_t],
                "y_model": [round(v, 6) for v in r_m],
                "ci": {"lo": [round(v, 6) for v in r_m], "hi": [round(v, 6) for v in r_m]},
            },
            "diverged": False,
            "details": {"K": 1, "note": "deterministic latent model; no sampling band",
                        "excluded": "qpos.rootx",
                        "noise_floor": "decoder holdout R2, see decoder block"},
        }],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(seed=args.seed)
