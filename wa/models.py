"""Ensemble dynamics models (PETS-style) + training loop + CRN model rollouts.

- Input (s_t, a_t): s = (qpos, qvel) flattened, a = applied-force channels
  (oracle.force_channels), so "the same poke" is well-defined on the model side.
- Target Δs = s_{t+1} − s_t; heads output Gaussian mean + diagonal log-variance;
  loss = NLL; inputs/targets normalized by training-set stats (stored in ckpt).
- Spectrum per scene: capacities {2×64, 3×256, 4×512} × budgets {5k, 100k}.
- CRN contract (§4): the poked and null rollouts for sample k use the same
  ensemble-member assignment and the same head-noise sequence, so the k-th
  contrast is a common-random-numbers estimate and the null contrast is
  exactly zero by construction.

All randomness flows through torch.Generator objects seeded via
wa.schema.derive_seed — no global torch seeding (invariant 3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wa import data as wa_data
from wa.oracle import Oracle, manifest_stamp
from wa.schema import Poke, RolloutPair, content_hash, derive_seed

MODELS_DIR = Path("experiments/models")
MANIFEST_DIR = Path("experiments/manifests")

CAPACITIES = {
    "2x64": (64, 64),
    "3x256": (256, 256, 256),
    "4x512": (512, 512, 512, 512),
}
E = 5  # ensemble size
LOGVAR_MIN, LOGVAR_MAX = -10.0, 2.0


class EnsembleLinear(nn.Module):
    """E parallel linear layers, batched as (E, B, ·) bmm."""

    def __init__(self, e: int, din: int, dout: int, gen: torch.Generator):
        super().__init__()
        scale = 1.0 / np.sqrt(din)
        self.w = nn.Parameter(torch.randn(e, din, dout, generator=gen) * scale)
        self.b = nn.Parameter(torch.zeros(e, 1, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (E, B, din) -> (E, B, dout)
        return torch.baddbmm(self.b, x, self.w)


class EnsembleDynamics(nn.Module):
    """E-member probabilistic MLP ensemble predicting normalized Δs."""

    def __init__(self, ds: int, da: int, hidden: tuple[int, ...], seed: int, e: int = E):
        super().__init__()
        self.ds, self.da, self.e, self.hidden = ds, da, e, tuple(hidden)
        gen = torch.Generator().manual_seed(derive_seed(seed, "init"))
        dims = [ds + da, *hidden]
        self.layers = nn.ModuleList(
            [EnsembleLinear(e, i, o, gen) for i, o in zip(dims[:-1], dims[1:])])
        self.head = EnsembleLinear(e, dims[-1], 2 * ds, gen)
        # normalizers (populated by fit_normalizer, saved with the checkpoint)
        for name, dim in [("in_mean", ds + da), ("in_std", ds + da),
                          ("tgt_mean", ds), ("tgt_std", ds)]:
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))

    def fit_normalizer(self, x: torch.Tensor, tgt: torch.Tensor) -> None:
        self.in_mean.copy_(x.mean(0))
        self.in_std.copy_(x.std(0).clamp_min(1e-8))
        self.tgt_mean.copy_(tgt.mean(0))
        self.tgt_std.copy_(tgt.std(0).clamp_min(1e-8))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (E, B, ds+da) raw units -> (mean, logvar) of normalized Δs, (E, B, ds)."""
        h = (x - self.in_mean) / self.in_std
        for layer in self.layers:
            h = torch.nn.functional.silu(layer(h))
        out = self.head(h)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = LOGVAR_MAX - torch.nn.functional.softplus(LOGVAR_MAX - logvar)
        logvar = LOGVAR_MIN + torch.nn.functional.softplus(logvar - LOGVAR_MIN)
        return mean, logvar

    def nll(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL of normalized targets, mean over members/batch/dims."""
        mean, logvar = self(x)
        tgt_n = (tgt - self.tgt_mean) / self.tgt_std
        return 0.5 * (((tgt_n - mean) ** 2) * torch.exp(-logvar) + logvar).mean()

    def predict_delta(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Sampled Δs in raw units; eps is the CRN head-noise, shape (E, B, ds)."""
        mean, logvar = self(x)
        d_n = mean + eps * torch.exp(0.5 * logvar)
        return d_n * self.tgt_std + self.tgt_mean


def train_cell(scene: str, capacity: str, budget: str, seed: int = 0,
               device: str | None = None, epochs: int = 40,
               batch: int = 256, lr: float = 1e-3,
               min_steps: int = 6000) -> Path:
    """Train one (capacity, budget) cell; returns checkpoint path.

    Small budgets get extra epochs so every cell sees ≥ min_steps gradient
    steps — otherwise the capacity axis confounds "too little data" with
    "too little training."
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = wa_data.load(scene, budget)
    x = torch.from_numpy(np.concatenate([d["obs"], d["act"]], axis=1)).float()
    tgt = torch.from_numpy(d["next_obs"] - d["obs"]).float()
    hx = torch.from_numpy(
        np.concatenate([d["holdout_obs"], d["holdout_act"]], axis=1)).float().to(device)
    htgt = torch.from_numpy(d["holdout_next_obs"] - d["holdout_obs"]).float().to(device)

    ds, da = d["obs"].shape[1], d["act"].shape[1]
    model_seed = derive_seed(seed, scene, capacity, budget)
    net = EnsembleDynamics(ds, da, CAPACITIES[capacity], seed=model_seed).to(device)
    net.fit_normalizer(x.to(device), tgt.to(device))

    x, tgt = x.to(device), tgt.to(device)
    n = x.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(derive_seed(model_seed, "batches"))
    batches_per_epoch = max(1, (n + batch - 1) // batch)
    epochs = max(epochs, (min_steps + batches_per_epoch - 1) // batches_per_epoch)
    for epoch in range(epochs):
        idx = torch.stack([torch.randint(n, (n,), generator=gen) for _ in range(E)])  # bootstrap
        for b0 in range(0, n, batch):
            bi = idx[:, b0:b0 + batch].to(device)
            loss = net.nll(x[bi], tgt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        holdout_nll = float(net.nll(hx.unsqueeze(0).expand(E, -1, -1), htgt))
        train_nll = float(net.nll(x[:10_000].unsqueeze(0).expand(E, -1, -1), tgt[:10_000]))
        hmean, _ = net(hx.unsqueeze(0).expand(E, -1, -1))
        htgt_n = (htgt - net.tgt_mean) / net.tgt_std
        holdout_mse = float(((hmean - htgt_n) ** 2).mean())

    model_id = f"{scene}-{capacity}-{budget}"
    manifest = {
        **manifest_stamp(Oracle(scene).scene),
        "kind": "model", "model_id": model_id,
        "capacity": capacity, "budget": budget, "ensemble": E,
        "seed": model_seed, "epochs": epochs, "batch": batch, "lr": lr,
        "data_path": d["path"], "torch": torch.__version__,
        "holdout_nll": holdout_nll, "train_nll": train_nll, "holdout_mse": holdout_mse,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_id}-{content_hash(manifest)}.pt"
    torch.save({
        "state_dict": net.cpu().state_dict(),
        "ds": ds, "da": da, "hidden": CAPACITIES[capacity], "e": E,
        "manifest": manifest,
    }, path)
    (MANIFEST_DIR / f"model-{model_id}-{content_hash(manifest)}.json").write_text(
        json.dumps(manifest, indent=2))
    return path


def load_model(path: str | Path) -> tuple[EnsembleDynamics, dict]:
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    net = EnsembleDynamics(ckpt["ds"], ckpt["da"], tuple(ckpt["hidden"]), seed=0, e=ckpt["e"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["manifest"]


@torch.no_grad()
def rollout_model(net: EnsembleDynamics, oracle: Oracle, init_state: np.ndarray,
                  poke: Poke | None, K: int, seed: int,
                  horizon: int | None = None, model_id: str = "model",
                  with_tracked: bool = True) -> list[RolloutPair]:
    """K iterated one-step rollouts under the CRN contract.

    Sample k draws its ensemble member and full head-noise sequence from a
    torch.Generator seeded by derive_seed(seed, "crn", k) — identical for the
    poked and null runs of the same (pair, k), making each contrast a CRN
    estimate (null contrast exactly zero by construction).

    Compounding one-step error is the phenomenon under audit, not a bug.
    """
    horizon = horizon or oracle.scene.horizon
    ds = oracle.nq + oracle.nv
    s0 = oracle.qpos_qvel_from_snapshot(init_state)
    a_seq = torch.from_numpy(oracle.force_channels(poke, horizon)).float()

    members = torch.empty(K, dtype=torch.long)
    eps = torch.empty(K, horizon, ds)
    for k in range(K):
        gk = torch.Generator().manual_seed(derive_seed(seed, "crn", k))
        members[k] = torch.randint(net.e, (1,), generator=gk)
        eps[k] = torch.randn(horizon, ds, generator=gk)

    s = torch.from_numpy(s0).float().unsqueeze(0).repeat(K, 1)  # (K, ds)
    states = torch.empty(K, horizon, ds)
    arange = torch.arange(K)
    for t in range(horizon):
        x = torch.cat([s, a_seq[t].unsqueeze(0).expand(K, -1)], dim=1)
        d_all = net.predict_delta(
            x.unsqueeze(0).expand(net.e, -1, -1),
            eps[:, t].unsqueeze(0).expand(net.e, -1, -1))  # (E, K, ds)
        s = s + d_all[members, arange]
        states[:, t] = s

    out = []
    for k in range(K):
        st = states[k].double().numpy()
        out.append(RolloutPair(
            scene=oracle.scene.name, model_id=model_id, poke=poke, states=st,
            tracked=oracle.tracked_from_states(st) if with_tracked else {},
            seed=derive_seed(seed, "crn", k),
            manifest=manifest_stamp(oracle.scene)))
    return out


def train_spectrum(scene: str, seed: int = 0) -> None:
    rows = []
    for budget in wa_data.BUDGETS:
        for capacity in CAPACITIES:
            path = train_cell(scene, capacity, budget, seed=seed)
            m = json.loads((MANIFEST_DIR / f"model-{path.stem}.json").read_text())
            rows.append((capacity, budget, m["holdout_nll"], m["holdout_mse"]))
            print(f"{scene:10s} {capacity:6s} {budget:5s} "
                  f"train_nll={m['train_nll']:+.4f} holdout_nll={m['holdout_nll']:+.4f} "
                  f"holdout_mse={m['holdout_mse']:.4f}")
    print("\nspectrum (holdout NLL / normalized MSE, lower = better):")
    for capacity, budget, h, mse in rows:
        print(f"  {capacity:6s} x {budget:5s} -> nll {h:+.4f}   mse {mse:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_spectrum(args.scene, seed=args.seed)
