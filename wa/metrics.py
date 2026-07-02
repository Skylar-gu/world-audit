"""Sub-test scores: pure functions of contrast outputs (spec §5).

Inputs are magnitude-ordered lists of ContrastSets — truth (K=1) and model
(K CRN samples) — plus scene-registry parameters. No physics is touched here.

Non-finite guard: diverged model rollouts yield capped scores and red lights;
curves are sanitized for JSON with a `diverged` flag rather than hidden.

Momentum caveat (verbatim requirement): the momentum score compares the
model's *change* in total momentum across the collision window to the truth's
change — never to zero — since MuJoCo contacts with friction/solver softness
are not exactly conservative. Site copy must say "vs. simulated ground
truth," not "vs. conservation law."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wa import config
from wa.contrast import ContrastSet
from wa.scenes import SceneSpec

_TINY = 1e-12


@dataclass
class MetricResult:
    name: str
    poke_id: str
    score: float                    # pre-cap score; may be inf
    light: str
    x: list                         # curve abscissa (magnitudes, or chain index)
    x_label: str
    y_truth: list
    y_model: list                   # K-sample mean
    ci_lo: list                     # K-sample band (10th pct)
    ci_hi: list                     # (90th pct)
    diverged: bool = False
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "poke_id": self.poke_id,
            "score": round(min(self.score, config.SCORE_CAP), 6)
                     if np.isfinite(self.score) else config.SCORE_CAP,
            "light": self.light,
            "curves": {
                "x": _san(self.x), "x_label": self.x_label,
                "y_truth": _san(self.y_truth), "y_model": _san(self.y_model),
                "ci": {"lo": _san(self.ci_lo), "hi": _san(self.ci_hi)},
            },
            "diverged": self.diverged,
            "details": self.details,
        }


def _san(xs) -> list:
    """JSON-safe curve values: non-finite -> None (site renders a gap)."""
    out = []
    for v in np.asarray(xs, dtype=np.float64).ravel():
        out.append(round(float(v), 6) if np.isfinite(v) else None)
    return out


def _finite(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=np.inf)


def _result(name: str, poke_id: str, score: float, x, x_label, y_truth, y_model,
            ci_lo, ci_hi, diverged: bool, details: dict | None = None) -> MetricResult:
    return MetricResult(
        name=name, poke_id=poke_id, score=float(score), light=config.light(score),
        x=list(x), x_label=x_label, y_truth=list(y_truth), y_model=list(y_model),
        ci_lo=list(ci_lo), ci_hi=list(ci_hi), diverged=diverged,
        details=details or {})


def _any_diverged(sets: list[ContrastSet]) -> bool:
    return any(cs.diverged_at is not None for cs in sets)


# ------------------------------------------------------------------ response

def response_curve(spec: SceneSpec, truth: list[ContrastSet],
                   model: list[ContrastSet]) -> MetricResult:
    """r(m) = ||Δx_target(H; m)||, target = poked body; score = normalized area
    between truth and model-mean curves over the magnitude grid."""
    mags = spec.magnitudes()
    target = truth[0].poke.site
    key = f"{target}.pos"
    r_t = np.array([np.linalg.norm(cs.contrast[key][0, -1]) for cs in truth])
    r_k = np.stack([np.linalg.norm(cs.contrast[key][:, -1], axis=-1) for cs in model],
                   axis=1)                      # (K, M)
    r_k = _finite(r_k)
    r_m = np.nanmean(np.where(np.isfinite(r_k), r_k, np.nan), axis=0)
    r_m_scored = np.where(np.isfinite(r_m), r_m, np.nanmax(r_t) * 100)
    score = float(np.trapezoid(np.abs(r_m_scored - r_t), mags)
                  / max(np.trapezoid(r_t, mags), _TINY))
    lo = np.percentile(np.where(np.isfinite(r_k), r_k, np.nan), 10, axis=0)
    hi = np.percentile(np.where(np.isfinite(r_k), r_k, np.nan), 90, axis=0)
    return _result("response", target, score, mags, "poke magnitude (N)",
                   r_t, r_m, lo, hi, _any_diverged(model),
                   {"horizon_steps": int(truth[0].contrast[key].shape[1]),
                    "target_body": target})


# ------------------------------------------------------------------ momentum

def momentum_check(spec: SceneSpec, truth: list[ContrastSet],
                   model: list[ContrastSet], window: int = 25) -> MetricResult:
    """Change in total linear momentum across a window bracketing the first
    collision; the model's Δp is compared to the truth's Δp (the reference is
    the simulated ground truth, never a conservation law)."""
    mags = spec.magnitudes()
    first_hit = spec.chain[1]  # first collision partner of the poked body
    bodies = spec.tracked_bodies

    def total_p(cs: ContrastSet) -> np.ndarray:  # (K, T, 3)
        return sum(cs.contrast[f"{b}.mom"] for b in bodies)

    y_t, y_m, lo, hi, per_mag = [], [], [], [], []
    for cs_t, cs_m in zip(truth, model):
        arr = np.linalg.norm(cs_t.contrast[f"{first_hit}.pos"][0], axis=-1)
        hit = np.flatnonzero(arr > spec.arrival_eps)
        if hit.size == 0:  # no collision at this magnitude: subtest undefined
            y_t.append(np.nan); y_m.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        t_c = int(hit[0])
        t0, t1 = max(t_c - window, 0), min(t_c + window, arr.shape[0] - 1)
        dp_t = total_p(cs_t)[0, t1] - total_p(cs_t)[0, t0]          # (3,)
        dp_m = total_p(cs_m)[:, t1] - total_p(cs_m)[:, t0]          # (K, 3)
        dp_m = _finite(dp_m)
        err = np.linalg.norm(dp_m.mean(axis=0) - dp_t) / max(np.linalg.norm(dp_t), _TINY)
        per_mag.append(err)
        norms = np.linalg.norm(dp_m, axis=-1)
        y_t.append(np.linalg.norm(dp_t)); y_m.append(np.mean(norms))
        lo.append(np.percentile(norms, 10)); hi.append(np.percentile(norms, 90))
    score = float(np.mean(per_mag)) if per_mag else np.inf
    return _result("momentum", truth[0].poke.site, score, mags, "poke magnitude (N)",
                   y_t, y_m, lo, hi, _any_diverged(model),
                   {"window_steps": window, "collision_body": first_hit,
                    "reference": "simulated ground truth",
                    "n_magnitudes_with_collision": len(per_mag)})


# ---------------------------------------------------------------- propagation

def _arrival_times(cs: ContrastSet, chain: tuple[str, ...], eps: float) -> np.ndarray:
    """(K, n_links) first step where ||Δx_i(t)|| > eps; horizon if never."""
    K = cs.K
    T = next(iter(cs.contrast.values())).shape[1]
    out = np.full((K, len(chain)), T, dtype=np.float64)
    for i, body in enumerate(chain):
        norms = _finite(np.linalg.norm(cs.contrast[f"{body}.pos"], axis=-1))  # (K, T)
        for k in range(K):
            idx = np.flatnonzero(norms[k] > eps)
            if idx.size:
                out[k, i] = idx[0]
    return out


def propagation(spec: SceneSpec, truth: list[ContrastSet],
                model: list[ContrastSet]) -> MetricResult:
    """Arrival-time profiles along the chain (slope = propagation speed);
    score = mean |t_M − t_T| over links and magnitudes, normalized by mean t_T."""
    per_mag_err = []
    for cs_t, cs_m in zip(truth, model):
        t_t = _arrival_times(cs_t, spec.chain, spec.arrival_eps)[0]     # (L,)
        t_m = _arrival_times(cs_m, spec.chain, spec.arrival_eps).mean(axis=0)
        per_mag_err.append(np.mean(np.abs(t_m - t_t)) / max(np.mean(t_t), 1.0))
    score = float(np.mean(per_mag_err))

    # site curve: arrival profile at the max magnitude
    t_t = _arrival_times(truth[-1], spec.chain, spec.arrival_eps)[0]
    prof = _arrival_times(model[-1], spec.chain, spec.arrival_eps)      # (K, L)
    return _result("propagation", truth[0].poke.site, score,
                   list(range(len(spec.chain))), "chain link (from poked body)",
                   t_t, prof.mean(axis=0),
                   np.percentile(prof, 10, axis=0), np.percentile(prof, 90, axis=0),
                   _any_diverged(model),
                   {"eps": spec.arrival_eps, "chain": list(spec.chain),
                    "profile_at_magnitude": float(spec.magnitudes()[-1]),
                    "per_magnitude_err": _san(per_mag_err)})


# ------------------------------------------------------------------ stability

def _threshold_on_grid(mags: np.ndarray, topple: np.ndarray) -> tuple[float, str]:
    """First non-topple→topple crossing; geometric midpoint of the bracket."""
    idx = np.flatnonzero(topple)
    if idx.size == 0:
        return float(mags[-1]), "never-topples"
    i = int(idx[0])
    if i == 0:
        return float(mags[0]), "topples-at-min"
    return float(np.sqrt(mags[i - 1] * mags[i])), "bracketed"


def stability(spec: SceneSpec, truth: list[ContrastSet],
              model: list[ContrastSet]) -> MetricResult:
    """Topple indicator vs. magnitude; score = |m*_M − m*_T| / m*_T."""
    mags = spec.magnitudes()
    key = f"{spec.topple_body}.pos"
    z_end_t = np.array([cs.poked[key][0, -1, 2] for cs in truth])
    topple_t = z_end_t < spec.topple_z
    z_end_m = _finite(np.stack([cs.poked[key][:, -1, 2] for cs in model], axis=1))  # (K, M)
    frac_m = (z_end_m < spec.topple_z).mean(axis=0)                                 # (M,)
    topple_m = frac_m >= 0.5

    m_t, kind_t = _threshold_on_grid(mags, topple_t)
    m_m, kind_m = _threshold_on_grid(mags, topple_m)
    score = abs(m_m - m_t) / max(m_t, _TINY)
    return _result("stability", truth[0].poke.site, score, mags, "poke magnitude (N)",
                   topple_t.astype(float), frac_m,
                   np.minimum(frac_m, topple_m.astype(float)),
                   np.maximum(frac_m, topple_m.astype(float)),
                   _any_diverged(model),
                   {"topple_body": spec.topple_body, "topple_z": spec.topple_z,
                    "m_star_truth": m_t, "m_star_model": m_m,
                    "truth_bracket": kind_t, "model_bracket": kind_m,
                    "disagreement_mags": _san(mags[topple_t != topple_m])})


SUBTESTS = {
    "response": response_curve,
    "momentum": momentum_check,
    "propagation": propagation,
    "stability": stability,
}


def run_subtests(spec: SceneSpec, truth: list[ContrastSet],
                 model: list[ContrastSet]) -> list[MetricResult]:
    return [SUBTESTS[name](spec, truth, model) for name in spec.subtests]
