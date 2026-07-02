"""Grid orchestration: scene × model × poke × magnitude → artifacts + manifest.

Per run:
1. Contrast null test (bitwise) for truth and EVERY model — abort on failure.
2. Truth contrasts across the magnitude grid (null rollout shared).
3. Per model: K-sample CRN contrasts, metrics, rollout npz assets (for render),
   one results JSON (content-addressed; existing key == skip, invariant 4).
4. Rebuild site/public/manifest.json from all results JSONs on disk, so partial
   runs always produce a servable site.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wa import config, metrics as wa_metrics
from wa.contrast import assert_null_contrast_zero, model_contrast, truth_contrast
from wa.models import MODELS_DIR, load_model
from wa.oracle import Oracle, manifest_stamp
from wa.schema import Poke, content_hash, derive_seed
from wa.scenes import REGISTRY, get_scene

ARTIFACTS = Path("experiments/artifacts")
RESULTS_DIR = ARTIFACTS / "results"
ROLLOUTS_DIR = ARTIFACTS / "rollouts"
MANIFEST_DIR = Path("experiments/manifests")
SITE_MANIFEST = Path("site/public/manifest.json")


def discover_models(scene: str) -> dict[str, Path]:
    """model_id -> newest checkpoint path."""
    out: dict[str, Path] = {}
    for p in sorted(MODELS_DIR.glob(f"{scene}-*.pt"), key=lambda p: p.stat().st_mtime):
        model_id = "-".join(p.stem.split("-")[:-1])  # strip content hash
        out[model_id] = p
    return out


def _audit_pokes(spec) -> tuple[str, list[Poke]]:
    site = spec.poke_sites[0]
    direction = np.array(spec.poke_directions[site])
    pokes = [Poke(site=site, direction=direction, magnitude=float(m),
                  t_start=spec.poke_t_start, duration=spec.poke_duration)
             for m in spec.magnitudes()]
    return site, pokes


def _save_states(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():  # immutable artifacts
        np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in arrays.items()})


def run_grid(scene: str, seed: int = config.AUDIT_SEED, K: int = config.K_SAMPLES,
             models: dict[str, object] | None = None,
             horizon: int | None = None) -> list[Path]:
    """Returns the list of results-JSON paths for this scene."""
    spec = get_scene(scene)
    oracle = Oracle(spec)
    horizon = horizon or spec.horizon
    site, pokes = _audit_pokes(spec)
    zero_poke = Poke(site=site, direction=np.array(spec.poke_directions[site]),
                     magnitude=0.0, t_start=spec.poke_t_start,
                     duration=spec.poke_duration)

    if models is None:
        found = discover_models(scene)
        if not found:
            raise FileNotFoundError(
                f"no checkpoints for {scene!r}; run `make train SCENE={scene}`")
        models = {mid: load_model(p) for mid, p in found.items()}
    # normalize injected values to (net, manifest) tuples
    models = {mid: v if isinstance(v, tuple) else (v, {"model_id": mid})
              for mid, v in models.items()}

    # ---- 1. contrast null tests, truth + every model (sacred; abort on failure)
    s0 = oracle.init_state(seed=derive_seed(seed, "audit-init"))
    assert_null_contrast_zero(truth_contrast(oracle, s0, zero_poke, horizon=300))
    for mid, (net, _) in models.items():
        assert_null_contrast_zero(model_contrast(
            net, oracle, s0, zero_poke, K=min(K, 4),
            seed=derive_seed(seed, "nulltest", mid), model_id=mid, horizon=300))
    print(f"[{scene}] contrast null test: PASS (truth + {len(models)} models)")

    # ---- 2. truth contrasts (shared null)
    truth_null = oracle.run(s0, None, horizon=horizon)
    truth_sets = [truth_contrast(oracle, s0, p, horizon=horizon, null=truth_null)
                  for p in pokes]
    for mi, cs in enumerate(truth_sets):
        _save_states(ROLLOUTS_DIR / scene / f"oracle__{site}__m{mi}.npz",
                     poked=cs.poked_states_k0, null=cs.null_states_k0)

    # ---- 3. per-model contrasts + metrics + artifacts
    out_paths = []
    stamp = manifest_stamp(spec)
    for mid, (net, model_manifest) in models.items():
        key = content_hash({
            "scene_hash": spec.xml_hash(), "model": model_manifest,
            "poke_site": site, "mags": [p.magnitude for p in pokes],
            "K": K, "seed": seed, "horizon": horizon,
            "thresholds": [config.LIGHT_GREEN, config.LIGHT_YELLOW],
        })
        rpath = RESULTS_DIR / scene / f"{mid}__{key}.json"
        out_paths.append(rpath)
        if rpath.exists():
            print(f"[{scene}] {mid}: results exist ({key}), skipping")
            continue
        t0 = time.time()
        model_sets = [model_contrast(net, oracle, s0, p, K=K,
                                     seed=derive_seed(seed, scene, mid, site, mi),
                                     model_id=mid, horizon=horizon)
                      for mi, p in enumerate(pokes)]
        for mi, cs in enumerate(model_sets):
            _save_states(ROLLOUTS_DIR / scene / f"{mid}__{site}__m{mi}.npz",
                         poked=cs.poked_states_k0)
        results = wa_metrics.run_subtests(spec, truth_sets, model_sets)
        diverged_at = [cs.diverged_at for cs in model_sets]
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps({
            **stamp, "kind": "grid-results", "model_id": mid, "key": key,
            "model_manifest": model_manifest, "poke_id": site,
            "magnitudes": [p.magnitude for p in pokes],
            "K": K, "seed": seed, "horizon": horizon,
            "null_test": "pass",
            "diverged_at_per_magnitude": diverged_at,
            "subtests": [r.to_json() for r in results],
        }, indent=2))
        lights = " ".join(f"{r.name}:{r.light}({min(r.score, config.SCORE_CAP):.3f})"
                          for r in results)
        print(f"[{scene}] {mid}: {lights}  [{time.time() - t0:.1f}s]")

    # ---- 4. site manifest + run manifest
    rebuild_site_manifest()
    run_manifest = {**stamp, "kind": "grid-run", "K": K, "seed": seed,
                    "horizon": horizon, "models": sorted(models),
                    "results": [str(p) for p in out_paths]}
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"grid-{scene}-{content_hash(run_manifest)}.json").write_text(
        json.dumps(run_manifest, indent=2))
    return out_paths


def rebuild_site_manifest() -> dict:
    """Site manifest from newest results per (scene, model): §2 schema.
    scenes[] -> models[] -> subtests[] -> entries[]."""
    newest: dict[tuple[str, str], dict] = {}
    for p in sorted(RESULTS_DIR.glob("*/*.json"), key=lambda p: p.stat().st_mtime):
        r = json.loads(p.read_text())
        newest[(r["scene"], r["model_id"])] = r

    scenes = []
    for scene in REGISTRY:
        model_rows = []
        for (sc, mid), r in sorted(newest.items()):
            if sc != scene:
                continue
            subtest_rows = []
            for st in r["subtests"]:
                entries = [{
                    "poke_id": st["poke_id"],
                    "magnitude": mag,
                    "clip_url": f"clips/{scene}/{mid}__{st['poke_id']}__m{mi}.webm",
                    "curves": st["curves"],
                    "score": st["score"],
                    "light": st["light"],
                } for mi, mag in enumerate(r["magnitudes"])]
                subtest_rows.append({
                    "name": st["name"], "score": st["score"], "light": st["light"],
                    "diverged": st["diverged"], "details": st["details"],
                    "entries": entries,
                })
            model_rows.append({
                "model_id": mid,
                "capacity": r["model_manifest"].get("capacity"),
                "budget": r["model_manifest"].get("budget"),
                "holdout_nll": r["model_manifest"].get("holdout_nll"),
                "null_test": r["null_test"],
                "subtests": subtest_rows,
            })
        if model_rows:
            scenes.append({"name": scene, "models": model_rows})

    manifest = {
        "generated_unix": int(time.time()),
        "config": {
            "light_green": config.LIGHT_GREEN, "light_yellow": config.LIGHT_YELLOW,
            "k_samples": config.K_SAMPLES, "score_cap": config.SCORE_CAP,
            "momentum_reference": "simulated ground truth",
        },
        "scenes": scenes,
    }
    SITE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SITE_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, default=config.AUDIT_SEED)
    ap.add_argument("--k", type=int, default=config.K_SAMPLES)
    args = ap.parse_args()
    run_grid(args.scene, seed=args.seed, K=args.k)
