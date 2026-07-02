"""Dataclasses + JSON (de)serialization. The only source of types.

All randomness in the codebase flows through explicit seeds declared here:
use `rng(seed)` / `derive_seed(...)` — never bare `np.random` or global torch
seeding (CLAUDE.md invariant 3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np


def rng(seed: int) -> np.random.Generator:
    """The only sanctioned way to obtain randomness."""
    return np.random.default_rng(seed)


def derive_seed(root_seed: int, *keys: object) -> int:
    """Deterministic child seed from a root seed and structured keys.

    Used to give every (pair, sample-k, purpose) its own independent stream.
    """
    h = hashlib.sha256(repr((root_seed, *keys)).encode()).digest()
    return int.from_bytes(h[:8], "little")


@dataclass
class Poke:
    """One intervention: a world-frame force held on a body for a step window."""

    site: str                 # named body/site in the scene registry
    direction: np.ndarray     # unit vector, world frame, shape (3,)
    magnitude: float          # Newtons — grid of 8 log-spaced values per scene
    t_start: int              # step index at which force turns on
    duration: int             # steps force is held (via xfrc_applied on the body)

    def __post_init__(self):
        self.direction = np.asarray(self.direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(self.direction))
        if not np.isclose(norm, 1.0):
            raise ValueError(f"poke direction must be a unit vector, got |d|={norm}")

    @property
    def force(self) -> np.ndarray:
        return self.direction * self.magnitude

    def to_json(self) -> dict:
        return {
            "site": self.site,
            "direction": self.direction.tolist(),
            "magnitude": self.magnitude,
            "t_start": self.t_start,
            "duration": self.duration,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Poke":
        return cls(
            site=d["site"],
            direction=np.asarray(d["direction"], dtype=np.float64),
            magnitude=float(d["magnitude"]),
            t_start=int(d["t_start"]),
            duration=int(d["duration"]),
        )


@dataclass
class RolloutPair:
    """The atomic audit object: one tracked trajectory from one source."""

    scene: str
    model_id: str             # "oracle" for ground truth
    poke: Poke | None         # None == null rollout
    states: np.ndarray        # (T, nq+nv) tracked full state trajectory
    tracked: dict[str, np.ndarray] = field(default_factory=dict)
    seed: int = 0
    manifest: dict = field(default_factory=dict)  # git SHA, lib versions, scene/config hash

    def to_json(self) -> dict:
        return {
            "scene": self.scene,
            "model_id": self.model_id,
            "poke": self.poke.to_json() if self.poke is not None else None,
            "states": self.states.tolist(),
            "tracked": {k: v.tolist() for k, v in self.tracked.items()},
            "seed": self.seed,
            "manifest": self.manifest,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RolloutPair":
        return cls(
            scene=d["scene"],
            model_id=d["model_id"],
            poke=Poke.from_json(d["poke"]) if d["poke"] is not None else None,
            states=np.asarray(d["states"], dtype=np.float64),
            tracked={k: np.asarray(v, dtype=np.float64) for k, v in d["tracked"].items()},
            seed=int(d["seed"]),
            manifest=dict(d["manifest"]),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_json())

    @classmethod
    def loads(cls, s: str) -> "RolloutPair":
        return cls.from_json(json.loads(s))


def content_hash(obj: dict) -> str:
    """Stable content hash for artifact addressing (invariant 4)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]
