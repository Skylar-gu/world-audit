"""Scene registry: scene → XML, tracked bodies, poke sites, magnitude range.

Magnitude ranges are calibrated so the max is dramatic but non-degenerate
(no bodies ejected from the arena); re-check with `tests/test_scenes.py`
whenever an XML changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCENES_DIR = Path(__file__).parent


@dataclass(frozen=True)
class SceneSpec:
    name: str
    xml: str                        # filename within wa/scenes/
    tracked_bodies: tuple[str, ...]
    poke_sites: tuple[str, ...]     # body names eligible for xfrc_applied pokes
    poke_directions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    magnitude_range: tuple[float, float] = (0.5, 20.0)  # Newtons, log-spaced grid endpoints
    n_magnitudes: int = 8
    horizon: int = 1500             # steps per rollout (3 s at timestep 0.002)
    settle_steps: int = 200         # pre-roll before init snapshot (warm-start non-trivial)
    poke_t_start: int = 10          # steps after init snapshot
    poke_duration: int = 10         # steps force is held
    arrival_eps: float = 1e-4       # propagation threshold; must be ≥10× null-test floor
    subtests: tuple[str, ...] = ("response",)
    chain: tuple[str, ...] = ()     # body order for propagation arrival profiles
    topple_body: str = ""           # blocks stability subtest
    topple_z: float = 0.0           # topple indicator: body z at horizon < topple_z

    @property
    def xml_path(self) -> Path:
        return SCENES_DIR / self.xml

    def xml_hash(self) -> str:
        return hashlib.sha256(self.xml_path.read_bytes()).hexdigest()[:16]

    def magnitudes(self) -> np.ndarray:
        lo, hi = self.magnitude_range
        return np.geomspace(lo, hi, self.n_magnitudes)


REGISTRY: dict[str, SceneSpec] = {
    s.name: s
    for s in [
        SceneSpec(
            name="billiards",
            xml="billiards.xml",
            tracked_bodies=("cue", "ball_1", "ball_2", "ball_3"),
            poke_sites=("cue",),
            poke_directions={"cue": (1.0, 0.0, 0.0)},
            magnitude_range=(0.5, 24.0),
            subtests=("response", "momentum", "divergence"),
            chain=("cue", "ball_1", "ball_2", "ball_3"),
        ),
        SceneSpec(
            name="blocks",
            xml="blocks.xml",
            tracked_bodies=("block_0", "block_1", "block_2", "block_3"),
            poke_sites=("block_3", "block_2"),
            poke_directions={"block_3": (1.0, 0.0, 0.0), "block_2": (1.0, 0.0, 0.0)},
            magnitude_range=(1.0, 60.0),
            subtests=("response", "propagation", "stability", "divergence"),
            chain=("block_3", "block_2", "block_1", "block_0"),
            topple_body="block_3",
            topple_z=0.20,  # initial top-block z is 0.28; below 0.20 == toppled
        ),
        SceneSpec(
            name="arm",
            xml="arm.xml",
            tracked_bodies=("link_1", "link_2", "link_3"),
            poke_sites=("link_3",),
            poke_directions={"link_3": (1.0, 0.0, 0.0)},
            magnitude_range=(0.5, 20.0),
            subtests=("response", "propagation", "divergence"),
            chain=("link_3", "link_2", "link_1"),
        ),
    ]
}


def get_scene(name: str) -> SceneSpec:
    return REGISTRY[name]
