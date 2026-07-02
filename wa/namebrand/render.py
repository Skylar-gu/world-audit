"""Side-by-side walker clips for the name-brand audit.

Left panel: dm_control ground truth. Right panel: the TD-MPC2 latent
prediction pushed through the decoder probe — so the right panel's error
includes the decoder's noise floor, and the burned-in label says so. Root x
is pinned to its snapshot value on the model side (unobservable by
construction); the model panel shows posture and gait, not forward travel.

Never steps physics: frames are kinematic playback of stored trajectories
(invariant 6), same contract as wa/render.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from wa import config
from wa.render import FPS, _annotate, encode  # shared annotate/encode machinery
from wa.namebrand.audit import POKE_DUR, POKE_T0, TASK, WalkerEnv

ROLLOUTS = Path(f"experiments/artifacts/rollouts-v{config.AUDIT_VERSION}/namebrand")
CLIPS_DIR = Path("site/public/clips") / f"tdmpc2-{TASK}"
PANEL = 480
AGENT_DT = 0.05          # 2 control steps x 0.025 s
CLIP_FPS = 20            # 100 agent steps -> 5 s clip, real time

TRUTH_LABEL = "ground truth (dm_control)"
MODEL_LABEL = "decoded latent prediction*"


def render_all(force: bool = False) -> list[Path]:
    from PIL import ImageFont
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    env = WalkerEnv(seed=1)  # kinematic playback + camera only
    physics = env.physics
    made = []
    for npz_path in sorted(ROLLOUTS.glob(f"{TASK}__ctrl__m*.npz")):
        out_base = CLIPS_DIR / npz_path.stem.replace(TASK, f"tdmpc2-{TASK}")
        if not force and out_base.with_suffix(".webm").exists():
            continue
        z = np.load(npz_path)
        truth_qpos, model_qpos = z["truth_qpos"], z["model_qpos"]
        frames = []
        for t in range(truth_qpos.shape[0]):
            poke_on = POKE_T0 <= t < POKE_T0 + POKE_DUR
            clock = t * AGENT_DT
            panels = []
            for qpos, label in ((truth_qpos[t], TRUTH_LABEL),
                                (model_qpos[t], MODEL_LABEL)):
                with physics.reset_context():
                    physics.data.qpos[:] = qpos
                frame = physics.render(PANEL, PANEL, camera_id=0)
                panels.append(_annotate(frame, label, clock, poke_on, font))
            combo = np.concatenate(panels, axis=1)
            combo[:, PANEL - 1:PANEL + 1] = (11, 11, 11)
            frames.append(combo)
        encode(frames, out_base, fps=CLIP_FPS)
        made.append(out_base)
        print(f"[namebrand] {out_base.name}: {len(frames)} frames, "
              f"webm {out_base.with_suffix('.webm').stat().st_size/1e6:.2f} MB")
    print(f"* model panel decodes the latent through the probe (R² noise floor); "
          f"root x pinned (unobservable) — posture, not travel")
    return made


if __name__ == "__main__":
    render_all()
