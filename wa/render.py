"""Offscreen EGL rendering: side-by-side truth|model clips from grid rollouts.

Invariant 6: this module never steps physics. It consumes stored trajectories
(rollout npz written by grid.py), writes qpos/qvel into a fresh MjData and
calls mj_forward per frame — kinematics for rendering only.

Model panels are labeled "model-predicted states, simulator renderer" (burned
into the frame). The poke window is flashed on both panels.

The render MjModel is built from the scene XML *string* with a <visual> block
injected for the 720p offscreen framebuffer — the on-disk XML (and therefore
every content hash) is untouched, and dynamics are irrelevant here anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wa import config
from wa.scenes import REGISTRY, SceneSpec, get_scene

os.environ.setdefault("MUJOCO_GL", "egl")

ROLLOUTS_DIR = Path(f"experiments/artifacts/rollouts-v{config.AUDIT_VERSION}")
CLIPS_DIR = Path("site/public/clips")

PANEL_W, PANEL_H = 640, 720          # composite = 1280x720
FPS = 25
STEP_STRIDE = 20                      # 500 Hz sim -> 25 fps at 2x slow motion
POKE_FLASH = (227, 73, 72)           # reference-palette red
INK = (11, 11, 11)
BANNER = (252, 252, 251)


@dataclass(frozen=True)
class CameraSpec:
    lookat: tuple[float, float, float]
    distance: float
    elevation: float
    azimuth: float


CAMERAS = {
    "billiards": CameraSpec((0.0, 0.0, 0.03), 1.05, -55, 90),
    "blocks": CameraSpec((0.0, 0.0, 0.16), 1.15, -18, 125),
    "arm": CameraSpec((0.0, 0.0, 0.45), 1.6, -12, 100),
}


def _render_model(spec: SceneSpec) -> mujoco.MjModel:
    xml = spec.xml_path.read_text()
    visual = '<visual><global offwidth="1280" offheight="720"/></visual>'
    return mujoco.MjModel.from_xml_string(xml.replace("</mujoco>", visual + "</mujoco>"))


class SceneRenderer:
    """One EGL renderer per scene, reused across clips."""

    def __init__(self, spec: SceneSpec):
        self.spec = spec
        self.model = _render_model(spec)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=PANEL_H, width=PANEL_W)
        cs = CAMERAS[spec.name]
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = cs.lookat
        self.cam.distance, self.cam.elevation, self.cam.azimuth = (
            cs.distance, cs.elevation, cs.azimuth)
        try:
            self._font = ImageFont.truetype(
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 17)
        except OSError:
            self._font = ImageFont.load_default()

    def frame(self, qpos_qvel: np.ndarray) -> np.ndarray:
        """(nq+nv,) -> (H, W, 3) uint8. Kinematics only — never steps."""
        nq = self.model.nq
        safe = np.clip(np.nan_to_num(qpos_qvel, nan=0.0, posinf=1e3, neginf=-1e3),
                       -1e3, 1e3)
        self.data.qpos[:] = safe[:nq]
        self.data.qvel[:] = safe[nq:]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()


def _annotate(frame: np.ndarray, label: str, clock_s: float, poke_on: bool,
              font) -> np.ndarray:
    h, w = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, 30], fill=BANNER)
    draw.text((8, 6), label, fill=INK, font=font)
    draw.text((w - 74, 6), f"t={clock_s:5.2f}s", fill=INK, font=font)
    if poke_on:
        for k in range(4):  # flash: thick border + tag
            draw.rectangle([k, k, w - 1 - k, h - 1 - k], outline=POKE_FLASH)
        draw.rectangle([8, 40, 78, 64], fill=POKE_FLASH)
        draw.text((16, 44), "POKE", fill=(255, 255, 255), font=font)
    return np.asarray(img)


def composite_clip(rend: SceneRenderer, truth: np.ndarray, model_pred: np.ndarray,
                   model_label: str, poke_window: tuple[int, int],
                   timestep: float) -> list[np.ndarray]:
    """Side-by-side frames: truth left, model right, synchronized clocks."""
    T = min(truth.shape[0], model_pred.shape[0])
    frames = []
    for t in range(0, T, STEP_STRIDE):
        poke_on = poke_window[0] <= t < poke_window[1]
        clock = t * timestep
        left = _annotate(rend.frame(truth[t]), "ground truth (MuJoCo)",
                         clock, poke_on, rend._font)
        right = _annotate(rend.frame(model_pred[t]),
                          "model-predicted states, simulator renderer",
                          clock, poke_on, rend._font)
        combo = np.concatenate([left, right], axis=1)
        combo[:, PANEL_W - 1:PANEL_W + 1] = INK  # panel divider
        frames.append(combo)
    return frames


def encode(frames: list[np.ndarray], out_base: Path, fps: int = FPS) -> None:
    """WebM (VP9) + MP4 (H.264), ~<1.5 MB per clip."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = frames[0].shape[:2]
    raw = np.stack(frames).tobytes()
    common = ["-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
              "-r", str(fps), "-i", "pipe:0"]
    jobs = {
        ".webm": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "40",
                  "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1"],
        ".mp4": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    }
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, codec in jobs.items():
        subprocess.run([ffmpeg, *common, *codec, str(out_base) + suffix],
                       input=raw, check=True, capture_output=True)


def render_scene(scene: str, only_model: str | None = None,
                 force: bool = False) -> list[Path]:
    spec = get_scene(scene)
    rend = SceneRenderer(spec)
    timestep = rend.model.opt.timestep
    src = ROLLOUTS_DIR / scene
    if not src.exists():
        raise FileNotFoundError(f"no rollouts at {src}; run `make grid SCENE={scene}`")
    truth_files = sorted(src.glob("oracle__*.npz"))
    made = []
    for tf in truth_files:
        _, site, mtag = tf.stem.split("__")
        truth_poked = np.load(tf)["poked"]
        t0 = spec.poke_t_start
        window = (t0, t0 + spec.poke_duration)
        for mf in sorted(src.glob(f"*__{site}__{mtag}.npz")):
            model_id = mf.stem.split("__")[0]
            if model_id == "oracle" or (only_model and model_id != only_model):
                continue
            out_base = CLIPS_DIR / scene / f"{model_id}__{site}__{mtag}"
            if not force and out_base.with_suffix(".webm").exists():
                continue
            pred = np.load(mf)["poked"]
            frames = composite_clip(rend, truth_poked, pred, model_id,
                                    window, timestep)
            encode(frames, out_base)
            made.append(out_base)
            print(f"[{scene}] {out_base.name}: {len(frames)} frames, "
                  f"webm {out_base.with_suffix('.webm').stat().st_size/1e6:.2f} MB, "
                  f"mp4 {out_base.with_suffix('.mp4').stat().st_size/1e6:.2f} MB")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    scenes = [args.scene] if args.scene else list(REGISTRY)
    total = []
    for sc in scenes:
        total += render_scene(sc, only_model=args.model, force=args.force)
    print(f"rendered {len(total)} clips")
