"""Render-module tests that need no GPU/EGL (invariant 5)."""

import inspect
from pathlib import Path

import numpy as np
from PIL import ImageFont

from wa import render as wa_render
from wa.render import PANEL_H, PANEL_W, POKE_FLASH, _annotate


def test_render_never_steps_physics():
    """Invariant 6, statically enforced: render.py must not step physics."""
    src = Path(inspect.getsourcefile(wa_render)).read_text()
    for forbidden in ("mj_step", "rollout.rollout", "mujoco.rollout"):
        assert forbidden not in src, f"render.py must never call {forbidden}"


def test_annotate_burns_label_and_poke_flash():
    font = ImageFont.load_default()
    frame = np.full((PANEL_H, PANEL_W, 3), 100, dtype=np.uint8)
    out = _annotate(frame.copy(), "model-predicted states, simulator renderer",
                    1.23, poke_on=True, font=font)
    assert out.shape == frame.shape
    assert not np.array_equal(out[:30], frame[:30]), "banner not drawn"
    assert (out[40] == POKE_FLASH).all(axis=-1).any(), "poke border missing"
    quiet = _annotate(frame.copy(), "x", 0.0, poke_on=False, font=font)
    assert not (quiet[40] == POKE_FLASH).all(axis=-1).any()


def test_composite_frame_count_and_divider():
    class FakeRenderer:
        _font = ImageFont.load_default()
        def frame(self, s):
            return np.full((PANEL_H, PANEL_W, 3), 100, dtype=np.uint8)

    T, ds = 200, 4
    truth = np.zeros((T, ds))
    pred = np.zeros((T, ds))
    frames = wa_render.composite_clip(FakeRenderer(), truth, pred, "m",
                                      poke_window=(10, 20), timestep=0.002)
    assert len(frames) == int(np.ceil(T / wa_render.STEP_STRIDE))
    assert frames[0].shape == (PANEL_H, 2 * PANEL_W, 3)
    assert (frames[0][100, PANEL_W - 1] == wa_render.INK).all()
