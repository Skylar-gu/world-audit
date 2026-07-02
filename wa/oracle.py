"""Ground-truth oracle: snapshot/restore + rollout wrapper + poke application.

Contracts (spec §3, CLAUDE.md):
- Snapshots are always `mjSTATE_INTEGRATION` — never qpos/qvel alone; warm-start
  restoration is required for bitwise reproducibility.
- Counterfactual pairs share the identical init snapshot byte-for-byte; the poke
  is injected through the rollout control interface (`control_spec` carrying
  `xfrc_applied`), never baked into the snapshot.
- Rolls via the official `mujoco.rollout` module. The only hand-rolled stepping
  loop is `run_raw`, which deliberately uses `mj_step` to cross-check `rollout`
  in the null-test variant.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from wa.schema import Poke, RolloutPair
from wa.scenes import SceneSpec, get_scene

INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION.value
FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS.value
WARMSTART = mujoco.mjtState.mjSTATE_WARMSTART.value
CTRL = mujoco.mjtState.mjSTATE_CTRL.value
XFRC_APPLIED = mujoco.mjtState.mjSTATE_XFRC_APPLIED.value


def snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    s = np.empty(mujoco.mj_stateSize(model, INTEGRATION), dtype=np.float64)
    mujoco.mj_getState(model, data, s, INTEGRATION)
    return s


def restore(model: mujoco.MjModel, data: mujoco.MjData, s: np.ndarray) -> None:
    mujoco.mj_setState(model, data, s, INTEGRATION)


def state_offset(model: mujoco.MjModel, spec: int, field_bit: int) -> int:
    """Offset of `field_bit`'s block within a `spec` state vector.

    mj_getState packs the selected fields in ascending bit order.
    """
    assert spec & field_bit, "field not in spec"
    off = 0
    bit = 1
    while bit < field_bit:
        if spec & bit:
            off += mujoco.mj_stateSize(model, bit)
        bit <<= 1
    return off


class Oracle:
    """One oracle per scene. Owns the MjModel; MjData is internal scratch."""

    def __init__(self, scene: str | SceneSpec):
        self.scene = scene if isinstance(scene, SceneSpec) else get_scene(scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene.xml_path))
        self.data = mujoco.MjData(self.model)
        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu
        # control vector layout for rollout: CTRL | XFRC_APPLIED, ascending bit order
        self.control_spec = XFRC_APPLIED | (CTRL if self.nu > 0 else 0)
        self.ncontrol = mujoco.mj_stateSize(self.model, self.control_spec)
        self._xfrc_off = state_offset(self.model, self.control_spec, XFRC_APPLIED)

    # ---------------------------------------------------------------- init states

    def init_state(self, seed: int = 0, jitter: float = 0.0) -> np.ndarray:
        """Settle the scene from its XML keyframe-free reset, then snapshot.

        `settle_steps ≥ 100` guarantees a non-trivial warm-start in the snapshot.
        Optional qpos jitter (for training-data diversity) is applied *before*
        settling, so the snapshot is always a dynamically consistent state.
        """
        mujoco.mj_resetData(self.model, self.data)
        if jitter > 0.0:
            from wa.schema import rng
            g = rng(seed)
            self.data.qpos[:] += g.uniform(-jitter, jitter, size=self.nq)
        for _ in range(self.scene.settle_steps):
            mujoco.mj_step(self.model, self.data)
        return snapshot(self.model, self.data)

    # ---------------------------------------------------------------- pokes

    def _control_array(self, horizon: int, ctrl_seq: np.ndarray | None,
                       poke: Poke | None) -> np.ndarray:
        """Per-step control vectors (T, ncontrol) carrying ctrl and xfrc_applied.

        The poke writes its world-frame force into xfrc_applied[body, :3] for
        t ∈ [t_start, t_start + duration), zero otherwise.
        """
        control = np.zeros((horizon, self.ncontrol), dtype=np.float64)
        if self.nu > 0 and ctrl_seq is not None:
            control[:, :self.nu] = np.asarray(ctrl_seq, dtype=np.float64).reshape(horizon, self.nu)
        if poke is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, poke.site)
            if body_id < 0:
                raise ValueError(f"poke site {poke.site!r} is not a body in {self.scene.name}")
            col = self._xfrc_off + 6 * body_id
            t0, t1 = poke.t_start, poke.t_start + poke.duration
            if t1 > horizon:
                raise ValueError("poke window exceeds horizon")
            control[t0:t1, col:col + 3] = poke.force
        return control

    # ---------------------------------------------------------------- rollouts

    def run(self, init_state: np.ndarray, poke: Poke | None = None,
            ctrl_seq: np.ndarray | None = None, horizon: int | None = None,
            seed: int = 0) -> RolloutPair:
        """Roll out via mujoco.rollout from an INTEGRATION snapshot."""
        horizon = horizon or self.scene.horizon
        restore(self.model, self.data, init_state)  # scratch restore to split the snapshot
        full0 = np.empty(mujoco.mj_stateSize(self.model, FULLPHYSICS), dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, full0, FULLPHYSICS)
        warm0 = self.data.qacc_warmstart.copy()

        control = self._control_array(horizon, ctrl_seq, poke)
        state, sensordata = mj_rollout.rollout(
            self.model, self.data,
            initial_state=full0[None],
            control=control[None],
            control_spec=self.control_spec,
            initial_warmstart=warm0[None],
        )
        return self._pair(state[0], sensordata[0], poke, seed)

    def run_raw(self, init_state: np.ndarray, poke: Poke | None = None,
                ctrl_seq: np.ndarray | None = None, horizon: int | None = None,
                seed: int = 0) -> RolloutPair:
        """Same rollout, deliberately hand-rolled with mj_step (null-test cross-check)."""
        horizon = horizon or self.scene.horizon
        restore(self.model, self.data, init_state)
        qoff, voff = self._qpos_off, self._qvel_off
        nfull = mujoco.mj_stateSize(self.model, FULLPHYSICS)
        state = np.empty((horizon, nfull), dtype=np.float64)
        sensordata = np.empty((horizon, self.model.nsensordata), dtype=np.float64)
        control = self._control_array(horizon, ctrl_seq, poke)
        for t in range(horizon):
            mujoco.mj_setState(self.model, self.data, control[t], self.control_spec)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_getState(self.model, self.data, state[t], FULLPHYSICS)
            sensordata[t] = self.data.sensordata
        return self._pair(state, sensordata, poke, seed)

    # ------------------------------------------------------- model-side helpers

    def qpos_qvel_from_snapshot(self, init_state: np.ndarray) -> np.ndarray:
        """(nq+nv,) flattened state out of an INTEGRATION snapshot."""
        restore(self.model, self.data, init_state)
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def force_channels(self, poke: Poke | None, horizon: int) -> np.ndarray:
        """(T, 3*len(poke_sites)) applied-force input channel, ordered by poke_sites.

        This is the action representation shared by training data and model
        rollouts, so "the same poke" is well-defined on the model side (§4).
        """
        out = np.zeros((horizon, 3 * len(self.scene.poke_sites)), dtype=np.float64)
        if poke is not None:
            i = self.scene.poke_sites.index(poke.site)
            t0, t1 = poke.t_start, poke.t_start + poke.duration
            out[t0:t1, 3 * i:3 * i + 3] = poke.force
        return out

    def tracked_from_states(self, states: np.ndarray) -> dict[str, np.ndarray]:
        """Tracked series from (T, nq+nv) states by kinematics only (mj_forward).

        Used for model-predicted trajectories; never steps physics (invariant 6).
        """
        data = mujoco.MjData(self.model)
        T = states.shape[0]
        sensordata = np.empty((T, self.model.nsensordata))
        # kinematic reconstruction only: sanitize diverged model states so
        # mj_forward stays finite; divergence itself is flagged upstream
        safe = np.clip(np.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6),
                       -1e6, 1e6)
        for t in range(T):
            data.qpos[:] = safe[t, :self.nq]
            data.qvel[:] = safe[t, self.nq:]
            mujoco.mj_forward(self.model, data)
            sensordata[t] = data.sensordata
        return self._tracked(sensordata)

    # ---------------------------------------------------------------- assembly

    @property
    def _qpos_off(self) -> int:
        return state_offset(self.model, FULLPHYSICS, mujoco.mjtState.mjSTATE_QPOS.value)

    @property
    def _qvel_off(self) -> int:
        return state_offset(self.model, FULLPHYSICS, mujoco.mjtState.mjSTATE_QVEL.value)

    def qpos_qvel(self, full_state: np.ndarray) -> np.ndarray:
        """(T, nq+nv) slice out of FULLPHYSICS trajectory rows."""
        q, v = self._qpos_off, self._qvel_off
        return np.concatenate(
            [full_state[..., q:q + self.nq], full_state[..., v:v + self.nv]], axis=-1)

    def _tracked(self, sensordata: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for body in self.scene.tracked_bodies:
            for kind in ("pos", "vel"):
                sid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{body}.{kind}")
                adr = self.model.sensor_adr[sid]
                dim = self.model.sensor_dim[sid]
                out[f"{body}.{kind}"] = sensordata[:, adr:adr + dim].copy()
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            mass = float(self.model.body_mass[body_id])
            out[f"{body}.mom"] = mass * out[f"{body}.vel"]
        return out

    def _pair(self, full_state: np.ndarray, sensordata: np.ndarray,
              poke: Poke | None, seed: int) -> RolloutPair:
        return RolloutPair(
            scene=self.scene.name,
            model_id="oracle",
            poke=poke,
            states=self.qpos_qvel(full_state),
            tracked=self._tracked(sensordata),
            seed=seed,
            manifest=manifest_stamp(self.scene),
        )


@lru_cache(maxsize=1)
def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def manifest_stamp(scene: SceneSpec) -> dict:
    return {
        "git_sha": _git_sha(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "scene": scene.name,
        "scene_hash": scene.xml_hash(),
    }
