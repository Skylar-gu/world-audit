import numpy as np
import pytest

from wa.schema import Poke, RolloutPair, content_hash, derive_seed, rng


def test_poke_requires_unit_direction():
    with pytest.raises(ValueError):
        Poke(site="cue", direction=np.array([1.0, 1.0, 0.0]), magnitude=1.0,
             t_start=0, duration=5)


def test_poke_json_roundtrip():
    p = Poke(site="cue", direction=np.array([0.0, 0.0, 1.0]), magnitude=3.5,
             t_start=10, duration=8)
    q = Poke.from_json(p.to_json())
    assert q.site == p.site and q.magnitude == p.magnitude
    assert np.array_equal(q.direction, p.direction)
    assert np.array_equal(q.force, np.array([0.0, 0.0, 3.5]))


def test_rolloutpair_json_roundtrip():
    p = Poke(site="cue", direction=np.array([1.0, 0.0, 0.0]), magnitude=2.0,
             t_start=1, duration=2)
    pair = RolloutPair(
        scene="billiards", model_id="oracle", poke=p,
        states=np.arange(12, dtype=np.float64).reshape(3, 4),
        tracked={"cue.pos": np.ones((3, 3))},
        seed=7, manifest={"git_sha": "x", "mujoco": "3.10.0"},
    )
    back = RolloutPair.loads(pair.dumps())
    assert back.scene == pair.scene and back.seed == 7
    assert np.array_equal(back.states, pair.states)
    assert np.array_equal(back.tracked["cue.pos"], pair.tracked["cue.pos"])
    null_pair = RolloutPair(scene="b", model_id="oracle", poke=None,
                            states=np.zeros((1, 1)))
    assert RolloutPair.loads(null_pair.dumps()).poke is None


def test_derive_seed_deterministic_and_distinct():
    assert derive_seed(0, "pair", 3) == derive_seed(0, "pair", 3)
    assert derive_seed(0, "pair", 3) != derive_seed(0, "pair", 4)
    assert derive_seed(0, "pair", 3) != derive_seed(1, "pair", 3)


def test_rng_streams_reproducible():
    assert rng(42).standard_normal(4).tolist() == rng(42).standard_normal(4).tolist()


def test_content_hash_stable():
    assert content_hash({"b": 1, "a": 2}) == content_hash({"a": 2, "b": 1})
