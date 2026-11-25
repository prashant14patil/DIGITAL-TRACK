"""Quick recording check on SYNTHETIC stepped turns (fixed camera, box turned by hand with pauses)."""

from __future__ import annotations

import pytest

from digitaltrack.synthetic.rgbd_scene import SceneConfig, make_sequence
from digitaltrack.validation.quickcheck import quick_check


def _seq(n_stops):
    return make_sequence(SceneConfig(n_stops=n_stops, step_deg=15.0, step_jitter_deg=1.0, frames_per_stop=7,
                                     transition_frames=4, noise_sigma_m=0.002, seed=4))


def _checks(r):
    return {c["id"]: c for c in r["checks"]}


@pytest.mark.slow
def test_full_turn_plus_one_step_passes():
    seq = _seq(25)  # 24 steps of ~15 deg = a full turn; the 25th stop is back at the start
    r = quick_check(seq.views, background_views=seq.background_views)
    c = _checks(r)
    assert r["stops"]["n"] == 25
    assert c["full_turn"]["status"] == "ok", c["full_turn"]
    assert c["empty_room"]["status"] == "ok" and c["floor"]["status"] == "ok"


@pytest.mark.slow
@pytest.mark.parametrize("n_stops", [22, 23])  # about 45 and 30 deg short of a full turn
def test_stopping_short_is_never_reported_as_a_full_turn(n_stops):
    seq = _seq(n_stops)
    r = quick_check(seq.views, background_views=seq.background_views)
    assert _checks(r)["full_turn"]["status"] == "warn", r.get("last_to_first")
    ltf = r["last_to_first"]
    # either the missing angle is measured, or the turn is explicitly unconfirmed; never "ok"
    assert ltf.get("confirmed") is False or ltf["short_of_start_deg"] > 8


@pytest.mark.slow
def test_overshoot_by_one_step_is_a_full_turn():
    seq = _seq(26)  # one step past the start
    r = quick_check(seq.views, background_views=seq.background_views)
    assert _checks(r)["full_turn"]["status"] == "ok"
    assert -25 < r["last_to_first"]["short_of_start_deg"] < -5


def test_no_pauses_fails_with_advice():
    seq = make_sequence(SceneConfig(n_stops=4, frames_per_stop=1, transition_frames=3))
    r = quick_check(seq.views, background_views=seq.background_views)
    assert r["verdict"] == "fail" and "pauses" in _checks(r)["stops"]["message"]


def test_missing_empty_room_is_a_warning():
    seq = make_sequence(SceneConfig(n_stops=4, frames_per_stop=7, transition_frames=4))
    r = quick_check(seq.views)
    assert _checks(r)["empty_room"]["status"] == "warn"
