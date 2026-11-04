"""CPU unit tests for the AS-Defend-Real-v0 macro layer (no Isaac needed).

    python -m pytest tests/test_defend_real_macro.py -q
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch

_PATH = (
    Path(__file__).resolve().parents[1] / "defend_task/macro.py"
)
_spec = importlib.util.spec_from_file_location("defend_real_macro", _PATH)
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)

CFG = M.MacroCfg()
SITE = torch.tensor([[0.0, 0.0, 1.25]])
PRIM_A, PRIM_B, SEC_A, SEC_B, GOALIE = 0, 1, 2, 3, 4
ALIVE2 = torch.ones(1, 2, dtype=torch.bool)


def _state(E=1):
    return M.MacroState(E, 3, 2, "cpu")


def _step(def_pos, att_pos, att_vel, alive=ALIVE2, state=None, cfg=CFG):
    state = state or _state(def_pos.shape[0])
    site = SITE.expand(def_pos.shape[0], 3)
    return M.macro_step(def_pos, torch.zeros_like(def_pos), att_pos, att_vel, alive, site, state, cfg), state


POSTS = torch.tensor([[[1.8, 0.0, 1.2], [-0.9, 1.56, 1.2], [-0.9, -1.56, 1.2]]])


# -- zones ----------------------------------------------------------------------------------


def test_tau_and_zones():
    st = _state()
    pos = torch.tensor([[[6.0, 0.0, 1.5], [0.0, 3.0, 1.5]]])
    vel = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    dist, tau = M.time_to_site(pos, vel, SITE, CFG)
    assert abs(tau[0, 0].item() - 4.5) < 1e-4
    assert math.isinf(tau[0, 1].item())
    zone = M.compute_zones(dist, tau, ALIVE2, st, CFG)
    assert zone.tolist() == [[M.ALERT, M.ENGAGE]]


def test_fast_attacker_is_engaged_early_by_tau():
    st = _state()
    pos = torch.tensor([[[4.8, 0.0, 1.5], [9.0, 9.0, 1.5]]])
    vel = torch.tensor([[[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # tau = 3.3 / 2 = 1.65 s
    dist, tau = M.time_to_site(pos, vel, SITE, CFG)
    zone = M.compute_zones(dist, tau, ALIVE2, st, CFG)
    assert zone[0, 0].item() == M.ENGAGE and zone[0, 1].item() == M.RELAXED


def test_engage_hysteresis():
    st = _state()
    still = torch.zeros(1, 2, 3)
    far = [9.0, 9.0, 1.5]
    inside = torch.tensor([[[3.4, 0.0, 1.5], far]])
    edge = torch.tensor([[[3.7, 0.0, 1.5], far]])
    out = torch.tensor([[[4.2, 0.0, 1.5], far]])
    for p, expect in ((edge, M.ALERT), (inside, M.ENGAGE), (edge, M.ENGAGE), (out, M.ALERT)):
        d, t = M.time_to_site(p, still, SITE, CFG)
        assert M.compute_zones(d, t, ALIVE2, st, CFG)[0, 0].item() == expect


def test_intercept_point_is_reachable_and_on_path():
    att_pos = torch.tensor([[[6.0, 0.0, 1.5], [0.0, 6.0, 1.5]]])
    att_vel = torch.tensor([[[-0.8, 0.0, 0.0], [0.0, -0.8, 0.0]]])
    _, tau = M.time_to_site(att_pos, att_vel, SITE, CFG)
    point, time, ok = M.intercept_points(POSTS, att_pos, att_vel, tau, CFG)
    assert ok[0, 0, 0]
    p, t = point[0, 0, 0], time[0, 0, 0].item()
    on_path = att_pos[0, 0] + att_vel[0, 0] * t
    assert torch.allclose(p[:2], on_path[:2], atol=1e-5)
    assert torch.linalg.norm(p - POSTS[0, 0]).item() / CFG.defender_speed <= t + 1e-5
    assert t <= tau[0, 0].item()


# -- slots ----------------------------------------------------------------------------------


def _two_threats(speed_a, speed_b=0.4):
    att_pos = torch.tensor([[[4.6, 0.3, 1.5], [-0.5, 4.8, 1.5]]])
    att_vel = torch.tensor([[[-speed_a, 0.0, 0.0], [0.0, -speed_b, 0.0]]])
    return att_pos, att_vel


def test_both_primaries_always_filled_and_every_drone_has_a_slot():
    out, _ = _step(POSTS, *_two_threats(0.5))
    slot = out["slot"][0].tolist()
    assert PRIM_A in slot and PRIM_B in slot
    assert M.POST not in slot  # third drone is never left without an objective
    assert sorted(out["assign"][0].tolist()) == sorted([slot.index(PRIM_A), slot.index(PRIM_B)])


def test_third_drone_guards_against_slow_attackers():
    out, _ = _step(POSTS, *_two_threats(0.5))
    assert GOALIE in out["slot"][0].tolist()
    assert not out["join_on"].item()


def test_third_drone_helps_against_a_fast_attacker():
    out, _ = _step(POSTS, *_two_threats(3.0))
    slot = out["slot"][0].tolist()
    assert SEC_A in slot, slot  # secondary on the fast attacker A, not the goalie
    assert out["join_on"].item()


def test_far_attackers_leave_drones_on_their_posts():
    att_pos = torch.tensor([[[9.0, 0.0, 1.5], [0.0, 9.0, 1.5]]])
    out, _ = _step(POSTS, att_pos, torch.zeros(1, 2, 3))
    assert out["slot"][0].tolist() == [M.POST] * 3
    assert torch.allclose(out["target"], M.formation_posts(SITE, 3, CFG))


def test_one_attacker_in_alert_gets_a_primary_and_a_goalie():
    att_pos = torch.tensor([[[4.6, 0.0, 1.5], [0.0, 9.0, 1.5]]])  # A alert (4.6 m), B relaxed
    out, _ = _step(POSTS, att_pos, torch.zeros(1, 2, 3))
    slot = out["slot"][0].tolist()
    assert slot.count(PRIM_A) == 1 and slot.count(GOALIE) == 1 and slot.count(M.POST) == 1
    assert PRIM_B not in slot and SEC_A not in slot  # secondaries only for engaged attackers


def test_dead_attacker_releases_its_slots():
    out, _ = _step(POSTS, *_two_threats(3.0), alive=torch.tensor([[True, False]]))
    slot = out["slot"][0].tolist()
    assert PRIM_B not in slot and SEC_B not in slot
    assert PRIM_A in slot


def test_engaged_primary_is_locked_to_its_drone():
    att_pos, att_vel = _two_threats(1.2)  # A engaged (tau 2.6 s), so its primary is locked
    out, st = _step(POSTS, att_pos, att_vel)
    holder_a = out["slot"][0].tolist().index(PRIM_A)
    # move the drones so another one is now much closer to attacker A: the lock must hold
    moved = POSTS.clone()
    moved[0, holder_a] = torch.tensor([-1.5, -1.5, 1.2])
    other = (holder_a + 1) % 3
    moved[0, other] = torch.tensor([3.8, 0.3, 1.5])
    out2, _ = _step(moved, att_pos, att_vel, state=st)
    assert out2["slot"][0, holder_a].item() == PRIM_A


def test_pincer_mode_offsets_from_the_primary():
    cfg = M.MacroCfg(secondary_mode="pincer")
    out, _ = _step(POSTS, *_two_threats(3.0), cfg=cfg)
    slot = out["slot"][0].tolist()
    if SEC_A in slot:
        sec = out["target"][0, slot.index(SEC_A)]
        prim = out["target"][0, slot.index(PRIM_A)]
        assert torch.linalg.norm(sec - prim).item() > 0.3


def test_base_actions_bounded_and_yaw_free():
    out, _ = _step(POSTS, *_two_threats(1.5))
    assert out["base"].shape == (1, 3, 4)
    assert out["base"].abs().max().item() <= 1.0 + 1e-6
    assert out["base"][..., 3].abs().max().item() == 0.0


def test_chase_base_points_at_nearest_attacker():
    def_pos = torch.tensor([[[0.0, 0.0, 1.0], [5.0, 5.0, 1.0], [0.0, 0.0, 1.0]]])
    att_pos = torch.tensor([[[3.0, 0.0, 1.0], [5.0, 8.0, 1.0]]])
    a = M.chase_base(def_pos, att_pos, ALIVE2, CFG)
    assert torch.allclose(a[0, 0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(a[0, 1, :3], torch.tensor([0.0, 1.0, 0.0]))


def test_batched_envs_are_independent():
    a1, v1 = _two_threats(0.5)
    a2, v2 = _two_threats(3.0)
    out, _ = _step(torch.cat([POSTS, POSTS]), torch.cat([a1, a2]), torch.cat([v1, v2]), alive=torch.ones(2, 2, dtype=torch.bool))
    assert out["join_on"].tolist() == [False, True]


def test_hungarian_matches_enumeration_with_switch_cost_and_lock():
    """The laptop's Hungarian solver picks an assignment as cheap as the training solver's,
    including the switch cost and the engaged-primary lock, over a sequence of steps."""
    import pytest

    pytest.importorskip("scipy")
    torch.manual_seed(0)
    cfg_e, cfg_h = M.MacroCfg(), M.MacroCfg(solver="hungarian")
    for _ in range(300):
        st_e, st_h = _state(), _state()
        for _step in range(4):
            cost = torch.rand(1, 3, 5) * 10
            cost[torch.rand(1, 3, 5) < 0.15] = M.UNUSABLE
            eng = torch.rand(1, 2) < 0.5
            prev = st_e.slot.clone()
            a = M.solve_slots(cost, eng, st_e, cfg_e)[0]
            st_h.slot = prev.clone()
            b = M.solve_slots(cost, eng, st_h, cfg_h)[0]

            def total(s):
                s = s.clamp(min=0)
                c = cost[0, torch.arange(3), s]
                ch = (prev[0] != M.POST) & (s != prev[0])
                return (c + cfg_e.switch_cost_s * ch.float()).sum().item()

            assert abs(total(a) - total(b)) < 1e-4
            st_h.slot = st_e.slot.clone()


def test_dead_defender_gets_no_slot_and_live_ones_cover_both_attackers():
    """A defender that is out (made contact) is never assigned; the two live ones take both primaries."""
    for solver in ("enumerate", "hungarian"):
        cfg = M.MacroCfg(solver=solver)
        st = _state()
        alive_def = torch.tensor([[True, False, True]])
        out = M.macro_step(POSTS, torch.zeros_like(POSTS), *_two_threats(1.2), ALIVE2,
                           SITE, st, cfg, def_alive=alive_def)
        slot = out["slot"][0].tolist()
        assert slot[1] == M.POST, (solver, slot)
        assert sorted([slot[0], slot[2]]) == [PRIM_A, PRIM_B], (solver, slot)


def test_last_defender_takes_a_primary():
    cfg = M.MacroCfg()
    out = M.macro_step(POSTS, torch.zeros_like(POSTS), *_two_threats(1.2), ALIVE2, SITE, _state(), cfg,
                       def_alive=torch.tensor([[False, False, True]]))
    slot = out["slot"][0].tolist()
    assert slot[:2] == [M.POST, M.POST] and slot[2] in (PRIM_A, PRIM_B)
