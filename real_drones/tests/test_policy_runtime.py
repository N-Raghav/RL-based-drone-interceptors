"""CPU checks for the deploy-time policy path (no Isaac, no skrl, no drones).

    python -m pytest drone_deployment_infra/defender/tests -q
"""
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from policy_runtime import SharedPolicy, build_obs, safety  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "macro_t", REPO / "defend_task/macro.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["macro_t"] = M
_spec.loader.exec_module(M)


def _fake_ckpt(path):
    """A checkpoint laid out like skrl's GaussianMixin + RunningStandardScaler."""
    torch.manual_seed(0)
    sd, dims = {}, [59, 256, 128, 64]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        sd[f"net_container.{2 * i}.weight"] = torch.randn(b, a) * 0.1
        sd[f"net_container.{2 * i}.bias"] = torch.zeros(b)
    sd["net_container.6.weight"] = torch.randn(4, 64) * 0.1
    sd["net_container.6.bias"] = torch.zeros(4)
    sd["log_std_parameter"] = torch.zeros(4)
    pre = {"running_mean": torch.zeros(59), "running_variance": torch.ones(59), "current_count": torch.tensor(1000.0)}
    torch.save({"policy": sd, "observation_preprocessor": pre}, path)


def _scene():
    site = torch.tensor([2.5, 0.0, 1.25])
    dp = torch.tensor([[4.3, 0.0, 1.2], [1.6, 1.56, 1.2], [1.6, -1.56, 1.2]])
    ap = torch.tensor([[7.0, 0.5, 1.5], [6.5, -2.0, 1.4]])
    av = torch.tensor([[-1.2, 0.0, 0.0], [-0.8, 0.4, 0.0]])
    return site, dp, torch.zeros_like(dp), ap, av


def test_policy_loads_and_acts_in_range(tmp_path):
    p = tmp_path / "ck.pt"
    _fake_ckpt(p)
    pol = SharedPolicy.load(str(p))
    site, dp, dv, ap, av = _scene()
    alive = torch.ones(2, dtype=torch.bool)
    st = M.MacroState(1, 3, 2, "cpu")
    out = M.macro_step(dp[None], dv[None], ap[None], av[None], alive[None], site[None], st, M.MacroCfg(solver="hungarian"))
    obs = build_obs(dp, dv, ap, av, alive, site, out, torch.zeros(3, 4), torch.full((3,), 0.2))
    assert obs.shape == (3, 59) and torch.isfinite(obs).all()
    a = pol.act(obs)
    assert a.shape == (3, 4) and a.abs().max() <= 1.0


def test_obs_layout_slots_and_mine_flags():
    site, dp, dv, ap, av = _scene()
    alive = torch.ones(2, dtype=torch.bool)
    out = M.macro_step(dp[None], dv[None], ap[None], av[None], alive[None], site[None], M.MacroState(1, 3, 2, "cpu"),
                       M.MacroCfg())
    obs = build_obs(dp, dv, ap, av, alive, site, out, torch.zeros(3, 4), torch.zeros(3))
    for i in range(3):
        slot = int(out["slot"][0, i])
        assert obs[i, 12 + slot + 1] == 1.0  # slot one-hot (index 0 = post) at 12..17
        role = int(out["role"][0, i])
        mine = [obs[i, 18 + 11], obs[i, 18 + 12 + 11]]  # last entry of each attacker block
        assert [float(m) for m in mine] == [1.0 if role == j else 0.0 for j in range(2)]


def test_dead_defender_flies_nothing_and_is_far_for_teammates():
    site, dp, dv, ap, av = _scene()
    alive = torch.ones(2, dtype=torch.bool)
    dalive = torch.tensor([True, False, True])
    out = M.macro_step(dp[None], dv[None], ap[None], av[None], alive[None], site[None], M.MacroState(1, 3, 2, "cpu"),
                       M.MacroCfg(), def_alive=dalive[None])
    obs = build_obs(dp, dv, ap, av, alive, site, out, torch.zeros(3, 4), torch.zeros(3), dalive)
    mate_of_0 = obs[0, 42:45]  # first teammate block of drone 0 is drone 1 (dead)
    assert mate_of_0.abs().max() >= 12.0 - 1e-5
    cmd = safety(torch.ones(3, 4), dp, dalive)
    assert cmd[1].abs().sum() == 0 and cmd[:, 3].abs().sum() == 0


def test_safety_repels_close_teammates_and_guards_floor():
    dp = torch.tensor([[0.0, 0.0, 0.3], [0.2, 0.0, 0.3], [5.0, 5.0, 1.2]])
    cmd = safety(torch.zeros(3, 4), dp)
    assert cmd[0, 0] < 0 and cmd[1, 0] > 0  # pushed apart along x
    assert cmd[0, 2] >= 0.3  # below the floor guard: climbs


def test_shipped_planner_is_the_simulators_planner():
    """defender/macro.py must be byte-identical to the one the policy was trained with."""
    shipped = (HERE / "macro.py").read_bytes()
    trained = (REPO / "defend_task/macro.py").read_bytes()
    assert shipped == trained, "defender/macro.py drifted from the simulator's macro.py: copy it again"
