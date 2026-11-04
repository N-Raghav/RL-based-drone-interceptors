"""Trained defender policy at deploy time: checkpoint loader, observation builder, safety layer.

Everything here mirrors AS-Defend-Role-v0 (simulation_infra/.../tello_defend_real/
tello_defend_real_env.py _get_observations, tello_defend_role_env.py _pre_physics_step) so the
network sees on the laptop exactly what it saw in training. Only torch and numpy are needed:
no Isaac, no skrl.

    policy = SharedPolicy.load("shared_stage3_best.pt")
    obs = build_obs(est_def_pos, est_def_vel, est_att_pos, est_att_vel, att_alive, site,
                    macro_out, applied, own_age_s, def_alive)          # (D, 59)
    act = policy.act(obs)                                             # (D, 4) in [-1, 1]
    cmd = safety(act, est_def_pos, def_alive)                         # what is flown

Shapes: D defenders, A attackers; positions/velocities are world frame (m, m/s), site (3,).
"""

from __future__ import annotations

import torch

# observation clipping and safety layer: the training configuration's values
OBS_CLIP_POS = 12.0  # tello_defend_env_cfg.obs_clip_pos
OBS_CLIP_VEL = 3.0  # tello_defend_env_cfg.obs_clip_vel
REPEL_RADIUS = 0.5  # tello_defend_role_env_cfg.safety_repel_radius
REPEL_GAIN = 2.0  # .safety_repel_gain
SAFETY_FLOOR = 0.45  # .safety_floor [m] above the floor
MAX_LIN_SPEED = 1.0  # command scale in training: action 1 = 1 m/s
DEAD_OFFSET = 1.0e3  # a defender that is out is shown far away (as in training)


class SharedPolicy:
    """The shared defender network from a skrl PPO checkpoint (train_shared.py).

    Policy MLP (ELU hidden layers, tanh on the mean), deterministic: the mean action.
    Input normalisation: skrl RunningStandardScaler, clamp((x - mean) / (sqrt(var) + 1e-8), -5, 5), exactly as skrl.
    """

    def __init__(self, layers, mean, var, clip=5.0, eps=1e-8):
        self.layers = layers  # [(W, b), ...]
        self.mean, self.var, self.clip, self.eps = mean, var, clip, eps

    @classmethod
    def load(cls, path: str) -> "SharedPolicy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["policy"]
        weights = [(k, v) for k, v in sd.items() if k.endswith("weight") and v.dim() == 2]
        layers = []
        for k, w in weights:
            b = sd.get(k[: -len("weight")] + "bias")
            if b is None:
                raise ValueError(f"no bias for {k}; checkpoint keys: {list(sd)}")
            layers.append((w.float(), b.float()))
        dims = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
        if dims[0] != 59 or dims[-1] != 4:
            raise ValueError(f"unexpected policy layout {dims} (keys {list(sd)})")
        pre = ck.get("observation_preprocessor") or ck.get("state_preprocessor") or {}  # skrl 2 / skrl 1 key
        mean = pre.get("running_mean")
        var = pre.get("running_variance")
        if mean is None or var is None:
            raise ValueError(f"checkpoint has no observation normaliser (keys {list(pre)})")
        print(f"[policy] loaded {path}: layers {dims}, normaliser over {int(pre.get('current_count', 0))} samples",
              flush=True)
        return cls(layers, mean.float(), var.float())

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        x = ((obs - self.mean) / (torch.sqrt(self.var) + self.eps)).clamp(-self.clip, self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return torch.tanh(x).clamp(-1.0, 1.0)


def build_obs(def_pos, def_vel, att_pos, att_vel, att_alive, site, out, applied, own_age_s, def_alive=None):
    """Per-defender observation, (D, 59), as in training (stage 3: the macro's own slots).

    def_pos/def_vel (D,3), att_pos/att_vel (A,3), att_alive (A,) bool, site (3,), out = macro_step
    output for one env (batch dim 0), applied (D,4) last flown command in action units,
    own_age_s (D,) age of each defender's own fix in seconds, def_alive (D,) bool or None.
    """
    D, A = def_pos.shape[0], att_pos.shape[0]
    cp, cv = OBS_CLIP_POS, OBS_CLIP_VEL
    pos, vel = def_pos.clone().float(), def_vel.clone().float()
    if def_alive is not None:  # teammates must not see a defender that is out
        pos = pos + (~def_alive).float()[:, None] * DEAD_OFFSET
        vel = torch.where(def_alive[:, None], vel, torch.zeros_like(vel))
    alive = att_alive.float()
    zone = out["zone"][0].clamp(min=0)
    zone_1h = torch.nn.functional.one_hot(zone, 3).float() * alive[:, None]
    tau = torch.nan_to_num(out["tau"][0], posinf=10.0).clamp(0.0, 10.0) / 10.0
    role = out["role"][0]
    slot_1h = torch.nn.functional.one_hot(out["slot"][0] + 1, 2 * A + 2).float()  # index 0 = post
    target, base = out["target"][0], out["base"][0]
    ids = torch.arange(A)
    rows = []
    for i in range(D):
        p = pos[i]
        att_rel = (att_pos - p).clamp(-cp, cp) * alive[:, None]
        a_vel = att_vel.clamp(-cv, cv) * alive[:, None]
        mine = (role[i] == ids).float()
        per_att = torch.cat([att_rel, a_vel, (tau * alive)[:, None], zone_1h, alive[:, None], mine[:, None]], -1).reshape(-1)
        mate = torch.cat([torch.cat([(pos[k] - p).clamp(-cp, cp), vel[k].clamp(-cv, cv)]) for k in range(D) if k != i])
        rows.append(torch.cat([
            (p - site).clamp(-cp, cp), vel[i].clamp(-cv, cv), (target[i] - p).clamp(-cp, cp), base[i, :3],
            slot_1h[i], per_att, mate, applied[i].float(), own_age_s[i : i + 1].float(),
        ]))
    obs = torch.stack(rows)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)


def safety(act, def_pos, def_alive=None, floor_z=0.0):
    """The training safety layer: yaw off, teammate repulsion, floor guard, clamp. (D, 4) action units."""
    cmd = act.clone().float()
    cmd[:, 3] = 0.0
    diff = def_pos[:, None] - def_pos[None]
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
    push = (REPEL_RADIUS - dist).clamp(min=0.0) * REPEL_GAIN * diff / dist
    if def_alive is not None:  # a downed drone pushes nobody
        push = push * def_alive[None, :, None].float()
    cmd[:, :3] = cmd[:, :3] + push.sum(1) / MAX_LIN_SPEED
    low = (def_pos[:, 2] - floor_z) < SAFETY_FLOOR
    cmd[:, 2] = torch.where(low, cmd[:, 2].clamp(min=0.3), cmd[:, 2])
    cmd = cmd.clamp(-1.0, 1.0)
    if def_alive is not None:
        cmd = torch.where(def_alive[:, None], cmd, torch.zeros_like(cmd))
    return cmd
