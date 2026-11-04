"""Scripted macro layer for the 3-defender task: zones, slot assignment, target points, base velocity.

Pure torch, no Isaac imports, so the same file can run in the simulator, in unit tests on a
CPU, and on the laptop at deploy time. Every input is an *estimate* (what the laptop would
have: delayed, noisy positions and finite-difference velocities), never ground truth.

Shapes: E envs, D defenders, A attackers, S = 2A + 1 slots. World frame, metres, z up.

Pipeline, once per policy step (13.5 Hz):

  zones       per attacker: relaxed (0), alert (1), engage (2), from BOTH horizontal distance
              to the site and time-to-site tau (closing speed); the more dangerous one wins.
              Engage has hysteresis so an attacker on the boundary does not flicker.
  slots       an augmented D x S assignment (3 x 5 for 3v2). Slots:
                0..A-1     primary intercept on attacker j
                A..2A-1    secondary intercept on attacker j (layered point nearer the site,
                           or a flanking pincer point: cfg.secondary_mode)
                2A         goalie (guard the site, towards the main threat)
              Primary slots of live, alert-or-engaged attackers carry a large bonus, so they are
              always filled. The remaining drone(s) pick between secondaries and the goalie:
              a secondary gets cheaper as its attacker's best intercept margin (tau - TTI)
              shrinks, so the third drone helps only against a threat the primary may miss.
              Solved exactly by enumerating every injective drone -> slot map (60 for 3 x 5).
              Unusable slots (dead or far attackers) are priced out; a drone left only with
              unusable options takes its formation post. Engaged primaries are locked to their
              drone; any other change pays a switch cost, so roles do not flicker.
  targets     post: formation point. primary: alert -> staging point on the site->attacker line
              (hold ground, anti-baiting); engage -> predicted intercept point. secondary: the
              layered or pincer point. goalie: near the site, towards the main threat.
  base        velocity = gain * (target - pos), capped, plus teammate repulsion, as actions in
              [-1, 1] for TelloVelocityController (world-frame velocity, yaw rate 0).

Every drone therefore always has a defined target, which is what the PPO residual observes.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

RELAXED, ALERT, ENGAGE = 0, 1, 2
POST = -1  # slot value for a drone with no usable slot
UNUSABLE = 1.0e4


@dataclass
class MacroCfg:
    # zones (horizontal distance to the site centre, m; time to the site edge, s)
    site_radius: float = 1.5
    r_alert: float = 5.0
    r_engage: float = 3.5
    tau_alert: float = 6.0
    tau_engage: float = 3.0
    engage_release_m: float = 0.5  # hysteresis: leave engage only this far beyond r_engage ...
    engage_release_s: float = 1.0  # ... and this much beyond tau_engage
    min_closing_speed: float = 0.05  # [m/s] below this tau is infinite (not approaching)
    # slot costs (seconds)
    primary_bonus: float = 100.0  # subtracted from primary slots: an unguarded attacker is never OK
    margin_weight: float = 2.0  # secondary cost grows with the attacker's intercept margin
    goalie_margin_s: float = 0.5  # goalie cost = travel + margin_weight * this: the margin at
    # which helping (secondary) and guarding (goalie) cost the same
    max_margin_s: float = 10.0
    travel_weight: float = 0.3  # weight of travel time in secondary and goalie costs: the
    # help-or-guard decision should turn on the threat (margin), not on which point is nearer
    switch_cost_s: float = 0.5  # extra cost for a drone changing slot
    solver: str = "enumerate"  # "enumerate" (batched, GPU training) or "hungarian" (scipy, laptop)
    secondary_mode: str = "layered"  # "layered" (second line nearer the site) or "pincer" (flank)
    pincer_offset: float = 0.8  # [m] sideways from the primary intercept point
    # defender motion model used for planning
    defender_speed: float = 0.9  # [m/s] planning speed (below the controller cap, for margin)
    site_prior: float = 0.5  # predicted heading = blend of measured heading and the direction
    # to the site (attackers always end at the site, so this ignores swerves); 0 = pure velocity
    intercept_horizon_s: float = 8.0
    intercept_samples: int = 33
    # target geometry
    post_radius: float = 1.8
    post_height: float = 1.2
    guard_radius: float = 2.6  # alert staging distance from the site
    last_line_radius: float = 1.9  # fallback when no intercept is feasible
    gk_radius: float = 1.0  # goalie distance from the site, towards the main threat
    layer_radius: float = 1.9  # secondary (layered) point distance from the site
    min_layer_separation: float = 1.0
    layer_altitude_offset: float = 0.4
    z_min: float = 0.6
    z_max: float = 2.2
    # aggressiveness
    stage_in_alert: bool = False  # alert primaries hold a staging point; False = go for the intercept
    terminal_radius: float = 0.0  # [m] within this of its attacker a primary flies straight at it
    # base controller
    gain: float = 5.0  # [1/s]; high: full speed until ~0.2 m out (swept on the CPU replica)
    max_speed: float = 1.0  # [m/s] cap on the base command
    repel_radius: float = 0.6
    repel_gain: float = 2.0
    max_lin_speed: float = 1.0  # TelloVelocityController scale: action 1 = this many m/s


def num_slots(num_attackers: int) -> int:
    return 2 * num_attackers + 1


class MacroState:
    """Per-env memory of the macro layer: each drone's slot and the engage latches."""

    def __init__(self, num_envs: int, num_defenders: int, num_attackers: int, device):
        self.E, self.D, self.A = num_envs, num_defenders, num_attackers
        self.S = num_slots(num_attackers)
        self.device = device
        self.slot = torch.full((num_envs, num_defenders), POST, dtype=torch.long, device=device)
        self.engaged = torch.zeros(num_envs, num_attackers, dtype=torch.bool, device=device)
        perms = list(itertools.permutations(range(self.S), num_defenders))
        self.perms = torch.tensor(perms, dtype=torch.long, device=device)  # (P, D): slot per drone

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.slot[idx] = POST
        self.engaged[idx] = False


# -- geometry -------------------------------------------------------------------------------


def time_to_site(att_pos, att_vel, site, cfg: MacroCfg):
    """Horizontal distance to the site centre and time to reach the site edge, (E, A) each."""
    rel = att_pos[..., :2] - site[:, None, :2]
    dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-6)
    closing = -(att_vel[..., :2] * rel).sum(-1) / dist
    tau = (dist - cfg.site_radius).clamp(min=0.0) / closing.clamp(min=cfg.min_closing_speed)
    tau = torch.where(closing > cfg.min_closing_speed, tau, torch.full_like(tau, float("inf")))
    return dist, tau


def compute_zones(dist, tau, alive, state: MacroState, cfg: MacroCfg):
    """Zone per attacker, (E, A) long (-1 for dead), with engage hysteresis kept in state.engaged."""
    enter = (dist < cfg.r_engage) | (tau < cfg.tau_engage)
    stay = (dist < cfg.r_engage + cfg.engage_release_m) | (tau < cfg.tau_engage + cfg.engage_release_s)
    engaged = alive & (enter | (state.engaged & stay))
    state.engaged = engaged
    alert = (dist < cfg.r_alert) | (tau < cfg.tau_alert)
    zone = torch.where(alert, ALERT, RELAXED)
    zone = torch.where(engaged, ENGAGE, zone)
    return torch.where(alive, zone, torch.full_like(zone, -1))


def intercept_points(def_pos, att_pos, att_vel, tau, cfg: MacroCfg):
    """Earliest point on each attacker's straight-line prediction each defender can reach first.

    Returns (point (E,D,A,3), time (E,D,A), feasible (E,D,A)). Only times before the attacker
    reaches the site count. Infeasible pairs get time = inf.
    """
    E, D, A = def_pos.shape[0], def_pos.shape[1], att_pos.shape[1]
    t = torch.linspace(0.0, cfg.intercept_horizon_s, cfg.intercept_samples, device=def_pos.device)  # (K,)
    pred = att_pos[:, None, :, None, :] + att_vel[:, None, :, None, :] * t[:, None]  # (E,1,A,K,3)
    pred = pred.clone()
    pred[..., 2] = pred[..., 2].clamp(cfg.z_min, cfg.z_max)
    reach = torch.linalg.norm(pred - def_pos[:, :, None, None, :], dim=-1) / cfg.defender_speed  # (E,D,A,K)
    before_site = t[None, None, None, :] <= tau[:, None, :, None]
    ok = (reach <= t) & before_site
    any_ok = ok.any(dim=-1)
    first = torch.argmax(ok.int(), dim=-1)  # first True (0 if none, masked below)
    pred_b = pred.expand(E, D, A, -1, 3)
    point = torch.gather(pred_b, 3, first[..., None, None].expand(E, D, A, 1, 3)).squeeze(3)
    time = torch.where(any_ok, t[first], torch.full_like(t[first], float("inf")))
    return point, time, any_ok


def predicted_velocity(att_pos, att_vel, site, cfg: MacroCfg):
    """Velocity used for the intercept prediction: measured speed, heading pulled toward the site."""
    speed = torch.linalg.norm(att_vel, dim=-1, keepdim=True)
    heading = att_vel / speed.clamp(min=1e-6)
    to_site = site[:, None] - att_pos
    to_site = to_site / torch.linalg.norm(to_site, dim=-1, keepdim=True).clamp(min=1e-6)
    mix = (1.0 - cfg.site_prior) * heading + cfg.site_prior * to_site
    mix = mix / torch.linalg.norm(mix, dim=-1, keepdim=True).clamp(min=1e-6)
    return mix * speed


def line_point(site, att_pos, radius, cfg: MacroCfg):
    """Point at `radius` from the site centre towards each attacker, at the attacker's height."""
    rel = att_pos[..., :2] - site[:, None, :2]
    direc = rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
    xy = site[:, None, :2] + radius * direc
    z = att_pos[..., 2:3].clamp(cfg.z_min, cfg.z_max)
    return torch.cat([xy, z], dim=-1)  # (E, A, 3)


def formation_posts(site, num_defenders: int, cfg: MacroCfg):
    ang = torch.arange(num_defenders, device=site.device, dtype=site.dtype) * (2 * math.pi / num_defenders)
    xy = site[:, None, :2] + cfg.post_radius * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)[None]
    z = torch.full_like(xy[..., :1], cfg.post_height)
    return torch.cat([xy, z], dim=-1)  # (E, D, 3)


# -- slot assignment --------------------------------------------------------------------------


def slot_costs(def_pos, zone, alive, tti, feasible, reach_now, margin, sec_point, gk_point, cfg: MacroCfg):
    """Cost of every drone in every slot, (E, D, S), in seconds; UNUSABLE marks forbidden slots."""
    E, D = def_pos.shape[:2]
    A = alive.shape[1]
    active = alive & (zone >= ALERT)
    engaged = alive & (zone == ENGAGE)
    big = torch.full_like(reach_now, UNUSABLE)

    prim = torch.where(feasible, tti, 30.0 + reach_now) - cfg.primary_bonus
    prim = torch.where(active[:, None], prim, big)

    t_sec = torch.linalg.norm(sec_point[:, None] - def_pos[:, :, None], dim=-1) / cfg.defender_speed  # (E, D, A)
    m = margin.clamp(min=0.0, max=cfg.max_margin_s)
    sec = cfg.travel_weight * t_sec + cfg.margin_weight * m[:, None]
    sec = torch.where(engaged[:, None], sec, big)

    t_gk = torch.linalg.norm(gk_point[:, None] - def_pos, dim=-1) / cfg.defender_speed  # (E, D)
    gk = cfg.travel_weight * t_gk + cfg.margin_weight * cfg.goalie_margin_s
    gk = torch.where(active.any(-1)[:, None], gk, torch.full_like(gk, UNUSABLE))
    return torch.cat([prim, sec, gk[..., None]], dim=-1)


def solve_slots(cost, engaged, state: MacroState, cfg: MacroCfg):
    """Exact min-cost injective drone -> slot map over all permutations. (E, D) slot or POST."""
    E, D, S = cost.shape
    A = engaged.shape[1]
    P = state.perms  # (P, D)
    if cfg.solver == "hungarian":
        return solve_slots_hungarian(cost, engaged, state, cfg)
    drone = torch.arange(D, device=cost.device)
    perm_cost = cost[:, drone[None, :], P]  # (E, P, D)
    total = perm_cost.sum(-1)
    prev = state.slot[:, None, :]  # (E, 1, D)
    changed = (prev != POST) & (P[None] != prev)
    total = total + cfg.switch_cost_s * changed.sum(-1)
    # lock: a drone holding the primary slot of an engaged attacker keeps it
    prev_c = state.slot.clamp(min=0)
    prev_is_engaged_primary = (state.slot >= 0) & (state.slot < A) & engaged.gather(1, prev_c.clamp(max=A - 1))
    lock_break = prev_is_engaged_primary[:, None, :] & changed
    total = total + 1.0e6 * lock_break.sum(-1)
    best = total.argmin(dim=-1)  # (E,)
    slot = P[best]  # (E, D)
    chosen = cost.gather(2, slot[..., None]).squeeze(-1)
    slot = torch.where(chosen >= UNUSABLE * 0.5, torch.full_like(slot, POST), slot)
    state.slot = slot
    return slot


def solve_slots_hungarian(cost, engaged, state: MacroState, cfg: MacroCfg):
    """Same assignment as solve_slots, via the Hungarian method (scipy), one env at a time.

    The switch cost and the engaged-primary lock depend only on (drone, slot), so they fold
    into the 3 x 5 matrix and the problem stays a plain rectangular assignment. ~1 us per
    matrix on a laptop CPU vs ~90 us for the 60-permutation enumeration; the enumeration wins
    in training, where it scores thousands of envs in one GPU call.
    """
    from scipy.optimize import linear_sum_assignment

    E, D, S = cost.shape
    A = engaged.shape[1]
    prev = state.slot  # (E, D)
    cols = torch.arange(S, device=cost.device)[None, None, :]
    changed = (prev[..., None] != POST) & (cols != prev[..., None])  # (E, D, S)
    c = cost + cfg.switch_cost_s * changed.float()
    prev_c = prev.clamp(min=0)
    locked = (prev >= 0) & (prev < A) & engaged.gather(1, prev_c.clamp(max=A - 1))
    c = c + 1.0e6 * (locked[..., None] & changed).float()
    c_np = c.detach().cpu().double().numpy()
    slot = torch.empty(E, D, dtype=torch.long)
    for e in range(E):
        rows, assigned = linear_sum_assignment(c_np[e])
        slot[e, torch.as_tensor(rows)] = torch.as_tensor(assigned, dtype=torch.long)
    slot = slot.to(cost.device)
    chosen = cost.gather(2, slot[..., None]).squeeze(-1)
    slot = torch.where(chosen >= UNUSABLE * 0.5, torch.full_like(slot, POST), slot)
    state.slot = slot
    return slot


# -- main entry -----------------------------------------------------------------------------


def macro_step(def_pos, def_vel, att_pos, att_vel, alive, site, state: MacroState, cfg: MacroCfg, def_alive=None):
    """One macro update. All inputs are estimates.

    def_pos/def_vel (E,D,3), att_pos/att_vel (E,A,3), alive (E,A) bool, site (E,3).
    Returns a dict of tensors used for control, observations and logging.
    """
    E, D, A = def_pos.shape[0], def_pos.shape[1], att_pos.shape[1]
    dev = def_pos.device
    ar = torch.arange(E, device=dev)
    if def_alive is not None:
        # a defender that is out (made contact, crashed, landed) cannot intercept anything
        def_pos = torch.where(def_alive[..., None], def_pos, def_pos + 1.0e3)
    dist, tau = time_to_site(att_pos, att_vel, site, cfg)
    zone = compute_zones(dist, tau, alive, state, cfg)
    pred_vel = predicted_velocity(att_pos, att_vel, site, cfg)
    point, tti, feasible = intercept_points(def_pos, att_pos, pred_vel, tau, cfg)
    reach_now = torch.linalg.norm(att_pos[:, None] - def_pos[:, :, None], dim=-1) / cfg.defender_speed

    # best achievable margin per attacker (independent of who ends up primary)
    best_tti = torch.where(feasible, tti, torch.full_like(tti, float("inf"))).min(dim=1).values  # (E, A)
    #   approaching, interceptable  -> tau - best TTI
    #   approaching, no intercept   -> -1 (the primary cannot make it: help is worth most)
    #   not approaching, or dead    -> max_margin (no threat)
    margin = torch.where(torch.isfinite(best_tti), tau - best_tti, torch.full_like(tau, -1.0))
    no_threat = ~alive | torch.isinf(tau)
    margin = torch.where(no_threat, torch.full_like(margin, cfg.max_margin_s), margin)

    # geometry of every slot's target
    posts = formation_posts(site, D, cfg)
    staging = line_point(site, att_pos, cfg.guard_radius, cfg)  # (E, A, 3)
    last_line = line_point(site, att_pos, cfg.last_line_radius, cfg)
    d_masked = torch.where(alive, dist, torch.full_like(dist, float("inf")))
    main_j = d_masked.argmin(dim=-1)
    gk_point = line_point(site, att_pos, cfg.gk_radius, cfg)[ar, main_j]  # (E, 3)
    # the attacker's best intercept point (whoever reaches first): reference for secondaries
    best_d = torch.where(feasible, tti, torch.full_like(tti, float("inf"))).argmin(dim=1)  # (E, A)
    best_point = point[ar[:, None], best_d, torch.arange(A, device=dev)[None]]  # (E, A, 3)
    best_point = torch.where(torch.isfinite(best_tti)[..., None], best_point, last_line)
    if cfg.secondary_mode == "pincer":
        to_site = site[:, None] - best_point
        side = torch.stack([-to_site[..., 1], to_site[..., 0], torch.zeros_like(to_site[..., 0])], dim=-1)
        side = side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp(min=1e-6)
        sec_point = best_point + cfg.pincer_offset * side
    else:
        sec_point = line_point(site, att_pos, cfg.layer_radius, cfg)
    # keep a secondary clear of its primary's point: lift it if they would converge
    close = torch.linalg.norm(sec_point - best_point, dim=-1) < cfg.min_layer_separation
    sec_point = sec_point.clone()
    sec_point[..., 2] = torch.where(
        close, (sec_point[..., 2] + cfg.layer_altitude_offset).clamp(max=cfg.z_max), sec_point[..., 2]
    )

    cost = slot_costs(def_pos, zone, alive, tti, feasible, reach_now, margin, sec_point, gk_point, cfg)
    if def_alive is not None:
        cost = torch.where(def_alive[..., None], cost, torch.full_like(cost, UNUSABLE))
    slot = solve_slots(cost, alive & (zone == ENGAGE), state, cfg)  # (E, D)

    # primary target per (defender, attacker): engage -> own intercept point, alert -> staging
    eng = (zone == ENGAGE)[:, None, :].expand(E, D, A)
    prim = torch.where(feasible[..., None], point, last_line[:, None].expand(E, D, A, 3))
    if cfg.stage_in_alert:
        prim = torch.where(eng[..., None], prim, staging[:, None].expand(E, D, A, 3))
    if cfg.terminal_radius > 0:
        near = (torch.linalg.norm(att_pos[:, None] - def_pos[:, :, None], dim=-1) < cfg.terminal_radius)
        prim = torch.where(near[..., None], att_pos[:, None].expand(E, D, A, 3), prim)

    target = posts.clone()
    is_prim = (slot >= 0) & (slot < A)
    is_sec = (slot >= A) & (slot < 2 * A)
    is_gk = slot == 2 * A
    j_prim = slot.clamp(0, A - 1)
    j_sec = (slot - A).clamp(0, A - 1)
    prim_t = torch.gather(prim, 2, j_prim[..., None, None].expand(E, D, 1, 3)).squeeze(2)
    sec_t = torch.gather(sec_point, 1, j_sec[..., None].expand(E, D, 3))
    target = torch.where(is_prim[..., None], prim_t, target)
    target = torch.where(is_sec[..., None], sec_t, target)
    target = torch.where(is_gk[..., None], gk_point[:, None].expand(E, D, 3), target)

    # derived views
    role = torch.where(is_prim, j_prim, torch.where(is_sec, j_sec, torch.full_like(slot, -1)))
    assign = torch.full((E, A), -1, dtype=torch.long, device=dev)
    for j in range(A):
        holder = slot == j
        has = holder.any(-1)
        assign[:, j] = torch.where(has, holder.int().argmax(-1), assign[:, j])
    gk_exists = is_gk.any(-1)
    goalkeeper = is_gk.int().argmax(-1)

    base = base_actions(def_pos, target, cfg)
    return {
        "zone": zone,  # (E, A)
        "dist": dist,
        "tau": tau,
        "margin": margin,  # (E, A) best achievable intercept margin, s
        "slot": slot,  # (E, D): 0..A-1 primary, A..2A-1 secondary, 2A goalie, -1 post
        "cost": cost,  # (E, D, S)
        "assign": assign,  # (E, A): drone holding each primary slot, or -1
        "role": role,  # (E, D): attacker a drone works on (primary or secondary), or -1
        "goalkeeper": goalkeeper,  # (E,)
        "gk_exists": gk_exists,
        "join_on": is_sec.any(-1),  # (E,): some drone is helping as a secondary
        "target": target,  # (E, D, 3)
        "base": base,  # (E, D, 4)
        # slot geometry, for callers that assign roles themselves (role-conditioned policy)
        "intercept": prim,  # (E, D, A, 3): each drone's primary target point per attacker
        "sec_point": sec_point,  # (E, A, 3)
        "gk_point": gk_point,  # (E, 3)
        "posts": posts,  # (E, D, 3)
    }


def base_actions(def_pos, target, cfg: MacroCfg):
    """Proportional velocity to target, capped, plus repulsion from teammates. (E, D, 4)."""
    v = cfg.gain * (target - def_pos)
    diff = def_pos[:, :, None] - def_pos[:, None, :]
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
    rep = ((cfg.repel_radius - dist).clamp(min=0.0) * cfg.repel_gain * diff / dist).sum(dim=2)
    v = v + rep
    speed = torch.linalg.norm(v, dim=-1, keepdim=True)
    v = v * (cfg.max_speed / speed.clamp(min=cfg.max_speed))
    yaw = torch.zeros_like(v[..., :1])
    return torch.cat([v / cfg.max_lin_speed, yaw], dim=-1).clamp(-1.0, 1.0)


def chase_base(def_pos, att_pos, alive, cfg: MacroCfg):
    """The repo's greedy baseline: every defender flies at full speed at its nearest attacker."""
    d = torch.linalg.norm(att_pos[:, None] - def_pos[:, :, None], dim=-1)  # (E, D, A)
    d = torch.where(alive[:, None], d, torch.full_like(d, float("inf")))
    j = d.argmin(dim=-1)  # (E, D)
    tgt = torch.gather(att_pos[:, None].expand(-1, def_pos.shape[1], -1, -1), 2, j[..., None, None].expand(-1, -1, 1, 3))
    rel = tgt.squeeze(2) - def_pos
    v = rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.where(alive.any(-1)[:, None, None], v, torch.zeros_like(v))
    return torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
