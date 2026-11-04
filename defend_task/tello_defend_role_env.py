"""AS-Defend-Role-v0: the macro layer decides WHAT each drone does, PPO learns HOW to fly it.

Same world, link model and sensing as AS-Defend-Real-v0 (3 defenders vs 2 attackers). Steps:

  1 zones       scripted (macro.compute_zones)
  2 slots       scripted 3x5 assignment (macro.solve_slots), or forced roles in training stages
  3 fly         PPO: the action IS the world-frame velocity command (no saturated base under it).
                A safety layer adds teammate repulsion and a floor guard; it is not a reward.

Training stages (cfg.stage), one network carried through all three:

  1 interception  every drone is a primary: d0 on attacker A, d1 on B, d2 on a random one.
                  Teaches contact against faster, swerving attackers over the real link.
  2 roles         roles forced at random per episode: both primaries filled, the third drone
                  is secondary A, secondary B or goalie. Teaches every role.
  3 full          the macro's own slot assignment, re-solved every step.

Reward per drone = shared team outcome + the reward of its current role:

  team       breach -10 and capture +5, to every drone
  primary    potential on the TRUE distance to its attacker; +10 to the drone that made contact
  secondary  potential on its layered point; +5 when its attacker goes down (anyone's contact)
  goalie     potential on the goalie point; small bonus per step while within 0.3 m
  post       potential on its formation post
  floor      -2 per step below floor_min_height
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import sample_uniform

from action_space_kit.tags.room_map import load_room_tags
from action_space_kit.tags.sensing import TagSensorCfg, camera_axes, detectable, sighting

from .macro import base_actions
from .tello_defend_real_env import TelloDefendRealEnv
from .tello_defend_role_env_cfg import TelloDefendRoleEnvCfg


class TelloDefendRoleEnv(TelloDefendRealEnv):
    cfg: TelloDefendRoleEnvCfg

    def __init__(self, cfg: TelloDefendRoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        E, D, A, dev = self.num_envs, cfg.num_defenders, cfg.num_attackers, self.device
        self._forced_slot = torch.full((E, D), -1, dtype=torch.long, device=dev)
        self._slot = torch.full((E, D), -1, dtype=torch.long, device=dev)
        self._role_point = torch.zeros(E, D, 3, device=dev)  # target for non-primary roles
        self._alive_at_sense = torch.ones(E, A, dtype=torch.bool, device=dev)
        self._prev_role_dist = torch.zeros(E, D, device=dev)
        self._homing = torch.ones(E, D, dtype=torch.bool, device=dev)
        self._sample_roles(torch.arange(E, device=dev))
        self._def_alive = torch.ones(E, D, dtype=torch.bool, device=dev)
        self._speed_scale = torch.ones(E, D, device=dev)
        self._def_down_now = torch.zeros(E, D, dtype=torch.bool, device=dev)
        # rc latency: previous command per defender, and each defender's latency this episode
        self._cmd_prev = torch.zeros(E, D, 4, device=dev)
        self._latency = torch.zeros(E, D, dtype=torch.bool, device=dev)
        if cfg.room:
            tags = load_room_tags(tag10_height=cfg.tag10_height)
            shift = torch.tensor([cfg.room_site_x, 0.0, 0.0], device=dev)  # world -> site-centred env frame
            ids = sorted(tags)
            self._tag_pos = torch.tensor([tags[i].pos for i in ids], dtype=torch.float32, device=dev) - shift
            self._tag_normal = torch.tensor([tags[i].normal for i in ids], dtype=torch.float32, device=dev)
            self._tag_size = torch.tensor([tags[i].size_m for i in ids], dtype=torch.float32, device=dev)
            self._tag10 = self._tag_pos[ids.index(10)]
            self._sensor = TagSensorCfg()
            self._cam_yaw = torch.zeros(E, D, device=dev)
            print(f"[role env] room mode: {len(ids)} tags {ids}, site {cfg.room_site_x} m from the tag wall", flush=True)

    # -- roles ----------------------------------------------------------------------------------

    def _sample_roles(self, env_ids: torch.Tensor) -> None:
        cfg, n, dev = self.cfg, len(env_ids), self.device
        A = cfg.num_attackers
        # both primaries always filled; the third drone is a primary too (stage 1) or a
        # secondary / goalie (stage 2). Drones are shuffled every episode: skrl's MAPPO keeps one
        # network per drone, so every drone has to see every role or the roles do not transfer.
        if cfg.stage == 1:
            third = torch.randint(0, A, (n,), device=dev)
        else:
            third = torch.randint(A, 2 * A + 1, (n,), device=dev)
        slots = torch.stack([torch.zeros(n, device=dev).long(), torch.ones(n, device=dev).long(), third], -1)
        if cfg.shuffle_roles:
            perm = torch.argsort(torch.rand(n, 3, device=dev), dim=-1)
            slots = torch.gather(slots, 1, perm)
        self._forced_slot[env_ids] = slots

    def _sense_and_plan(self) -> None:
        super()._sense_and_plan()
        if self.cfg.defender_attrition:
            # teammates must not see a defender that is out at its last (stale) fix
            far = torch.where(self._def_alive[..., None], torch.zeros_like(self._est_def_pos), torch.full_like(self._est_def_pos, 1.0e3))
            self._est_def_pos = self._est_def_pos + far
            self._est_def_vel = torch.where(self._def_alive[..., None], self._est_def_vel, torch.zeros_like(self._est_def_vel))
        cfg, out = self.cfg, self._macro_out
        A = cfg.num_attackers
        E, D = self.num_envs, cfg.num_defenders
        alive = self._attacker_alive
        if cfg.stage == 3:
            slot = out["slot"].clone()
        else:
            slot = self._forced_slot.clone()
            # a role whose attacker is down falls back: primary/secondary -> goalie if any
            # attacker is left, else post
            j = torch.where(slot < A, slot, slot - A).clamp(0, A - 1)
            on_att = (slot >= 0) & (slot < 2 * A)
            dead = on_att & ~alive.gather(1, j)
            fallback = torch.where(alive.any(-1, keepdim=True), 2 * A, -1).expand(E, D)
            slot = torch.where(dead, fallback, slot)
        self._slot = slot
        is_prim = (slot >= 0) & (slot < A)
        is_sec = (slot >= A) & (slot < 2 * A)
        is_gk = slot == 2 * A
        j_prim = slot.clamp(0, A - 1)
        j_sec = (slot - A).clamp(0, A - 1)

        # targets the policy observes (estimates)
        prim_t = torch.gather(out["intercept"], 2, j_prim[..., None, None].expand(E, D, 1, 3)).squeeze(2)
        sec_t = torch.gather(out["sec_point"], 1, j_sec[..., None].expand(E, D, 3))
        target = out["posts"].clone()
        target = torch.where(is_prim[..., None], prim_t, target)
        target = torch.where(is_sec[..., None], sec_t, target)
        target = torch.where(is_gk[..., None], out["gk_point"][:, None].expand(E, D, 3), target)
        self._target = target
        self._role_point = target
        out["slot"] = slot
        out["role"] = torch.where(is_prim, j_prim, torch.where(is_sec, j_sec, torch.full_like(slot, -1)))
        # the scripted command toward the role target: an observation (hint), never applied
        self._base = base_actions(self._est_def_pos, target, cfg.macro)

        # potentials measured now, on TRUE positions. A primary's target is fixed for this step:
        # the intercept point (lead the attacker), or the attacker itself once within
        # homing_radius (terminal dash). Fixing it here keeps the shaping potential-based.
        self._alive_at_sense = alive.clone()
        self._homing = torch.ones_like(slot, dtype=torch.bool)
        if cfg.prim_target == "intercept":
            self._homing = self._dist_to_attacker() < cfg.homing_radius
        self._prev_role_dist = self._role_distance()

    # -- room: tag-limited own fixes, look-at-tag heading --------------------------------------

    def _own_fix_model(self, delayed: torch.Tensor):
        cfg = self.cfg
        if not cfg.room:
            return super()._own_fix_model(delayed)
        E, D = delayed.shape[:2]
        local = delayed - self.scene.env_origins[:, None]  # site-centred env frame
        if cfg.heading_mode == "forward":
            v = self._est_def_vel[..., :2]
            moving = torch.linalg.norm(v, dim=-1) > 0.2
            yaw = torch.where(moving, torch.atan2(v[..., 1], v[..., 0]), self._cam_yaw)
        else:
            to_tag = self._tag10[None, None, :2] - local[..., :2]
            yaw = torch.atan2(to_tag[..., 1], to_tag[..., 0])  # the adapter's yaw loop: look at tag 10
        yaw_rate = torch.remainder(yaw - self._cam_yaw + torch.pi, 2 * torch.pi) - torch.pi
        yaw_rate = torch.where(self._just_reset[:, None], torch.zeros_like(yaw_rate), yaw_rate) / self.step_dt
        self._cam_yaw = yaw
        c, s = torch.cos(yaw).reshape(-1), torch.sin(yaw).reshape(-1)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        body = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
        off = torch.tensor(self._sensor.camera_offset, device=self.device)
        cam_pos = local.reshape(-1, 3) + body @ off
        cam_rot = camera_axes(body, self._sensor.camera_pitch)
        speed = torch.linalg.norm(self._est_def_vel, dim=-1).reshape(-1)
        fix, seen, _ = sighting(self._sensor, cam_pos, cam_rot, self._tag_pos, self._tag_normal, self._tag_size,
                                noise=True, speed=speed, yaw_rate=yaw_rate.reshape(-1))
        geo, _ = detectable(self._sensor, cam_pos, cam_rot, self._tag_pos, self._tag_normal, self._tag_size)
        self._tag_visible = geo.any(dim=1).reshape(E, D)
        fix = (fix - body @ off).reshape(E, D, 3) + self.scene.env_origins[:, None] + self._own_bias
        return fix, seen.reshape(E, D)

    def _resolve_contacts(self, contact, captured, capturer):
        """Random outcome per contact (cfg.contact_outcome_probs): both down / the attacker survives
        (the rammer is down) / the defender survives (the attacker is down). Only the first
        defender in contact (the capturer) bears the outcome. With cfg.defender_single_use the capturer
        is out after any contact, whatever happens to the attacker (it cannot engage the next one)."""
        cfg = self.cfg
        E, D, A = contact.shape
        none = torch.zeros(E, D, dtype=torch.bool, device=self.device)
        if not cfg.defender_attrition:
            return captured, none
        p = torch.tensor(cfg.contact_outcome_probs, dtype=torch.float32, device=self.device)
        p = p / p.sum()
        u = torch.rand(E, A, device=self.device)
        both = captured & (u < p[0])
        att_survives = captured & (u >= p[0]) & (u < p[0] + p[1])
        def_survives = captured & (u >= p[0] + p[1])
        attacker_down = both | def_survives
        rammer = torch.nn.functional.one_hot(capturer, D).bool()  # (E, A, D)
        rammer_out = captured if cfg.defender_single_use else (both | att_survives)  # single use: every rammer is spent
        defender_down = (rammer & rammer_out[..., None]).any(dim=1)  # (E, D)
        self._contact_outcomes = torch.stack([both, att_survives, def_survives], -1).float().sum(1)  # (E, 3)
        return attacker_down, defender_down

    def _park_defenders(self, out: torch.Tensor) -> None:
        """Move defenders that are out far away (like downed attackers)."""
        for i, drone in enumerate(self._defenders):
            ids = out[:, i].nonzero(as_tuple=False).flatten()
            if len(ids) == 0:
                continue
            state = drone.data.default_root_state[ids].clone()
            state[:, :3] = self.scene.env_origins[ids]
            state[:, 0] -= 60.0 + 3.0 * i
            state[:, 2] += 1.0
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], ids)
            drone.write_root_velocity_to_sim(state[:, 7:], ids)
            self._defender_ctl[i].reset(ids)

    def _wall_push(self, local_pos: torch.Tensor, cmd: torch.Tensor, push: float = 0.4) -> torch.Tensor:
        """Soft walls of the room on x/y velocity commands (action units)."""
        cfg = self.cfg
        x = local_pos[..., 0] + cfg.room_site_x  # distance from the tag wall
        y = local_pos[..., 1]
        m, hw = cfg.room_margin, 0.5 * cfg.room_width
        cmd = cmd.clone()
        cmd[..., 0] = torch.where(x < m, cmd[..., 0].clamp(min=push), cmd[..., 0])
        cmd[..., 0] = torch.where(x > cfg.room_depth - m, cmd[..., 0].clamp(max=-push), cmd[..., 0])
        cmd[..., 1] = torch.where(y < -hw + m, cmd[..., 1].clamp(min=push), cmd[..., 1])
        cmd[..., 1] = torch.where(y > hw - m, cmd[..., 1].clamp(max=-push), cmd[..., 1])
        return cmd

    def _scripted_attacker_actions(self):
        out = super()._scripted_attacker_actions()
        if not self.cfg.room:
            return out
        res = []
        for i, a in enumerate(out):
            local = self._attackers[i].data.root_pos_w - self.scene.env_origins
            moving = a[:, :3].abs().sum(-1, keepdim=True) > 0
            res.append(torch.where(moving, self._wall_push(local, a), a))
        return res

    def _dist_to_attacker(self) -> torch.Tensor:
        """Per drone: true distance to the attacker of its slot (primary or secondary). (E, D)."""
        A = self.cfg.num_attackers
        dp, _, ap, _ = self._true_states()
        slot = self._slot
        j = torch.where(slot < A, slot, slot - A).clamp(0, A - 1)
        att = torch.gather(ap, 1, j[..., None].expand(-1, -1, 3))
        return torch.linalg.norm(dp - att, dim=-1)

    def _role_distance(self) -> torch.Tensor:
        """Per drone: distance the role reward wants to shrink (true positions). (E, D).

        primary   prim_target "intercept": the macro's intercept point (lead the attacker), and
                  the attacker itself once within homing_radius; "attacker": always the attacker
        others    their role point (layered point, goalie point, post)
        """
        cfg = self.cfg
        A = cfg.num_attackers
        dp, _, _, _ = self._true_states()
        slot = self._slot
        is_prim = (slot >= 0) & (slot < A)
        d_att = self._dist_to_attacker()
        d_point = torch.linalg.norm(dp - self._role_point, dim=-1)
        d_prim = torch.where(self._homing, d_att, d_point)
        return torch.where(is_prim, d_prim, d_point)

    # -- control --------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        cfg = self.cfg
        act = torch.stack([actions[f"defender_{i}"] for i in range(cfg.num_defenders)], dim=1).clamp(-1.0, 1.0)
        act = act.clone()
        act[..., 3] = 0.0  # yaw belongs to the scripted yaw loop
        # safety layer: teammate repulsion and a floor guard (estimates only, like the laptop)
        pos = self._est_def_pos
        diff = pos[:, :, None] - pos[:, None, :]
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
        rep = ((cfg.safety_repel_radius - dist).clamp(min=0.0) * cfg.safety_repel_gain * diff / dist).sum(dim=2)
        cmd = act.clone()
        cmd[..., :3] = cmd[..., :3] + rep / cfg.command.max_lin_speed
        low = (pos[..., 2] - self.scene.env_origins[:, None, 2]) < cfg.safety_floor
        cmd[..., 2] = torch.where(low, cmd[..., 2].clamp(min=0.3), cmd[..., 2])
        if cfg.room:
            cmd = self._wall_push(pos - self.scene.env_origins[:, None], cmd)
        cmd = cmd.clamp(-1.0, 1.0)
        cmd = torch.where(self._def_alive[..., None], cmd, torch.zeros_like(cmd))
        # rc latency: a defender with one step of latency flies last step's command
        delayed = torch.where(self._latency[..., None], self._cmd_prev, cmd)
        self._cmd_prev = cmd
        cmd = delayed
        lost = torch.rand_like(self._rc_loss) < self._rc_loss
        cmd = torch.where(lost[..., None], self._applied, cmd)
        self._prev_applied = self._applied
        self._applied = cmd
        self._prev_residual = self._residual
        self._residual = act
        # the real top speed is uncertain (drone_gui rc cap, Tello firmware): each defender flies
        # its command scaled by a per-episode factor; the policy only sees the response
        flown = cmd.clone()
        flown[..., :3] = flown[..., :3] * self._speed_scale[..., None]
        for i in range(cfg.num_defenders):
            self._actions[f"defender_{i}"] = flown[:, i]

    # -- NaN guard ------------------------------------------------------------------------------

    def _nonfinite_envs(self) -> torch.Tensor:
        dp, dv, ap, av = self._true_states()
        bad = ~torch.isfinite(torch.cat([dp, dv, ap, av], dim=1).reshape(self.num_envs, -1)).all(dim=-1)
        bad |= dv.abs().amax(dim=(1, 2)) > 50.0  # a drone flung by a physics blow-up
        return bad

    def _get_dones(self):
        terminated, time_outs = super()._get_dones()
        if self.cfg.defender_attrition:
            hit = self._def_down_now & self._def_alive
            if hit.any():
                self._def_alive &= ~hit
                self._park_defenders(hit)
        bad = self._nonfinite_envs()
        self._bad_envs = bad
        self.outcome_valid = self.outcome_valid & ~bad  # a blown-up episode is not an outcome
        finished = terminated[self.cfg.possible_agents[0]] | time_outs[self.cfg.possible_agents[0]]
        self._success_now = finished & (self._episode_breaches == 0) & ~bad
        # overlap monitor (bodies touching): any pair of the five drones within 0.12 m
        dp, _, ap, _ = self._true_states()
        allp = torch.cat([dp, ap], dim=1)
        d = torch.linalg.norm(allp[:, :, None] - allp[:, None], dim=-1)
        d = d + 1e3 * torch.eye(allp.shape[1], device=self.device)
        alive = torch.cat([torch.ones_like(self._attacker_alive[:, :1]).expand(-1, dp.shape[1]), self._attacker_alive], 1)
        pair_ok = alive[:, :, None] & alive[:, None]
        self._ep_overlap = getattr(self, "_ep_overlap", torch.zeros(self.num_envs, device=self.device))
        self._ep_overlap += ((d < 0.12) & pair_ok).any(-1).any(-1).float()
        if bad.any():
            self.nan_resets = getattr(self, "nan_resets", 0) + int(bad.sum().item())
            if self._tb is not None:
                self._tb.add_scalar("Episode/nan_resets_total", self.nan_resets, int(self.common_step_counter))
            time_outs = {a: v | bad for a, v in time_outs.items()}
        return terminated, time_outs

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = super()._get_observations()
        return {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0) for k, v in obs.items()}

    def _get_states(self) -> torch.Tensor:
        return torch.nan_to_num(super()._get_states(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)

    # -- rewards --------------------------------------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        A = cfg.num_attackers
        dp, _, _, _ = self._true_states()
        slot = self._slot
        is_prim = (slot >= 0) & (slot < A)
        is_sec = (slot >= A) & (slot < 2 * A)
        is_gk = slot == 2 * A
        j = torch.where(is_sec, slot - A, slot).clamp(0, A - 1)
        att_alive_now = self._attacker_alive.gather(1, j)
        att_alive_then = self._alive_at_sense.gather(1, j)
        att_downed = att_alive_then & ~att_alive_now
        on_att = is_prim | is_sec

        # potential on the role distance; an attacker that went down this step pays no progress
        # (it was parked 60 m away), the capture terms pay for it instead
        dist = self._role_distance()
        progress = (self._prev_role_dist - cfg.discount * dist).clamp(-0.3, 0.3)
        progress = torch.where(on_att & ~att_alive_now, torch.zeros_like(progress), progress)
        w = torch.where(is_prim, cfg.rew_prim_progress, torch.where(is_sec, cfg.rew_sec_progress, cfg.rew_hold_progress))

        captured_by_me = (self._step_credit > 0).float()
        floor = (dp[..., 2] - self.scene.env_origins[:, None, 2] < cfg.floor_min_height).float()
        D = cfg.num_defenders
        terms = {
            "team_breach": (cfg.rew_breach * self._step_breaches)[:, None].expand(-1, D),
            "team_capture": (cfg.rew_team_capture * self._step_captures)[:, None].expand(-1, D),
            "contact": cfg.rew_contact * captured_by_me,
            "progress": w * progress,
            "sec_downed": cfg.rew_sec_downed * (is_sec & att_downed).float(),
            "goalie_hold": cfg.rew_goalie_hold * (is_gk & (dist < 0.3)).float(),
            "floor": cfg.rew_floor * floor,
        }
        # v2: success, self-collision, no tag in view
        succ = getattr(self, "_success_now", None)
        if succ is not None:
            terms["success"] = (cfg.rew_success * succ.float())[:, None].expand(-1, D)
        pair = torch.linalg.norm(dp[:, :, None] - dp[:, None], dim=-1) + 1e3 * torch.eye(D, device=self.device)
        terms["self_collision"] = cfg.rew_self_collision * (pair < cfg.self_collision_radius).any(-1).float()
        if cfg.room and hasattr(self, "_tag_visible"):
            terms["no_tag"] = cfg.rew_no_tag * (~self._tag_visible).float()
        bad = getattr(self, "_bad_envs", None)
        if bad is not None and bad.any():
            terms = {k: torch.where(bad[:, None], torch.zeros_like(v), v) for k, v in terms.items()}
        if self.cfg.defender_attrition:
            # a defender that is out earns nothing more (its contact this step was credited first)
            gone = ~self._def_alive & ~self._def_down_now  # a drone downed this step keeps this step's reward
            terms = {k: torch.where(gone, torch.zeros_like(v), v) for k, v in terms.items()}
        terms = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in terms.items()}
        total = sum(terms.values())
        for k, v in terms.items():
            self._ep_terms[k] = self._ep_terms.get(k, torch.zeros(self.num_envs, device=self.device)) + v.mean(-1)
        return {f"defender_{i}": total[:, i] for i in range(D)}

    # -- reset ----------------------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if hasattr(self, "_forced_slot"):
            ids = self._defenders[0]._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device)
            bad = getattr(self, "_bad_envs", None)
            if bad is not None and self._ep_terms and len(ids):
                # pre-subtract blown-up episodes so the parent's tally of these resets nets them out
                b = ids[bad[ids]]
                if len(b):
                    self._stats["episodes"] -= float(len(b))
                    self._stats["breaches"] -= self._episode_breaches[b].sum().item()
                    self._stats["captures"] -= self._episode_captures[b].sum().item()
                    self._stats["breached_eps"] -= (self._episode_breaches[b] > 0).float().sum().item()
                    for k, v in self._ep_terms.items():
                        self._stat_terms[k] = self._stat_terms.get(k, 0.0) - v[b].sum().item()
            ov = getattr(self, "_ep_overlap", None)
            if ov is not None and len(ids):
                good = ids if bad is None else ids[~bad[ids]]
                self._stat_terms["overlap_steps"] = self._stat_terms.get("overlap_steps", 0.0) + ov[good].sum().item()
                ov[ids] = 0.0
        super()._reset_idx(env_ids)
        if not hasattr(self, "_forced_slot"):
            return
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._sample_roles(env_ids)
        self._def_alive[env_ids] = True
        self._speed_scale[env_ids] = sample_uniform(*self.cfg.defender_speed_range, (len(env_ids), self.cfg.num_defenders), self.device)
        cfg = self.cfg
        n, dev = len(env_ids), self.device
        lo, hi = cfg.rc_latency_steps
        self._latency[env_ids] = torch.randint(lo, hi + 1, (n, cfg.num_defenders), device=dev) > 0
        self._cmd_prev[env_ids] = 0.0
        if cfg.room:
            # attackers launch from the far part of the room
            for i, drone in enumerate(self._attackers):
                state = drone.data.default_root_state[env_ids].clone()
                x = sample_uniform(*cfg.attacker_spawn_x, (n,), dev) - cfg.room_site_x
                hw = 0.5 * cfg.room_width - cfg.room_margin - 0.2
                y = sample_uniform(-hw, hw, (n,), dev)
                state[:, 0] = self.scene.env_origins[env_ids, 0] + x
                state[:, 1] = self.scene.env_origins[env_ids, 1] + y
                state[:, 2] = self._spawn_att[env_ids, i, 2]
                state[:, 7:] = 0.0
                drone.write_root_pose_to_sim(state[:, :7], env_ids)
                drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
                self._spawn_att[env_ids, i] = state[:, :3]
            self._feed[env_ids] = self._spawn_att[env_ids]
            self._cam_yaw[env_ids] = torch.pi if cfg.heading_mode == "forward" else 0.0
        self._alive_at_sense[env_ids] = True
        self._prev_role_dist[env_ids] = 0.0
