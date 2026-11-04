"""AS-Defend-Real-v0: 3 Tello defenders vs 2 scripted attackers over the measured link.

Per policy step (13.5 Hz):

  sense     true poses go into a short history. Each defender's own fix is read
            ``own_delay`` steps back, with noise and a per-episode bias, and is lost (held,
            age grows) with the episode's video loss rate. The attacker feed is read
            ``feed_delay`` steps back with noise; attacker velocity is a finite difference
            over ``vel_window_steps`` of that feed. Everything is then predicted to "now"
            by its age. The policy and the macro layer only ever see these estimates.
  macro     macro.macro_step: zones, 3x5 slot assignment (primaries, secondaries, goalie),
            target points and a proportional base command per defender.
  act       applied = clip(base + residual_scale * policy, -1, 1) on x/y/z, yaw 0; a lost rc
            packet (per drone rc loss rate) repeats the previous applied command.
  events    contact (true centres within contact_radius) downs an attacker; an attacker
            inside the site cylinder is a breach. Downed attackers are parked far away.
            The episode ends when no attacker is left, or at the time limit.

Rewards, the critic's true state and every other knob are in tello_defend_real_env_cfg.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectMARLEnv  # noqa: F401  (type reference for the MRO note below)
from isaaclab.utils.math import sample_uniform

from action_space_kit.assets import TELLO_MASS
from action_space_kit.control import TelloVelocityController

from .base_defend_env import TelloDefendEnv
from .macro import ALERT, ENGAGE, MacroState, chase_base, macro_step
from .tello_defend_real_env_cfg import TelloDefendRealEnvCfg

PARK_OFFSET = 60.0  # [m] downed attackers are moved this far away in x


class TelloDefendRealEnv(TelloDefendEnv):
    cfg: TelloDefendRealEnvCfg

    def __init__(self, cfg: TelloDefendRealEnvCfg, render_mode: str | None = None, **kwargs):
        # the macro planner must agree with the env on the site and the action scale
        cfg.macro.site_radius = cfg.site_radius
        cfg.macro.max_lin_speed = cfg.command.max_lin_speed
        super().__init__(cfg, render_mode, **kwargs)
        E, D, A, H = self.num_envs, self.cfg.num_defenders, self.cfg.num_attackers, self.cfg.history_steps
        assert H > max(cfg.own_delay_steps[1], cfg.feed_delay_steps[1]) + cfg.vel_window_steps, "history too short"
        dev = self.device

        # attackers get their own, faster speed cap
        inertia_zz = float(self._attackers[0].root_physx_view.get_inertias()[0].reshape(3, 3)[2, 2])
        self._attacker_ctl = [
            TelloVelocityController(cfg=cfg.attacker_command, num_envs=E, mass=TELLO_MASS, inertia_zz=inertia_zz, device=dev)
            for _ in range(A)
        ]

        z3 = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        # sensing history (newest first) and estimates
        self._hist_def = z3(H, E, D, 3)
        self._hist_att = z3(H, E, A, 3)
        self._hist_feed = z3(H, E, A, 3)
        self._own_fix = z3(E, D, 3)
        self._own_age = torch.zeros(E, D, dtype=torch.long, device=dev)
        self._feed = z3(E, A, 3)
        self._feed_age = torch.zeros(E, A, dtype=torch.long, device=dev)
        self._est_def_pos = z3(E, D, 3)
        self._est_def_vel = z3(E, D, 3)
        self._est_att_pos = z3(E, A, 3)
        self._est_att_vel = z3(E, A, 3)
        # per-episode link and sensing parameters
        self._own_delay = torch.zeros(E, D, dtype=torch.long, device=dev)
        self._feed_delay = torch.zeros(E, A, dtype=torch.long, device=dev)
        self._own_bias = z3(E, D, 3)
        self._rc_loss = z3(E, D)
        self._video_loss = z3(E, D)
        # attacker script
        self._swerve = z3(E, A, 3)  # amplitude, frequency, phase (side sign folded into amplitude)
        self._pattern = torch.zeros(E, A, dtype=torch.long, device=dev)  # see cfg.pattern_probs
        self._weave2 = z3(E, A, 2)  # second weave sinusoid: frequency ratio, phase
        self._jink = z3(E, A, 2)  # jink period [s], per-attacker seed
        self._alt = z3(E, A, 3)  # altitude variation: amplitude (0 = off), frequency, phase
        self._dodge = torch.zeros(E, A, dtype=torch.bool, device=dev)
        self._aim = z3(E, A, 3)  # aim point offset from the site centre
        self._launch = z3(E, A)  # [s] launch time
        self._feint_t = z3(E, A)  # [s] after launch, when a feint flips sides
        self._burst = torch.stack([torch.full((E, A), 1e9, device=dev), torch.ones(E, A, device=dev)], -1)
        # spawn positions, used for the first observation after a reset (sim data lags a step)
        self._spawn_def = z3(E, D, 3)
        self._spawn_att = z3(E, A, 3)
        self._just_reset = torch.ones(E, dtype=torch.bool, device=dev)
        # control
        self._macro = MacroState(E, D, A, dev)
        self._macro_out: dict[str, torch.Tensor] = {}
        self._base = z3(E, D, 4)
        self._target = z3(E, D, 3)
        self._residual = z3(E, D, 4)
        self._prev_residual = z3(E, D, 4)
        self._applied = z3(E, D, 4)
        self._prev_applied = z3(E, D, 4)
        self._prev_target = z3(E, D, 3)
        self._prev_dist = z3(E, D)
        # events of the current step and episode tallies
        self._step_credit = z3(E, D)
        self._step_captures = z3(E)
        self._step_breaches = z3(E)
        self._episode_breaches = z3(E)
        self.outcome_breaches = z3(E)
        self._last_obs: dict[str, torch.Tensor] | None = None
        self._ep_terms: dict[str, torch.Tensor] = {}
        # own TensorBoard writer: skrl's multi-agent trainer drops extras["log"]
        self._stats = {"episodes": 0.0, "breaches": 0.0, "captures": 0.0, "breached_eps": 0.0}
        self._stat_terms: dict[str, float] = {}
        self._tb = None
        self._tb_step = 0
        if cfg.stats_dir:
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            run = os.path.join(cfg.stats_dir, time.strftime("%Y-%m-%d_%H-%M-%S") + "_episode_stats")
            self._tb = SummaryWriter(run)

    # -- helpers --------------------------------------------------------------------------------

    def _site(self) -> torch.Tensor:
        site = self.scene.env_origins.clone()
        site[:, 2] += self.cfg.site_height * 0.5
        return site

    def _true_states(self):
        dp = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)
        dv = torch.stack([d.data.root_lin_vel_w for d in self._defenders], dim=1)
        ap = torch.stack([d.data.root_pos_w for d in self._attackers], dim=1)
        av = torch.stack([d.data.root_lin_vel_w for d in self._attackers], dim=1)
        return dp, dv, ap, av

    @staticmethod
    def _read_hist(hist: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """hist (H, E, N, 3), steps (E, N) -> (E, N, 3), the entry `steps` back."""
        h = hist.permute(1, 2, 0, 3)  # (E, N, H, 3)
        return torch.gather(h, 2, steps[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)

    # -- sensing and macro (once per policy step, from _get_observations) ------------------------

    def _own_fix_model(self, delayed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Own fix from the delayed true position: (fix (E, D, 3), fresh (E, D)). Random video loss."""
        cfg = self.cfg
        fresh = torch.rand_like(self._rc_loss) >= self._video_loss
        fix = delayed + cfg.own_fix_noise * torch.randn_like(delayed) + self._own_bias
        return fix, fresh

    def _sense_and_plan(self) -> None:
        cfg, dt = self.cfg, self.step_dt
        dp, dv, ap, av = self._true_states()
        jr = self._just_reset
        dp = torch.where(jr[:, None, None], self._spawn_def, dp)
        ap = torch.where(jr[:, None, None], self._spawn_att, ap)
        dv = torch.where(jr[:, None, None], torch.zeros_like(dv), dv)

        # true history, newest first; a fresh episode fills it with its spawn
        self._hist_def = torch.cat([dp[None], self._hist_def[:-1]])
        self._hist_att = torch.cat([ap[None], self._hist_att[:-1]])
        self._hist_def[:, jr] = dp[jr]
        self._hist_att[:, jr] = ap[jr]

        # own fixes: delayed, noisy, biased, sometimes lost (held while lost)
        delayed = self._read_hist(self._hist_def, self._own_delay)
        fix, fresh = self._own_fix_model(delayed)
        fresh |= jr[:, None]
        self._own_fix = torch.where(fresh[..., None], fix, self._own_fix)
        self._own_age = torch.where(fresh, self._own_delay, self._own_age + 1)
        own_vel = dv + cfg.own_vel_noise * torch.randn_like(dv)
        self._est_def_vel = own_vel
        self._est_def_pos = self._own_fix + own_vel * (self._own_age.float() * dt)[..., None]

        # attacker feed: delayed, noisy, sometimes dropped; velocity by finite difference
        delayed = self._read_hist(self._hist_att, self._feed_delay)
        got = (torch.rand(self.num_envs, cfg.num_attackers, device=self.device) >= cfg.feed_loss) | jr[:, None]
        self._feed = torch.where(got[..., None], delayed + cfg.feed_noise * torch.randn_like(delayed), self._feed)
        self._feed_age = torch.where(got, self._feed_delay, self._feed_age + 1)
        self._hist_feed = torch.cat([self._feed[None], self._hist_feed[:-1]])
        self._hist_feed[:, jr] = self._feed[jr]
        w = cfg.vel_window_steps
        vel = (self._hist_feed[0] - self._hist_feed[w]) / (w * dt)
        self._est_att_vel = vel
        self._est_att_pos = self._feed + vel * (self._feed_age.float() * dt)[..., None]
        self._just_reset[:] = False

        # macro layer and base command
        site = self._site()
        out = macro_step(
            self._est_def_pos, self._est_def_vel, self._est_att_pos, self._est_att_vel,
            self._attacker_alive, site, self._macro, cfg.macro,
            def_alive=getattr(self, "_def_alive", None),
        )
        self._macro_out = out
        self._target = out["target"]
        # potential for this step's shaping, measured against the target this action will fly to
        self._prev_dist = torch.linalg.norm(dp - self._target, dim=-1)
        if cfg.base_controller == "intercept":
            self._base = out["base"]
        elif cfg.base_controller == "chase":
            alive = self._attacker_alive
            self._base = chase_base(self._est_def_pos, self._est_att_pos, alive, cfg.macro)
        else:
            self._base = torch.zeros_like(out["base"])

    # -- stepping -------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        cfg = self.cfg
        res = torch.stack([actions[f"defender_{i}"] for i in range(cfg.num_defenders)], dim=1).clamp(-1.0, 1.0)
        res = res.clone()
        res[..., 3] = 0.0  # yaw belongs to the scripted yaw loop
        scale = cfg.residual_scale if cfg.base_controller != "none" else 1.0
        cmd = (self._base + scale * res).clamp(-1.0, 1.0)
        lost = torch.rand_like(self._rc_loss) < self._rc_loss
        cmd = torch.where(lost[..., None], self._applied, cmd)
        self._prev_applied = self._applied
        self._applied = cmd
        self._prev_residual = self._residual
        self._residual = res
        for i in range(cfg.num_defenders):
            self._actions[f"defender_{i}"] = cmd[:, i]

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        """Each attacker flies its episode's pattern at its aim point (see the cfg's attacker block)."""
        cfg = self.cfg
        t = self.episode_length_buf.float() * self.step_dt
        site = self._site()
        out = []
        for i, drone in enumerate(self._attackers):
            t_fly = t - self._launch[:, i]
            launched = t_fly >= 0.0
            t_fly = t_fly.clamp(min=0.0)
            to_aim = site + self._aim[:, i] - drone.data.root_pos_w
            direction = to_aim / torch.linalg.norm(to_aim, dim=-1, keepdim=True).clamp(min=1e-6)
            lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=-1)
            amp, freq, phase = self._swerve[:, i].unbind(-1)
            pat = self._pattern[:, i]
            side = torch.zeros_like(amp)
            side = torch.where(pat == 1, amp * torch.sin(freq * t_fly + phase), side)
            side = torch.where(pat == 2, amp, side)
            flip = torch.where(t_fly < self._feint_t[:, i], 1.0, -1.0)
            side = torch.where(pat == 3, amp * flip, side)
            ratio, phase2 = self._weave2[:, i].unbind(-1)
            weave = amp * (torch.sin(freq * t_fly + phase) + 0.6 * torch.sin(ratio * freq * t_fly + phase2)) / 1.6
            side = torch.where(pat == 4, weave, side)
            period, seed = self._jink[:, i].unbind(-1)
            seg = torch.floor(t_fly / period.clamp(min=0.1))
            jink = amp * torch.sign(torch.sin(12.9898 * seg + 78.233 * seed) + 1e-3)
            side = torch.where(pat == 5, jink, side)
            direction = direction + side[:, None] * lateral
            # altitude variation and a soft band that keeps attackers inside the room
            a_amp, a_freq, a_phase = self._alt[:, i].unbind(-1)
            vz = a_amp * torch.sin(a_freq * t_fly + a_phase)
            z = drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
            lo, hi = cfg.attacker_z_range
            vz = torch.where(z < lo + 0.15, vz.abs() + 0.3, vz)
            vz = torch.where(z > hi - 0.15, -vz.abs() - 0.3, vz)
            direction = direction + vz[:, None] * torch.tensor([0.0, 0.0, 1.0], device=self.device)
            # reactive dodge: sidestep the nearest defender, perpendicular to its line of approach
            dp = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)
            rel = drone.data.root_pos_w[:, None] - dp  # defender -> attacker
            dist = torch.linalg.norm(rel, dim=-1)
            k = dist.argmin(dim=-1)
            r = rel[torch.arange(self.num_envs, device=self.device), k]
            dk = dist.gather(1, k[:, None]).squeeze(1)
            away = torch.stack([-r[:, 1], r[:, 0], torch.zeros_like(r[:, 0])], dim=-1)
            away = away / torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
            away = away * torch.sign((away * lateral).sum(-1, keepdim=True) + 1e-3)  # keep its sidestep side stable
            w = (cfg.dodge_gain * (1.0 - dk / cfg.dodge_radius)).clamp(min=0.0) * self._dodge[:, i].float()
            direction = direction + w[:, None] * away
            direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
            burst_t, burst_m = self._burst[:, i].unbind(-1)
            speed = self._attacker_speed[:, i] * torch.where(t_fly >= burst_t, burst_m, torch.ones_like(burst_m))
            speed = (speed / cfg.attacker_command.max_lin_speed)[:, None]
            action = torch.cat([direction * speed, torch.zeros_like(speed)], dim=-1)
            action = action * (self._attacker_alive[:, i] & launched)[:, None].float()
            out.append(action.clamp(-1.0, 1.0))
        return out

    # -- MDP ------------------------------------------------------------------------------------

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        cfg = self.cfg
        dp, _, ap, _ = self._true_states()
        alive = self._attacker_alive
        site = self._site()

        d = torch.linalg.norm(ap[:, None] - dp[:, :, None], dim=-1)  # (E, D, A)
        contact = (d < cfg.contact_radius) & alive[:, None]
        captured = contact.any(dim=1)  # (E, A)
        capturer = torch.argmax(contact.int(), dim=1)  # first defender in contact, (E, A)
        # what each contact does (default: the attacker goes down, the defender flies on)
        captured, self._def_down_now = self._resolve_contacts(contact, captured, capturer)
        radial = torch.linalg.norm(ap[..., :2] - site[:, None, :2], dim=-1)
        breached = (radial < cfg.site_radius) & alive & ~captured

        mult = torch.where(
            radial > cfg.macro.r_engage,
            cfg.capture_mult_far,
            torch.where(radial > cfg.near_radius, cfg.capture_mult_mid, cfg.capture_mult_near),
        )
        credit = torch.zeros(self.num_envs, cfg.num_defenders, device=self.device)
        credit.scatter_add_(1, capturer, (captured.float() * mult))
        self._step_credit = credit
        self._step_captures = captured.float().sum(-1)
        self._step_breaches = breached.float().sum(-1)
        self._episode_captures += self._step_captures
        self._episode_breaches += self._step_breaches

        down = captured | breached
        self._attacker_alive = alive & ~down
        if down.any():
            self._park_attackers(down)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = ~self._attacker_alive.any(dim=-1)
        finished = done | time_out
        self.outcome_valid = finished
        self.outcome_breached = torch.where(finished, self._episode_breaches > 0, self.outcome_breached)
        self.outcome_captures = torch.where(finished, self._episode_captures, self.outcome_captures)
        self.outcome_breaches = torch.where(finished, self._episode_breaches, self.outcome_breaches)
        self._breached = breached.any(dim=-1)
        terminated = {a: done for a in cfg.possible_agents}
        time_outs = {a: time_out for a in cfg.possible_agents}
        return terminated, time_outs

    def _resolve_contacts(self, contact, captured, capturer):
        """Outcome of this step's contacts: (attacker down (E, A), defender down (E, D))."""
        return captured, torch.zeros(contact.shape[:2], dtype=torch.bool, device=self.device)

    def _park_attackers(self, down: torch.Tensor) -> None:
        """Move downed attackers far away so they stop interacting with the defenders."""
        for i, drone in enumerate(self._attackers):
            ids = down[:, i].nonzero(as_tuple=False).flatten()
            if len(ids) == 0:
                continue
            state = drone.data.default_root_state[ids].clone()
            state[:, :3] = self.scene.env_origins[ids]
            state[:, 0] += PARK_OFFSET + 3.0 * i
            state[:, 2] += 1.0
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], ids)
            drone.write_root_velocity_to_sim(state[:, 7:], ids)
            self._attacker_ctl[i].reset(ids)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        dp, _, _, _ = self._true_states()

        # potential-based progress, Phi = -distance to the target this action flew to (both ends
        # of the difference use the same target, so a target that moves does not pay or charge)
        dist = torch.linalg.norm(dp - self._target, dim=-1)
        progress = (self._prev_dist - cfg.discount * dist).clamp(-0.3, 0.3)

        residual = (self._residual[..., :3] ** 2).sum(-1)
        # smoothness of PPO's own correction only: penalising the applied command would teach
        # PPO to damp the (deliberately aggressive) base and let attackers through
        smooth = ((self._residual - self._prev_residual)[..., :3] ** 2).sum(-1)
        pair = torch.linalg.norm(dp[:, :, None] - dp[:, None, :], dim=-1)
        pair = pair + 1e3 * torch.eye(cfg.num_defenders, device=self.device)
        collide = (pair < cfg.defender_collision_radius).any(dim=-1).float()
        floor = (dp[..., 2] - self.scene.env_origins[:, None, 2] < cfg.floor_min_height).float()
        settle = torch.zeros_like(dist)
        if self._macro_out:
            slot = self._macro_out["slot"]
            A = cfg.num_attackers
            secondary = (slot >= A) & (slot < 2 * A)
            settle = (secondary & (dist < 0.3)).float()

        terms = {
            "capture": cfg.rew_capture * self._step_credit,
            "capture_team": (cfg.rew_capture_team * self._step_captures)[:, None].expand_as(dist),
            "breach": (cfg.rew_breach * self._step_breaches)[:, None].expand_as(dist),
            "progress": cfg.rew_progress * progress,
            "residual": cfg.rew_residual * residual,
            "smooth": cfg.rew_smooth * smooth,
            "collision": cfg.rew_collision * collide,
            "floor": cfg.rew_floor * floor,
            "settle": cfg.rew_settle * settle,
        }
        total = sum(terms.values())
        for k, v in terms.items():
            self._ep_terms[k] = self._ep_terms.get(k, torch.zeros(self.num_envs, device=self.device)) + v.mean(-1)
        return {f"defender_{i}": total[:, i] for i in range(cfg.num_defenders)}

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if not hasattr(self, "_macro"):  # called from the parent constructor, before our buffers
            return {a: torch.zeros(self.num_envs, self.cfg.observation_spaces[a], device=self.device)
                    for a in self.cfg.possible_agents}
        self._sense_and_plan()
        cfg = self.cfg
        out = self._macro_out
        E, D, A = self.num_envs, cfg.num_defenders, cfg.num_attackers
        site = self._site()
        alive = self._attacker_alive.float()
        zone = out["zone"].clamp(min=0)
        zone_1h = torch.nn.functional.one_hot(zone, 3).float() * alive[..., None]
        tau = torch.nan_to_num(out["tau"], posinf=10.0).clamp(0.0, 10.0) / 10.0
        role = out["role"]
        slot_1h = torch.nn.functional.one_hot(out["slot"] + 1, 2 * A + 2).float()  # index 0 = post
        att_ids = torch.arange(A, device=self.device)
        cp, cv = cfg.obs_clip_pos, cfg.obs_clip_vel
        pos, vel = self._est_def_pos, self._est_def_vel

        obs = {}
        for i in range(D):
            p = pos[:, i]
            att_rel = (self._est_att_pos - p[:, None]).clamp(-cp, cp) * alive[..., None]
            att_vel = self._est_att_vel.clamp(-cv, cv) * alive[..., None]
            mine = (role[:, i : i + 1] == att_ids[None]).float()
            per_att = torch.cat(
                [att_rel, att_vel, tau[..., None] * alive[..., None], zone_1h, alive[..., None], mine[..., None]],
                dim=-1,
            ).reshape(E, -1)
            mates = [k for k in range(D) if k != i]
            mate = torch.cat(
                [torch.cat([(pos[:, k] - p).clamp(-cp, cp), vel[:, k].clamp(-cv, cv)], dim=-1) for k in mates], dim=-1
            )
            obs[f"defender_{i}"] = torch.cat(
                [
                    (p - site).clamp(-cp, cp),
                    vel[:, i].clamp(-cv, cv),
                    (self._target[:, i] - p).clamp(-cp, cp),
                    self._base[:, i, :3],
                    slot_1h[:, i],
                    per_att,
                    mate,
                    self._applied[:, i],
                    (self._own_age[:, i].float() * self.step_dt)[:, None],
                ],
                dim=-1,
            )
        self._last_obs = obs
        return obs

    def _get_terminal_observations(self) -> dict[str, torch.Tensor]:
        """Read-only: the observations already built this step (sensing must not advance twice)."""
        return self._last_obs if self._last_obs is not None else self._get_observations()

    def _get_states(self) -> torch.Tensor:
        """Ground truth for the asymmetric critic: all five drones, attacker script, macro state."""
        cfg = self.cfg
        if not hasattr(self, "_macro"):
            return torch.zeros(self.num_envs, cfg.state_space, device=self.device)
        site = self._site()
        dp, dv, ap, av = self._true_states()
        alive = self._attacker_alive.float()
        ap_rel = torch.where(alive[..., None] > 0, ap - site[:, None], torch.zeros_like(ap))
        av = av * alive[..., None]
        drones = torch.cat(
            [torch.cat([dp - site[:, None], dv], dim=-1), torch.cat([ap_rel, av], dim=-1)], dim=1
        ).reshape(self.num_envs, -1)
        amp, freq, phase = self._swerve.unbind(-1)
        t = (self.episode_length_buf.float() * self.step_dt)[:, None]
        burst_t, burst_m = self._burst.unbind(-1)
        pattern_1h = torch.nn.functional.one_hot(self._pattern, 6).float().reshape(self.num_envs, -1)
        script = torch.cat(
            [
                alive,
                self._attacker_speed / cfg.attacker_command.max_lin_speed,
                amp,
                freq / 2.5,
                phase / (2 * math.pi),
                pattern_1h,
                (t >= self._launch).float(),
                ((self._launch - t).clamp(min=0.0) / 4.0),
                (burst_t.clamp(max=10.0) / 10.0),
                burst_m / 2.0,
                self._alt[..., 0],
                self._dodge.float(),
            ],
            dim=-1,
        )
        n_slot = 2 * cfg.num_attackers + 2
        if not self._macro_out:
            assign_1h = torch.zeros(self.num_envs, cfg.num_defenders * n_slot, device=self.device)
            zone = torch.zeros(self.num_envs, cfg.num_attackers, device=self.device)
        else:
            slot = self._macro_out["slot"]
            assign_1h = torch.nn.functional.one_hot(slot + 1, n_slot).float().reshape(self.num_envs, -1)
            zone = self._macro_out["zone"].clamp(min=0).float() / 2.0
        state = torch.cat([drones.clamp(-cfg.obs_clip_pos, cfg.obs_clip_pos), script, assign_1h, zone], dim=-1)
        return state

    def _flush_stats(self) -> None:
        """Every stats_every finished episodes, write their means to TensorBoard."""
        n = self._stats["episodes"]
        if self._tb is None or n < self.cfg.stats_every:
            return
        self._tb_step += 1
        step = int(self.common_step_counter)
        self._tb.add_scalar("Episode/breaches_per_episode", self._stats["breaches"] / n, step)
        self._tb.add_scalar("Episode/breach_rate", self._stats["breached_eps"] / n, step)
        self._tb.add_scalar("Episode/interceptions_per_episode", self._stats["captures"] / n, step)
        for k, v in self._stat_terms.items():
            self._tb.add_scalar(f"Episode_Reward/{k}", v / n, step)
        self._tb.flush()
        self._stats = {k: 0.0 for k in self._stats}
        self._stat_terms = {}

    # -- reset ----------------------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if not hasattr(self, "_macro"):  # called from the parent constructor, before our buffers
            return super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)
        # episode reward terms, for logging (Isaac Lab convention)
        if self._ep_terms and len(env_ids):
            n_done = float(len(env_ids))
            self._stats["episodes"] += n_done
            self._stats["breaches"] += self._episode_breaches[env_ids].sum().item()
            self._stats["captures"] += self._episode_captures[env_ids].sum().item()
            self._stats["breached_eps"] += (self._episode_breaches[env_ids] > 0).float().sum().item()
            for k, v in self._ep_terms.items():
                self._stat_terms[k] = self._stat_terms.get(k, 0.0) + v[env_ids].sum().item()
                v[env_ids] = 0.0
            self._flush_stats()
        # skip TelloDefendEnv._reset_idx; keep the terminal-observation mixin and DirectMARLEnv
        super(TelloDefendEnv, self)._reset_idx(env_ids)
        cfg = self.cfg
        n, dev = len(env_ids), self.device
        D, A = cfg.num_defenders, cfg.num_attackers
        origins = self.scene.env_origins[env_ids]
        site = origins.clone()
        site[:, 2] += cfg.site_height * 0.5

        # defenders: around their formation posts
        ang0 = torch.arange(D, device=dev) * (2 * math.pi / D)
        for i, drone in enumerate(self._defenders):
            state = drone.data.default_root_state[env_ids].clone()
            jitter = sample_uniform(-cfg.defender_start_jitter, cfg.defender_start_jitter, (n, 2), dev)
            state[:, 0] = origins[:, 0] + cfg.macro.post_radius * torch.cos(ang0[i]) + jitter[:, 0]
            state[:, 1] = origins[:, 1] + cfg.macro.post_radius * torch.sin(ang0[i]) + jitter[:, 1]
            state[:, 2] = origins[:, 2] + cfg.macro.post_height
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._defender_ctl[i].reset(env_ids)
            self._spawn_def[env_ids, i] = state[:, :3]

        # attackers: on the spawn ring, own heights, speeds and scripts
        for i, drone in enumerate(self._attackers):
            angle = sample_uniform(0.0, 2 * math.pi, (n,), dev)
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + cfg.spawn_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + cfg.spawn_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(*cfg.attacker_height, (n,), dev)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._attacker_ctl[i].reset(env_ids)
            self._spawn_att[env_ids, i] = state[:, :3]
        self._attacker_alive[env_ids] = True
        self._attacker_speed[env_ids] = sample_uniform(cfg.attacker_speed[0], cfg.attacker_speed[1], (n, A), dev)
        probs = torch.tensor(cfg.pattern_probs, device=dev, dtype=torch.float)
        pattern = torch.multinomial(probs, n * A, replacement=True).reshape(n, A)
        side = torch.where(torch.rand(n, A, device=dev) < 0.5, -1.0, 1.0)
        amp = torch.zeros(n, A, device=dev)
        amp = torch.where(pattern == 1, sample_uniform(*cfg.swerve_amplitude, (n, A), dev), amp)
        amp = torch.where(pattern == 2, side * sample_uniform(*cfg.arc_amplitude, (n, A), dev), amp)
        amp = torch.where(pattern == 3, side * sample_uniform(*cfg.feint_amplitude, (n, A), dev), amp)
        amp = torch.where(pattern == 4, side * sample_uniform(*cfg.weave_amplitude, (n, A), dev), amp)
        amp = torch.where(pattern == 5, sample_uniform(*cfg.jink_amplitude, (n, A), dev), amp)
        self._weave2[env_ids] = torch.stack(
            [sample_uniform(1.7, 3.1, (n, A), dev), sample_uniform(0.0, 2 * math.pi, (n, A), dev)], dim=-1
        )
        self._jink[env_ids] = torch.stack(
            [sample_uniform(*cfg.jink_period_s, (n, A), dev), sample_uniform(0.0, 100.0, (n, A), dev)], dim=-1
        )
        alt_on = (torch.rand(n, A, device=dev) < cfg.alt_prob).float()
        self._alt[env_ids] = torch.stack(
            [
                alt_on * sample_uniform(*cfg.alt_amplitude, (n, A), dev),
                sample_uniform(*cfg.alt_frequency, (n, A), dev),
                sample_uniform(0.0, 2 * math.pi, (n, A), dev),
            ],
            dim=-1,
        )
        self._dodge[env_ids] = torch.rand(n, A, device=dev) < cfg.dodge_prob
        freq = sample_uniform(*cfg.swerve_frequency, (n, A), dev)
        phase = sample_uniform(0.0, 2 * math.pi, (n, A), dev)
        self._pattern[env_ids] = pattern
        self._swerve[env_ids] = torch.stack([amp, freq, phase], dim=-1)
        self._feint_t[env_ids] = sample_uniform(*cfg.feint_switch_s, (n, A), dev)
        r = cfg.aim_offset * torch.sqrt(torch.rand(n, A, device=dev))
        th = sample_uniform(0.0, 2 * math.pi, (n, A), dev)
        self._aim[env_ids] = torch.stack([r * torch.cos(th), r * torch.sin(th), torch.zeros_like(r)], dim=-1)
        self._launch[env_ids] = sample_uniform(*cfg.launch_delay_s, (n, A), dev)
        bursting = torch.rand(n, A, device=dev) < cfg.burst_prob
        burst_t = torch.where(bursting, sample_uniform(*cfg.burst_time_s, (n, A), dev), torch.full((n, A), 1e9, device=dev))
        burst_m = torch.where(bursting, sample_uniform(*cfg.burst_mult, (n, A), dev), torch.ones(n, A, device=dev))
        self._burst[env_ids] = torch.stack([burst_t, burst_m], dim=-1)

        # link and sensing, per drone per episode
        self._rc_loss[env_ids] = sample_uniform(*cfg.rc_loss_range, (n, D), dev)
        self._video_loss[env_ids] = sample_uniform(*cfg.video_loss_range, (n, D), dev)
        lo, hi = cfg.own_delay_steps
        self._own_delay[env_ids] = torch.randint(lo, hi + 1, (n, D), device=dev)
        lo, hi = cfg.feed_delay_steps
        self._feed_delay[env_ids] = torch.randint(lo, hi + 1, (n, A), device=dev)
        self._own_bias[env_ids] = cfg.own_fix_bias * torch.randn(n, D, 3, device=dev)
        self._own_age[env_ids] = 0
        self._feed_age[env_ids] = 0
        self._own_fix[env_ids] = self._spawn_def[env_ids]
        self._feed[env_ids] = self._spawn_att[env_ids]
        self._just_reset[env_ids] = True

        # control and bookkeeping
        self._macro.reset(env_ids)
        self._applied[env_ids] = 0.0
        self._prev_applied[env_ids] = 0.0
        self._residual[env_ids] = 0.0
        self._prev_residual[env_ids] = 0.0
        self._base[env_ids] = 0.0
        self._target[env_ids] = self._spawn_def[env_ids]
        self._prev_target[env_ids] = self._spawn_def[env_ids]
        self._prev_dist[env_ids] = 0.0
        for agent in cfg.possible_agents:
            self._actions[agent][env_ids] = 0.0
        self._breached[env_ids] = False
        self._episode_captures[env_ids] = 0.0
        self._episode_breaches[env_ids] = 0.0
        self._step_credit[env_ids] = 0.0
        self._step_captures[env_ids] = 0.0
        self._step_breaches[env_ids] = 0.0
        self._prev_distance[env_ids] = cfg.spawn_radius
