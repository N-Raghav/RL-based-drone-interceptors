"""Configuration for AS-Defend-Role-v0 (see tello_defend_role_env.py for the design).

Everything about the world, link and sensing is inherited from AS-Defend-Real-v0, so
observations and the critic state have the same shape and one network carries through
all three stages:  env.stage=1  ->  env.stage=2  ->  env.stage=3  (resume with --checkpoint).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .tello_defend_real_env_cfg import TelloDefendRealEnvCfg


@configclass
class TelloDefendRoleEnvCfg(TelloDefendRealEnvCfg):
    stage = 3  # 1 interception, 2 forced random roles, 3 macro assigns roles
    shuffle_roles = True  # shuffle which drone gets which forced role (stages 1-2)

    # room realism (see tello_defend_role_env.py): real tag map, tag-limited fixes, walls
    room = False
    room_site_x = 2.5  # [m] site centre in front of the tag-10 wall (adapter --site X)
    room_depth = 8.0  # [m] room extent in front of the tag wall
    room_width = 8.0  # [m] room extent along the wall, centred on tag 10
    room_margin = 0.4  # [m] soft walls start this far inside the room
    attacker_spawn_x = (5.6, 7.6)  # [m] attackers launch from this band (distance from the tag wall)
    tag10_height = 1.22  # [m] tag 10 centre above the floor
    rc_latency_steps = (0, 0)  # per defender per episode, in policy steps (room runs: (0, 1))
    defender_speed_range = (1.0, 1.0)  # per defender per episode: real m/s at action 1 (room runs: (0.4, 1.0))

    # the PPO action is the command; nothing scripted is added under it
    base_controller = "intercept"  # only used to build the observed hint
    stats_dir = "logs/skrl/tello_defend_role"

    # safety layer (applied to the command, not a reward)
    safety_repel_radius = 0.5  # [m]
    safety_repel_gain = 2.0
    safety_floor = 0.45  # [m] below this the command cannot descend

    # rewards: team outcome + role
    rew_breach = -10.0  # every drone, per breach
    rew_team_capture = 5.0  # every drone, per interception
    rew_contact = 10.0  # the drone that made contact
    rew_prim_progress = 1.0  # potential on the primary's target distance (m), see prim_target
    prim_target = "intercept"  # "intercept": lead the attacker (macro intercept point), "attacker": chase it
    homing_radius = 1.0  # [m] within this of its attacker a primary's target becomes the attacker itself
    rew_sec_progress = 0.5  # potential on its layered point
    rew_hold_progress = 0.3  # goalie and post: potential on their point
    rew_sec_downed = 5.0  # secondary, when its attacker goes down
    rew_goalie_hold = 0.02  # per step within 0.3 m of the goalie point
    rew_floor = -2.0
    defender_attrition = True  # contacts can take defenders out (see contact_outcome_probs)
    contact_outcome_probs = (1 / 3, 1 / 3, 1 / 3)  # per contact: both down, attacker survives, defender survives
    defender_single_use = True  # a defender that made contact is out even when it survives (no second engagement)
    # v2
    rew_success = 10.0  # every defender, when an episode ends with no breach
    rew_self_collision = -1.0  # per step, defender within self_collision_radius of another defender
    self_collision_radius = 0.3  # [m] centre to centre
    rew_no_tag = -0.02  # per step without any mapped tag in the camera image (room mode)
    heading_mode = "forward"  # face the flight direction ("look_at_tag" = face home, tag 10, kept for comparison)
    discount = 0.995
