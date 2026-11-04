"""Configuration for AS-Defend-Real-v0: 3 defenders vs 2 attackers over the measured Tello link.

Differences from AS-Defend-v0 (the idealised task):

  rate        13.5 Hz policy (one rc packet per 74 ms), physics 135 Hz.
  link        per drone, per episode rc loss 0-15% (a lost packet repeats the last command);
              own-pose fixes lost 3-19% of steps (video), held while lost.
  sensing     own pose: 2-3 steps late (0.15-0.22 s), 3 cm noise per fix, 3 cm bias per episode.
              attackers: the organisers' feed, 1-3 steps late, 3 cm noise, small dropout.
              attacker velocity is NOT given: it is a finite difference of the delayed feed.
  attackers   scripted, randomised per attacker per episode: straight, swerve, arc or feint, a
              random aim point inside the site, a launch delay (staggered raids) and an optional
              mid-flight speed burst; their OWN speed cap lets them outrun the defenders.
  contact     an interception needs real contact: centres within contact_radius (0.2 m).
  control     a scripted macro layer (zones, 3x5 augmented slot assignment, target points) and a
              proportional base controller fly every defender; PPO outputs a clipped residual on
              the translation channels. Yaw is held at 0 (the scripted yaw loop keeps tags in view).
  critic      asymmetric: the shared state (MAPPO critic) is the TRUE state of all five drones.

Zero residual actions reproduce the scripted controller exactly, so `--baseline zero` in
scripts/evaluate.py scores it; `env.base_controller=chase` / `none` give the other baselines.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.control import TelloCommandCfg

from .base_defend_env_cfg import TelloDefendEnvCfg
from .macro import MacroCfg, num_slots

NUM_DEFENDERS = 3
NUM_ATTACKERS = 2

# per defender, all from estimates:
#   own pos rel site 3, own vel 3, target rel own pos 3, base action xyz 3,
#   slot one-hot (post + 2A+1 slots = 6 for 3v2: post, primary A/B, secondary A/B, goalie)
#   per attacker: rel pos 3, est vel 3, tau 1, zone one-hot 3, alive 1, my-attacker 1 = 12
#   per teammate: rel pos 3, vel 3 = 6
#   last applied action 4, own fix age 1
SLOT_DIM = num_slots(NUM_ATTACKERS) + 1
OBS_DIM = 3 + 3 + 3 + 3 + SLOT_DIM + 12 * NUM_ATTACKERS + 6 * (NUM_DEFENDERS - 1) + 4 + 1
# critic state, ground truth: 5 drones x (pos rel site 3, vel 3); per attacker: alive, speed,
# amplitude, frequency, phase, pattern one-hot (4), launched, time to launch, burst time, burst
# mult (13); every drone's slot one-hot (D x SLOT_DIM); zone per attacker
NUM_PATTERNS = 6
# alive, speed, amp, freq, phase, launched, time to launch, burst time, burst mult, pattern one-hot,
# altitude amplitude, dodges
ATTACKER_SCRIPT_DIM = 9 + NUM_PATTERNS + 2
STATE_DIM = 6 * (NUM_DEFENDERS + NUM_ATTACKERS) + ATTACKER_SCRIPT_DIM * NUM_ATTACKERS + NUM_DEFENDERS * SLOT_DIM + NUM_ATTACKERS



@configclass
class DefendMacroCfg(MacroCfg):
    """MacroCfg as an Isaac Lab configclass, so Hydra overrides like env.macro.r_engage=3.0 work."""


POLICY_HZ = 13.5
PHYSICS_DECIMATION = 10


@configclass
class TelloDefendRealEnvCfg(TelloDefendEnvCfg):
    # -- rate: 13.5 Hz policy, 135 Hz physics
    decimation = PHYSICS_DECIMATION
    episode_length_s = 20.0
    sim: SimulationCfg = SimulationCfg(dt=1 / (POLICY_HZ * PHYSICS_DECIMATION), render_interval=PHYSICS_DECIMATION)

    # -- agents
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    action_spaces = {f"defender_{i}": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for i in range(NUM_DEFENDERS)}
    observation_spaces = {f"defender_{i}": OBS_DIM for i in range(NUM_DEFENDERS)}
    state_space = STATE_DIM

    # -- geometry
    site_radius = 1.5
    site_height = 2.5
    spawn_radius = 7.0
    attacker_height = (1.0, 2.0)  # [m] indoor ceiling
    defender_start_jitter = 0.3  # [m] around each formation post
    contact_radius = 0.2  # [m] centre to centre: real contact between two Tellos
    defender_collision_radius = 0.3  # [m] defenders this close are penalised
    floor_min_height = 0.3

    # -- attackers: own speed cap, randomised patterns (fixed within an episode)
    # Defenders cap at command.max_lin_speed = 1.0 m/s; attacker_speed goes past it, and a burst
    # can push an attacker further (still capped by attacker_command.max_lin_speed).
    attacker_speed = [0.5, 1.6]  # [m/s] base speed; curriculum: [0.4, 0.9] -> [0.5, 1.6] -> [0.8, 2.2]
    attacker_command: TelloCommandCfg = TelloCommandCfg(max_lin_speed=2.5)
    # pattern family per attacker, per episode (sideways motion relative to the dive direction):
    #   0 straight  dive at the aim point
    #   1 swerve    one sinusoid sideways
    #   2 arc       constant sideways bias, the path spirals in
    #   3 feint     sideways one way, flips once at t_switch
    #   4 weave     two sinusoids mixed (irregular S-curves, hard to fit with one frequency)
    #   5 jink      sharp random side changes every jink_period (piecewise constant, no warning)
    pattern_probs = (0.15, 0.2, 0.15, 0.15, 0.2, 0.15)
    weave_amplitude = (0.4, 1.0)
    jink_amplitude = (0.6, 1.2)
    jink_period_s = (0.4, 1.2)
    # modifiers, independent of the pattern
    alt_prob = 0.5  # altitude variation: vertical sinusoid on top of the pattern
    alt_amplitude = (0.2, 0.6)  # vertical weight relative to the dive direction
    alt_frequency = (0.6, 1.8)  # [rad/s]
    attacker_z_range = (0.6, 2.2)  # [m] attackers are pushed back inside this band
    dodge_prob = 0.4  # reactive dodging: sidestep a defender that comes within dodge_radius
    dodge_radius = 1.2  # [m]
    dodge_gain = 1.2  # sidestep weight at contact distance (fades to 0 at dodge_radius)
    swerve_amplitude = (0.3, 1.0)  # sideways weight relative to the dive direction
    swerve_frequency = (0.8, 2.5)  # [rad/s]
    arc_amplitude = (0.2, 0.7)
    feint_amplitude = (0.6, 1.2)
    feint_switch_s = (1.0, 4.0)  # [s] after launch
    aim_offset = 1.0  # [m] each attacker aims at a random point within this of the site centre
    launch_delay_s = (0.0, 4.0)  # [s] attackers hover at spawn until launch (staggered raids)
    burst_prob = 0.3  # chance an attacker speeds up mid-flight
    burst_time_s = (1.5, 5.0)  # [s] after launch
    burst_mult = (1.3, 1.8)

    # -- link and sensing (ranges sampled per drone, per episode)
    rc_loss_range = (0.0, 0.15)
    video_loss_range = (0.03, 0.19)
    own_delay_steps = (2, 3)
    own_fix_noise = 0.03
    own_fix_bias = 0.03
    own_vel_noise = 0.05  # Tello vgx/vgy/vgz, state stream (no video delay)
    feed_delay_steps = (1, 3)
    feed_noise = 0.03
    feed_loss = 0.05
    vel_window_steps = 4  # finite-difference window for attacker velocity (0.3 s)
    history_steps = 8  # must exceed the largest delay + vel window

    # -- control
    base_controller = "intercept"  # "intercept" (scripted macro), "chase" (repo baseline), "none"
    residual_scale = 0.7  # PPO correction, in action units, added to the base command
    macro: DefendMacroCfg = DefendMacroCfg()

    # -- logging: episode stats written straight to TensorBoard ("" turns it off)
    stats_dir = "logs/skrl/tello_defend_real"
    stats_every = 2048  # finished episodes per data point

    # -- rewards (per defender)
    rew_capture = 10.0  # to the defender that made contact, times the zone multiplier
    capture_mult_far = 1.5  # beyond r_engage
    capture_mult_mid = 1.0
    capture_mult_near = 0.7  # within 2.2 m of the site
    near_radius = 2.2
    rew_capture_team = 2.0  # to every defender, per interception
    rew_breach = -10.0  # to every defender, per attacker reaching the site
    rew_progress = 0.3  # potential-based, toward the macro target (metres)
    discount = 0.99
    rew_residual = -0.005  # times |residual|^2
    rew_smooth = -0.02  # times |r_t - r_{t-1}|^2 of the PPO residual (not the base)
    rew_collision = -2.0  # per step within defender_collision_radius of a teammate
    rew_floor = -2.0  # per step below floor_min_height
    rew_settle = 0.02  # a secondary within 0.3 m of its point, per step (arrive early, settled)
