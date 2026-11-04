# 3. How we trained it

## Where it practised
We trained in NVIDIA Isaac Lab (a physics simulator), on top of the organisers' Tello drone
model. Our task is called **AS-Defend-Role-v0** (code: `defend_task/`). We trained with 2,048
copies of the room running at once.

We made the simulation as close to the real setup as we could:
- **Late, noisy positions.** Each drone learns where it is 0.15 to 0.22 s late, with about
  3 cm of error, and sometimes not at all (3 to 19% of the time).
- **Lost commands.** Up to 15% of stick commands are dropped, so the drone keeps doing the
  last thing it was told.
- **Attacker speed is guessed**, not given, from how their position changes.
- **Realistic Tello response:** about 0.45 s to react, limited acceleration.
- **Tricky attackers.** Each one flies straight, swerves, arcs, fakes one way, weaves or
  jinks, with height changes, sudden bursts of speed and dodging. They can be faster than
  our drones.
- **Real contact.** An attacker only goes down if a defender actually touches it (centres
  within 0.2 m).

## How it learned
We used **PPO**, a standard trial-and-error learning method: the network tries things, gets
points for good outcomes, and slowly shifts towards what scores well. A helper
(`defend_task/shared_wrapper.py`) treats all three drones as practice for the same
network. During training only, a "coach" part of the system sees the true positions of
everything. That helps it judge how well things are going, but it isn't used when flying.

## Three stages, easy to hard
Each stage starts from the best network of the previous one.

| Stage | What changes | How close counts as a hit |
|---|---|---|
| 1. Learn to intercept | Every drone chases an attacker. Simpler attackers, no dodging. | 0.35 m |
| 2. Learn every job | Jobs are handed out at random, so the network practises all of them. | 0.27 m |
| 3. Real conditions | The planner hands out jobs, exactly as on the real drones. | 0.2 m |

## Points (rewards), per drone
| Event | Points |
|---|---|
| An attacker reaches the site | -10 to every defender |
| Round ends with no breach | +10 to every defender |
| Any attacker is taken down | +5 to every defender |
| This drone made the hit | +10 |
| Getting closer to its job's target | +1 per metre (primary), +0.5 (backup), +0.3 (goalie, post) |
| Backup's attacker goes down | +5 |
| Goalie in position | +0.02 each tick |
| Too close to a teammate | -1 each tick |
| No tag in camera view (room mode) | -0.02 each tick |
| Flying too low (near the floor) | -2 each tick |


## Results
See the table in [1_overview.md](1_overview.md). The submitted policy is
`policy/shared_stage3_best.pt`. We also trained a version from scratch with "a hit can
knock out our defender too" switched on the whole time. It scored worse on every test, so we
kept the stage 3 version.

## Do it yourself
Set up Isaac Lab once with `organiser_base/simulation_infra/setup.sh ~/isaac` (NVIDIA GPU
needed). Then, from the repo root:

```bash
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd

# stage 1
python training/train_shared.py --task AS-Defend-Role-v0 --agent skrl_ppo_shared_cfg_entry_point \
  --headless --num_envs 2048 agent.trainer.timesteps=6000 env.stage=1 env.contact_radius=0.35
# stage 2: add --checkpoint <best from stage 1>, and use env.stage=2 env.contact_radius=0.27 agent.trainer.timesteps=7000
# stage 3: add --checkpoint <best from stage 2>, and use env.stage=3 env.contact_radius=0.2 agent.trainer.timesteps=9000

# score a checkpoint (512 rounds)
python training/evaluate_defend.py --task AS-Defend-Role-v0 --headless --num_envs 256 \
  --episodes 512 --seed 1234 --shared --checkpoint policy/shared_stage3_best.pt --env_set stage=3
```

Training output goes to `logs/` (not saved in git). We trained on one H100 GPU.

**If Isaac Lab was installed before the folders were reorganised**, run this once so the
simulator finds the kit at its new location:
`uv pip install -e organiser_base/simulation_infra/source/action_space_kit`

Our task code lives in `defend_task/`. The simulator only finds tasks inside the
organisers' kit, so `organiser_base/.../tasks/direct/tello_defend_real` is a link pointing to
`defend_task/`. Edit the files in `defend_task/`.
