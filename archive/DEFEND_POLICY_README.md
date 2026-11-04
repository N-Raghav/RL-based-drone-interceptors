# Defend policy add-on: render / evaluate the shared checkpoints

Unzip at the root of the Action_Space repo. Every file is NEW: nothing in the kit is
overwritten, and the task package carries its own frozen copy of the tello_defend base env,
so it works even where tello_defend has changed.

## What is in it
- `simulation_infra/.../tasks/direct/tello_defend_real/`: the tasks (auto-registered by
  action_space_kit.tasks):
  - **AS-Defend-Role-v0**: what `shared_*_best.pt` were trained on (3 defenders vs 2 attackers,
    macro planner + shared PPO policy, 59 inputs per defender, 143-dim critic state)
  - AS-Defend-Real-v0: the older residual task (not for these checkpoints)
  - `agents/skrl_ppo_shared_cfg.yaml`: registered as `skrl_ppo_shared_cfg_entry_point`
  - `shared_wrapper.py`: one policy for all defenders (parameter sharing)
- `simulation_infra/scripts/evaluate_defend.py`: the evaluator (`--shared`, `--env_set`, `--seed`,
  `--baseline scripted`, `--traj_out`, `--video`)
- `simulation_infra/scripts/skrl/train_shared.py`: the trainer that produced the checkpoints
- `drone_deployment_infra/defender/`: the hardware runner (not needed for rendering)

## Render (needs an RTX GPU), from simulation_infra/
Use **AS-Defend-Role-v0** with the shared checkpoints:

```bash
# best policy (stage 3: open arena, full-speed attackers). defender_attrition=False = as trained
python -u scripts/evaluate_defend.py --task AS-Defend-Role-v0 --shared \
  --checkpoint ../shared_stage3_best.pt --num_envs 4 --episodes 8 --seed 0 \
  --headless --enable_cameras --video --video_length 400 --video_dir videos/defend_stage3 \
  --env_set stage=3 defender_attrition=False 'stats_dir=""'

# room-tuned policy (real tag map, walls, Tello lag), as trained
python -u scripts/evaluate_defend.py --task AS-Defend-Role-v0 --shared \
  --checkpoint ../shared_room_best.pt --num_envs 4 --episodes 8 --seed 0 \
  --headless --enable_cameras --video --video_length 400 --video_dir videos/defend_room \
  --env_set stage=3 room=True "rc_latency_steps=[0,1]" command.response_time=0.45 \
  command.max_lin_accel=3.0 defender_attrition=False "heading_mode='look_at_tag'" 'stats_dir=""'

# scripted planner baseline on the same seed (no checkpoint)
python -u scripts/evaluate_defend.py --task AS-Defend-Real-v0 --baseline scripted \
  --num_envs 4 --episodes 8 --seed 0 --headless --enable_cameras --video --video_dir videos/defend_scripted
```
Statistics only (no video): drop the four video flags and use `--num_envs 256 --episodes 512`.

## Harder / "aggressive" attackers
This task has no `aggressive` switch; its attacker knobs (all `--env_set`):
`attacker_speed=[lo,hi]` (m/s, defenders cap at 1.0), `pattern_probs=[straight,swerve,arc,feint,weave,jink]`,
`dodge_prob=0.4`, `dodge_radius=1.2`, `dodge_gain=1.2`, `burst_prob=0.3`, `"burst_mult=[1.3,1.8]"`,
`alt_prob=0.5`, `"launch_delay_s=[0.0,4.0]"`. The stage-3 defaults already include all six patterns,
40 % dodging and bursts. Example: `"attacker_speed=[0.4,1.0]" dodge_prob=1.0 dodge_gain=1.0`.
