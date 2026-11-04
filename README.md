# Team 8: Drone Defenders

Three DJI Tello drones guard a spot in a room against two attacking drones.
A planner decides which defender covers which attacker, and a small trained neural
network (our "policy") flies each defender there.

**Result:** in simulation, attackers are kept out in **61.3%** of rounds (38.7% breach rate).
Watch it: [reports/videos/](reports/videos/) ([what the videos show](reports/5_sim_videos.md)).
**On real drones it did not work yet:** see [reports/4_real_drones.md](reports/4_real_drones.md) and `reports/videos/realtesting.mp4`.

## What's where

| Folder | What's inside |
|---|---|
| `defend_task/` | Our simulator task: attackers, planner, rewards |
| `training/` | Train the policy and score it |
| `real_drones/` | Fly the trained policy on real Tellos |
| `policy/` | Trained policies: `shared_stage3_best.pt` (submitted) and `scratch_attr_s1.pt` (newer, see reports/5) |
| `tests/` | Quick checks (no simulator needed) |
| `reports/` | Plain-English write-ups, simulation videos and the full PDF |
| `archive/` | Old notes and the full git history |
| `organiser_base/` | The organisers' starter kit, unchanged |

## Run it

**Quick checks** (any laptop):
```bash
pip3 install torch numpy pyyaml scipy pytest
python3 -m pytest tests real_drones/tests -q
```

**Score the policy in simulation** (NVIDIA GPU, Isaac Lab installed with
`organiser_base/simulation_infra/setup.sh ~/isaac`):
```bash
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd
python training/evaluate_defend.py --task AS-Defend-Role-v0 --headless --num_envs 256 \
  --episodes 512 --seed 1234 --shared --checkpoint policy/shared_stage3_best.pt --env_set stage=3
```
If you installed Isaac Lab before the folders were reorganised, first run this once:
`uv pip install -e organiser_base/simulation_infra/source/action_space_kit`

**Fly real drones** (start the organisers' `drone_gui` first and take off from it):
```bash
pip3 install -r real_drones/requirements.txt
python3 real_drones/run_defender.py --defenders TELLO-1,TELLO-2,TELLO-3 \
  --attackers TELLO-7,TELLO-8 --policy policy/shared_stage3_best.pt --dry-run
```
Remove `--dry-run` to actually send commands. Press Ctrl+C to stop; the drones hover.

How to train it yourself: [reports/3_training.md](reports/3_training.md).
