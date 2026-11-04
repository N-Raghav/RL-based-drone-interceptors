# Defender (deliverable)

Flies 3 Tello defenders against 2 attackers. Everything needed is in this folder, plus a
trained checkpoint and the room's tag map. Nothing in the kit (drone_gui, config, ...) is modified.

| File | What it is |
|---|---|
| `run_defender.py` | the loop, 13.5 Hz: read positions, plan, fly, send sticks through drone_gui |
| `macro.py` | the planner: zones + 3x5 Hungarian job assignment (identical to the simulator's) |
| `policy_runtime.py` | the trained network: checkpoint loader, the 59 inputs, the training safety layer |
| `requirements.txt` | torch (CPU), numpy, pyyaml, scipy |
| `tests/` | `python -m pytest tests -q` |

## Run

drone_gui must be running and the defenders airborne (take-off and landing stay in drone_gui).

```bash
pip install -r requirements.txt
# dry run: plans and logs, sends no sticks
python run_defender.py --defenders TELLO-1,TELLO-2,TELLO-7 --attackers TELLO-3,TELLO-4 \
    --site 2.5,0,1.25 --policy shared_stage3_best.pt --dry-run
# fly: same command without --dry-run. Ctrl+C stops (drones hover), then land from drone_gui.
```

Options: `--policy` (omit for the scripted planner), `--attacker-feed gui|udp` with `--feed-port`
and `--feed-frame map|world` (external attacker positions, format in `read_attacker_feed`),
`--map` (tag map, default `organiser_base/drone_deployment_infra/config/map.yaml`), `--dry-run`, `--rc-cap`, `--rc-step`, `--hz`.
Test with drone_gui's simulated drones: `--shared-frame --drive-attackers`.

Each tick is logged to `outputs/defender_log.jsonl`.
