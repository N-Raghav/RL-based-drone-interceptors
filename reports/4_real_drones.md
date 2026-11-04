# 4. Flying it on real drones

The code is in `real_drones/`. It runs on an ordinary laptop CPU: no GPU and no simulator.

## How it works
`run_defender.py` works alongside the organisers' **drone_gui**, which handles the Wi-Fi
links, take-off and landing. 13.5 times a second it:

1. Reads every drone's position from drone_gui (each drone works out its position from the
   AprilTags its camera sees).
2. Converts those into one shared room map using `organiser_base/drone_deployment_infra/config/map.yaml`.
3. Runs the same planner as in training (zones and jobs, `macro.py`).
4. Runs the trained network for each defender (`policy_runtime.py`).
5. Applies the safety layer (keep apart, stay off the floor).
6. Turns each command into the Tello's four stick values and sends them through drone_gui.

It **never takes off or lands**: you do that in drone_gui. It only sends commands to
defenders that are in the air. If the script stops or crashes, drone_gui zeroes the sticks
within 0.35 s and the drones hover. Every tick is logged to `outputs/defender_log.jsonl`.

## Run it
```bash
pip3 install -r real_drones/requirements.txt

# practice run: plans and logs, sends nothing
python3 real_drones/run_defender.py --defenders TELLO-1,TELLO-2,TELLO-3 \
  --attackers TELLO-7,TELLO-8 --policy policy/shared_stage3_best.pt --dry-run

# real run: same command without --dry-run. Ctrl+C stops; then land from drone_gui.
```

Useful options:

| Option | What it does |
|---|---|
| `--site X,Y,Z` | Where the protected spot is, in metres (default 2.5,0,1.25) |
| `--dry-run` | Work everything out but send no commands |
| `--rc-cap`, `--rc-step` | Limit how hard and how suddenly the sticks move (defaults 40 and 15) |
| `--attacker-feed udp` | Take attacker positions from an outside feed instead of drone_gui |
| `--map` | Use a different room map file |
| Leave out `--policy` | Fly with the hand-written planner only, no network |

To test with drone_gui's simulated drones instead of real ones, add `--shared-frame --drive-attackers`.

## What we checked before flying
- **Same network, same answers:** our lightweight loader gives exactly the same outputs as
  the training software (largest difference 0.0 over 4,096 random inputs).
- **Full loop without drones:** `real_drones/tests/fake_gui.py` pretends to be drone_gui with
  5 fake drones. The script sent commands about 13 times a second to each defender and none
  to the attackers.
- **Directions are right:** fake defenders flown by the script reached their attackers
  (0.0 m and 0.06 m away), so left/right and forward/back aren't mixed up.
- **Unit tests:** `python3 -m pytest tests real_drones/tests -q`

## Real-world test
Video: `reports/videos/realtesting.mp4` (about 50 s, filmed in the room with the tag boards).

**Result: it did not work.**

What the video shows: drones start on the floor among the floor tags. Partway through, drones
take off and fly towards the far wall with the tag boards. At the end, two drones are
hovering near the boards and two are on the floor.

No log or written notes from this test are in the repo, so the cause isn't recorded here.
