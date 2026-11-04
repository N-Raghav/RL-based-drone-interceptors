# 1. Overview: what we built

## The challenge
Two attacking drones fly at a protected spot (the "site") in an 8 m x 8 m room. Three of our
DJI Tello drones have to stop them by touching them before they get there. If an attacker
reaches the site, that's a **breach**. A round is a win if no attacker gets through.

The hard parts:
- The attackers can be **faster** than our drones, and they dodge, swerve, fake one way and
  speed up mid-flight.
- Our drones only know where they are from **AprilTags** (printed markers on the walls) seen
  by their cameras. That information arrives late, is a bit noisy and sometimes drops out.
- Commands to a Tello go over Wi-Fi about 13 times a second, and some get lost.

## Our approach in one picture

```
 positions of all drones
        |
        v
 1. ZONES       how dangerous is each attacker? (far / getting close / about to arrive)
        |
        v
 2. JOBS        give each defender one job: chase attacker A or B, back up a teammate,
                guard the site ("goalie"), or wait at a post
        |
        v
 3. FLYING      one small neural network turns "my job + what I see" into a speed command
        |
        v
 4. SAFETY      keep defenders apart and off the floor, then send stick commands to the Tello
```

Steps 1, 2 and 4 are hand-written rules. Step 3 is learned. The same three defenders all
use **one shared network**, so whichever drone gets a job flies it the same way.

Details: [2_how_it_works.md](2_how_it_works.md).

## Results for `shared_stage3_best.pt` (simulation, 512 rounds each, seed 1234)

Numbers as reported in `ActionSpaceTeam8_Documentation.pdf`. "Attrition on" means a hit can
also knock out our defender.

| Test | Rounds with a breach (lower is better) |
|---|---|
| Stage 3 conditions (all attacker patterns, 0.2 m contact) | **38.7%** |
| Training mix, attrition on | 42.4% |
| Aggressive attackers at equal speed, attrition on | 17.3% |
| Faster attackers, attrition on | 66.0% |
| Room (walls, tag-limited sensing), attrition on | 67.8% |

A second policy, `policy/scratch_attr_s1.pt` (labelled "new PPO" in the videos), is also in
the repo. Side-by-side replays against the submitted policy are in [5_sim_videos.md](5_sim_videos.md). No 512-round scores were recorded for it,
so it isn't in this table.

## Real drones
In the real-world test (`reports/videos/realtesting.mp4`) it did not work. Details in
[4_real_drones.md](4_real_drones.md).

## What's left to improve (from the PDF)
- **Faster attackers** are the biggest gap. We can't catch what we can't outrun, so the fix is
  to meet them further out.
- **The realistic room** is the weakest case. Next step: more training in room mode.
- **Scoring rules matter a lot.** Changing what happens on contact changes the numbers a lot,
  so compare results only under the same rules.
- To run inside the organisers' `swarm_ops` setup, we'd need a small adapter that reads
  attacker positions from their message bus.

## Where to read more
- [2_how_it_works.md](2_how_it_works.md): the planner and the network
- [3_training.md](3_training.md): how we trained it
- [4_real_drones.md](4_real_drones.md): flying it on real Tellos
- [5_sim_videos.md](5_sim_videos.md): side-by-side simulation replays
- `ActionSpaceTeam8_Documentation.pdf`: the full technical write-up
