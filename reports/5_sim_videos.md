# 5. Simulation videos

Two short top-down replays (about 12 s each) in `reports/videos/`. Each one plays the **same
round twice, side by side**, so you can compare two controllers on identical attacks.

Both use the setting the videos are titled with: **aggressive attackers flying as fast as our
drones**, with **attrition on** (a hit can also knock out our defender). Over 512 rounds in
this setting, the submitted policy's breach rate was 17.3% (from the PDF).

**How to read them**
- Grey circle in the middle: the site we're protecting.
- Circles: our defenders (D0, D1, D2), labelled with their current job (primary, goalie, post...).
- Triangles: attackers (A0, A1). Lines behind each drone show where it has been.
- Red/orange X: attacker taken down. Star: attacker reached the site (breach).
- "D1 out": that defender was knocked out by its own hit.

## Video 1: new policy vs our submitted policy
`sim1_new_ppo_vs_submitted_policy.mp4`

| Left: new policy (`scratch_attr_s1`) | Right: submitted policy (`shared_stage3_best`) |
|---|---|
| D1 takes down A0 (and is knocked out doing it). D2 takes down A1 well before the site. D0 stays back as goalie. **Held** at 8.8 s: both stopped. | D0 takes down A0 (and is knocked out). A1 gets past D2 and its backup D1 and reaches the site at 10.1 s. **Breached.** |

## Video 2: new policy vs the planner alone (no network)
`sim2_new_ppo_vs_planner_only.mp4`

| Left: new policy | Right: hand-written planner only ("classical macro") |
|---|---|
| Same as the left side of video 1. **Held** at 8.8 s. | D0 takes down A0 (and is knocked out), D1 takes down A1, and D2 guards the site. **Held** at 7.6 s. |

Both hold this round: the new policy at 8.8 s, the planner alone at 7.6 s.

## Keep in mind
- Each video is **one round**. One round can't show which controller is better overall. For
  that, use the breach rates over 512 rounds (see [1_overview.md](1_overview.md)).
- The new policy is saved as `policy/scratch_attr_s1.pt`. It has the same network shape as
  the submitted one, so every command that takes `--policy` or `--checkpoint` works with it.
  Its 512-round breach rates haven't been added to the results yet.
