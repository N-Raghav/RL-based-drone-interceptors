# 2. How the defenders decide what to do

Everything below runs 13.5 times a second, the rate a Tello accepts commands.
The code is in `defend_task/macro.py` (planner) and `defend_task/tello_defend_role_env.py`
(the task the network learns on).

## Step 1: Zones (how worried should we be?)
Each attacker gets one of three labels, based on how far it is from the site and how soon it
would arrive at its current speed:

| Zone | Meaning |
|---|---|
| Relaxed | Far away or not heading in. Defenders stay put. |
| Alert | Getting closer. A defender lines up between it and the site. |
| Engage | About to arrive. Go and intercept it. |

Whichever measure (distance or time) is more worrying wins. An attacker sitting right on the
line between two zones doesn't keep flipping back and forth, because leaving "engage" takes
a bit more margin than entering it.

## Step 2: Jobs (who covers what?)
There are 5 jobs for 3 drones:

| Job | What the drone does |
|---|---|
| Primary on A / primary on B | Fly to where the attacker *will be* and hit it. Within 1 m, go straight at it. |
| Secondary on A / secondary on B | Wait at a backup point between that attacker and the site, in case the primary misses. |
| Goalie | Guard the site, on the side of the biggest threat. |
| Post (no job) | Hold a formation spot. |

The planner tries every possible way of handing out the jobs (only 60 options for 3 drones)
and picks the cheapest one. "Cost" here is roughly *time*: who can get there first. The rules:
- Both primary jobs are always filled while attackers are a threat.
- The third drone backs up whichever primary is most likely to miss, or plays goalie.
- A drone that's already closing in keeps its job, and switching jobs costs a little extra,
  so drones don't keep changing their minds.

## Step 3: Flying (the learned part)
One neural network (59 inputs, 4 outputs) is shared by all three defenders and runs once
per drone per tick.

**What it sees** (all from the drone's own, slightly delayed view, never perfect truth):
its position and speed, where its job wants it to go, the planner's suggested speed, which
job it has, where both attackers are and how they're moving, where its teammates are, the
command it sent last time, and how old its own position fix is.

**What it outputs:** a speed command left/right, forward/back, up/down (1.0 = 1 m/s), plus a
turn rate we don't use on real drones (they turn to face where they're flying so the camera
keeps seeing tags).

The planner already suggests a speed towards each target (one of the network's inputs).
The network decides the actual command, and was trained in simulation on late, noisy
positions like the real ones.

## Step 4: Safety layer (not learned)
Before a command is sent:
- Defenders closer than 0.5 m are pushed apart.
- A drone below 0.45 m can't fly any lower.

Then the speed command is turned into the Tello's four stick values and sent.

## Same code in simulation and on real drones
`real_drones/macro.py` is a byte-for-byte copy of `defend_task/macro.py`, and a test checks
that they stay identical. The real drones make exactly the same decisions the network was
trained with.
