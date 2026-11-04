#!/usr/bin/env bash
# Speed-ratio sweep on AS-Defend-Real-v0: breach rate per controller per attacker speed.
#
#   training/sweep_defend_real.sh [checkpoint.pt] [episodes]
#
# Defenders cap at 1.0 m/s, so attacker speed (m/s) = attacker/defender speed ratio.
# Every controller sees the same seed. Writes outputs/sweep_defend_real.csv and prints a table.
# Each point boots Isaac Sim (~1 min), so the full sweep takes about 20-30 minutes.
set -u
CKPT="${1:-}"
EPISODES="${2:-512}"
SEED=1234
OUT=outputs/sweep_defend_real.csv
mkdir -p outputs/sweep_logs
echo "controller,speed,breach_rate_pct,breaches_per_episode,interception_pct" > "$OUT"

controllers=(zero chase scripted)
[ -n "$CKPT" ] && controllers+=(ppo)

for speed in 0.6 0.9 1.2 1.5 1.8; do
  for c in "${controllers[@]}"; do
    log="outputs/sweep_logs/${c}_${speed}.log"
    if [ "$c" = ppo ]; then
      src=(--checkpoint "$CKPT" --algorithm MAPPO)
    else
      src=(--baseline "$c")
    fi
    python -u "$(dirname "$0")/evaluate_defend.py" --task AS-Defend-Real-v0 --headless --num_envs 256 --episodes "$EPISODES" \
      --seed "$SEED" "${src[@]}" --env_set "attacker_speed=[$speed,$speed]" > "$log" 2>&1
    br=$(grep -oP "breach rate:\s+\K[0-9.]+" "$log" | tail -1)
    bpe=$(grep -oP "breaches/episode:\s+\K[0-9.]+" "$log" | tail -1)
    ir=$(grep -oP "interception rate:\s+\K[0-9.]+" "$log" | tail -1)
    echo "$c,$speed,${br:-NA},${bpe:-NA},${ir:-NA}" | tee -a "$OUT"
  done
done

echo
column -s, -t "$OUT"
