# ActionSpaceTeam8

Team 8's full Action Space code base (branch `defend-real`), built to run the
`shared_stage3_best.pt` policy (shared PPO, stage 3, 38.7% breach in sim).

## Where things are

- Deployment: `drone_deployment_infra/defender/run_defender.py` (planner + network + safety, sends rc via drone_gui)
- Policy loader: `drone_deployment_infra/defender/policy_runtime.py` (matches skrl exactly, verified on the checkpoint)
- Training env: `simulation_infra/source/action_space_kit/action_space_kit/tasks/direct/tello_defend_real/` (AS-Defend-Role-v0)
- Training / eval: `simulation_infra/scripts/skrl/train_shared.py`, `simulation_infra/scripts/evaluate_defend.py`
- Full git history: `ActionSpaceTeam8_history.bundle` (`git clone ActionSpaceTeam8_history.bundle repo`)

## Checkpoint

The checkpoint ships at `policy/shared_stage3_best.pt`:

    cd drone_deployment_infra/defender
    python run_defender.py --defenders D1,D2,D3 --attackers A1,A2 --policy ../../policy/shared_stage3_best.pt

The checkpoint is compatible with this code: the observation (59), the network (59-256-128-64-4) and the
planner are unchanged since it was trained. The newer rule switches (`defender_attrition`,
`defender_single_use`) only change training and evaluation, not deployment.

## Missing from this copy

`.vscode/` and `.pre-commit-config.yaml` could not be written by the remote tools; they are in the
history bundle.
