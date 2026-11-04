"""AS-Defend-Real-v0: 3 defenders vs 2 attackers over the measured Tello link (13.5 Hz,
rc and video loss, delayed noisy feeds), scripted macro layer + PPO residual."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Defend-Real-v0",
    entry_point=f"{__name__}.tello_defend_real_env:TelloDefendRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_real_env_cfg:TelloDefendRealEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="AS-Defend-Role-v0",
    entry_point=f"{__name__}.tello_defend_role_env:TelloDefendRoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_role_env_cfg:TelloDefendRoleEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_role_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_role_cfg.yaml",
        "skrl_ppo_shared_cfg_entry_point": f"{agents.__name__}:skrl_ppo_shared_cfg.yaml",
    },
)
