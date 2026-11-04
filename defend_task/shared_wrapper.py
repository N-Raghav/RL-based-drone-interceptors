"""One shared policy for every defender (parameter sharing), for skrl's single-agent PPO.

The 3-defender DirectMARLEnv with E envs is presented to skrl as ONE agent in 3E envs: row
d*E + e is defender d in env e. Every drone's experience trains the same network, and the
network cannot tell drones apart except through its own observation (role one-hot, its own
pose...), so whichever Tello the macro gives a role flies it the same way. On the laptop this
is one network evaluated three times per tick.

  observation (policy)  the defender's own 59-dim observation, exactly as before
  state (critic)        its own observation + the TRUE global state (asymmetric critic), so the
                        value knows whose return it is predicting
"""

from __future__ import annotations

from typing import Any

import gymnasium
import torch

from skrl.envs.wrappers.torch.base import Wrapper


class SharedDefenderWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        u = self._unwrapped
        self._agents = list(u.cfg.possible_agents)
        self._D = len(self._agents)
        self._E = u.num_envs
        self._obs_dim = int(u.cfg.observation_spaces[self._agents[0]])
        self._state_dim = self._obs_dim + int(u.cfg.state_space)
        self._obs = None
        self._states = None
        self._reset_once = True
        self._info: dict[str, Any] = {}

    # -- spaces -----------------------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._D * self._E

    @property
    def num_agents(self) -> int:
        return 1

    @property
    def observation_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(-float("inf"), float("inf"), (self._obs_dim,))

    @property
    def state_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(-float("inf"), float("inf"), (self._state_dim,))

    @property
    def action_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(-1.0, 1.0, (4,))

    # -- packing ----------------------------------------------------------------------------------

    def _pack(self, obs: dict[str, torch.Tensor]) -> None:
        o = torch.cat([obs[a] for a in self._agents], dim=0)  # (D*E, obs)
        g = self._unwrapped.state()  # (E, S) true global state
        self._obs = o
        self._states = torch.cat([o, g.repeat(self._D, 1)], dim=-1)

    def _stack(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([d[a].reshape(self._E) for a in self._agents], dim=0).view(-1, 1)

    # -- api --------------------------------------------------------------------------------------

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._reset_once:
            obs, self._info = self._env.reset()
            self._pack(obs)
            self._reset_once = False
        return self._obs, self._info

    def step(self, actions: torch.Tensor):
        a = actions.view(self._D, self._E, -1)
        act = {name: a[i] for i, name in enumerate(self._agents)}
        with torch.no_grad():
            obs, rew, term, trunc, self._info = self._env.step(act)
        self._pack(obs)
        return self._obs, self._stack(rew), self._stack(term).bool(), self._stack(trunc).bool(), self._info

    def state(self) -> torch.Tensor:
        return self._states

    def render(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        self._env.close()
