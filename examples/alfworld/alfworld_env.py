"""Small ALFWorld TextWorld wrapper used by the slime ALFWorld example.

The wrapper intentionally keeps ALFWorld imports lazy so repository-side syntax
checks do not require the heavy environment package to be installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


@dataclass
class StepResult:
    observation: str
    admissible_actions: list[str]
    reward: float
    done: bool
    won: bool
    info: dict[str, Any]


def _load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as reader:
        return yaml.safe_load(reader)


def _squeeze_info(info: dict[str, Any]) -> dict[str, Any]:
    squeezed = {}
    for key, value in info.items():
        if isinstance(value, (list, tuple)) and len(value) == 1:
            squeezed[key] = value[0]
        else:
            squeezed[key] = value
    return squeezed


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


class AlfWorldTextEpisode:
    """Run one ALFWorld TextWorld episode, optionally pinned to one game file."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        split: str = "train",
        gamefile: str | None = None,
        seed: int = 0,
    ) -> None:
        self.config_path = str(config_path)
        self.split = split
        self.gamefile = gamefile
        self.seed = seed
        self._base_env = None
        self._env = None

    def __enter__(self) -> "AlfWorldTextEpisode":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _open(self) -> None:
        try:
            from alfworld.agents.environment import get_environment
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise ImportError(
                "ALFWorld is not installed. Install examples/alfworld/requirements.txt "
                "and run `alfworld-download -f` before launching this example."
            ) from exc

        config = _load_config(self.config_path)
        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        if env_type != "AlfredTWEnv":
            raise ValueError(f"Only AlfredTWEnv text mode is supported by this example, got {env_type!r}.")

        self._base_env = get_environment(env_type)(config, train_eval=self.split)
        if self.gamefile:
            gamefile = os.path.expandvars(os.path.expanduser(self.gamefile))
            if not Path(gamefile).exists():
                raise FileNotFoundError(f"ALFWorld gamefile does not exist: {gamefile}")
            # Pin this episode to the dataset row selected by slime. AlfredTWEnv
            # collects all games in __init__, but init_env() reads self.game_files,
            # so replacing it before init_env keeps the game source explicit.
            self._base_env.game_files = [gamefile]
            self._base_env.num_games = 1

        self._env = self._base_env.init_env(batch_size=1)
        if hasattr(self._env, "seed"):
            self._env.seed(self.seed)

    def reset(self) -> StepResult:
        assert self._env is not None, "Episode is not open"
        observations, infos = self._env.reset()
        info = _squeeze_info(infos)
        return StepResult(
            observation=_first(observations),
            admissible_actions=list(info.get("admissible_commands", [])),
            reward=0.0,
            done=False,
            won=bool(info.get("won", False)),
            info=info,
        )

    def step(self, action: str) -> StepResult:
        assert self._env is not None, "Episode is not open"
        observations, _scores, dones, infos = self._env.step([action])
        info = _squeeze_info(infos)
        won = bool(info.get("won", False))
        reward = 1.0 * float(won)
        return StepResult(
            observation=_first(observations),
            admissible_actions=list(info.get("admissible_commands", [])),
            reward=reward,
            done=bool(_first(dones)),
            won=won,
            info=info,
        )

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None
        self._base_env = None
