"""Small ScienceWorld wrapper used by the slime ScienceWorld example.

The wrapper intentionally keeps ScienceWorld imports lazy so repository-side
syntax checks do not require the Java/py4j environment package to be installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from prompts import normalize_action


@dataclass
class StepResult:
    observation: str
    admissible_actions: list[str]
    reward: float
    score: float
    done: bool
    completed: bool
    info: dict[str, Any]


def _dedupe_actions(actions: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for action in actions:
        normalized = normalize_action(str(action))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _actions_from_env(env) -> list[str]:
    """Prefer ScienceWorld state-valid actions, falling back to possible actions."""
    try:
        valid = env.get_valid_action_object_combinations_with_templates()
        actions = []
        for item in valid:
            if isinstance(item, dict) and item.get("action"):
                actions.append(item["action"])
            elif isinstance(item, str):
                actions.append(item)
    except Exception:
        actions = []
    if not actions:
        actions = list(env.get_possible_actions())
    return _dedupe_actions(actions)


def _safe_env_call(env, method_name: str, default: Any = None) -> Any:
    method = getattr(env, method_name, None)
    if not callable(method):
        return default
    try:
        return method()
    except Exception:
        return default


def _load_scienceworld_env_class():
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise ImportError(
            "ScienceWorld is not installed. Run `mamba install -c conda-forge openjdk=11` "
            "and `pip install -r examples/scienceworld/requirements.txt` before launching this example."
        ) from exc
    return ScienceWorldEnv


class ScienceWorldTextEpisode:
    """Run one ScienceWorld episode pinned to one task variation."""

    def __init__(self, *, env_step_limit: int = 100, jar_path: str | None = None) -> None:
        self.env_step_limit = env_step_limit
        self.jar_path = jar_path or os.environ.get("SCIENCEWORLD_JAR_PATH") or None
        self._env = None

    def __enter__(self) -> ScienceWorldTextEpisode:
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _open(self) -> None:
        if self._env is not None:
            return
        env_cls = _load_scienceworld_env_class()
        self._env = env_cls("", self.jar_path, envStepLimit=self.env_step_limit)

    @property
    def env(self):
        self._open()
        return self._env

    def task_names(self) -> list[str]:
        return list(self.env.get_task_names())

    def max_variations(self, task_name: str) -> int:
        return int(self.env.get_max_variations(task_name))

    def variation_splits(self, task_name: str, simplification: str) -> dict[str, list[int]]:
        self.env.load(task_name, 0, simplification)
        return {
            "train": list(self.env.get_variations_train()),
            "eval": list(self.env.get_variations_dev()),
            "test": list(self.env.get_variations_test()),
        }

    def reset(self, *, task_name: str, variation_idx: int, simplification: str = "easy") -> StepResult:
        self.env.load(task_name, int(variation_idx), simplification)
        observation, info = self.env.reset()
        info = dict(info or {})
        info.setdefault("score", _safe_env_call(self.env, "get_score", 0))
        info.setdefault("reward", 0)
        info.setdefault("taskDesc", _safe_env_call(self.env, "get_task_description", ""))
        info.setdefault("inv", _safe_env_call(self.env, "inventory", ""))
        info.update(
            {
                "taskName": task_name,
                "variationIdx": int(variation_idx),
                "simplificationStr": simplification,
                "admissible_actions": _actions_from_env(self.env),
                "completed": False,
            }
        )
        return StepResult(
            observation=observation,
            admissible_actions=info["admissible_actions"],
            reward=float(info.get("reward", 0.0)),
            score=float(info.get("score", 0.0)),
            done=False,
            completed=False,
            info=info,
        )

    def step(self, action: str) -> StepResult:
        observation, reward, done, info = self.env.step(action)
        info = dict(info or {})
        info.setdefault("score", _safe_env_call(self.env, "get_score", 0.0))
        info.setdefault("inv", _safe_env_call(self.env, "inventory", ""))
        score = float(info.get("score", 0.0))
        info.update(
            {
                "admissible_actions": _actions_from_env(self.env),
                "completed": bool(done),
            }
        )
        return StepResult(
            observation=observation,
            admissible_actions=info["admissible_actions"],
            reward=float(reward),
            score=score,
            done=bool(done),
            completed=bool(done),
            info=info,
        )

    def close(self) -> None:
        if self._env is not None:
            close_fn = getattr(self._env, "close", None) or getattr(self._env, "shutdown", None)
            if callable(close_fn):
                close_fn()
        self._env = None
