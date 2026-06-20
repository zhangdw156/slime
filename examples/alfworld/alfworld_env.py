"""Small ALFWorld TextWorld wrapper used by the slime ALFWorld example.

The wrapper intentionally keeps ALFWorld imports lazy so repository-side syntax
checks do not require the heavy environment package to be installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ENV_CHILD_ATTRS = ("batch_env", "envs", "_wrapped_env")
_SCALAR_TYPES = (str, bytes, bytearray, int, float, bool, type(None))

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


def _pin_config_to_gamefile(config: dict[str, Any], split: str, gamefile: Path) -> None:
    """Limit ALFWorld's dataset scan to the selected game directory."""
    dataset = config.setdefault("dataset", {})
    game_root = str(gamefile.parent)

    if split == "train":
        dataset["data_path"] = game_root
        dataset["num_train_games"] = 1
    elif split == "eval_in_distribution":
        dataset["eval_id_data_path"] = game_root
        dataset["num_eval_games"] = 1
    elif split == "eval_out_of_distribution":
        dataset["eval_ood_data_path"] = game_root
        dataset["num_eval_games"] = 1
    else:
        raise ValueError(f"Unsupported ALFWorld split: {split!r}.")


def _close_fast_downward_libs(root: Any) -> int:
    """Release TextWorld PDDL native libraries before env wrappers drop refs.

    ALFWorld 0.4.2 uses TextWorld's ``PddlEnv``, whose constructor calls
    ``fast_downward.load_lib()``. That function copies ``libdownward.so`` into
    a temporary directory and ``dlopen``s the copy. TextWorld's PDDL env does
    not implement a matching ``close()`` method, so long-lived Ray actors keep
    accumulating ``libdownward.so (deleted)`` mappings unless we explicitly call
    ``fast_downward.close_lib()`` while the underlying PDDL env is still
    reachable.

    The traversal is intentionally narrow: it only follows the wrapper/batch
    attributes used by TextWorld's gym env stack and only touches objects that
    directly own a ``downward_lib`` attribute. This keeps the cleanup local to
    ALFWorld/TextWorld and avoids walking arbitrary user data.
    """

    visited_objects: set[int] = set()
    visited_libs: set[int] = set()
    closed_libs: set[int] = set()
    missing = object()
    close_lib: Any = missing
    closed = 0

    def get_close_lib():
        nonlocal close_lib
        if close_lib is missing:
            try:
                import fast_downward  # type: ignore[import-not-found]
            except ImportError:
                close_lib = None
            else:
                close_lib = getattr(fast_downward, "close_lib", None)
        return close_lib

    def visit(obj: Any) -> None:
        nonlocal closed

        if isinstance(obj, _SCALAR_TYPES):
            return
        if isinstance(obj, dict):
            for child in obj.values():
                visit(child)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for child in obj:
                visit(child)
            return

        obj_id = id(obj)
        if obj_id in visited_objects:
            return
        visited_objects.add(obj_id)

        state = getattr(obj, "__dict__", None)
        if not isinstance(state, dict):
            return

        lib = state.get("downward_lib")
        if lib is not None:
            lib_id = id(lib)
            if lib_id not in visited_libs:
                visited_libs.add(lib_id)
                # Current ALFWorld/TextWorld dependencies leave PddlEnv.close()
                # inherited from textworld.Environment, where it is a no-op. If
                # a future PddlEnv owns an explicit close(), let TextWorld handle
                # the native resource to avoid double-closing it here.
                has_own_close = "close" in type(obj).__dict__
                closer = None if has_own_close else get_close_lib()
                if closer is not None:
                    try:
                        closer(lib)
                    except Exception:
                        logger.exception("Failed to close TextWorld fast_downward native library; continuing cleanup.")
                    else:
                        closed += 1
                        closed_libs.add(lib_id)
            if lib_id in closed_libs:
                state["downward_lib"] = None

        for attr in _ENV_CHILD_ATTRS:
            child = state.get(attr)
            if child is not None:
                visit(child)

    visit(root)
    return closed


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

    def __enter__(self) -> AlfWorldTextEpisode:
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

        pinned_gamefile = None
        if self.gamefile:
            pinned_gamefile = Path(os.path.expandvars(os.path.expanduser(self.gamefile)))
            if not pinned_gamefile.exists():
                raise FileNotFoundError(f"ALFWorld gamefile does not exist: {pinned_gamefile}")
            _pin_config_to_gamefile(config, self.split, pinned_gamefile)

        self._base_env = get_environment(env_type)(config, train_eval=self.split)
        if pinned_gamefile is not None:
            # Keep an explicit pin as a guard even though the config now points to
            # the selected game directory before AlfredTWEnv scans the dataset.
            self._base_env.game_files = [str(pinned_gamefile)]
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
        env = self._env
        try:
            if env is not None:
                _close_fast_downward_libs(env)
                if hasattr(env, "close"):
                    env.close()
        finally:
            self._env = None
            self._base_env = None
