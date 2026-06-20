from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ALFWORLD_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "alfworld"
if str(ALFWORLD_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(ALFWORLD_EXAMPLE_DIR))

from alfworld_env import AlfWorldTextEpisode, _close_fast_downward_libs  # noqa: E402


class _PddlLikeEnv:
    def __init__(self, lib: object) -> None:
        self.downward_lib = lib


class _Wrapper:
    def __init__(self, child: object) -> None:
        self._wrapped_env = child


class _SyncBatchLikeEnv:
    def __init__(self, child: object) -> None:
        self.envs = [child]


class _GymLikeEnv:
    def __init__(self, child: object) -> None:
        self.batch_env = _SyncBatchLikeEnv(child)


@pytest.mark.unit
def test_close_fast_downward_libs_releases_nested_textworld_lib_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setitem(sys.modules, "fast_downward", types.SimpleNamespace(close_lib=lambda lib: calls.append(lib)))

    lib = object()
    root = _GymLikeEnv(_Wrapper(_PddlLikeEnv(lib)))

    assert _close_fast_downward_libs(root) == 1
    assert calls == [lib]
    assert root.batch_env.envs[0]._wrapped_env.downward_lib is None

    assert _close_fast_downward_libs(root) == 0
    assert calls == [lib]


@pytest.mark.unit
def test_episode_close_releases_fast_downward_before_env_close(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setitem(sys.modules, "fast_downward", types.SimpleNamespace(close_lib=lambda lib: calls.append(lib)))

    lib = object()
    pddl_env = _PddlLikeEnv(lib)

    class Env(_GymLikeEnv):
        closed = False

        def close(self) -> None:
            assert pddl_env.downward_lib is None
            self.closed = True

    env = Env(_Wrapper(pddl_env))
    episode = AlfWorldTextEpisode("unused.yaml")
    episode._env = env
    episode._base_env = object()

    episode.close()

    assert calls == [lib]
    assert env.closed
    assert episode._env is None
    assert episode._base_env is None

