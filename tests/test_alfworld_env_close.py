from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ALFWORLD_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "alfworld"
if str(ALFWORLD_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(ALFWORLD_EXAMPLE_DIR))

from alfworld_env import AlfWorldTextEpisode  # noqa: E402


class _Env:
    closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_episode_close_does_not_dlclose_fast_downward(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_close_lib(_lib: object) -> None:
        raise AssertionError("fast_downward.close_lib should not be called in-process")

    monkeypatch.setitem(sys.modules, "fast_downward", types.SimpleNamespace(close_lib=fail_close_lib))

    env = _Env()
    episode = AlfWorldTextEpisode("unused.yaml")
    episode._env = env
    episode._base_env = object()

    episode.close()

    assert env.closed
    assert episode._env is None
    assert episode._base_env is None
