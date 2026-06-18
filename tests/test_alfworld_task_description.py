from __future__ import annotations

import sys
from pathlib import Path

import pytest

ALFWORLD_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "alfworld"
if str(ALFWORLD_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(ALFWORLD_EXAMPLE_DIR))

from prompts import _task_description_from_reset, build_observation_prompt  # noqa: E402


@pytest.mark.unit
def test_task_description_extracts_reset_goal() -> None:
    observation = (
        "-= Welcome to TextWorld, ALFRED! =-\n\n"
        "You are in the middle of a room.\n\n"
        "Your task is to: find two toiletpaper and put them in drawer."
    )

    task = _task_description_from_reset(observation)

    assert task == "find two toiletpaper and put them in drawer."


@pytest.mark.unit
def test_late_history_prompt_keeps_full_goal_after_initial_observation_slides_out() -> None:
    observation = (
        "-= Welcome to TextWorld, ALFRED! =-\n\n"
        "You are in the middle of a room.\n\n"
        "Your task is to: put a soapbottle in garbagecan."
    )
    task = _task_description_from_reset(observation)
    history = [
        {"observation": f"later observation {i}", "action": f"action {i}"}
        for i in range(5)
    ]

    prompt = build_observation_prompt(
        current_observation="current later observation",
        admissible_actions=["look", "go to garbagecan 1"],
        task_description=task,
        history=history,
        history_length=4,
    )

    assert "Your task is to: put a soapbottle in garbagecan." in prompt
    assert "Your task is to: pick_and_place_simple" not in prompt
    assert "later observation 0" not in prompt
    assert "later observation 1" in prompt


@pytest.mark.unit
def test_task_description_fails_fast_without_reset_goal() -> None:
    with pytest.raises(ValueError, match="task goal"):
        _task_description_from_reset("You are in the middle of a room, but no explicit goal is present.")
