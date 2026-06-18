"""Prompt templates and parsers for the ALFWorld slime example."""

from __future__ import annotations

import re
from dataclasses import dataclass

ALFWORLD_SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED embodied environment. "
    "At every turn, return exactly one action from the admissible action list."
)

ALFWORLD_TEMPLATE_NO_HIS = """Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one admissible action for the current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and corresponding actions:
{action_history}

You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one admissible action for the current step and present it within <action> </action> tags.
"""

_TASK_PREFIX = "Your task is to: "
_ACTION_RE = re.compile(r"<action>(.*?)</action>", flags=re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ParsedAction:
    """Result of parsing a model response into an ALFWorld action."""

    action: str
    valid_format: bool
    valid_admissible: bool
    invalid_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.valid_format and self.valid_admissible


def extract_task(observation: str) -> str:
    """Extract the ALFWorld goal string from the initial observation when present."""
    start = observation.find(_TASK_PREFIX)
    if start == -1:
        return ""
    return observation[start + len(_TASK_PREFIX) :].strip()


def _task_description_from_reset(reset_observation: str) -> str:
    """Return the natural-language ALFWorld goal for subsequent turns.

    ``metadata["task_type"]`` is only a coarse ALFWorld category such as
    ``pick_two_obj_and_place``.  The reset observation must contain the real
    natural-language task goal; fail fast if it does not.
    """
    task = extract_task(reset_observation).strip()
    if not task:
        raise ValueError("ALFWorld reset observation did not contain a task goal.")
    return task


def format_admissible_actions(actions: list[str]) -> str:
    """Format ALFWorld admissible commands for prompt insertion."""
    return "\n ".join(f"'{action}'" for action in actions if action != "help")


def build_observation_prompt(
    *,
    current_observation: str,
    admissible_actions: list[str],
    task_description: str = "",
    history: list[dict[str, str]] | None = None,
    history_length: int = 4,
) -> str:
    """Build the per-turn user prompt, following the SDAR/verl-agent ALFWorld style."""
    formatted_actions = format_admissible_actions(admissible_actions)
    history = history or []
    recent_history = history[-history_length:] if history_length > 0 else []

    if not recent_history:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=current_observation,
            admissible_actions=formatted_actions,
        )

    action_history_parts = []
    start_step = len(history) - len(recent_history) + 1
    for offset, item in enumerate(recent_history):
        step = start_step + offset
        action_history_parts.append(
            f"Observation {step}:\n{item['observation']}\n\nAction {step}:\n{item['action']}"
        )

    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_length=len(recent_history),
        action_history="\n\n".join(action_history_parts),
        current_step=len(history) + 1,
        current_observation=current_observation,
        admissible_actions=formatted_actions,
    )


def parse_action(response: str, admissible_actions: list[str]) -> ParsedAction:
    """Parse and validate the model's ALFWorld action response."""
    original_response = response
    response = response.strip()
    match = _ACTION_RE.search(response)
    has_think = _THINK_RE.search(response) is not None
    has_chinese = _CHINESE_RE.search(original_response) is not None

    if match is None:
        fallback = response[-30:].strip().lower() or "look"
        return ParsedAction(
            action=fallback,
            valid_format=False,
            valid_admissible=False,
            invalid_reason="missing_action_tag",
        )

    action = match.group(1).strip().lower()
    valid_format = bool(action) and has_think and not has_chinese
    admissible_set = {item.lower() for item in admissible_actions}
    valid_admissible = action in admissible_set

    invalid_reason = None
    if not has_think:
        invalid_reason = "missing_think_tag"
    elif has_chinese:
        invalid_reason = "contains_chinese"
    elif not action:
        invalid_reason = "empty_action"
    elif not valid_admissible:
        invalid_reason = "not_in_admissible_actions"

    return ParsedAction(
        action=action,
        valid_format=valid_format,
        valid_admissible=valid_admissible,
        invalid_reason=invalid_reason,
    )
