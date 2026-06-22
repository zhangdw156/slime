"""Prompt templates and parsers for the ScienceWorld slime example."""

from __future__ import annotations

import re
from dataclasses import dataclass

SCIENCEWORLD_SYSTEM_PROMPT = (
    "You are an expert autonomous agent operating in the ScienceWorld text environment. "
    "At every turn, return exactly one action from the admissible action list."
)

SCIENCEWORLD_TEMPLATE_NO_HIS = """Your science task is: {task_description}

Your current observation is: {current_observation}
Your current inventory is: {inventory}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one admissible action for the current step and present it within <action> </action> tags.
"""

SCIENCEWORLD_TEMPLATE = """Your science task is: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and corresponding actions:
{action_history}

You are now at step {current_step} and your current observation is: {current_observation}
Your current inventory is: {inventory}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one admissible action for the current step and present it within <action> </action> tags.
"""

_ACTION_RE = re.compile(r"<action>(.*?)</action>", flags=re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ParsedAction:
    """Result of parsing a model response into a ScienceWorld action."""

    action: str
    valid_format: bool
    valid_admissible: bool
    invalid_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.valid_format and self.valid_admissible


def normalize_action(action: str) -> str:
    """Normalize ScienceWorld action strings for matching."""
    return " ".join(action.strip().lower().split())


def format_admissible_actions(actions: list[str]) -> str:
    """Format ScienceWorld admissible commands for prompt insertion."""
    return "\n ".join(f"'{normalize_action(action)}'" for action in actions)


def build_observation_prompt(
    *,
    current_observation: str,
    admissible_actions: list[str],
    task_description: str,
    inventory: str = "",
    history: list[dict[str, str]] | None = None,
    history_length: int = 4,
    max_prompt_chars: int = 12000,
) -> tuple[str, int]:
    """Build the per-turn user prompt with bounded recent history.

    If the history-bearing prompt grows past ``max_prompt_chars``, fall back to
    a no-history prompt.  This mirrors the WebShop/ALFWorld-style bounded
    context organization while keeping long ScienceWorld observations/actions
    from exploding prompt length.
    """
    formatted_actions = format_admissible_actions(admissible_actions)
    history = history or []
    recent_history = history[-history_length:] if history_length > 0 else []

    if not recent_history:
        return (
            SCIENCEWORLD_TEMPLATE_NO_HIS.format(
                task_description=task_description,
                current_observation=current_observation,
                inventory=inventory or "empty",
                admissible_actions=formatted_actions,
            ),
            0,
        )

    action_history_parts = []
    start_step = len(history) - len(recent_history) + 1
    for offset, item in enumerate(recent_history):
        step = start_step + offset
        action_history_parts.append(
            f"Observation {step}:\n{item['observation']}\n\nAction {step}:\n{item['action']}"
        )

    prompt = SCIENCEWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_length=len(recent_history),
        action_history="\n\n".join(action_history_parts),
        current_step=len(history) + 1,
        current_observation=current_observation,
        inventory=inventory or "empty",
        admissible_actions=formatted_actions,
    )
    if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
        return (
            SCIENCEWORLD_TEMPLATE_NO_HIS.format(
                task_description=task_description,
                current_observation=current_observation,
                inventory=inventory or "empty",
                admissible_actions=formatted_actions,
            ),
            0,
        )
    return prompt, len(recent_history)


def parse_action(response: str, admissible_actions: list[str]) -> ParsedAction:
    """Parse and validate the model's ScienceWorld action response."""
    original_response = response
    response = response.strip()
    match = _ACTION_RE.search(response)
    has_think = _THINK_RE.search(response) is not None
    has_chinese = _CHINESE_RE.search(original_response) is not None

    if match is None:
        fallback = normalize_action(response[-50:].strip() or "look around")
        return ParsedAction(
            action=fallback,
            valid_format=False,
            valid_admissible=False,
            invalid_reason="missing_action_tag",
        )

    action = normalize_action(match.group(1))
    valid_format = bool(action) and has_think and not has_chinese
    admissible_set = {normalize_action(item) for item in admissible_actions}
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
