"""Prompt templates and action parsing for the WebShop slime example."""

from __future__ import annotations

import re
from dataclasses import dataclass

WEBSHOP_SYSTEM_PROMPT = (
    "You are an expert shopping agent operating in the WebShop text environment. "
    "At every turn, return exactly one valid WebShop action."
)

WEBSHOP_SEARCH_GUIDANCE = """WebShop search guidance:
- Use search[<your query>] with a short core product query, such as the product type or category.
- Do not put color, size, price, or every requested attribute into search[<your query>]. Handle those by opening a product page and selecting/clicking options when available.
- If a search returns zero results, retry with a shorter broader product query, not a longer query.
- The goal is to inspect/select a matching product and eventually click[buy now]."""


WEBSHOP_TEMPLATE_NO_HIS = """Your task is to: {instruction_text}.

Your current observation is:
{current_observation}

Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one admissible action for the current step and present it within <action> </action> tags.

{search_guidance}
"""

WEBSHOP_TEMPLATE = """Your task is to: {instruction_text}.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {current_step} and your current observation is:
{current_observation}

Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one admissible action for the current step and present it within <action> </action> tags.

{search_guidance}
"""

_ACTION_RE = re.compile(r"<action>(.*?)</action>", flags=re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_BRACKET_ACTION_RE = re.compile(r"^(search|click)\[(.*)\]$", flags=re.IGNORECASE | re.DOTALL)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ParsedAction:
    action: str
    valid_format: bool
    valid_admissible: bool
    invalid_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.valid_format and self.valid_admissible


def normalize_clickable(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _clickables(available_actions: dict) -> set[str]:
    return {normalize_clickable(action) for action in available_actions.get("clickables", [])}


def format_available_actions(available_actions: dict) -> str:
    actions: list[str] = []
    if available_actions.get("has_search_bar"):
        actions.append("search[<your query>]")
    clickables = [normalize_clickable(item) for item in available_actions.get("clickables", [])]
    for clickable in clickables:
        if clickable and clickable != "search":
            actions.append(f"click[{clickable}]")
    return "\n".join(f"'{action}'," for action in actions) if actions else "'no valid actions',"


def build_observation_prompt(
    *,
    instruction_text: str,
    current_observation: str,
    available_actions: dict,
    history: list[dict[str, str]] | None = None,
    history_length: int = 4,
) -> tuple[str, int]:
    history = history or []
    recent_history = history[-history_length:] if history_length > 0 else []
    formatted_actions = format_available_actions(available_actions)

    if not recent_history:
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
            instruction_text=instruction_text,
            current_observation=current_observation,
            available_actions=formatted_actions,
            search_guidance=WEBSHOP_SEARCH_GUIDANCE,
        )
        return prompt, 0

    start_step = len(history) - len(recent_history) + 1
    history_parts = []
    for offset, item in enumerate(recent_history):
        step = start_step + offset
        history_parts.append(
            f"Step {step} observation:\n{item['observation']}\n\nStep {step} action:\n{item['action']}"
        )
    prompt = WEBSHOP_TEMPLATE.format(
        instruction_text=instruction_text,
        step_count=len(history),
        history_length=len(recent_history),
        action_history="\n\n".join(history_parts),
        current_step=len(history) + 1,
        current_observation=current_observation,
        available_actions=formatted_actions,
        search_guidance=WEBSHOP_SEARCH_GUIDANCE,
    )
    return prompt, len(recent_history)


def parse_action(response: str, available_actions: dict) -> ParsedAction:
    original_response = response
    response = response.strip()
    match = _ACTION_RE.search(response)
    has_think = _THINK_RE.search(response) is not None
    has_chinese = _CHINESE_RE.search(original_response) is not None
    if match is None:
        return ParsedAction(
            action="",
            valid_format=False,
            valid_admissible=False,
            invalid_reason="missing_action_tag",
        )

    action = " ".join(match.group(1).strip().split())
    action_match = _BRACKET_ACTION_RE.match(action)
    valid_format = bool(action_match) and has_think and not has_chinese
    if not action_match:
        return ParsedAction(
            action=action.lower(),
            valid_format=False,
            valid_admissible=False,
            invalid_reason="malformed_action",
        )

    action_name = action_match.group(1).lower()
    action_arg = normalize_clickable(action_match.group(2))
    normalized_action = f"{action_name}[{action_arg}]"

    if action_name == "search":
        valid_admissible = bool(action_arg) and bool(available_actions.get("has_search_bar"))
    else:
        valid_admissible = action_arg in _clickables(available_actions) and action_arg != "search"

    invalid_reason = None
    if not has_think:
        invalid_reason = "missing_think_tag"
    elif has_chinese:
        invalid_reason = "contains_chinese"
    elif not valid_admissible:
        invalid_reason = "invalid_action"

    return ParsedAction(
        action=normalized_action,
        valid_format=valid_format,
        valid_admissible=valid_admissible,
        invalid_reason=invalid_reason,
    )
