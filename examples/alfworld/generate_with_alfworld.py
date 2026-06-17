"""ALFWorld custom generation function for slime GRPO training."""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import Any

from alfworld_env import AlfWorldTextEpisode
from prompts import ALFWORLD_SYSTEM_PROMPT, build_observation_prompt, extract_task, parse_action

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 50
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_STEP_MAX_TOKENS = 512
DEFAULT_INVALID_ACTION_PENALTY = 0.01


def _get_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    return metadata


def _get_env_config_path() -> str:
    config_path = os.environ.get("ALFWORLD_CONFIG_PATH")
    if config_path:
        return config_path
    return os.path.join(os.path.dirname(__file__), "configs", "config_tw.yaml")


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, os.environ.get(name), default)
        return default


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %s", name, os.environ.get(name), default)
        return default


def _apply_chat_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool, args: Namespace) -> str:
    kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _token_delta(tokenizer, messages: list[dict[str, str]], args: Namespace) -> tuple[list[int], list[int]]:
    """Return newly added tokens and loss mask after appending the latest message."""
    curr = _apply_chat_template(tokenizer, messages, add_generation_prompt=False, args=args)
    if messages[-1]["role"] == "assistant":
        prev = _apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True, args=args)
        new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
        return new_tokens, [1] * len(new_tokens)

    prev = _apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=False, args=args)
    new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
    return new_tokens, [0] * len(new_tokens)


def _safe_action_for_env(parsed_action: str, admissible_actions: list[str]) -> str:
    """Choose an action to send to ALFWorld even when the model output is malformed."""
    if parsed_action:
        return parsed_action
    if "look" in admissible_actions:
        return "look"
    if admissible_actions:
        return admissible_actions[0]
    return "look"


def _router_headers(args: Namespace, sample: Sample) -> dict[str, str] | None:
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": sample.session_id}
    return None


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Run one complete ALFWorld episode and convert it to a slime Sample."""
    assert not args.partial_rollout, "Partial rollout is not supported for ALFWorld episodes."

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    metadata = _get_metadata(sample)

    max_steps = _get_int_env("ALFWORLD_MAX_STEPS", DEFAULT_MAX_STEPS)
    history_length = _get_int_env("ALFWORLD_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)
    step_max_tokens = _get_int_env("ALFWORLD_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)
    invalid_action_penalty = _get_float_env("ALFWORLD_INVALID_ACTION_PENALTY", DEFAULT_INVALID_ACTION_PENALTY)
    config_path = _get_env_config_path()

    split = metadata.get("split", "train")
    # ALFWorld/SDAR uses eval_in_distribution and eval_out_of_distribution names at env-construction time.
    if split == "valid_seen":
        env_split = "eval_in_distribution"
    elif split == "valid_unseen":
        env_split = "eval_out_of_distribution"
    else:
        env_split = "train"

    seed = int(getattr(args, "rollout_seed", 0)) + int(sample.index or 0)
    gamefile = metadata.get("gamefile")

    prompt_tokens: list[int] = []
    response_tokens: list[int] = []
    loss_mask: list[int] = []
    assistant_responses: list[str] = []
    trajectory: list[dict[str, Any]] = []
    invalid_action_count = 0
    total_reward = 0.0
    done = False
    won = False

    with AlfWorldTextEpisode(config_path, split=env_split, gamefile=gamefile, seed=seed) as episode:
        reset_result = episode.reset()
        task_description = metadata.get("task_type") or extract_task(reset_result.observation)
        history: list[dict[str, str]] = []
        messages = [
            {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_observation_prompt(
                    current_observation=reset_result.observation,
                    admissible_actions=reset_result.admissible_actions,
                    task_description=task_description,
                    history=history,
                    history_length=history_length,
                ),
            },
        ]
        initial_prompt = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, args=args)
        prompt_tokens = tokenizer.encode(initial_prompt, add_special_tokens=False)

        current_observation = reset_result.observation
        admissible_actions = reset_result.admissible_actions
        last_info = reset_result.info

        for step_id in range(max_steps):
            step_sampling_params = dict(sampling_params)
            step_sampling_params["max_new_tokens"] = min(
                int(step_sampling_params.get("max_new_tokens", step_max_tokens)),
                step_max_tokens,
            )

            prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, args=args)
            payload = {"text": prompt_text, "sampling_params": step_sampling_params}
            output = await post(url, payload, headers=_router_headers(args, sample))
            sample.update_from_meta_info(args, output.get("meta_info", {}))

            if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
                sample.status = Sample.Status.ABORTED
                break

            response = output.get("text", "")
            if response.endswith("<|im_end|>"):
                response = response[: -len("<|im_end|>")]

            messages.append({"role": "assistant", "content": response})
            new_tokens, new_loss_mask = _token_delta(tokenizer, messages, args)
            response_tokens.extend(new_tokens)
            loss_mask.extend(new_loss_mask)
            assistant_responses.append(response)

            parsed = parse_action(response, admissible_actions)
            env_action = _safe_action_for_env(parsed.action, admissible_actions)
            if not parsed.is_valid:
                invalid_action_count += 1

            step_result = episode.step(env_action)
            done = step_result.done
            won = step_result.won
            total_reward = step_result.reward
            last_info = step_result.info
            trajectory.append(
                {
                    "step": step_id + 1,
                    "observation": current_observation,
                    "model_response": response,
                    "action": env_action,
                    "valid_action": parsed.is_valid,
                    "invalid_reason": parsed.invalid_reason,
                    "reward": step_result.reward,
                    "done": done,
                    "won": won,
                }
            )

            if done:
                break

            history.append({"observation": current_observation, "action": env_action})
            current_observation = step_result.observation
            admissible_actions = step_result.admissible_actions
            messages.append(
                {
                    "role": "user",
                    "content": build_observation_prompt(
                        current_observation=current_observation,
                        admissible_actions=admissible_actions,
                        task_description=task_description,
                        history=history,
                        history_length=history_length,
                    ),
                }
            )
            env_tokens, env_loss_mask = _token_delta(tokenizer, messages, args)
            response_tokens.extend(env_tokens)
            loss_mask.extend(env_loss_mask)

    if sample.status != Sample.Status.ABORTED:
        sample.status = Sample.Status.COMPLETED if done else Sample.Status.TRUNCATED
    sample.tokens = prompt_tokens + response_tokens
    sample.response_length = len(response_tokens)
    sample.loss_mask = loss_mask
    sample.response = "\n".join(assistant_responses)
    sample.reward = float(total_reward) - invalid_action_penalty * invalid_action_count
    sample.metadata = {
        **metadata,
        "alfworld": {
            "evaluation": evaluation,
            "done": done,
            "won": won,
            "steps": len(trajectory),
            "invalid_action_count": invalid_action_count,
            "raw_episode_reward": float(total_reward),
            "invalid_action_penalty": invalid_action_penalty,
            "final_info": last_info,
            "trajectory": trajectory,
        },
    }

    if sample.response_length != len(sample.loss_mask):
        raise RuntimeError(
            f"ALFWorld sample has response_length={sample.response_length} but loss_mask={len(sample.loss_mask)}"
        )
    if not sample.tokens or sample.response_length <= 0:
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0

    return sample
