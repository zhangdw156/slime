"""Single-episode ScienceWorld generation helpers for slime."""

from __future__ import annotations

import logging
import os
import uuid
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from prompts import SCIENCEWORLD_SYSTEM_PROMPT, build_observation_prompt, parse_action
from scienceworld_env import ScienceWorldTextEpisode

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 50
DEFAULT_ENV_STEP_LIMIT = 100
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_STEP_MAX_TOKENS = 512
DEFAULT_INVALID_ACTION_PENALTY = 0.01
DEFAULT_PROMPT_CHAR_LIMIT = 12000
DEFAULT_SIMPLIFICATION = "easy"


def _get_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    return metadata


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


def _safe_action_for_env(parsed_action: str, admissible_actions: list[str]) -> str:
    """Choose an action to send to ScienceWorld even after malformed output."""
    if parsed_action:
        return parsed_action
    if "look around" in admissible_actions:
        return "look around"
    return admissible_actions[0] if admissible_actions else "look around"


def _router_headers(args: Namespace, sample: Sample) -> dict[str, str] | None:
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": sample.session_id}
    return None


def _build_step_prompt(
    *,
    tokenizer,
    args: Namespace,
    current_observation: str,
    admissible_actions: list[str],
    task_description: str,
    inventory: str,
    history: list[dict[str, str]],
    history_length: int,
    max_prompt_chars: int,
) -> tuple[str, list[int], int]:
    """Build the current ScienceWorld step prompt with bounded recent history."""
    prompt_body, prompt_history_used = build_observation_prompt(
        current_observation=current_observation,
        admissible_actions=admissible_actions,
        task_description=task_description,
        inventory=inventory,
        history=history,
        history_length=history_length,
        max_prompt_chars=max_prompt_chars,
    )
    messages = [
        {"role": "system", "content": SCIENCEWORLD_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_body},
    ]
    prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, args=args)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, prompt_tokens, prompt_history_used


def _response_tokens_and_logprobs(tokenizer, response: str, meta_info: dict[str, Any]) -> tuple[list[int], list[float] | None]:
    """Use SGLang token IDs when available; fall back to tokenizer encoding."""
    token_logprobs = meta_info.get("output_token_logprobs") or []
    if token_logprobs:
        tokens = [item[1] for item in token_logprobs]
        log_probs = [item[0] for item in token_logprobs]
        return tokens, log_probs

    tokens = tokenizer.encode(response, add_special_tokens=False)
    return tokens, None


def _make_placeholder_step_sample(sample: Sample, tokenizer, metadata: dict[str, Any], reason: str) -> Sample:
    """Create a zero-loss placeholder for rare aborted-before-token cases."""
    placeholder = Sample()
    placeholder.index = sample.index
    placeholder.group_index = sample.group_index
    placeholder.group_id = sample.index if sample.index is not None else sample.group_index
    placeholder.prompt = ""
    placeholder.tokens = [tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0]
    placeholder.response = ""
    placeholder.response_length = 1
    placeholder.loss_mask = [0]
    placeholder.reward = 0.0
    placeholder.status = Sample.Status.ABORTED
    placeholder.remove_sample = True
    scienceworld_metadata = metadata.get("scienceworld") if isinstance(metadata, dict) else None
    if not isinstance(scienceworld_metadata, dict):
        scienceworld_metadata = {}
    placeholder.metadata = {**metadata, "scienceworld": {**scienceworld_metadata, "aborted_reason": reason}}
    return placeholder


def _final_episode_metadata(
    *,
    metadata: dict[str, Any],
    evaluation: bool,
    done: bool,
    completed: bool,
    trajectory: list[dict[str, Any]],
    invalid_action_count: int,
    final_score: float,
    total_delta_reward: float,
    invalid_action_penalty: float,
    last_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **metadata,
        "raw_reward": float(final_score),
        "scienceworld": {
            "evaluation": evaluation,
            "done": done,
            "completed": completed,
            "steps": len(trajectory),
            "invalid_action_count": invalid_action_count,
            "final_score": float(final_score),
            "total_delta_reward": float(total_delta_reward),
            "invalid_action_penalty": invalid_action_penalty,
            "final_info": last_info,
            "trajectory": trajectory,
        },
    }


def _set_eval_episode_token_fields(sample: Sample, response_tokens: list[int]) -> None:
    """Populate eval token metrics from generated assistant/action tokens only."""
    sample.tokens = list(response_tokens)
    sample.response_length = len(response_tokens)
    sample.loss_mask = [1] * sample.response_length


def _task_from_metadata(metadata: dict[str, Any]) -> tuple[str, int, str, int]:
    task_name = metadata.get("task_name")
    if not task_name:
        raise ValueError("ScienceWorld sample metadata must contain task_name.")
    variation_idx = int(metadata.get("variation_idx", 0))
    simplification = metadata.get("simplification") or os.environ.get(
        "SCIENCEWORLD_SIMPLIFICATION",
        DEFAULT_SIMPLIFICATION,
    )
    env_step_limit = int(
        metadata.get("env_step_limit") or _get_int_env("SCIENCEWORLD_ENV_STEP_LIMIT", DEFAULT_ENV_STEP_LIMIT)
    )
    return str(task_name), variation_idx, str(simplification), env_step_limit


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one ScienceWorld episode.

    Training returns step-level samples; evaluation returns one episode-level
    summary sample so metrics are counted per trajectory.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for ScienceWorld episodes."

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    metadata = _get_metadata(sample)
    if sample.session_id is None:
        sample.session_id = str(uuid.uuid4())

    max_steps = _get_int_env("SCIENCEWORLD_MAX_STEPS", DEFAULT_MAX_STEPS)
    history_length = _get_int_env("SCIENCEWORLD_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)
    step_max_tokens = _get_int_env("SCIENCEWORLD_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)
    invalid_action_penalty = _get_float_env("SCIENCEWORLD_INVALID_ACTION_PENALTY", DEFAULT_INVALID_ACTION_PENALTY)
    max_prompt_chars = _get_int_env("SCIENCEWORLD_MAX_PROMPT_CHARS", DEFAULT_PROMPT_CHAR_LIMIT)
    task_name, variation_idx, simplification, env_step_limit = _task_from_metadata(metadata)
    trajectory_id = sample.index if sample.index is not None else sample.group_index

    step_samples: list[Sample] = []
    assistant_responses: list[str] = []
    assistant_response_tokens: list[int] = []
    trajectory: list[dict[str, Any]] = []
    invalid_action_count = 0
    total_delta_reward = 0.0
    final_score = 0.0
    done = False
    completed = False
    aborted = False
    last_info: dict[str, Any] | None = None

    with ScienceWorldTextEpisode(env_step_limit=env_step_limit) as episode:
        reset_result = episode.reset(
            task_name=task_name,
            variation_idx=variation_idx,
            simplification=simplification,
        )
        task_description = str(reset_result.info.get("taskDesc") or "")
        history: list[dict[str, str]] = []
        current_observation = reset_result.observation
        inventory = str(reset_result.info.get("inv") or "")
        admissible_actions = reset_result.admissible_actions
        final_score = float(reset_result.score)
        total_delta_reward += float(reset_result.reward)
        last_info = reset_result.info

        for step_id in range(max_steps):
            step_sampling_params = dict(sampling_params)
            step_sampling_params["max_new_tokens"] = min(
                int(step_sampling_params.get("max_new_tokens", step_max_tokens)),
                step_max_tokens,
            )

            prompt_text, prompt_tokens, prompt_history_used = _build_step_prompt(
                tokenizer=tokenizer,
                args=args,
                current_observation=current_observation,
                admissible_actions=admissible_actions,
                task_description=task_description,
                inventory=inventory,
                history=history,
                history_length=history_length,
                max_prompt_chars=max_prompt_chars,
            )
            payload = {
                "input_ids": prompt_tokens,
                "sampling_params": step_sampling_params,
                "return_logprob": True,
            }
            output = await post(url, payload, headers=_router_headers(args, sample))
            meta_info = output.get("meta_info", {})
            if "finish_reason" in meta_info:
                sample.update_from_meta_info(args, meta_info)

            finish_type = meta_info.get("finish_reason", {}).get("type")
            if finish_type == "abort":
                aborted = True
                break
            response = output.get("text", "")
            if response.endswith("<|im_end|>"):
                response = response[: -len("<|im_end|>")]

            response_tokens, response_log_probs = _response_tokens_and_logprobs(tokenizer, response, meta_info)
            assistant_responses.append(response)
            assistant_response_tokens.extend(response_tokens)

            parsed = parse_action(response, admissible_actions)
            env_action = _safe_action_for_env(parsed.action, admissible_actions)
            if not parsed.is_valid:
                invalid_action_count += 1

            step_result = episode.step(env_action)
            done = step_result.done
            completed = step_result.completed
            total_delta_reward += float(step_result.reward)
            final_score = float(step_result.score)
            last_info = step_result.info

            step_record = {
                "step": step_id + 1,
                "observation": current_observation,
                "inventory": inventory,
                "model_response": response,
                "action": env_action,
                "valid_action": parsed.is_valid,
                "invalid_reason": parsed.invalid_reason,
                "reward": step_result.reward,
                "score": step_result.score,
                "done": done,
                "completed": completed,
                "prompt_tokens": len(prompt_tokens),
                "response_tokens": len(response_tokens),
                "history_used": prompt_history_used,
            }
            trajectory.append(step_record)

            if response_tokens:
                step_sample = Sample()
                step_sample.index = (sample.index or 0) * (max_steps + 1) + step_id
                step_sample.group_index = sample.group_index
                step_sample.group_id = trajectory_id
                step_sample.prompt = prompt_text
                step_sample.tokens = prompt_tokens + response_tokens
                step_sample.response = response
                step_sample.response_length = len(response_tokens)
                step_sample.loss_mask = [1] * len(response_tokens)
                step_sample.session_id = sample.session_id
                step_sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
                if response_log_probs is not None and len(response_log_probs) == len(response_tokens):
                    step_sample.rollout_log_probs = response_log_probs
                step_sample.metadata = {
                    **metadata,
                    "episode_id": trajectory_id,
                    "turn_step": step_id,
                    "scienceworld_step": step_record,
                }
                step_samples.append(step_sample)

            if done:
                break

            history.append({"observation": current_observation, "action": env_action})
            current_observation = step_result.observation
            inventory = str(step_result.info.get("inv") or "")
            admissible_actions = step_result.admissible_actions

    final_reward = float(final_score) - invalid_action_penalty * invalid_action_count
    episode_metadata = _final_episode_metadata(
        metadata=metadata,
        evaluation=evaluation,
        done=done,
        completed=completed,
        trajectory=trajectory,
        invalid_action_count=invalid_action_count,
        final_score=final_score,
        total_delta_reward=total_delta_reward,
        invalid_action_penalty=invalid_action_penalty,
        last_info=last_info,
    )

    if evaluation:
        sample.response = "\n".join(assistant_responses)
        _set_eval_episode_token_fields(sample, assistant_response_tokens)
        sample.reward = final_reward
        sample.status = Sample.Status.ABORTED if aborted else (Sample.Status.COMPLETED if done else Sample.Status.TRUNCATED)
        sample.metadata = episode_metadata
        return sample

    if not step_samples:
        reason = "sglang_abort" if aborted else "empty_episode_response"
        return [_make_placeholder_step_sample(sample, tokenizer, episode_metadata, reason)]

    if any(step_sample.rollout_log_probs is None for step_sample in step_samples):
        for step_sample in step_samples:
            step_sample.rollout_log_probs = None

    for step_sample in step_samples:
        step_sample.reward = final_reward
        step_sample.metadata = {
            **episode_metadata,
            "episode_id": trajectory_id,
            "turn_step": step_sample.metadata["turn_step"],
            "scienceworld_step": step_sample.metadata["scienceworld_step"],
        }

    return step_samples


def _trajectory_units(group: Iterable[Sample | list[Sample]]) -> list[list[Sample]]:
    units = []
    for item in group:
        if isinstance(item, list):
            units.append(item)
        else:
            units.append([item])
    return units


def check_episode_reward_nonzero_std(args, samples: list[Sample] | list[list[Sample]], **kwargs) -> DynamicFilterOutput:
    """Dynamic filter that treats each compact list as one ScienceWorld trajectory."""
    rewards = []
    for trajectory_samples in _trajectory_units(samples):
        if not trajectory_samples:
            continue
        rewards.append(trajectory_samples[0].get_reward_value(args))

    if len(rewards) <= 1:
        return DynamicFilterOutput(keep=True)

    import torch

    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    keep = bool(reward_tensor.std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(float(rewards[0]), 3)}",
    )


def grpo_normalize_scienceworld_steps(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Normalize GRPO rewards by trajectory, then broadcast to step samples."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", True)
    ):
        return raw_rewards, raw_rewards

    import torch

    prompt_groups: dict[int, dict[int, float]] = defaultdict(dict)
    sample_keys: list[tuple[int, int]] = []
    for sample, reward in zip(samples, raw_rewards, strict=True):
        prompt_key = sample.group_index if sample.group_index is not None else sample.index
        traj_key = sample.group_id if sample.group_id is not None else sample.index
        prompt_groups[prompt_key][traj_key] = float(reward)
        sample_keys.append((prompt_key, traj_key))

    normalized_by_traj: dict[tuple[int, int], float] = {}
    use_std = getattr(args, "advantage_estimator", None) in ["grpo", "gspo"] and getattr(
        args,
        "grpo_std_normalization",
        True,
    )
    for prompt_key, traj_rewards in prompt_groups.items():
        traj_ids = list(traj_rewards.keys())
        rewards = torch.tensor([traj_rewards[traj_id] for traj_id in traj_ids], dtype=torch.float)
        normalized = rewards - rewards.mean()
        if use_std and len(traj_ids) > 1:
            normalized = normalized / (rewards.std() + 1e-6)
        for traj_id, value in zip(traj_ids, normalized.tolist(), strict=True):
            normalized_by_traj[(prompt_key, traj_id)] = float(value)

    processed = [normalized_by_traj[key] for key in sample_keys]
    return raw_rewards, processed
