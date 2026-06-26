"""WebShop custom generation function for slime GRPO training.

This path talks to a separately deployed WebShop HTTP service.  The slime
training environment therefore only needs the thin client in this directory;
WebShop's heavier dependencies live in the service environment.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from collections.abc import Iterable
from argparse import Namespace
from typing import Any

from client import close_session, reset_session, step_session
from prompts import (
    WEBSHOP_SYSTEM_PROMPT,
    build_observation_prompt,
    parse_action,
    safe_action_for_service,
)

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 30
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_STEP_MAX_TOKENS = 256
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_INVALID_ACTION_PENALTY = 0.0


def _get_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    return metadata if isinstance(metadata, dict) else {}


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


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _apply_chat_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool, args: Namespace) -> str:
    kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _build_step_prompt(
    *,
    tokenizer,
    args: Namespace,
    instruction_text: str,
    current_observation: str,
    available_actions: dict,
    history: list[dict[str, str]],
    history_length: int,
    max_prompt_chars: int,
) -> tuple[str, list[int], int]:
    user_prompt, prompt_history_used = build_observation_prompt(
        instruction_text=instruction_text,
        current_observation=current_observation,
        available_actions=available_actions,
        history=history,
        history_length=history_length,
        max_prompt_chars=max_prompt_chars,
    )
    messages = [
        {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, args=args)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, prompt_tokens, prompt_history_used


def _response_tokens_and_logprobs(tokenizer, response: str, meta_info: dict[str, Any]) -> tuple[list[int], list[float] | None]:
    token_logprobs = meta_info.get("output_token_logprobs") or []
    if token_logprobs:
        return [item[1] for item in token_logprobs], [item[0] for item in token_logprobs]
    return tokenizer.encode(response, add_special_tokens=False), None


def _goal_idx_from_sample(sample: Sample, metadata: dict[str, Any]) -> int | None:
    for key in ("webshop_goal_idx", "goal_idx", "goal_id"):
        if key in metadata and metadata[key] is not None:
            return int(metadata[key])
    if _get_bool_env("WEBSHOP_USE_SAMPLE_INDEX_AS_GOAL", False) and sample.index is not None:
        return int(sample.index)
    return None


def _trajectory_id(sample: Sample) -> int:
    if sample.index is not None:
        return int(sample.index)
    if sample.group_index is not None:
        return int(sample.group_index)
    return 0


def _router_headers(args: Namespace, sample: Sample) -> dict[str, str] | None:
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": sample.session_id}
    return None


def _make_placeholder_sample(sample: Sample, tokenizer, metadata: dict[str, Any], reason: str) -> Sample:
    placeholder = Sample()
    placeholder.index = sample.index
    placeholder.group_index = sample.group_index
    placeholder.group_id = _trajectory_id(sample)
    placeholder.prompt = sample.prompt
    placeholder.response = ""
    placeholder.tokens = tokenizer.encode(str(sample.prompt or ""), add_special_tokens=False)
    placeholder.response_length = 0
    placeholder.loss_mask = []
    placeholder.reward = 0.0
    placeholder.status = Sample.Status.ABORTED
    placeholder.session_id = sample.session_id
    placeholder.metadata = {**metadata, "webshop_error": reason, "raw_reward": 0.0}
    return placeholder


def _episode_metadata(
    *,
    metadata: dict[str, Any],
    session_id: str,
    instruction_text: str,
    trajectory: list[dict[str, Any]],
    final_state: dict[str, Any],
    raw_reward: float,
    final_reward: float,
    invalid_action_count: int,
    invalid_action_penalty: float,
    done: bool,
) -> dict[str, Any]:
    info = final_state.get("info", {}) if isinstance(final_state, dict) else {}
    return {
        **metadata,
        "session_id": session_id,
        "instruction_text": instruction_text,
        "trajectory": trajectory,
        "raw_reward": raw_reward,
        "final_reward": final_reward,
        "done": done,
        "invalid_action_count": invalid_action_count,
        "invalid_action_penalty": invalid_action_penalty,
        "reward_info": info.get("reward_info"),
        "webshop_url": info.get("url"),
    }


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one WebShop episode through the external service."""

    assert not args.partial_rollout, "Partial rollout is not supported for WebShop episodes."

    state = GenerateState(args)
    tokenizer = state.tokenizer
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    metadata = _get_metadata(sample)
    if sample.session_id is None:
        sample.session_id = str(uuid.uuid4())

    max_steps = _get_int_env("WEBSHOP_MAX_STEPS", DEFAULT_MAX_STEPS)
    history_length = _get_int_env("WEBSHOP_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)
    step_max_tokens = _get_int_env("WEBSHOP_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)
    max_prompt_chars = _get_int_env("WEBSHOP_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS)
    invalid_action_penalty = _get_float_env("WEBSHOP_INVALID_ACTION_PENALTY", DEFAULT_INVALID_ACTION_PENALTY)
    close_on_done = _get_bool_env("WEBSHOP_CLOSE_SESSION_ON_DONE", True)
    observation_mode = os.environ.get("WEBSHOP_OBSERVATION_MODE")

    goal_idx = _goal_idx_from_sample(sample, metadata)
    reset_state = await reset_session(
        session_id=sample.session_id,
        goal_idx=goal_idx,
        observation_mode=observation_mode,
    )
    instruction_text = reset_state["instruction_text"]
    current_observation = reset_state["observation"]
    available_actions = reset_state.get("available_actions", {})

    trajectory_id = _trajectory_id(sample)
    history: list[dict[str, str]] = []
    trajectory: list[dict[str, Any]] = []
    step_samples: list[Sample] = []
    assistant_responses: list[str] = []
    assistant_response_tokens: list[int] = []
    raw_reward = 0.0
    final_state = reset_state
    done = False
    aborted = False
    invalid_action_count = 0

    try:
        for step_id in range(max_steps):
            step_sampling_params = dict(sampling_params)
            step_sampling_params["max_new_tokens"] = min(
                int(step_sampling_params.get("max_new_tokens") or step_max_tokens),
                step_max_tokens,
            )
            prompt_text, prompt_tokens, prompt_history_used = _build_step_prompt(
                tokenizer=tokenizer,
                args=args,
                instruction_text=instruction_text,
                current_observation=current_observation,
                available_actions=available_actions,
                history=history,
                history_length=history_length,
                max_prompt_chars=max_prompt_chars,
            )
            output = await post(
                sglang_url,
                {
                    "input_ids": prompt_tokens,
                    "sampling_params": step_sampling_params,
                    "return_logprob": True,
                },
                headers=_router_headers(args, sample),
            )
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

            parsed = parse_action(response, available_actions)
            service_action = safe_action_for_service(parsed, available_actions, instruction_text)
            if not parsed.is_valid:
                invalid_action_count += 1

            final_state = await step_session(session_id=sample.session_id, action=service_action)
            raw_reward = float(final_state.get("reward", 0.0))
            done = bool(final_state.get("done", False))

            step_record = {
                "step": step_id + 1,
                "observation": current_observation,
                "model_response": response,
                "parsed_action": parsed.action,
                "service_action": service_action,
                "valid_action": parsed.is_valid,
                "invalid_reason": parsed.invalid_reason,
                "reward": raw_reward,
                "done": done,
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
                    "webshop_step": step_record,
                }
                step_samples.append(step_sample)

            if done:
                break

            history.append({"observation": current_observation, "action": service_action})
            current_observation = final_state["observation"]
            available_actions = final_state.get("available_actions", {})
    finally:
        if close_on_done:
            await close_session(sample.session_id)

    final_reward = max(0.0, raw_reward - invalid_action_penalty * invalid_action_count)
    episode_metadata = _episode_metadata(
        metadata=metadata,
        session_id=sample.session_id,
        instruction_text=instruction_text,
        trajectory=trajectory,
        final_state=final_state,
        raw_reward=raw_reward,
        final_reward=final_reward,
        invalid_action_count=invalid_action_count,
        invalid_action_penalty=invalid_action_penalty,
        done=done,
    )

    if evaluation:
        sample.response = "\n".join(assistant_responses)
        sample.tokens = assistant_response_tokens
        sample.response_length = len(assistant_response_tokens)
        sample.loss_mask = [1] * len(assistant_response_tokens)
        sample.reward = final_reward
        sample.status = Sample.Status.ABORTED if aborted else (Sample.Status.COMPLETED if done else Sample.Status.TRUNCATED)
        sample.metadata = episode_metadata
        return sample

    if not step_samples:
        return [_make_placeholder_sample(sample, tokenizer, episode_metadata, "empty_or_aborted_episode")]

    if any(step_sample.rollout_log_probs is None for step_sample in step_samples):
        for step_sample in step_samples:
            step_sample.rollout_log_probs = None

    for step_sample in step_samples:
        step_sample.reward = final_reward
        step_sample.metadata = {
            **episode_metadata,
            "episode_id": trajectory_id,
            "turn_step": step_sample.metadata["turn_step"],
            "webshop_step": step_sample.metadata["webshop_step"],
        }

    return step_samples


def _trajectory_units(group: Iterable[Sample | list[Sample]]) -> list[list[Sample]]:
    """Return compact trajectory units from a possibly fan-out sample group."""

    units = []
    for item in group:
        if isinstance(item, list):
            units.append(item)
        else:
            units.append([item])
    return units


def check_episode_reward_nonzero_std(args, samples: list[Sample] | list[list[Sample]], **kwargs) -> DynamicFilterOutput:
    """Dynamic filter for WebShop GRPO over step-level trajectories.

    ``generate`` returns one list of step samples per trajectory.  The default
    reward-std filter would see repeated step rewards from the same trajectory;
    for GRPO we need to compare the trajectory-level rewards among the
    ``n_samples_per_prompt`` siblings for the same prompt.
    """

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
        reason=None if keep else f"zero_std_{round(float(rewards[0]), 4)}",
    )


def grpo_normalize_webshop_steps(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Normalize GRPO rewards by trajectory and broadcast to step samples.

    All step samples from one WebShop episode share ``group_id`` and the final
    episode reward.  Normalize one reward per trajectory inside each original
    prompt group, then assign that normalized value back to every step from the
    same trajectory.  This prevents longer WebShop episodes from receiving more
    total advantage mass only because they produced more step samples.
    """

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
        args, "grpo_std_normalization", True
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


def _unique_trajectory_samples(samples: list[Sample]) -> list[Sample]:
    seen = set()
    unique = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        key = (
            sample.group_index,
            sample.group_id if sample.group_id is not None else sample.index,
            metadata.get("session_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def _webshop_summary_from_samples(samples: list[Sample]) -> dict[str, float]:
    trajectories = _unique_trajectory_samples(samples)
    if not trajectories:
        return {}

    raw_rewards = []
    final_rewards = []
    steps = []
    invalid_counts = []
    done_count = 0
    purchase_count = 0
    for sample in trajectories:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_rewards.append(float(metadata.get("raw_reward", sample.reward or 0.0)))
        final_rewards.append(float(metadata.get("final_reward", sample.reward or 0.0)))
        trajectory = metadata.get("trajectory") or []
        steps.append(float(len(trajectory)))
        invalid_counts.append(float(metadata.get("invalid_action_count", 0)))
        if metadata.get("done"):
            done_count += 1
        if float(metadata.get("raw_reward", sample.reward or 0.0)) > 0:
            purchase_count += 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "webshop/episodes": float(len(trajectories)),
        "webshop/raw_reward_mean": mean(raw_rewards),
        "webshop/final_reward_mean": mean(final_rewards),
        "webshop/success_rate": mean([1.0 if reward > 0 else 0.0 for reward in raw_rewards]),
        "webshop/done_rate": done_count / len(trajectories),
        "webshop/purchase_rate": purchase_count / len(trajectories),
        "webshop/avg_steps": mean(steps),
        "webshop/invalid_actions_per_episode": mean(invalid_counts),
    }


def log_webshop_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Attach WebShop episode-level metrics, then let slime do default logging."""

    metrics = _webshop_summary_from_samples(samples)
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(metrics)
    return False


def log_webshop_eval_rollout(rollout_id, args, data, extra_metrics) -> bool:
    """Log default eval metrics plus WebShop episode-level summaries.

    slime currently passes ``None`` for eval rollout extra metrics in the
    default SGLang rollout path, so mutating ``extra_metrics`` is not enough.
    This hook mirrors the default eval logger, adds WebShop summaries, logs the
    result, and returns True to skip duplicate default logging.
    """

    from slime.ray.rollout import compute_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, dict_add_prefix

    log_dict = extra_metrics or {}
    for dataset_name, payload in data.items():
        rewards = payload["rewards"]
        log_dict[f"eval/{dataset_name}"] = sum(rewards) / len(rewards) if rewards else 0.0
        samples = payload.get("samples") or []
        if samples:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{dataset_name}/")
            for key, value in _webshop_summary_from_samples(samples).items():
                log_dict[f"eval/{dataset_name}/{key}"] = value
        if "truncated" in payload:
            truncated = payload["truncated"]
            log_dict[f"eval/{dataset_name}-truncated_ratio"] = sum(truncated) / len(truncated) if truncated else 0.0
        if getattr(args, "log_passrate", False):
            log_dict |= dict_add_prefix(
                compute_pass_rate(flat_rewards=rewards, group_size=args.n_samples_per_eval_prompt),
                f"eval/{dataset_name}-",
            )

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logger.info("eval %s: %s", rollout_id, log_dict)
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
