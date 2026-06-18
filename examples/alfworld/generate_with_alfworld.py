"""ALFWorld custom generation function for slime GRPO training.

The training path intentionally mirrors SDAR/verl-agent: one ALFWorld
trajectory is generated online, but it is returned as multiple step-level
``Sample`` objects.  Each step sample contains only the prompt/response for one
model action, while all step samples from the same trajectory share the final
episode reward and ``group_id``.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections import defaultdict
from typing import Any, Iterable

from alfworld_env import AlfWorldTextEpisode
from prompts import ALFWORLD_SYSTEM_PROMPT, _task_description_from_reset, build_observation_prompt, parse_action

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
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



def _build_step_prompt(
    *,
    tokenizer,
    args: Namespace,
    current_observation: str,
    admissible_actions: list[str],
    task_description: str,
    history: list[dict[str, str]],
    history_length: int,
) -> tuple[str, list[int], int]:
    """Build the current ALFWorld step prompt with bounded recent history."""
    keep_history = min(history_length, len(history)) if history_length > 0 else 0
    prompt_history = history[-keep_history:] if keep_history else []
    messages = [
        {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_observation_prompt(
                current_observation=current_observation,
                admissible_actions=admissible_actions,
                task_description=task_description,
                history=prompt_history,
                history_length=keep_history,
            ),
        },
    ]
    prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, args=args)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, prompt_tokens, keep_history


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
    placeholder.metadata = {**metadata, "alfworld": {"aborted_reason": reason}}
    return placeholder


def _final_episode_metadata(
    *,
    metadata: dict[str, Any],
    evaluation: bool,
    done: bool,
    won: bool,
    trajectory: list[dict[str, Any]],
    invalid_action_count: int,
    total_reward: float,
    invalid_action_penalty: float,
    last_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **metadata,
        "raw_reward": float(total_reward),
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


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one ALFWorld episode.

    Training returns step-level samples so both generation and training context
    lengths stay bounded.  Evaluation returns one episode-level summary sample so
    eval metrics are counted per trajectory rather than per step.
    """
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
    trajectory_id = sample.index if sample.index is not None else sample.group_index

    step_samples: list[Sample] = []
    assistant_responses: list[str] = []
    trajectory: list[dict[str, Any]] = []
    invalid_action_count = 0
    total_reward = 0.0
    done = False
    won = False
    aborted = False
    last_info: dict[str, Any] | None = None

    with AlfWorldTextEpisode(config_path, split=env_split, gamefile=gamefile, seed=seed) as episode:
        reset_result = episode.reset()
        task_description = _task_description_from_reset(reset_result.observation)
        history: list[dict[str, str]] = []
        current_observation = reset_result.observation
        admissible_actions = reset_result.admissible_actions
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
                history=history,
                history_length=history_length,
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

            parsed = parse_action(response, admissible_actions)
            env_action = _safe_action_for_env(parsed.action, admissible_actions)
            if not parsed.is_valid:
                invalid_action_count += 1

            step_result = episode.step(env_action)
            done = step_result.done
            won = step_result.won
            total_reward += float(step_result.reward)
            last_info = step_result.info

            step_record = {
                "step": step_id + 1,
                "observation": current_observation,
                "model_response": response,
                "action": env_action,
                "valid_action": parsed.is_valid,
                "invalid_reason": parsed.invalid_reason,
                "reward": step_result.reward,
                "done": done,
                "won": won,
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
                step_sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
                if response_log_probs is not None and len(response_log_probs) == len(response_tokens):
                    step_sample.rollout_log_probs = response_log_probs
                step_sample.metadata = {
                    **metadata,
                    "episode_id": trajectory_id,
                    "turn_step": step_id,
                    "alfworld_step": step_record,
                }
                step_samples.append(step_sample)

            if done:
                break

            history.append({"observation": current_observation, "action": env_action})
            current_observation = step_result.observation
            admissible_actions = step_result.admissible_actions

    final_reward = float(total_reward) - invalid_action_penalty * invalid_action_count
    episode_metadata = _final_episode_metadata(
        metadata=metadata,
        evaluation=evaluation,
        done=done,
        won=won,
        trajectory=trajectory,
        invalid_action_count=invalid_action_count,
        total_reward=total_reward,
        invalid_action_penalty=invalid_action_penalty,
        last_info=last_info,
    )

    if evaluation:
        # TODO: Populate response_length/loss_mask/tokens for eval summaries so
        # eval response_len metrics reflect generated ALFWorld action tokens.
        sample.response = "\n".join(assistant_responses)
        sample.reward = final_reward
        sample.status = Sample.Status.ABORTED if aborted else (Sample.Status.COMPLETED if done else Sample.Status.TRUNCATED)
        sample.metadata = episode_metadata
        return sample

    if not step_samples:
        reason = "sglang_abort" if aborted else "empty_episode_response"
        return [_make_placeholder_step_sample(sample, tokenizer, episode_metadata, reason)]

    # Keep the rollout-logprob field all-or-nothing.  The train-data converter
    # decides whether to include this column by checking the first sample, so a
    # rare mixed SGLang response would otherwise create a partially-None column.
    if any(step_sample.rollout_log_probs is None for step_sample in step_samples):
        for step_sample in step_samples:
            step_sample.rollout_log_probs = None

    for step_sample in step_samples:
        step_sample.reward = final_reward
        step_sample.metadata = {
            **episode_metadata,
            "episode_id": trajectory_id,
            "turn_step": step_sample.metadata["turn_step"],
            "alfworld_step": step_sample.metadata["alfworld_step"],
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
    """Dynamic filter that treats each compact list as one trajectory.

    The default filter sees step-level fanout as many samples and would compute
    std over repeated step rewards.  For ALFWorld+GRPO we need std over the n
    trajectory rewards for the same original prompt.
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
        reason=None if keep else f"zero_std_{round(float(rewards[0]), 1)}",
    )


def grpo_normalize_alfworld_steps(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Normalize GRPO rewards by trajectory, then broadcast to step samples.

    ``samples`` is already flattened by slime.  Multiple step samples from one
    trajectory share ``group_id`` and reward.  We first compute one reward per
    trajectory inside each original prompt group (``group_index``), normalize
    those n trajectory rewards, then assign the normalized value back to every
    step sample from that trajectory.
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
