"""Batched ALFWorld rollout function for slime GRPO training.

This module replaces the per-sample ``custom_generate`` path for ALFWorld.
It keeps the same step-level training ``Sample`` contract as
``generate_with_alfworld.generate``, but coordinates many active episodes in a
single rollout function so model requests are issued in step batches and
ALFWorld environment steps run inside Ray actors.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pybase64
import ray
from tqdm import tqdm

from alfworld_env import AlfWorldTextEpisode, StepResult
from generate_with_alfworld import (
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_INVALID_ACTION_PENALTY,
    DEFAULT_MAX_STEPS,
    DEFAULT_STEP_MAX_TOKENS,
    _build_step_prompt,
    _final_episode_metadata,
    _get_env_config_path,
    _get_float_env,
    _get_int_env,
    _get_metadata,
    _make_placeholder_step_sample,
    _response_tokens_and_logprobs,
    _router_headers,
    _safe_action_for_env,
)
from prompts import extract_task, parse_action

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.http_utils import post
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@ray.remote
class AlfWorldEpisodeWorker:
    """Ray actor that owns one mutable ALFWorld episode."""

    def __init__(self) -> None:
        self._episode: AlfWorldTextEpisode | None = None

    def reset(self, config_path: str, split: str, gamefile: str | None, seed: int) -> StepResult:
        self.close()
        self._episode = AlfWorldTextEpisode(config_path, split=split, gamefile=gamefile, seed=seed)
        self._episode.__enter__()
        return self._episode.reset()

    def step(self, action: str) -> StepResult:
        if self._episode is None:
            raise RuntimeError("ALFWorld episode worker has not been reset.")
        return self._episode.step(action)

    def close(self) -> None:
        if self._episode is not None:
            self._episode.close()
        self._episode = None


class AlfWorldWorkerPool:
    """Grow-only actor pool reused by the rollout manager process."""

    def __init__(self) -> None:
        self._workers: list[Any] = []

    def ensure(self, size: int) -> list[Any]:
        while len(self._workers) < size:
            num_cpus = _get_float_env("ALFWORLD_ENV_WORKER_CPUS", 0.1)
            self._workers.append(AlfWorldEpisodeWorker.options(num_cpus=num_cpus, num_gpus=0).remote())
        return self._workers[:size]


_WORKER_POOL = AlfWorldWorkerPool()


@dataclass
class TrajectoryState:
    sample: Sample
    worker: Any
    metadata: dict[str, Any]
    trajectory_id: int
    evaluation: bool
    task_description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    current_observation: str = ""
    admissible_actions: list[str] = field(default_factory=list)
    last_info: dict[str, Any] | None = None
    step_samples: list[Sample] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    invalid_action_count: int = 0
    total_reward: float = 0.0
    done: bool = False
    won: bool = False
    aborted: bool = False
    env_time: float = 0.0


def _env_split_from_metadata(metadata: dict[str, Any]) -> str:
    split = metadata.get("split", "train")
    if split == "valid_seen":
        return "eval_in_distribution"
    if split == "valid_unseen":
        return "eval_out_of_distribution"
    return "train"


def _trajectory_id(sample: Sample) -> int:
    if sample.index is not None:
        return int(sample.index)
    if sample.group_index is not None:
        return int(sample.group_index)
    return 0


def _deterministic_seed(args: Namespace, sample: Sample) -> int:
    offset = int(sample.index or 0) % max(1, int(getattr(args, "n_samples_per_prompt", 1)))
    return int(getattr(args, "rollout_seed", 0)) + offset


def _step_sampling_params(
    args: Namespace,
    base_sampling_params: dict[str, Any],
    step_max_tokens: int,
    sample: Sample,
) -> dict[str, Any]:
    params = dict(base_sampling_params)
    max_new_tokens = params.get("max_new_tokens") or step_max_tokens
    params["max_new_tokens"] = min(int(max_new_tokens), step_max_tokens)
    if getattr(args, "sglang_enable_deterministic_inference", False):
        params["sampling_seed"] = _deterministic_seed(args, sample)
    return params


async def _generate_one_step(
    *,
    args: Namespace,
    generate_state: GenerateState,
    trajectory: TrajectoryState,
    base_sampling_params: dict[str, Any],
    step_id: int,
    history_length: int,
    step_max_tokens: int,
) -> dict[str, Any]:
    tokenizer = generate_state.tokenizer
    prompt_text, prompt_tokens, prompt_history_used = _build_step_prompt(
        tokenizer=tokenizer,
        args=args,
        current_observation=trajectory.current_observation,
        admissible_actions=trajectory.admissible_actions,
        task_description=trajectory.task_description,
        history=trajectory.history,
        history_length=history_length,
    )
    payload = {
        "input_ids": prompt_tokens,
        "sampling_params": _step_sampling_params(args, base_sampling_params, step_max_tokens, trajectory.sample),
        "return_logprob": True,
    }
    if getattr(args, "use_rollout_routing_replay", False):
        payload["return_routed_experts"] = True

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with generate_state.semaphore:
        with generate_state.dp_rank_context():
            output = await post(url, payload, headers=_router_headers(args, trajectory.sample))

    return {
        "trajectory": trajectory,
        "step_id": step_id,
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "prompt_history_used": prompt_history_used,
        "output": output,
    }


def _apply_generation_result(args: Namespace, generate_state: GenerateState, result: dict[str, Any]) -> str | None:
    trajectory: TrajectoryState = result["trajectory"]
    sample = trajectory.sample
    output = result["output"]
    meta_info = output.get("meta_info", {})
    finish_reason = meta_info.get("finish_reason") or {}
    if finish_reason:
        sample.update_from_meta_info(args, meta_info)

    finish_type = finish_reason.get("type")
    if finish_type == "abort":
        trajectory.aborted = True
        return None

    response = output.get("text", "")
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")]

    response_tokens, response_log_probs = _response_tokens_and_logprobs(generate_state.tokenizer, response, meta_info)
    trajectory.assistant_responses.append(response)

    parsed = parse_action(response, trajectory.admissible_actions)
    env_action = _safe_action_for_env(parsed.action, trajectory.admissible_actions)
    if not parsed.is_valid:
        trajectory.invalid_action_count += 1

    result["response"] = response
    result["response_tokens"] = response_tokens
    result["response_log_probs"] = response_log_probs
    result["meta_info"] = meta_info
    result["parsed_action"] = parsed
    result["env_action"] = env_action
    result["finish_type"] = finish_type
    return env_action


def _record_step_sample(
    args: Namespace,
    trajectory: TrajectoryState,
    result: dict[str, Any],
    step_result: StepResult,
    max_steps: int,
) -> None:
    sample = trajectory.sample
    step_id = result["step_id"]
    parsed = result["parsed_action"]
    response_tokens = result["response_tokens"]

    trajectory.done = step_result.done
    trajectory.won = step_result.won
    trajectory.total_reward += float(step_result.reward)
    trajectory.last_info = step_result.info

    step_record = {
        "step": step_id + 1,
        "observation": trajectory.current_observation,
        "model_response": result["response"],
        "action": result["env_action"],
        "valid_action": parsed.is_valid,
        "invalid_reason": parsed.invalid_reason,
        "reward": step_result.reward,
        "done": step_result.done,
        "won": step_result.won,
        "prompt_tokens": len(result["prompt_tokens"]),
        "response_tokens": len(response_tokens),
        "history_used": result["prompt_history_used"],
    }
    trajectory.trajectory.append(step_record)

    if response_tokens:
        step_sample = Sample()
        step_sample.index = int(sample.index or 0) * (max_steps + 1) + step_id
        step_sample.group_index = sample.group_index
        step_sample.group_id = trajectory.trajectory_id
        step_sample.prompt = result["prompt_text"]
        step_sample.tokens = result["prompt_tokens"] + response_tokens
        step_sample.response = result["response"]
        step_sample.response_length = len(response_tokens)
        step_sample.loss_mask = [1] * len(response_tokens)
        step_sample.status = Sample.Status.TRUNCATED if result["finish_type"] == "length" else Sample.Status.COMPLETED
        if result["response_log_probs"] is not None and len(result["response_log_probs"]) == len(response_tokens):
            step_sample.rollout_log_probs = result["response_log_probs"]
        meta_info = result["meta_info"]
        if "routed_experts" in meta_info:
            step_sample.rollout_routed_experts = np.frombuffer(
                pybase64.b64decode(meta_info["routed_experts"].encode("ascii")),
                dtype=np.int32,
            ).reshape(
                len(step_sample.tokens) - 1,
                args.num_layers,
                args.moe_router_topk,
            )
        step_sample.non_generation_time = result.get("non_generation_time", 0.0)
        step_sample.metadata = {
            **trajectory.metadata,
            "episode_id": trajectory.trajectory_id,
            "turn_step": step_id,
            "alfworld_step": step_record,
        }
        trajectory.step_samples.append(step_sample)

    if not step_result.done:
        trajectory.history.append({"observation": trajectory.current_observation, "action": result["env_action"]})
        trajectory.current_observation = step_result.observation
        trajectory.admissible_actions = step_result.admissible_actions


def _finalize_trajectory(
    generate_state: GenerateState,
    trajectory: TrajectoryState,
    invalid_action_penalty: float,
) -> Sample | list[Sample]:
    final_reward = float(trajectory.total_reward) - invalid_action_penalty * trajectory.invalid_action_count
    episode_metadata = _final_episode_metadata(
        metadata=trajectory.metadata,
        evaluation=trajectory.evaluation,
        done=trajectory.done,
        won=trajectory.won,
        trajectory=trajectory.trajectory,
        invalid_action_count=trajectory.invalid_action_count,
        total_reward=trajectory.total_reward,
        invalid_action_penalty=invalid_action_penalty,
        last_info=trajectory.last_info,
    )

    if trajectory.evaluation:
        sample = trajectory.sample
        sample.response = "\n".join(trajectory.assistant_responses)
        sample.response_length = sum(step["response_tokens"] for step in trajectory.trajectory)
        sample.reward = final_reward
        sample.status = (
            Sample.Status.ABORTED
            if trajectory.aborted
            else (Sample.Status.COMPLETED if trajectory.done else Sample.Status.TRUNCATED)
        )
        sample.non_generation_time = trajectory.env_time
        sample.metadata = episode_metadata
        return sample

    if not trajectory.step_samples:
        reason = "sglang_abort" if trajectory.aborted else "empty_episode_response"
        return [_make_placeholder_step_sample(trajectory.sample, generate_state.tokenizer, episode_metadata, reason)]

    if any(step_sample.rollout_log_probs is None for step_sample in trajectory.step_samples):
        for step_sample in trajectory.step_samples:
            step_sample.rollout_log_probs = None

    for step_sample in trajectory.step_samples:
        step_sample.reward = final_reward
        step_sample.metadata = {
            **episode_metadata,
            "episode_id": trajectory.trajectory_id,
            "turn_step": step_sample.metadata["turn_step"],
            "alfworld_step": step_sample.metadata["alfworld_step"],
        }

    return trajectory.step_samples


async def _run_batched_episodes(
    args: Namespace,
    samples: list[Sample],
    base_sampling_params: dict[str, Any],
    *,
    evaluation: bool,
) -> list[Sample | list[Sample]]:
    assert not args.partial_rollout, "Partial rollout is not supported for batched ALFWorld episodes."

    generate_state = GenerateState(args)
    max_steps = _get_int_env("ALFWORLD_MAX_STEPS", DEFAULT_MAX_STEPS)
    history_length = _get_int_env("ALFWORLD_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)
    step_max_tokens = _get_int_env("ALFWORLD_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)
    invalid_action_penalty = _get_float_env("ALFWORLD_INVALID_ACTION_PENALTY", DEFAULT_INVALID_ACTION_PENALTY)
    config_path = _get_env_config_path()

    workers = _WORKER_POOL.ensure(len(samples))
    trajectories: list[TrajectoryState] = []
    reset_refs = []
    for worker, sample in zip(workers, samples, strict=True):
        metadata = _get_metadata(sample)
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())
        trajectory_id = _trajectory_id(sample)
        trajectories.append(
            TrajectoryState(
                sample=sample,
                worker=worker,
                metadata=metadata,
                trajectory_id=trajectory_id,
                evaluation=evaluation,
            )
        )
        reset_refs.append(
            worker.reset.remote(
                config_path,
                _env_split_from_metadata(metadata),
                metadata.get("gamefile"),
                int(getattr(args, "rollout_seed", 0)) + int(sample.index or 0),
            )
        )

    reset_start = time.perf_counter()
    reset_results = ray.get(reset_refs)
    reset_time = time.perf_counter() - reset_start
    for trajectory, reset_result in zip(trajectories, reset_results, strict=True):
        trajectory.env_time += reset_time / max(1, len(trajectories))
        trajectory.task_description = trajectory.metadata.get("task_type") or extract_task(reset_result.observation)
        trajectory.current_observation = reset_result.observation
        trajectory.admissible_actions = reset_result.admissible_actions
        trajectory.last_info = reset_result.info

    for step_id in range(max_steps):
        active = [trajectory for trajectory in trajectories if not trajectory.done and not trajectory.aborted]
        if not active:
            break

        generation_results = await asyncio.gather(
            *[
                _generate_one_step(
                    args=args,
                    generate_state=generate_state,
                    trajectory=trajectory,
                    base_sampling_params=base_sampling_params,
                    step_id=step_id,
                    history_length=history_length,
                    step_max_tokens=step_max_tokens,
                )
                for trajectory in active
            ]
        )

        step_items = []
        step_refs = []
        for result in generation_results:
            trajectory = result["trajectory"]
            env_action = _apply_generation_result(args, generate_state, result)
            if env_action is None:
                continue
            step_items.append(result)
            step_refs.append(trajectory.worker.step.remote(env_action))

        if not step_refs:
            continue

        step_start = time.perf_counter()
        step_results = ray.get(step_refs)
        step_time = time.perf_counter() - step_start
        per_trajectory_step_time = step_time / max(1, len(step_items))
        for result, step_result in zip(step_items, step_results, strict=True):
            trajectory = result["trajectory"]
            previous_env_time = trajectory.env_time
            trajectory.env_time += per_trajectory_step_time
            result["non_generation_time"] = (
                trajectory.env_time if result["step_id"] == 0 else trajectory.env_time - previous_env_time
            )
            _record_step_sample(args, trajectory, result, step_result, max_steps)

    return [_finalize_trajectory(generate_state, trajectory, invalid_action_penalty) for trajectory in trajectories]


def _first_sample(group: list[Sample | list[Sample]]) -> Sample:
    first = group[0]
    return first[0] if isinstance(first, list) else first


def _flatten_prompt_groups(prompt_groups: list[list[Sample]]) -> list[Sample]:
    return [sample for group in prompt_groups for sample in group]


async def _generate_train_rollout(args: Namespace, rollout_id: int, data_source: Any) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    generate_state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    data: list[list[list[Sample]]] = []
    all_data: list[list[list[Sample]]] = []
    do_print = True
    pbar = tqdm(total=args.rollout_batch_size * args.n_samples_per_prompt, desc="Batched ALFWorld rollout")
    collect_start = time.perf_counter()

    while len(data) < args.rollout_batch_size:
        needed = args.rollout_batch_size - len(data)
        prompt_groups = data_source.get_samples(max(needed, args.over_sampling_batch_size))
        flat_samples = _flatten_prompt_groups(prompt_groups)
        flat_results = await _run_batched_episodes(
            args,
            flat_samples,
            generate_state.sampling_params.copy(),
            evaluation=False,
        )

        offset = 0
        for prompt_group in prompt_groups:
            trajectory_count = len(prompt_group)
            group = flat_results[offset : offset + trajectory_count]
            offset += trajectory_count
            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)

            if do_print:
                sample = _first_sample(group)
                logger.info(
                    "First batched ALFWorld rollout sample: %s label=%s reward=%s",
                    [str(sample.prompt) + sample.response],
                    str(sample.label)[:100],
                    sample.reward,
                )
                do_print = False

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            if len(data) < args.rollout_batch_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = _first_sample(data[-1])
    logger.info(
        "Finish batched ALFWorld rollout: %s label=%s reward=%s",
        [str(sample.prompt) + sample.response],
        str(sample.label)[:100],
        sample.reward,
    )

    data = sorted(data, key=lambda group: _first_sample(group).index)
    all_data = sorted(all_data, key=lambda group: _first_sample(group).index)

    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_data, data_source)

    metrics = metric_gatherer.collect()
    metrics["rollout/alfworld_batched_episodes"] = sum(len(group) for group in data)
    metrics["rollout/alfworld_worker_pool_size"] = len(_WORKER_POOL._workers)
    metrics["rollout/alfworld_collect_time"] = time.perf_counter() - collect_start
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


async def _generate_eval_rollout(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    assert not args.group_rm, "Group RM is not supported for eval rollout."

    generate_state = GenerateState(args)
    results: dict[str, dict[str, Any]] = {}
    reward_key = args.eval_reward_key or args.reward_key
    eval_batch_size = _get_int_env(
        "ALFWORLD_EVAL_BATCH_SIZE",
        max(1, args.rollout_batch_size * args.n_samples_per_prompt),
    )

    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        dataset = Dataset(
            path=dataset_cfg.path,
            tokenizer=generate_state.tokenizer,
            processor=generate_state.processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
        base_sampling_params = dict(
            temperature=dataset_cfg.temperature,
            top_p=dataset_cfg.top_p,
            top_k=dataset_cfg.top_k,
            max_new_tokens=dataset_cfg.max_response_len,
            stop=dataset_cfg.stop or args.rollout_stop,
            stop_token_ids=dataset_cfg.stop_token_ids or args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        samples: list[Sample] = []
        sample_index = 0
        for prompt_sample in dataset.samples:
            for j in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample.group_index = sample_index // max(1, dataset_cfg.n_samples_per_eval_prompt)
                sample_index += 1
                sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
                sample.metadata["eval_sample_offset"] = j
                samples.append(sample)

        data: list[Sample] = []
        pbar = tqdm(total=len(samples), desc=f"Eval {dataset_cfg.name}")
        for start in range(0, len(samples), eval_batch_size):
            chunk = samples[start : start + eval_batch_size]
            chunk_results = await _run_batched_episodes(args, chunk, base_sampling_params, evaluation=True)
            data.extend(chunk_results)  # type: ignore[arg-type]
            pbar.update(len(chunk))
        pbar.close()

        data.sort(key=lambda sample: sample.index)
        if data:
            logger.info(
                "Batched ALFWorld eval %s example: %s reward=%s",
                dataset_cfg.name,
                [str(data[0].prompt) + data[0].response],
                data[0].reward,
            )
        results[dataset_cfg.name] = {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }

    # TODO: Aggregate ALFWorld eval success rates from sample.metadata["alfworld"]["won"]
    # and metadata["task_type"], then return them via RolloutFnEvalOutput.metrics so
    # SwanLab can show SDAR-style per-task success-rate curves.
    return RolloutFnEvalOutput(data=results)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        return run(_generate_eval_rollout(args, rollout_id))
    return run(_generate_train_rollout(args, rollout_id, data_source))
