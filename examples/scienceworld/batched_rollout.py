"""Batched ScienceWorld rollout function for slime GRPO training.

This module coordinates many active ScienceWorld episodes in one rollout
function. Model action requests are issued concurrently at each environment
step, while each mutable ScienceWorld Java environment lives in a Ray actor.
The returned training data uses the same step-level ``Sample`` contract as
``generate_with_scienceworld.generate``: every episode becomes a compact list of
assistant-turn samples that share a trajectory ``group_id`` and receive the same
final episode reward.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pybase64
import ray
from tqdm import tqdm

from generate_with_scienceworld import (
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_INVALID_ACTION_PENALTY,
    DEFAULT_MAX_STEPS,
    DEFAULT_PROMPT_CHAR_LIMIT,
    DEFAULT_STEP_MAX_TOKENS,
    _build_step_prompt,
    _final_episode_metadata,
    _get_float_env,
    _get_int_env,
    _get_metadata,
    _make_placeholder_step_sample,
    _response_tokens_and_logprobs,
    _router_headers,
    _safe_action_for_env,
    _task_from_metadata,
)
from prompts import parse_action
from scienceworld_env import ScienceWorldTextEpisode, StepResult

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
class ScienceWorldEpisodeWorker:
    """Ray actor that owns one mutable ScienceWorld episode."""

    def __init__(self) -> None:
        self._episode: ScienceWorldTextEpisode | None = None

    def reset(
        self,
        task_name: str,
        variation_idx: int,
        simplification: str,
        env_step_limit: int,
    ) -> StepResult:
        self.close()
        self._episode = ScienceWorldTextEpisode(env_step_limit=env_step_limit)
        self._episode.__enter__()
        return self._episode.reset(
            task_name=task_name,
            variation_idx=variation_idx,
            simplification=simplification,
        )

    def step(self, action: str) -> StepResult:
        if self._episode is None:
            raise RuntimeError("ScienceWorld episode worker has not been reset.")
        return self._episode.step(action)

    def close(self) -> None:
        if self._episode is not None:
            self._episode.close()
        self._episode = None


class ScienceWorldWorkerPool:
    """Bounded-lifetime actor pool for ScienceWorld Java environments."""

    def __init__(self) -> None:
        self._idle_workers: list[Any] = []
        self._episode_counts: dict[int, int] = {}
        self._created_count = 0

    @property
    def created_count(self) -> int:
        return self._created_count

    @property
    def idle_count(self) -> int:
        return len(self._idle_workers)

    @staticmethod
    def max_episodes_per_worker() -> int:
        return max(1, _get_int_env("SCIENCEWORLD_ENV_WORKER_MAX_EPISODES", 1))

    def _new_worker(self) -> Any:
        num_cpus = _get_float_env("SCIENCEWORLD_ENV_WORKER_CPUS", 0.25)
        self._created_count += 1
        return ScienceWorldEpisodeWorker.options(num_cpus=num_cpus, num_gpus=0).remote()

    def acquire(self, size: int) -> list[Any]:
        workers: list[Any] = []
        while len(workers) < size:
            worker = self._idle_workers.pop() if self._idle_workers else self._new_worker()
            self._episode_counts[id(worker)] = self._episode_counts.get(id(worker), 0) + 1
            workers.append(worker)
        return workers

    def release(self, workers: list[Any]) -> None:
        if not workers:
            return

        close_failed = False
        close_refs = []
        for worker in workers:
            try:
                close_refs.append(worker.close.remote())
            except Exception:
                close_failed = True
                logger.warning("Failed to schedule ScienceWorld worker close; retiring worker.", exc_info=True)

        if close_refs:
            try:
                ray.get(close_refs)
            except Exception:
                close_failed = True
                logger.warning(
                    "Failed to close one or more ScienceWorld workers; retiring this worker batch.",
                    exc_info=True,
                )

        max_episodes = self.max_episodes_per_worker()
        for worker in workers:
            count_key = id(worker)
            episode_count = self._episode_counts.get(count_key, 0)
            should_retire = close_failed or episode_count >= max_episodes
            if should_retire:
                self._episode_counts.pop(count_key, None)
                try:
                    ray.kill(worker, no_restart=True)
                except Exception:
                    logger.debug("Failed to kill retired ScienceWorld worker; it may already be dead.", exc_info=True)
            else:
                self._idle_workers.append(worker)


_WORKER_POOL = ScienceWorldWorkerPool()


@dataclass
class TrajectoryState:
    sample: Sample
    worker: Any
    metadata: dict[str, Any]
    trajectory_id: int
    evaluation: bool
    task_name: str
    variation_idx: int
    simplification: str
    env_step_limit: int
    task_description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    current_observation: str = ""
    inventory: str = ""
    admissible_actions: list[str] = field(default_factory=list)
    last_info: dict[str, Any] | None = None
    step_samples: list[Sample] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)
    assistant_response_tokens: list[int] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    invalid_action_count: int = 0
    total_delta_reward: float = 0.0
    final_score: float = 0.0
    done: bool = False
    completed: bool = False
    aborted: bool = False
    env_time: float = 0.0


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
    max_prompt_chars: int,
) -> dict[str, Any]:
    tokenizer = generate_state.tokenizer
    prompt_text, prompt_tokens, prompt_history_used = _build_step_prompt(
        tokenizer=tokenizer,
        args=args,
        current_observation=trajectory.current_observation,
        admissible_actions=trajectory.admissible_actions,
        task_description=trajectory.task_description,
        inventory=trajectory.inventory,
        history=trajectory.history,
        history_length=history_length,
        max_prompt_chars=max_prompt_chars,
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
    trajectory.assistant_response_tokens.extend(response_tokens)

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
) -> Sample | None:
    sample = trajectory.sample
    step_id = result["step_id"]
    parsed = result["parsed_action"]
    response_tokens = result["response_tokens"]

    trajectory.done = step_result.done
    trajectory.completed = step_result.completed
    trajectory.total_delta_reward += float(step_result.reward)
    trajectory.final_score = float(step_result.score)
    trajectory.last_info = step_result.info

    step_record = {
        "step": step_id + 1,
        "observation": trajectory.current_observation,
        "inventory": trajectory.inventory,
        "model_response": result["response"],
        "action": result["env_action"],
        "valid_action": parsed.is_valid,
        "invalid_reason": parsed.invalid_reason,
        "reward": step_result.reward,
        "score": step_result.score,
        "done": step_result.done,
        "completed": step_result.completed,
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
        step_sample.session_id = sample.session_id
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
            "scienceworld_step": step_record,
        }
        trajectory.step_samples.append(step_sample)
    else:
        step_sample = None

    if not step_result.done:
        trajectory.history.append({"observation": trajectory.current_observation, "action": result["env_action"]})
        trajectory.current_observation = step_result.observation
        trajectory.inventory = str(step_result.info.get("inv") or "")
        trajectory.admissible_actions = step_result.admissible_actions

    return step_sample


def _finalize_trajectory(
    generate_state: GenerateState,
    trajectory: TrajectoryState,
    invalid_action_penalty: float,
) -> Sample | list[Sample]:
    final_reward = float(trajectory.final_score) - invalid_action_penalty * trajectory.invalid_action_count
    episode_metadata = _final_episode_metadata(
        metadata=trajectory.metadata,
        evaluation=trajectory.evaluation,
        done=trajectory.done,
        completed=trajectory.completed,
        trajectory=trajectory.trajectory,
        invalid_action_count=trajectory.invalid_action_count,
        final_score=trajectory.final_score,
        total_delta_reward=trajectory.total_delta_reward,
        invalid_action_penalty=invalid_action_penalty,
        last_info=trajectory.last_info,
    )

    if trajectory.evaluation:
        sample = trajectory.sample
        sample.response = "\n".join(trajectory.assistant_responses)
        sample.tokens = list(trajectory.assistant_response_tokens)
        sample.response_length = len(trajectory.assistant_response_tokens)
        sample.loss_mask = [1] * sample.response_length
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

    # Keep the rollout-logprob field all-or-nothing. The train-data converter
    # decides whether to include this column by checking the first sample.
    if any(step_sample.rollout_log_probs is None for step_sample in trajectory.step_samples):
        for step_sample in trajectory.step_samples:
            step_sample.rollout_log_probs = None

    for step_sample in trajectory.step_samples:
        step_sample.reward = final_reward
        step_sample.metadata = {
            **episode_metadata,
            "episode_id": trajectory.trajectory_id,
            "turn_step": step_sample.metadata["turn_step"],
            "scienceworld_step": step_sample.metadata["scienceworld_step"],
        }

    return trajectory.step_samples


async def _run_batched_episodes(
    args: Namespace,
    samples: list[Sample],
    base_sampling_params: dict[str, Any],
    *,
    evaluation: bool,
) -> list[Sample | list[Sample]]:
    assert not args.partial_rollout, "Partial rollout is not supported for batched ScienceWorld episodes."

    generate_state = GenerateState(args)
    max_steps = _get_int_env("SCIENCEWORLD_MAX_STEPS", DEFAULT_MAX_STEPS)
    history_length = _get_int_env("SCIENCEWORLD_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)
    step_max_tokens = _get_int_env("SCIENCEWORLD_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)
    invalid_action_penalty = _get_float_env("SCIENCEWORLD_INVALID_ACTION_PENALTY", DEFAULT_INVALID_ACTION_PENALTY)
    max_prompt_chars = _get_int_env("SCIENCEWORLD_MAX_PROMPT_CHARS", DEFAULT_PROMPT_CHAR_LIMIT)

    workers: list[Any] = []
    trajectories: list[TrajectoryState] = []
    try:
        workers = _WORKER_POOL.acquire(len(samples))
        reset_refs = []
        for worker, sample in zip(workers, samples, strict=True):
            metadata = _get_metadata(sample)
            if sample.session_id is None:
                sample.session_id = str(uuid.uuid4())
            task_name, variation_idx, simplification, env_step_limit = _task_from_metadata(metadata)
            trajectory = TrajectoryState(
                sample=sample,
                worker=worker,
                metadata=metadata,
                trajectory_id=_trajectory_id(sample),
                evaluation=evaluation,
                task_name=task_name,
                variation_idx=variation_idx,
                simplification=simplification,
                env_step_limit=env_step_limit,
            )
            trajectories.append(trajectory)
            reset_refs.append(worker.reset.remote(task_name, variation_idx, simplification, env_step_limit))

        reset_start = time.perf_counter()
        reset_results = ray.get(reset_refs)
        reset_time = time.perf_counter() - reset_start
        for trajectory, reset_result in zip(trajectories, reset_results, strict=True):
            trajectory.env_time += reset_time / max(1, len(trajectories))
            trajectory.task_description = str(reset_result.info.get("taskDesc") or "")
            trajectory.current_observation = reset_result.observation
            trajectory.inventory = str(reset_result.info.get("inv") or "")
            trajectory.admissible_actions = reset_result.admissible_actions
            trajectory.final_score = float(reset_result.score)
            trajectory.total_delta_reward += float(reset_result.reward)
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
                        max_prompt_chars=max_prompt_chars,
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
    finally:
        _WORKER_POOL.release(workers)


def _first_sample(group: list[Sample | list[Sample]]) -> Sample:
    first = group[0]
    return first[0] if isinstance(first, list) else first


def _flatten_prompt_groups(prompt_groups: list[list[Sample]]) -> list[Sample]:
    return [sample for group in prompt_groups for sample in group]


def _episode_samples_from_groups(groups: list[list[Sample | list[Sample]]]) -> list[Sample]:
    samples: list[Sample] = []
    for group in groups:
        for item in group:
            samples.append(item[0] if isinstance(item, list) else item)
    return samples


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
        return bool(numeric_value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
    return None


def _metric_key_part(value: Any) -> str:
    text = str(value).strip() if value is not None else "unknown"
    if not text:
        return "unknown"
    return "_".join(text.replace("/", " ").split()) or "unknown"


def _add_scienceworld_metrics(metrics: dict[str, Any], prefix: str, samples: list[Sample]) -> None:
    """Add episode-level ScienceWorld score/completion metrics."""
    episode_count = 0
    completed_count = 0
    missing_count = 0
    score_sum = 0.0
    invalid_action_sum = 0
    step_sum = 0
    by_task: dict[str, dict[str, float]] = {}

    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            missing_count += 1
            continue

        scienceworld_metadata = metadata.get("scienceworld")
        if not isinstance(scienceworld_metadata, dict):
            missing_count += 1
            continue

        score = _coerce_float(scienceworld_metadata.get("final_score"))
        completed = _coerce_bool(scienceworld_metadata.get("completed"))
        if score is None or completed is None:
            missing_count += 1
            continue

        steps = int(scienceworld_metadata.get("steps") or 0)
        invalid_actions = int(scienceworld_metadata.get("invalid_action_count") or 0)
        task_name = _metric_key_part(metadata.get("task_name"))
        bucket = by_task.setdefault(
            task_name,
            {"episode_count": 0.0, "completed_count": 0.0, "score_sum": 0.0},
        )

        episode_count += 1
        score_sum += score
        invalid_action_sum += invalid_actions
        step_sum += steps
        bucket["episode_count"] += 1
        bucket["score_sum"] += score
        if completed:
            completed_count += 1
            bucket["completed_count"] += 1

    metrics[f"{prefix}/scienceworld/sample_count"] = len(samples)
    metrics[f"{prefix}/scienceworld/episode_count"] = episode_count
    metrics[f"{prefix}/scienceworld/completed_count"] = completed_count
    metrics[f"{prefix}/scienceworld/missing_metadata_count"] = missing_count
    if episode_count > 0:
        metrics[f"{prefix}/scienceworld/avg_score"] = score_sum / episode_count
        metrics[f"{prefix}/scienceworld/completion_rate"] = completed_count / episode_count
        metrics[f"{prefix}/scienceworld/avg_invalid_actions"] = invalid_action_sum / episode_count
        metrics[f"{prefix}/scienceworld/avg_steps"] = step_sum / episode_count

    for task_name, counts in sorted(by_task.items()):
        task_episode_count = counts["episode_count"]
        task_prefix = f"{prefix}/scienceworld/task/{task_name}"
        metrics[f"{task_prefix}/episode_count"] = task_episode_count
        metrics[f"{task_prefix}/avg_score"] = counts["score_sum"] / task_episode_count
        metrics[f"{task_prefix}/completion_rate"] = counts["completed_count"] / task_episode_count


async def _generate_train_rollout(args: Namespace, rollout_id: int, data_source: Any) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    generate_state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    data: list[list[Sample | list[Sample]]] = []
    all_data: list[list[Sample | list[Sample]]] = []
    do_print = True
    pbar = tqdm(total=args.rollout_batch_size * args.n_samples_per_prompt, desc="Batched ScienceWorld rollout")
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
                    "First batched ScienceWorld rollout sample: %s label=%s reward=%s",
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
        "Finish batched ScienceWorld rollout: %s label=%s reward=%s",
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
    metrics["rollout/scienceworld_batched_episodes"] = sum(len(group) for group in data)
    metrics["rollout/scienceworld_worker_pool_size"] = _WORKER_POOL.idle_count
    metrics["rollout/scienceworld_worker_processes_created"] = _WORKER_POOL.created_count
    metrics["rollout/scienceworld_worker_max_episodes"] = _WORKER_POOL.max_episodes_per_worker()
    metrics["rollout/scienceworld_collect_time"] = time.perf_counter() - collect_start
    _add_scienceworld_metrics(metrics, "rollout", _episode_samples_from_groups(data))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


async def _generate_eval_rollout(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    assert not args.group_rm, "Group RM is not supported for eval rollout."

    generate_state = GenerateState(args)
    results: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    all_eval_samples: list[Sample] = []
    reward_key = args.eval_reward_key or args.reward_key
    eval_batch_size = _get_int_env(
        "SCIENCEWORLD_EVAL_BATCH_SIZE",
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
        _add_scienceworld_metrics(metrics, f"eval/{dataset_cfg.name}", data)
        all_eval_samples.extend(data)
        if data:
            logger.info(
                "Batched ScienceWorld eval %s example: %s reward=%s",
                dataset_cfg.name,
                [str(data[0].prompt) + data[0].response],
                data[0].reward,
            )
        else:
            logger.warning(
                "Batched ScienceWorld eval %s produced no samples; skipping default reward metrics for this dataset.",
                dataset_cfg.name,
            )
            continue
        results[dataset_cfg.name] = {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }

    _add_scienceworld_metrics(metrics, "eval", all_eval_samples)
    return RolloutFnEvalOutput(data=results, metrics=metrics)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        return run(_generate_eval_rollout(args, rollout_id))
    return run(_generate_train_rollout(args, rollout_id, data_source))
