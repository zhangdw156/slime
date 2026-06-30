#!/usr/bin/env python3
"""Collect ALFWorld teacher trajectories for analysis and offline SFT distillation.

This collector mirrors the batched ALFWorld rollout structure used during
training: sample rows are expanded into repeated sampling tasks, each active
task owns one Ray-managed ALFWorld environment, and every batch advances those
environments step-by-step while sending concurrent requests to an already
running SGLang teacher.

The collector writes every completed sampling attempt to one append-only JSONL
ledger, regardless of whether the teacher wins, fails, truncates, aborts, or
hits an error. Filtering/selection for SFT is handled by
``build_sft_from_teacher_trajectories.py`` rather than by the collector.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


from prompts import ALFWORLD_SYSTEM_PROMPT, _task_description_from_reset, build_observation_prompt, parse_action

if TYPE_CHECKING:
    import httpx

    from alfworld_env import AlfWorldTextEpisode, StepResult

try:  # pragma: no cover - optional cosmetic dependency in minimal local checks.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 50
DEFAULT_SAMPLES_PER_TASK = 8
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_STEP_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT = 0.0
DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_SLEEP = 1.0

_RAY: Any | None = None
_REMOTE_ENV_WORKER: Any | None = None


def _get_env_config_path() -> str:
    config_path = os.environ.get("ALFWORLD_CONFIG_PATH")
    if config_path:
        return config_path
    return str(Path(__file__).resolve().parent / "configs" / "config_tw.yaml")


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as reader:
        for line_num, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
    return rows


@dataclass(frozen=True)
class SampleRow:
    """One original ALFWorld sample row from train_games.jsonl."""

    row_id: int
    sample_id: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SamplingTask:
    """One repeated sampling task derived from an original sample row."""

    sample: SampleRow
    sample_repeat_id: int


@dataclass
class TrajectoryState:
    task: SamplingTask
    worker: Any
    seed: int
    trajectory: dict[str, Any]
    task_description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    current_observation: str = ""
    admissible_actions: list[str] = field(default_factory=list)
    last_info: dict[str, Any] | None = None
    done: bool = False
    won: bool = False
    aborted: bool = False
    closed: bool = False
    progress_counted: bool = False
    row_written: bool = False


class _FallbackProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.count = 0

    def update(self, n: int = 1) -> None:
        self.count += n
        print(f"collect {self.count}/{self.total}", flush=True)

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


def _make_progress(total: int):
    if tqdm is None:
        return _FallbackProgress(total)
    return tqdm(total=total, desc="collect", unit="traj", dynamic_ncols=True)


def _load_samples(path: Path, *, start_index: int, limit: int | None) -> list[SampleRow]:
    raw_rows = _read_jsonl(path)
    samples: list[SampleRow] = []
    for row_id, row in enumerate(raw_rows):
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Row {row_id} has non-dict metadata: {type(metadata)}")
        if not metadata.get("gamefile"):
            raise ValueError(f"Row {row_id} is missing metadata.gamefile")
        samples.append(SampleRow(row_id=row_id, sample_id=int(row.get("index", row_id)), metadata=metadata))

    if start_index:
        samples = samples[start_index:]
    if limit is not None and limit >= 0:
        samples = samples[:limit]
    return samples


def _expand_sampling_tasks(samples: list[SampleRow], samples_per_task: int) -> list[SamplingTask]:
    return [SamplingTask(sample=sample, sample_repeat_id=i) for sample in samples for i in range(samples_per_task)]


def _env_split_from_metadata(metadata: dict[str, Any]) -> str:
    split = metadata.get("split", "train")
    if split == "valid_seen":
        return "eval_in_distribution"
    if split == "valid_unseen":
        return "eval_out_of_distribution"
    return "train"


def _trajectory_id(task: SamplingTask) -> str:
    split = str(task.sample.metadata.get("split", "train"))
    return f"{split}_{task.sample.sample_id:06d}_sample_{task.sample_repeat_id:02d}"


def _load_existing_trajectory_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            trajectory_id = row.get("trajectory_id") or row.get("episode_id")
            if trajectory_id is not None:
                ids.add(str(trajectory_id))
    return ids


def _prepare_outputs(args: argparse.Namespace) -> tuple[Path, Path, set[str]]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_output = Path(args.trajectory_output).expanduser() if args.trajectory_output else output_dir / "all_trajectories.jsonl"
    summary_output = Path(args.summary_output).expanduser() if args.summary_output else output_dir / "collect_summary.json"
    trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        trajectory_output.write_text("", encoding="utf-8")
        if summary_output.exists():
            summary_output.unlink()
        return trajectory_output, summary_output, set()

    if trajectory_output.exists() and trajectory_output.stat().st_size > 0 and not args.resume:
        raise FileExistsError(f"Output already exists: {trajectory_output}. Use --resume or --overwrite.")

    seen_ids = _load_existing_trajectory_ids(trajectory_output) if args.resume else set()
    return trajectory_output, summary_output, seen_ids


class JsonlTrajectoryWriter:
    """Write trajectory rows immediately to one all-attempt JSONL file."""

    def __init__(self, trajectory_output: Path, *, mode: str, fsync_every: int = 0) -> None:
        self.trajectory_output = trajectory_output
        self.fsync_every = fsync_every
        self._trajectory_writer = trajectory_output.open(mode, encoding="utf-8")
        self._write_count = 0
        self._closed = False

    def __enter__(self) -> JsonlTrajectoryWriter:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    def write(self, row: dict[str, Any]) -> None:
        line = _json_dumps(row) + "\n"
        self._trajectory_writer.write(line)
        self._trajectory_writer.flush()

        self._write_count += 1
        if self.fsync_every and self._write_count % self.fsync_every == 0:
            os.fsync(self._trajectory_writer.fileno())

    def close(self) -> None:
        if self._closed:
            return
        self._trajectory_writer.flush()
        if self.fsync_every:
            os.fsync(self._trajectory_writer.fileno())
        self._trajectory_writer.close()
        self._closed = True


def _safe_action_for_env(parsed_action: str, admissible_actions: list[str]) -> str:
    if parsed_action:
        return parsed_action
    if "look" in admissible_actions:
        return "look"
    if admissible_actions:
        return admissible_actions[0]
    return "look"


def _response_token_count(tokenizer: Any, response: str, meta_info: dict[str, Any]) -> int:
    token_logprobs = meta_info.get("output_token_logprobs") or []
    if token_logprobs:
        return len(token_logprobs)
    return len(tokenizer.encode(response, add_special_tokens=False))


def _build_step_messages(
    *,
    current_observation: str,
    admissible_actions: list[str],
    task_description: str,
    history: list[dict[str, str]],
    history_length: int,
) -> tuple[list[dict[str, str]], str, int]:
    keep_history = min(history_length, len(history)) if history_length > 0 else 0
    prompt_history = history[-keep_history:] if keep_history else []
    user_prompt = build_observation_prompt(
        current_observation=current_observation,
        admissible_actions=admissible_actions,
        task_description=task_description,
        history=prompt_history,
        history_length=keep_history,
    )
    messages = [
        {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, user_prompt, keep_history


def _ensure_ray() -> Any:
    global _RAY
    if _RAY is None:
        try:
            import ray  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - requires optional runtime dependency.
            raise ImportError("Ray is required for ALFWorld trajectory collection. Install ray before running.") from exc
        _RAY = ray
    return _RAY


class _AlfWorldEnvWorker:
    """Ray worker that owns at most one mutable ALFWorld episode."""

    def __init__(self) -> None:
        self._episode: AlfWorldTextEpisode | None = None

    def reset(self, config_path: str, split: str, gamefile: str | None, seed: int) -> StepResult:
        from alfworld_env import AlfWorldTextEpisode  # noqa: PLC0415

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


def _remote_env_worker_class() -> Any:
    global _REMOTE_ENV_WORKER
    if _REMOTE_ENV_WORKER is None:
        ray = _ensure_ray()
        _REMOTE_ENV_WORKER = ray.remote(_AlfWorldEnvWorker)
    return _REMOTE_ENV_WORKER


class RayEnvWorkerPool:
    """Small Ray actor pool with bounded actor lifetime.

    By default each actor is retired after one episode, matching the safe mode in
    examples/alfworld/batched_rollout.py. This bounds TextWorld/Fast-Downward
    native-library lifetime without requiring the outer shell script to chunk and
    restart the whole collector process.
    """

    def __init__(self) -> None:
        self._idle_workers: list[Any] = []
        self._episode_counts: dict[int, int] = {}
        self._created_count = 0
        self._remote_cls = _remote_env_worker_class()
        self._ray = _ensure_ray()

    @property
    def created_count(self) -> int:
        return self._created_count

    @property
    def idle_count(self) -> int:
        return len(self._idle_workers)

    @staticmethod
    def max_episodes_per_worker() -> int:
        return max(1, _get_int_env("ALFWORLD_ENV_WORKER_MAX_EPISODES", 1))

    @staticmethod
    def worker_cpus() -> float:
        return max(0.0, _get_float_env("ALFWORLD_ENV_WORKER_CPUS", 0.1))

    def _new_worker(self) -> Any:
        self._created_count += 1
        return self._remote_cls.options(num_cpus=self.worker_cpus(), num_gpus=0).remote()

    def acquire(self, size: int) -> list[Any]:
        workers: list[Any] = []
        while len(workers) < size:
            worker = self._idle_workers.pop() if self._idle_workers else self._new_worker()
            self._episode_counts[id(worker)] = self._episode_counts.get(id(worker), 0) + 1
            workers.append(worker)
        return workers

    async def close_worker_envs(self, workers: list[Any]) -> None:
        if not workers:
            return
        refs = []
        for worker in workers:
            try:
                refs.append(worker.close.remote())
            except Exception:
                logger.debug("failed to schedule ALFWorld worker close", exc_info=True)
        if refs:
            results = await asyncio.gather(*[_ray_get(ref) for ref in refs], return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.debug("failed to close ALFWorld worker", exc_info=result)

    async def release(self, workers: list[Any]) -> None:
        if not workers:
            return
        await self.close_worker_envs(workers)
        max_episodes = self.max_episodes_per_worker()
        for worker in workers:
            count_key = id(worker)
            episode_count = self._episode_counts.get(count_key, 0)
            should_retire = episode_count >= max_episodes
            if should_retire:
                self._episode_counts.pop(count_key, None)
                try:
                    self._ray.kill(worker, no_restart=True)
                except Exception:
                    logger.debug("failed to kill retired ALFWorld worker", exc_info=True)
            else:
                self._idle_workers.append(worker)

    def shutdown(self) -> None:
        for worker in self._idle_workers:
            try:
                self._ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("failed to kill idle ALFWorld worker", exc_info=True)
        self._idle_workers.clear()
        self._episode_counts.clear()


async def _ray_get(ref: Any) -> Any:
    ray = _ensure_ray()
    return await asyncio.to_thread(ray.get, ref)


async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Teacher response is not a JSON object: {type(data)}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry all request failures.
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


async def _generate_one_step(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    client: httpx.AsyncClient,
    trajectory: TrajectoryState,
    step_id: int,
) -> dict[str, Any]:
    messages, user_prompt, history_used = _build_step_messages(
        current_observation=trajectory.current_observation,
        admissible_actions=trajectory.admissible_actions,
        task_description=trajectory.task_description,
        history=trajectory.history,
        history_length=args.history_length,
    )
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.step_max_tokens,
        "skip_special_tokens": False,
    }
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k

    output = await _post_with_retries(
        client,
        args.teacher_url,
        {"input_ids": prompt_tokens, "sampling_params": sampling_params, "return_logprob": False},
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    return {
        "trajectory": trajectory,
        "step_id": step_id,
        "user_prompt": user_prompt,
        "history_used": history_used,
        "prompt_tokens": len(prompt_tokens),
        "output": output,
    }


def _make_initial_trajectory(args: argparse.Namespace, task: SamplingTask, seed: int) -> dict[str, Any]:
    sample = task.sample
    metadata = sample.metadata
    return {
        "schema_version": 2,
        "trajectory_id": _trajectory_id(task),
        "sample_id": sample.sample_id,
        "sample_row_id": sample.row_id,
        "sample_repeat_id": task.sample_repeat_id,
        "split": metadata.get("split", "train"),
        "gamefile": metadata.get("gamefile"),
        "task_type": metadata.get("task_type"),
        "task_root": metadata.get("task_root"),
        "teacher_url": args.teacher_url,
        "teacher_model": args.teacher_model_name,
        "tokenizer_path": args.tokenizer_path,
        "system_prompt": ALFWORLD_SYSTEM_PROMPT,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "step_max_tokens": args.step_max_tokens,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.step_max_tokens,
        },
        "seed": seed,
        "steps": [],
        "done": False,
        "won": False,
        "truncated": False,
        "aborted": False,
        "status": "running",
        "terminal_reason": None,
        "invalid_action_count": 0,
        "reward": 0.0,
        "penalized_reward": 0.0,
        "error": None,
    }


def _trajectory_status(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "error"
    if row.get("aborted"):
        return "aborted"
    if row.get("reward", 0.0) == 1.0 or row.get("won") is True:
        return "success"
    if row.get("done"):
        return "failed"
    if row.get("truncated"):
        return "truncated"
    return "unknown"


def _terminal_reason(row: dict[str, Any]) -> str:
    status = str(row.get("status") or _trajectory_status(row))
    if status == "error":
        return "exception"
    if status == "aborted":
        return "sglang_abort"
    if status == "success":
        return "env_won"
    if status == "failed":
        return "env_done_without_reward"
    if status == "truncated":
        return "max_steps"
    return "unknown"


def _mark_finished(progress: Any, counters: Counter[str], trajectory: TrajectoryState) -> None:
    if trajectory.progress_counted:
        return
    trajectory.progress_counted = True
    status = str(trajectory.trajectory.get("status") or _trajectory_status(trajectory.trajectory))
    counters["completed"] += 1
    counters[f"status:{status}"] += 1
    if status == "failed":
        counters["failed"] += 1
    if status == "error":
        counters["errors"] += 1
    if status == "success":
        counters["success"] += 1
    progress.update(1)
    progress.set_postfix(success=counters["success"], failed=counters["failed"], errors=counters["errors"], refresh=False)


async def _close_finished_envs(pool: RayEnvWorkerPool, trajectories: list[TrajectoryState]) -> None:
    workers = []
    for trajectory in trajectories:
        if trajectory.closed:
            continue
        if trajectory.done or trajectory.aborted or trajectory.trajectory.get("error"):
            workers.append(trajectory.worker)
            trajectory.closed = True
    await pool.close_worker_envs(workers)


def _apply_step_result(tokenizer: Any, result: dict[str, Any], step_result: StepResult) -> None:
    trajectory: TrajectoryState = result["trajectory"]
    output = result["output"]
    meta_info = output.get("meta_info", {}) or {}
    finish_reason = meta_info.get("finish_reason") or {}
    finish_type = finish_reason.get("type")
    if finish_type == "abort":
        trajectory.aborted = True
        trajectory.trajectory["aborted"] = True
        return

    response = output.get("text", "") or ""
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")]

    parsed = parse_action(response, trajectory.admissible_actions)
    env_action = _safe_action_for_env(parsed.action, trajectory.admissible_actions)
    if not parsed.is_valid:
        trajectory.trajectory["invalid_action_count"] += 1

    trajectory.done = bool(step_result.done)
    trajectory.won = bool(step_result.won)
    trajectory.last_info = step_result.info
    trajectory.trajectory["done"] = trajectory.done
    trajectory.trajectory["won"] = trajectory.won
    trajectory.trajectory["reward"] = 1.0 if trajectory.won else 0.0

    step_record = {
        "step": result["step_id"] + 1,
        "system_prompt": ALFWORLD_SYSTEM_PROMPT,
        "user_prompt": result["user_prompt"],
        "teacher_response": response,
        "action": env_action,
        "parsed_action": parsed.action,
        "valid_action": parsed.is_valid,
        "valid_format": parsed.valid_format,
        "valid_admissible": parsed.valid_admissible,
        "invalid_reason": parsed.invalid_reason,
        "observation": trajectory.current_observation,
        "admissible_actions": trajectory.admissible_actions,
        "reward": step_result.reward,
        "done": step_result.done,
        "won": step_result.won,
        "finish_type": finish_type,
        "prompt_tokens": result["prompt_tokens"],
        "response_tokens": _response_token_count(tokenizer, response, meta_info),
        "history_used": result["history_used"],
    }
    trajectory.trajectory["steps"].append(step_record)

    if not step_result.done:
        trajectory.history.append({"observation": trajectory.current_observation, "action": env_action})
        trajectory.current_observation = step_result.observation
        trajectory.admissible_actions = step_result.admissible_actions


def _finalize_trajectory(trajectory: TrajectoryState, elapsed_sec: float) -> dict[str, Any]:
    row = trajectory.trajectory
    if not row.get("done") and not row.get("aborted") and not row.get("error"):
        row["truncated"] = True
    row["num_steps"] = len(row.get("steps") or [])
    row["last_info"] = trajectory.last_info
    row["penalized_reward"] = float(row.get("reward", 0.0)) - 0.01 * int(row.get("invalid_action_count", 0))
    row["elapsed_sec"] = elapsed_sec
    row["status"] = _trajectory_status(row)
    row["terminal_reason"] = _terminal_reason(row)
    return row


def _finish_and_write_trajectory(
    *,
    trajectory: TrajectoryState,
    elapsed_sec: float,
    writer: JsonlTrajectoryWriter,
    progress: Any,
    counters: Counter[str],
) -> None:
    if trajectory.row_written:
        return
    row = _finalize_trajectory(trajectory, elapsed_sec)
    writer.write(row)
    trajectory.row_written = True
    _mark_finished(progress, counters, trajectory)


async def _run_task_batch(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    client: httpx.AsyncClient,
    pool: RayEnvWorkerPool,
    tasks: list[SamplingTask],
    progress: Any,
    counters: Counter[str],
    writer: JsonlTrajectoryWriter,
) -> None:
    batch_start = time.perf_counter()
    workers = pool.acquire(len(tasks))
    trajectories: list[TrajectoryState] = []
    try:
        reset_refs = []
        for worker, task in zip(workers, tasks, strict=True):
            seed = int(args.seed) + task.sample.sample_id * 1009 + task.sample_repeat_id
            trajectory = _make_initial_trajectory(args, task, seed)
            state = TrajectoryState(task=task, worker=worker, seed=seed, trajectory=trajectory)
            trajectories.append(state)
            reset_refs.append(
                worker.reset.remote(
                    args.config_path,
                    _env_split_from_metadata(task.sample.metadata),
                    task.sample.metadata.get("gamefile"),
                    seed,
                )
            )

        reset_results = await asyncio.gather(*[_ray_get(ref) for ref in reset_refs], return_exceptions=True)
        for state, reset_result in zip(trajectories, reset_results, strict=True):
            if isinstance(reset_result, Exception):
                state.trajectory["error"] = repr(reset_result)
                state.aborted = True
                _finish_and_write_trajectory(
                    trajectory=state,
                    elapsed_sec=time.perf_counter() - batch_start,
                    writer=writer,
                    progress=progress,
                    counters=counters,
                )
                continue
            state.task_description = _task_description_from_reset(reset_result.observation)
            state.current_observation = reset_result.observation
            state.admissible_actions = reset_result.admissible_actions
            state.last_info = reset_result.info
            state.trajectory["task_description"] = state.task_description
            state.trajectory["reset_info"] = reset_result.info

        await _close_finished_envs(pool, trajectories)

        for step_id in range(args.max_steps):
            active = [state for state in trajectories if not state.row_written and not state.done and not state.aborted and not state.trajectory.get("error")]
            if not active:
                break

            generation_results = await asyncio.gather(
                *[
                    _generate_one_step(
                        args=args,
                        tokenizer=tokenizer,
                        client=client,
                        trajectory=state,
                        step_id=step_id,
                    )
                    for state in active
                ],
                return_exceptions=True,
            )

            step_items = []
            step_refs = []
            for state, gen_result in zip(active, generation_results, strict=True):
                if isinstance(gen_result, Exception):
                    state.trajectory["error"] = repr(gen_result)
                    state.aborted = True
                    _finish_and_write_trajectory(
                        trajectory=state,
                        elapsed_sec=time.perf_counter() - batch_start,
                        writer=writer,
                        progress=progress,
                        counters=counters,
                    )
                    continue

                output = gen_result["output"]
                meta_info = output.get("meta_info", {}) or {}
                finish_reason = meta_info.get("finish_reason") or {}
                if finish_reason.get("type") == "abort":
                    state.aborted = True
                    state.trajectory["aborted"] = True
                    _finish_and_write_trajectory(
                        trajectory=state,
                        elapsed_sec=time.perf_counter() - batch_start,
                        writer=writer,
                        progress=progress,
                        counters=counters,
                    )
                    continue

                response = output.get("text", "") or ""
                if response.endswith("<|im_end|>"):
                    response = response[: -len("<|im_end|>")]
                parsed = parse_action(response, state.admissible_actions)
                env_action = _safe_action_for_env(parsed.action, state.admissible_actions)
                gen_result["preparsed_response"] = response
                gen_result["preparsed_action"] = parsed
                gen_result["env_action"] = env_action
                step_items.append(gen_result)
                step_refs.append(state.worker.step.remote(env_action))

            step_results = await asyncio.gather(*[_ray_get(ref) for ref in step_refs], return_exceptions=True)
            for item, step_result in zip(step_items, step_results, strict=True):
                state = item["trajectory"]
                if isinstance(step_result, Exception):
                    state.trajectory["error"] = repr(step_result)
                    state.aborted = True
                    _finish_and_write_trajectory(
                        trajectory=state,
                        elapsed_sec=time.perf_counter() - batch_start,
                        writer=writer,
                        progress=progress,
                        counters=counters,
                    )
                    continue
                _apply_step_result(tokenizer, item, step_result)
                if state.done:
                    _finish_and_write_trajectory(
                        trajectory=state,
                        elapsed_sec=time.perf_counter() - batch_start,
                        writer=writer,
                        progress=progress,
                        counters=counters,
                    )

            await _close_finished_envs(pool, trajectories)

        elapsed_sec = time.perf_counter() - batch_start
        for state in trajectories:
            _finish_and_write_trajectory(
                trajectory=state,
                elapsed_sec=elapsed_sec,
                writer=writer,
                progress=progress,
                counters=counters,
            )
    finally:
        await pool.release(workers)


def _summarize(counters: Counter[str], elapsed_sec: float, args: argparse.Namespace, *, worker_count: int) -> dict[str, Any]:
    completed = counters["completed"]
    status_counts = {key.removeprefix("status:"): value for key, value in sorted(counters.items()) if key.startswith("status:")}
    return {
        "elapsed_sec": elapsed_sec,
        "input_samples": counters["input_samples"],
        "requested_tasks": counters["requested_tasks"],
        "total_tasks": counters["total_tasks"],
        "completed": completed,
        "success": counters["success"],
        "failed": counters["failed"],
        "errors": counters["errors"],
        "status_counts": status_counts,
        "skipped_existing": counters["skipped_existing"],
        "success_rate": counters["success"] / completed if completed else 0.0,
        "samples_per_task": args.samples_per_task,
        "max_concurrent_tasks": args.max_concurrent_tasks,
        "ray_workers_created": worker_count,
        "args": {
            "teacher_url": args.teacher_url,
            "task_file": args.task_file,
            "output_dir": args.output_dir,
            "trajectory_output": args.trajectory_output,
            "tokenizer_path": args.tokenizer_path,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "step_max_tokens": args.step_max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "fsync_every": args.fsync_every,
        },
    }


async def _collect(args: argparse.Namespace) -> None:
    import httpx  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    samples = _load_samples(Path(args.task_file).expanduser(), start_index=args.start_index, limit=args.limit)
    tasks = _expand_sampling_tasks(samples, args.samples_per_task)
    requested_tasks = len(tasks)
    trajectory_output, summary_output, seen_ids = _prepare_outputs(args)
    if seen_ids:
        before = len(tasks)
        tasks = [task for task in tasks if _trajectory_id(task) not in seen_ids]
        skipped = before - len(tasks)
    else:
        skipped = 0

    counters: Counter[str] = Counter(input_samples=len(samples), requested_tasks=requested_tasks, total_tasks=len(tasks), skipped_existing=skipped)
    if not tasks:
        summary_output.write_text(
            json.dumps(_summarize(counters, 0.0, args, worker_count=0), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    ray = _ensure_ray()
    if not ray.is_initialized():
        init_kwargs: dict[str, Any] = {"ignore_reinit_error": True, "include_dashboard": False, "log_to_driver": False}
        if args.ray_address:
            init_kwargs["address"] = args.ray_address
        ray_tmpdir = os.environ.get("RAY_TMPDIR")
        if ray_tmpdir and not args.ray_address:
            init_kwargs["_temp_dir"] = ray_tmpdir
        ray.init(**init_kwargs)

    pool = RayEnvWorkerPool()
    timeout = httpx.Timeout(None if args.request_timeout <= 0 else args.request_timeout)
    limits = httpx.Limits(max_connections=args.max_concurrent_tasks, max_keepalive_connections=args.max_concurrent_tasks)
    start_time = time.perf_counter()
    mode = "a" if args.resume else "w"
    progress = _make_progress(len(tasks))
    try:
        with JsonlTrajectoryWriter(trajectory_output, mode=mode, fsync_every=args.fsync_every) as writer:
            async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
                for offset in range(0, len(tasks), args.max_concurrent_tasks):
                    batch = tasks[offset : offset + args.max_concurrent_tasks]
                    await _run_task_batch(
                        args=args,
                        tokenizer=tokenizer,
                        client=client,
                        pool=pool,
                        tasks=batch,
                        progress=progress,
                        counters=counters,
                        writer=writer,
                    )
    finally:
        progress.close()
        pool.shutdown()

    elapsed_sec = time.perf_counter() - start_time
    summary = _summarize(counters, elapsed_sec, args, worker_count=pool.created_count)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"done: completed={summary['completed']} success={summary['success']} failed={summary['failed']} errors={summary['errors']} elapsed={elapsed_sec:.1f}s output={trajectory_output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all ALFWorld teacher trajectories with Ray env workers.")

    # Core inputs.
    parser.add_argument("--teacher-url", default=os.environ.get("TEACHER_URL", "http://127.0.0.1:30000/generate"))
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer/model dir matching the teacher chat template.")
    parser.add_argument("--task-file", required=True, help="ALFWorld train_games.jsonl generated by prepare_alfworld_data.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectory-output", default=None, help="JSONL path for all trajectory attempts.")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--config-path", default=_get_env_config_path())
    parser.add_argument("--teacher-model-name", default=os.environ.get("TEACHER_MODEL_NAME", "qwen2.5-3b-alfworld-teacher"))

    # ALFWorld/task and server knobs.
    parser.add_argument("--samples-per-task", type=int, default=int(os.environ.get("SAMPLES_PER_TASK", DEFAULT_SAMPLES_PER_TASK)))
    parser.add_argument("--max-concurrent-tasks", type=int, default=int(os.environ.get("MAX_CONCURRENT_TASKS", 128)))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("ALFWORLD_MAX_STEPS", DEFAULT_MAX_STEPS)))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("ALFWORLD_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)))
    parser.add_argument("--step-max-tokens", type=int, default=int(os.environ.get("ALFWORLD_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.7)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 0.95)))
    parser.add_argument("--top-k", type=int, default=None)

    # Save/range controls.
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsync-every", type=int, default=0, help="Fsync output files every N completed trajectories; 0 disables fsync.")

    # Runtime defaults, normally left untouched.
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", ""))
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, help="<=0 means no timeout.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP)

    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    if args.max_concurrent_tasks <= 0:
        parser.error("--max-concurrent-tasks must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be non-negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        asyncio.run(_collect(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
