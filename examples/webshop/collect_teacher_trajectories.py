#!/usr/bin/env python3
"""Collect WebShop teacher trajectories for offline analysis and SFT distillation.

This standalone collector mirrors ``examples/alfworld/collect_teacher_trajectories.py``:
it reads the fixed WebShop ``train.jsonl`` goal index, expands each goal into
multiple teacher sampling attempts, runs each attempt against a separately
deployed WebShop HTTP service, and appends every completed attempt to one
``all_trajectories.jsonl`` ledger.  The ledger intentionally keeps successes,
partial purchases, failures, truncations, aborts, and errors; downstream SFT
builders can decide which attempts to keep.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional cosmetic dependency in minimal local checks.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from prompts import WEBSHOP_SYSTEM_PROMPT, build_observation_prompt, parse_action

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_URL = "http://127.0.0.1:30000/generate"
DEFAULT_WEBSHOP_SERVICE_URL = "http://127.0.0.1:3001"
DEFAULT_MAX_STEPS = 15
DEFAULT_SAMPLES_PER_TASK = 8
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_STEP_MAX_TOKENS = 512
DEFAULT_MAX_PROMPT_CHARS = 13000
DEFAULT_INVALID_ACTION_PENALTY = 0.1
DEFAULT_REWARD_MODE = "dense"
SUPPORTED_REWARD_MODES = {"binary", "dense"}
DEFAULT_REQUEST_TIMEOUT = 0.0
DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_SLEEP = 1.0


@dataclass(frozen=True)
class SampleRow:
    """One original WebShop row from ``train.jsonl``."""

    row_id: int
    sample_id: int
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SamplingTask:
    """One repeated teacher sampling attempt for an original WebShop row."""

    sample: SampleRow
    sample_repeat_id: int


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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_num}, got {type(row)}")
            rows.append(row)
    return rows


def load_samples(path: Path, *, start_index: int, limit: int | None) -> list[SampleRow]:
    """Load WebShop goal rows while preserving stable original row ids."""

    raw_rows = _read_jsonl(path)
    samples: list[SampleRow] = []
    for row_id, row in enumerate(raw_rows):
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Row {row_id} has non-dict metadata: {type(metadata)}")
        samples.append(
            SampleRow(
                row_id=row_id,
                sample_id=int(row.get("index", row_id)),
                text=str(row.get("text", f"webshop goal {metadata.get('goal_idx', row_id)}")),
                metadata=metadata,
            )
        )

    if start_index:
        samples = samples[start_index:]
    if limit is not None and limit >= 0:
        samples = samples[:limit]
    return samples


def expand_sampling_tasks(samples: list[SampleRow], samples_per_task: int) -> list[SamplingTask]:
    return [SamplingTask(sample=sample, sample_repeat_id=i) for sample in samples for i in range(samples_per_task)]


def trajectory_id(task: SamplingTask) -> str:
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
            row_id = row.get("trajectory_id") or row.get("episode_id")
            if row_id is not None:
                ids.add(str(row_id))
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
        self._writer = trajectory_output.open(mode, encoding="utf-8")
        self._write_count = 0
        self._closed = False

    def __enter__(self) -> JsonlTrajectoryWriter:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.write(_json_dumps(row) + "\n")
        self._writer.flush()
        self._write_count += 1
        if self.fsync_every and self._write_count % self.fsync_every == 0:
            os.fsync(self._writer.fileno())

    def close(self) -> None:
        if self._closed:
            return
        self._writer.flush()
        if self.fsync_every:
            os.fsync(self._writer.fileno())
        self._writer.close()
        self._closed = True


def _goal_idx_from_metadata(metadata: dict[str, Any]) -> int | None:
    if metadata.get("goal_idx") is None:
        return None
    return int(metadata["goal_idx"])


def _goal_seed_from_metadata(metadata: dict[str, Any]) -> int | None:
    if metadata.get("goal_seed") is None:
        return None
    return int(metadata["goal_seed"])


def _episode_reward_from_raw(*, raw_reward: float, done: bool, reward_mode: str) -> float:
    if not done:
        return 0.0
    if reward_mode == "binary":
        return 10.0 if raw_reward >= 1.0 else 0.0
    return 10.0 * max(0.0, min(1.0, float(raw_reward)))


def _normalized_reward_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"Unsupported reward mode {value!r}; supported modes: {sorted(SUPPORTED_REWARD_MODES)}")
    return mode


def _project_action_like_sdar(response: str) -> tuple[str, bool, str | None]:
    """Mirror WebShop rollout's SDAR-style action projection."""

    original_response = response
    lowered = response.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        action = lowered[-20:].strip() or "invalid"
        return action, False, "missing_action_tag"

    action = lowered[start_idx + len(start_tag) : end_idx].strip() or "invalid"
    if original_response.find("<think>") == -1 or original_response.find("</think>") == -1:
        return action, False, "missing_think_tag"
    if any("\u4e00" <= char <= "\u9fff" for char in original_response):
        return action, False, "contains_chinese"
    return action, True, None


def _service_action_from_projection(parsed_action: str, projected_action: str) -> str:
    if parsed_action.startswith(("search[", "click[")):
        return parsed_action
    return projected_action


def trajectory_status(*, done: bool, raw_reward: float, aborted: bool, error: str | None) -> str:
    if error:
        return "error"
    if aborted:
        return "aborted"
    if done and raw_reward >= 1.0:
        return "success"
    if done and raw_reward > 0.0:
        return "partial"
    if done:
        return "failed"
    return "truncated"


def _terminal_reason(status: str) -> str:
    return {
        "success": "full_reward",
        "partial": "partial_reward",
        "failed": "done_without_reward",
        "truncated": "max_steps",
        "aborted": "sglang_abort",
        "error": "exception",
    }.get(status, "unknown")


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _response_token_count(tokenizer: Any, response: str, meta_info: dict[str, Any]) -> int:
    token_logprobs = meta_info.get("output_token_logprobs") or []
    if token_logprobs:
        return len(token_logprobs)
    return len(tokenizer.encode(response, add_special_tokens=False))


def _build_step_prompt(
    *,
    tokenizer: Any,
    instruction_text: str,
    current_observation: str,
    available_actions: dict[str, Any],
    history: list[dict[str, str]],
    history_length: int,
    max_prompt_chars: int,
) -> tuple[str, str, list[int], int]:
    user_prompt, history_used = build_observation_prompt(
        instruction_text=instruction_text,
        current_observation=current_observation,
        available_actions=available_actions,
        history=history,
        history_length=history_length,
        max_prompt_chars=max_prompt_chars,
    )
    prompt_text = _apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return user_prompt, prompt_text, prompt_tokens, history_used


def make_step_record(
    *,
    step_id: int,
    system_prompt: str,
    user_prompt: str,
    teacher_response: str,
    parsed_action: str,
    projected_action: str,
    service_action: str,
    valid_action: bool,
    valid_format: bool,
    valid_admissible: bool,
    invalid_reason: str | None,
    observation: str,
    available_actions: dict[str, Any],
    reward: float,
    done: bool,
    finish_type: str | None,
    prompt_tokens: int,
    response_tokens: int,
    history_used: int,
) -> dict[str, Any]:
    """Build one SFT-ready per-step trajectory record."""

    return {
        "step": step_id + 1,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "teacher_response": teacher_response,
        "model_response": teacher_response,
        "action": service_action,
        "parsed_action": parsed_action,
        "projected_action": projected_action,
        "service_action": service_action,
        "valid_action": valid_action,
        "valid_format": valid_format,
        "valid_admissible": valid_admissible,
        "invalid_reason": invalid_reason,
        "observation": observation,
        "available_actions": available_actions,
        "reward": reward,
        "done": done,
        "finish_type": finish_type,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "history_used": history_used,
    }


def _make_initial_trajectory(args: argparse.Namespace, task: SamplingTask, session_id: str) -> dict[str, Any]:
    sample = task.sample
    metadata = sample.metadata
    return {
        "schema_version": 1,
        "trajectory_id": trajectory_id(task),
        "episode_id": trajectory_id(task),
        "sample_id": sample.sample_id,
        "sample_row_id": sample.row_id,
        "sample_repeat_id": task.sample_repeat_id,
        "text": sample.text,
        "split": metadata.get("split", "train"),
        "goal_idx": _goal_idx_from_metadata(metadata),
        "goal_seed": _goal_seed_from_metadata(metadata),
        "teacher_url": args.teacher_url,
        "teacher_model": args.teacher_model_name,
        "tokenizer_path": args.tokenizer_path,
        "webshop_service_url": args.service_url,
        "session_id": session_id,
        "system_prompt": WEBSHOP_SYSTEM_PROMPT,
        "instruction_text": None,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "step_max_tokens": args.step_max_tokens,
        "max_prompt_chars": args.max_prompt_chars,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.step_max_tokens,
            "skip_special_tokens": False,
        },
        "steps": [],
        "done": False,
        "status": "running",
        "terminal_reason": None,
        "raw_reward": 0.0,
        "episode_reward": 0.0,
        "final_reward": 0.0,
        "reward_mode": args.reward_mode,
        "invalid_action_count": 0,
        "projection_invalid_action_count": 0,
        "invalid_action_penalty": args.invalid_action_penalty,
        "num_steps": 0,
        "reward_info": None,
        "webshop_url": None,
        "elapsed_sec": 0.0,
        "aborted": False,
        "error": None,
    }


async def _post_json(
    client: Any,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        response = None
        try:
            response = await client.post(url, json=payload or {})
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Response from {url} is not a JSON object: {type(data)}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry all request failures.
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(retry_sleep)
        finally:
            if response is not None:
                await response.aclose()
    assert last_exc is not None
    raise last_exc


async def _delete_session(client: Any, service_url: str, session_id: str) -> None:
    url = f"{service_url.rstrip('/')}/v1/session/{session_id}"
    response = None
    try:
        response = await client.delete(url)
        response.raise_for_status()
    except Exception:
        logger.debug("Failed to close WebShop service session %s", session_id, exc_info=True)
    finally:
        if response is not None:
            await response.aclose()


async def _reset_session(client: Any, args: argparse.Namespace, task: SamplingTask, session_id: str) -> dict[str, Any]:
    metadata = task.sample.metadata
    payload: dict[str, Any] = {"session_id": session_id}
    goal_idx = _goal_idx_from_metadata(metadata)
    goal_seed = _goal_seed_from_metadata(metadata)
    if goal_idx is not None:
        payload["goal_idx"] = goal_idx
    if goal_seed is not None:
        payload["goal_seed"] = goal_seed
    if args.observation_mode:
        payload["observation_mode"] = args.observation_mode
    return await _post_json(
        client,
        f"{args.service_url.rstrip('/')}/v1/reset",
        payload,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )


async def _step_session(client: Any, args: argparse.Namespace, *, session_id: str, action: str) -> dict[str, Any]:
    return await _post_json(
        client,
        f"{args.service_url.rstrip('/')}/v1/step",
        {"session_id": session_id, "action": action},
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )


async def _generate_one_step(
    client: Any,
    args: argparse.Namespace,
    *,
    prompt_tokens: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": max_new_tokens,
        "skip_special_tokens": False,
    }
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    return await _post_json(
        client,
        args.teacher_url,
        {"input_ids": prompt_tokens, "sampling_params": sampling_params, "return_logprob": False},
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )


def _finalize_trajectory(
    row: dict[str, Any],
    *,
    elapsed_sec: float,
    raw_reward: float,
    done: bool,
    aborted: bool,
    error: str | None,
    final_state: dict[str, Any] | None,
) -> dict[str, Any]:
    status = trajectory_status(done=done, raw_reward=raw_reward, aborted=aborted, error=error)
    info = final_state.get("info", {}) if isinstance(final_state, dict) else {}
    projection_invalid_count = int(row.get("projection_invalid_action_count", 0) or 0)
    episode_reward = _episode_reward_from_raw(raw_reward=raw_reward, done=done, reward_mode=row["reward_mode"])
    row.update(
        {
            "done": done,
            "aborted": aborted,
            "error": error,
            "raw_reward": raw_reward,
            "episode_reward": episode_reward,
            "final_reward": episode_reward - float(row["invalid_action_penalty"]) * projection_invalid_count,
            "status": status,
            "terminal_reason": _terminal_reason(status),
            "num_steps": len(row.get("steps") or []),
            "reward_info": info.get("reward_info"),
            "webshop_url": info.get("url"),
            "last_state": final_state,
            "elapsed_sec": elapsed_sec,
        }
    )
    return row


async def collect_one_trajectory(client: Any, tokenizer: Any, args: argparse.Namespace, task: SamplingTask) -> dict[str, Any]:
    """Collect one complete WebShop teacher attempt."""

    start_time = time.perf_counter()
    session_id = str(uuid.uuid4())
    row = _make_initial_trajectory(args, task, session_id)
    raw_reward = 0.0
    final_state: dict[str, Any] | None = None
    done = False
    aborted = False
    error: str | None = None

    try:
        reset_state = await _reset_session(client, args, task, session_id)
        final_state = reset_state
        instruction_text = str(reset_state["instruction_text"])
        current_observation = str(reset_state["observation"])
        available_actions = reset_state.get("available_actions", {})
        if not isinstance(available_actions, dict):
            available_actions = {}
        row["instruction_text"] = instruction_text
        row["reset_state"] = reset_state

        history: list[dict[str, str]] = []
        for step_id in range(args.max_steps):
            user_prompt, _prompt_text, prompt_tokens, history_used = _build_step_prompt(
                tokenizer=tokenizer,
                instruction_text=instruction_text,
                current_observation=current_observation,
                available_actions=available_actions,
                history=history,
                history_length=args.history_length,
                max_prompt_chars=args.max_prompt_chars,
            )
            output = await _generate_one_step(
                client,
                args,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.step_max_tokens,
            )
            meta_info = output.get("meta_info", {}) or {}
            finish_reason = meta_info.get("finish_reason") or {}
            finish_type = finish_reason.get("type")
            if finish_type == "abort":
                aborted = True
                break

            response = output.get("text", "") or ""
            if response.endswith("<|im_end|>"):
                response = response[: -len("<|im_end|>")]
            response_tokens = _response_token_count(tokenizer, response, meta_info)

            parsed = parse_action(response, available_actions)
            projected_action, valid_for_penalty, projection_invalid_reason = _project_action_like_sdar(response)
            service_action = _service_action_from_projection(parsed.action, projected_action)
            valid_action = parsed.is_valid
            if not valid_action:
                row["invalid_action_count"] += 1
            if not valid_for_penalty:
                row["projection_invalid_action_count"] += 1

            final_state = await _step_session(client, args, session_id=session_id, action=service_action)
            raw_reward = float(final_state.get("reward", 0.0))
            done = bool(final_state.get("done", False))

            step_record = make_step_record(
                step_id=step_id,
                system_prompt=WEBSHOP_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                teacher_response=response,
                parsed_action=parsed.action,
                projected_action=projected_action,
                service_action=service_action,
                valid_action=valid_action,
                valid_format=parsed.valid_format,
                valid_admissible=parsed.valid_admissible,
                invalid_reason=projection_invalid_reason or parsed.invalid_reason,
                observation=current_observation,
                available_actions=available_actions,
                reward=raw_reward,
                done=done,
                finish_type=finish_type,
                prompt_tokens=len(prompt_tokens),
                response_tokens=response_tokens,
                history_used=history_used,
            )
            step_record["valid_for_penalty"] = valid_for_penalty
            row["steps"].append(step_record)

            if done:
                break

            history.append({"observation": current_observation, "action": service_action})
            current_observation = str(final_state.get("observation", ""))
            available_actions = final_state.get("available_actions", {})
            if not isinstance(available_actions, dict):
                available_actions = {}
    except Exception as exc:  # noqa: BLE001 - ledger should preserve failed attempts.
        error = repr(exc)
        logger.exception("WebShop teacher trajectory failed: %s", row["trajectory_id"])
    finally:
        await _delete_session(client, args.service_url, session_id)

    return _finalize_trajectory(
        row,
        elapsed_sec=time.perf_counter() - start_time,
        raw_reward=raw_reward,
        done=done,
        aborted=aborted,
        error=error,
        final_state=final_state,
    )


def _update_counters(counters: Counter[str], row: dict[str, Any]) -> None:
    status = str(row.get("status", "unknown"))
    counters["completed"] += 1
    counters[f"status:{status}"] += 1
    if status == "success":
        counters["success"] += 1
    if status == "partial":
        counters["partial"] += 1
    if status == "failed":
        counters["failed"] += 1
    if status == "truncated":
        counters["truncated"] += 1
    if status == "aborted":
        counters["aborted"] += 1
    if status == "error":
        counters["errors"] += 1


async def _worker(
    *,
    queue: asyncio.Queue[SamplingTask],
    client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    writer: JsonlTrajectoryWriter,
    writer_lock: asyncio.Lock,
    progress: Any,
    counters: Counter[str],
) -> None:
    while True:
        task = await queue.get()
        try:
            row = await collect_one_trajectory(client, tokenizer, args, task)
            async with writer_lock:
                writer.write(row)
                _update_counters(counters, row)
                progress.update(1)
                progress.set_postfix(
                    success=counters["success"],
                    partial=counters["partial"],
                    failed=counters["failed"],
                    errors=counters["errors"],
                    refresh=False,
                )
        finally:
            queue.task_done()


def _summarize(counters: Counter[str], elapsed_sec: float, args: argparse.Namespace) -> dict[str, Any]:
    completed = counters["completed"]
    status_counts = {key.removeprefix("status:"): value for key, value in sorted(counters.items()) if key.startswith("status:")}
    return {
        "elapsed_sec": elapsed_sec,
        "input_samples": counters["input_samples"],
        "requested_tasks": counters["requested_tasks"],
        "total_tasks": counters["total_tasks"],
        "completed": completed,
        "success": counters["success"],
        "partial": counters["partial"],
        "failed": counters["failed"],
        "truncated": counters["truncated"],
        "aborted": counters["aborted"],
        "errors": counters["errors"],
        "status_counts": status_counts,
        "skipped_existing": counters["skipped_existing"],
        "success_rate": counters["success"] / completed if completed else 0.0,
        "partial_or_success_rate": (counters["success"] + counters["partial"]) / completed if completed else 0.0,
        "samples_per_task": args.samples_per_task,
        "max_concurrent_tasks": args.max_concurrent_tasks,
        "args": {
            "teacher_url": args.teacher_url,
            "service_url": args.service_url,
            "task_file": args.task_file,
            "output_dir": args.output_dir,
            "trajectory_output": args.trajectory_output,
            "tokenizer_path": args.tokenizer_path,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "step_max_tokens": args.step_max_tokens,
            "max_prompt_chars": args.max_prompt_chars,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "reward_mode": args.reward_mode,
            "invalid_action_penalty": args.invalid_action_penalty,
            "fsync_every": args.fsync_every,
        },
    }


async def _collect(args: argparse.Namespace) -> None:
    import httpx  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    samples = load_samples(Path(args.task_file).expanduser(), start_index=args.start_index, limit=args.limit)
    tasks = expand_sampling_tasks(samples, args.samples_per_task)
    requested_tasks = len(tasks)
    trajectory_output, summary_output, seen_ids = _prepare_outputs(args)
    if seen_ids:
        before = len(tasks)
        tasks = [task for task in tasks if trajectory_id(task) not in seen_ids]
        skipped = before - len(tasks)
    else:
        skipped = 0

    counters: Counter[str] = Counter(input_samples=len(samples), requested_tasks=requested_tasks, total_tasks=len(tasks), skipped_existing=skipped)
    if not tasks:
        summary_output.write_text(
            json.dumps(_summarize(counters, 0.0, args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    timeout = httpx.Timeout(None if args.request_timeout <= 0 else args.request_timeout)
    limits = httpx.Limits(max_connections=args.max_concurrent_tasks * 2, max_keepalive_connections=args.max_concurrent_tasks * 2)
    queue: asyncio.Queue[SamplingTask] = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)

    mode = "a" if args.resume else "w"
    start_time = time.perf_counter()
    progress = _make_progress(len(tasks))
    writer_lock = asyncio.Lock()
    try:
        with JsonlTrajectoryWriter(trajectory_output, mode=mode, fsync_every=args.fsync_every) as writer:
            async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
                workers = [
                    asyncio.create_task(
                        _worker(
                            queue=queue,
                            client=client,
                            tokenizer=tokenizer,
                            args=args,
                            writer=writer,
                            writer_lock=writer_lock,
                            progress=progress,
                            counters=counters,
                        )
                    )
                    for _ in range(args.max_concurrent_tasks)
                ]
                await queue.join()
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
    finally:
        progress.close()

    elapsed_sec = time.perf_counter() - start_time
    summary = _summarize(counters, elapsed_sec, args)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"done: completed={summary['completed']} success={summary['success']} partial={summary['partial']} "
        f"failed={summary['failed']} errors={summary['errors']} elapsed={elapsed_sec:.1f}s output={trajectory_output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all WebShop teacher trajectories into an append-only JSONL ledger.")

    parser.add_argument("--teacher-url", default=os.environ.get("TEACHER_URL", DEFAULT_TEACHER_URL))
    parser.add_argument("--service-url", default=os.environ.get("WEBSHOP_SERVICE_URL", DEFAULT_WEBSHOP_SERVICE_URL))
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer/model dir matching the teacher chat template.")
    parser.add_argument("--task-file", required=True, help="WebShop train.jsonl generated by prepare_webshop_data.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectory-output", default=None, help="JSONL path for all trajectory attempts.")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--teacher-model-name", default=os.environ.get("TEACHER_MODEL_NAME", "qwen2.5-3b-webshop-teacher"))

    parser.add_argument("--samples-per-task", type=int, default=int(os.environ.get("SAMPLES_PER_TASK", DEFAULT_SAMPLES_PER_TASK)))
    parser.add_argument("--max-concurrent-tasks", type=int, default=int(os.environ.get("MAX_CONCURRENT_TASKS", 128)))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("WEBSHOP_MAX_STEPS", DEFAULT_MAX_STEPS)))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("WEBSHOP_HISTORY_LENGTH", DEFAULT_HISTORY_LENGTH)))
    parser.add_argument("--step-max-tokens", type=int, default=int(os.environ.get("WEBSHOP_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)))
    parser.add_argument("--max-prompt-chars", type=int, default=int(os.environ.get("WEBSHOP_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS)))
    parser.add_argument("--invalid-action-penalty", type=float, default=float(os.environ.get("WEBSHOP_INVALID_ACTION_PENALTY", DEFAULT_INVALID_ACTION_PENALTY)))
    parser.add_argument("--reward-mode", default=os.environ.get("WEBSHOP_REWARD_MODE", DEFAULT_REWARD_MODE))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.7)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 0.95)))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--observation-mode", default=None)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsync-every", type=int, default=0, help="Fsync output files every N completed trajectories; 0 disables fsync.")

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
    if args.step_max_tokens <= 0:
        parser.error("--step-max-tokens must be positive")
    if args.max_prompt_chars < 0:
        parser.error("--max-prompt-chars must be non-negative")
    if args.invalid_action_penalty < 0:
        parser.error("--invalid-action-penalty must be non-negative")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be non-negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    args.reward_mode = _normalized_reward_mode(args.reward_mode)
    args.service_url = args.service_url.rstrip("/")
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
