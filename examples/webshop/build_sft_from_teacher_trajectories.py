#!/usr/bin/env python3
"""Build step-level SFT data from collected WebShop teacher trajectories.

The collector stores every sampled WebShop teacher attempt.  This script mirrors
the ALFWorld SFT builder while keeping WebShop's success semantics explicit:

1. group trajectories by original ``sample_id``;
2. keep only full-success trajectories for SFT by default;
3. select the shortest successful trajectory per sample;
4. emit one messages-format SFT row per usable step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prompts import WEBSHOP_SYSTEM_PROMPT

try:  # pragma: no cover - optional cosmetic dependency in minimal checks.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


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


def _is_successful(trajectory: dict[str, Any]) -> bool:
    """Return whether a WebShop trajectory is a full-success demonstration."""

    if trajectory.get("error") or trajectory.get("aborted"):
        return False
    status = str(trajectory.get("status") or "").lower()
    if status == "success":
        return True
    try:
        raw_reward = float(trajectory.get("raw_reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(trajectory.get("done")) and raw_reward >= 1.0


def _sample_key(trajectory: dict[str, Any]) -> str:
    if trajectory.get("sample_id") is not None:
        return str(trajectory["sample_id"])
    if trajectory.get("goal_idx") is not None:
        return f"goal:{trajectory['goal_idx']}"
    return str(trajectory.get("trajectory_id") or trajectory.get("episode_id"))


def _trajectory_sort_key(trajectory: dict[str, Any]) -> tuple[int, int, int, str]:
    steps = trajectory.get("steps") or []
    num_steps = int(trajectory.get("num_steps") or len(steps) or 10**9)
    response_tokens = sum(int(step.get("response_tokens") or 0) for step in steps)
    repeat_id = int(trajectory.get("sample_repeat_id", trajectory.get("attempt_id", 0)) or 0)
    trajectory_id = str(trajectory.get("trajectory_id") or trajectory.get("episode_id") or "")
    return num_steps, response_tokens, repeat_id, trajectory_id


def _select_shortest_per_sample(trajectories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        if _is_successful(trajectory) and trajectory.get("steps"):
            grouped[_sample_key(trajectory)].append(trajectory)

    selected = [sorted(group, key=_trajectory_sort_key)[0] for group in grouped.values()]
    selected.sort(key=lambda item: int(item.get("sample_id", item.get("goal_idx", 0)) or 0))
    groups_with_multiple = sum(1 for group in grouped.values() if len(group) > 1)
    return selected, groups_with_multiple


def _step_filter_reason(step: dict[str, Any]) -> str | None:
    response = step.get("teacher_response")
    if not isinstance(response, str) or not response.strip():
        return "empty_response"
    user_prompt = step.get("user_prompt")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        return "empty_user_prompt"
    if step.get("finish_type") == "length":
        return "finish_length"
    if step.get("finish_type") == "abort":
        return "finish_abort"
    if int(step.get("response_tokens") or 0) <= 0:
        return "empty_response_tokens"
    if step.get("valid_format") is not True:
        return str(step.get("invalid_reason") or "invalid_format")
    if step.get("valid_admissible") is not True:
        return str(step.get("invalid_reason") or "invalid_admissible")
    if step.get("valid_action") is not True:
        return str(step.get("invalid_reason") or "invalid_action")
    return None


def _sft_row(trajectory: dict[str, Any], step: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    system_prompt = step.get("system_prompt") or trajectory.get("system_prompt") or WEBSHOP_SYSTEM_PROMPT
    metadata = {
        "source": args.source,
        "teacher_model": trajectory.get("teacher_model"),
        "trajectory_id": trajectory.get("trajectory_id") or trajectory.get("episode_id"),
        "sample_id": trajectory.get("sample_id"),
        "sample_repeat_id": trajectory.get("sample_repeat_id", trajectory.get("attempt_id")),
        "split": trajectory.get("split"),
        "goal_idx": trajectory.get("goal_idx"),
        "goal_seed": trajectory.get("goal_seed"),
        "instruction_text": trajectory.get("instruction_text"),
        "step": step.get("step"),
        "action": step.get("action"),
        "parsed_action": step.get("parsed_action"),
        "service_action": step.get("service_action"),
        "num_steps": trajectory.get("num_steps"),
        "raw_reward": trajectory.get("raw_reward"),
        "episode_reward": trajectory.get("episode_reward"),
        "final_reward": trajectory.get("final_reward"),
        "status": trajectory.get("status"),
        "reward_mode": trajectory.get("reward_mode"),
        "invalid_action_count": trajectory.get("invalid_action_count"),
        "webshop_url": trajectory.get("webshop_url"),
    }
    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": step["user_prompt"]},
            {"role": "assistant", "content": step["teacher_response"]},
        ],
        "metadata": metadata,
    }
    if args.include_step_observation:
        row["metadata"]["observation"] = step.get("observation")
    if args.include_available_actions:
        row["metadata"]["available_actions"] = step.get("available_actions")
    return row


def _write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise ImportError("Writing parquet requires pyarrow. Use a .jsonl output or install pyarrow.") from exc
        pq.write_table(pa.Table.from_pylist(rows), output_path)
        return

    with output_path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(_json_dumps(row) + "\n")


def _status_counts(trajectories: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("status") or "unknown") for row in trajectories)
    return dict(sorted(counts.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    trajectories = _read_jsonl(Path(args.input).expanduser())
    selected, groups_with_multiple = _select_shortest_per_sample(trajectories)
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]

    rows: list[dict[str, Any]] = []
    skipped_steps = Counter()
    iterator = selected if tqdm is None else tqdm(selected, desc="build-sft", unit="traj", dynamic_ncols=True)
    for trajectory in iterator:
        for step in trajectory.get("steps") or []:
            reason = _step_filter_reason(step)
            if reason is not None:
                skipped_steps[reason] += 1
                continue
            rows.append(_sft_row(trajectory, step, args))
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
        if args.max_samples is not None and len(rows) >= args.max_samples:
            rows = rows[: args.max_samples]
            break

    output_path = Path(args.output).expanduser()
    _write_rows(rows, output_path)

    summary = {
        "input": args.input,
        "output": str(output_path),
        "input_trajectories": len(trajectories),
        "status_counts": _status_counts(trajectories),
        "successful_trajectories": sum(1 for item in trajectories if _is_successful(item)),
        "selected_trajectories": len(selected),
        "samples_with_multiple_successes": groups_with_multiple,
        "sft_samples": len(rows),
        "skipped_steps": dict(sorted(skipped_steps.items())),
    }
    if args.summary_output:
        summary_path = Path(args.summary_output).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"done: selected={summary['selected_trajectories']} sft_samples={summary['sft_samples']} output={output_path}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT rows from collected WebShop teacher trajectories.")
    parser.add_argument("--input", required=True, help="all_trajectories.jsonl from the WebShop collector.")
    parser.add_argument("--output", required=True, help="Output .jsonl or .parquet SFT dataset.")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--source", default="qwen2.5-3b-webshop-teacher")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-step-observation", action="store_true")
    parser.add_argument("--include-available-actions", action="store_true")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive when set")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive when set")
    return args


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
