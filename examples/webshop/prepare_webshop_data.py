#!/usr/bin/env python3
"""Prepare slime WebShop prompt JSONL files.

The WebShop service owns the actual goals and instructions. Slime only needs a
stable list of goal metadata; the custom WebShop rollout uses that metadata to
reset the remote service to the requested goal.

This script writes the fixed slime WebShop goal schedule used by the
examples in this directory:

* ``train.jsonl``: training goal indices are sampled from ``[500, goal_count)``;
* ``valid.jsonl``: validation goal indices are the prefix of ``[0, 500)``;
* each row carries the WebShop goal seed needed to reproduce per-worker
  synthetic-goal ordering.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def _fetch_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - operator-provided service URL.
        return json.loads(response.read().decode("utf-8"))


def _goal_count_from_payload(path: str, payload: dict[str, Any]) -> int:
    for key in ("goal_count", "goals"):
        if key in payload and isinstance(payload[key], int):
            return int(payload[key])
    raise KeyError(f"{path} response did not include a goal count: {payload}")


def _fetch_goal_count(service_url: str, timeout: float) -> int:
    base = service_url.rstrip("/")
    first_error: Exception | None = None
    for path in ("/v1/goals?limit=0", "/health"):
        try:
            return _goal_count_from_payload(path, _fetch_json(base + path, timeout))
        except Exception as exc:  # fall back for older service builds that only expose /health
            if first_error is None:
                first_error = exc
    raise RuntimeError(f"failed to fetch WebShop goal count from {service_url}: {first_error}")


def _record(goal_idx: int, split: str, **metadata: Any) -> dict[str, Any]:
    return {
        "text": f"webshop goal {goal_idx}",
        "metadata": {
            "goal_idx": int(goal_idx),
            "split": split,
            **metadata,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_slime_webshop_rows(
    args: argparse.Namespace, goal_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the static slime WebShop rollout schedule.

    The default schedule uses ``env_seed=0``, train batch size ``16``,
    validation size ``100``, training goals from ``range(500, goal_count)``,
    and validation goals from ``range(100)``.
    Worker j uses goal order seed ``env_seed + j`` for train and
    ``env_seed + 1000 + j`` for validation.
    """

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("slime WebShop sampling requires numpy for np.random.RandomState scheduling") from exc

    train_start = args.train_start
    if goal_count <= train_start:
        raise ValueError(f"goal_count {goal_count} must exceed train_start {train_start}")
    if args.train_batch_size > goal_count - train_start:
        raise ValueError("train_batch_size exceeds WebShop train goal pool")
    if args.valid_size > train_start:
        raise ValueError("valid_size exceeds WebShop validation pool [0, train_start)")

    train_pool = np.arange(train_start, goal_count)
    train_rng = np.random.RandomState(args.env_seed)
    train_rows: list[dict[str, Any]] = []
    for rollout_id in range(args.total_rollouts):
        goal_indices = train_rng.choice(train_pool, size=args.train_batch_size, replace=False)
        for env_id, goal_idx in enumerate(goal_indices.tolist()):
            goal_seed = args.env_seed + env_id
            train_rows.append(
                _record(
                    goal_idx,
                    "train",
                    goal_seed=goal_seed,
                    rollout_id=rollout_id,
                    env_id=env_id,
                )
            )

    valid_indices = np.arange(0, args.valid_size)
    valid_rows = [
        _record(
            goal_idx,
            "valid",
            goal_seed=args.env_seed + 1000 + env_id,
            env_id=env_id,
        )
        for env_id, goal_idx in enumerate(valid_indices.tolist())
    ]

    summary = {
        "goal_count": goal_count,
        "schedule": {
            "env_seed": args.env_seed,
            "train_start": train_start,
            "train_pool": [train_start, goal_count],
            "valid_pool": [0, train_start],
            "train_batch_size": args.train_batch_size,
            "total_rollouts": args.total_rollouts,
            "valid_size": args.valid_size,
        },
        "splits": {
            "train": {"count": len(train_rows), "first_goal_idx": train_rows[0]["metadata"]["goal_idx"] if train_rows else None},
            "valid": {"count": len(valid_rows), "first_goal_idx": valid_rows[0]["metadata"]["goal_idx"] if valid_rows else None},
        },
    }
    return train_rows, valid_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create slime WebShop goal-index JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.jsonl, valid.jsonl, and summary.json.")
    parser.add_argument("--service-url", default=None, help="Running WebShop service URL, e.g. http://127.0.0.1:3001")
    parser.add_argument("--num-goals", type=int, default=None, help="Goal count when --service-url is not available.")
    parser.add_argument("--timeout", type=float, default=10.0)

    parser.add_argument("--env-seed", type=int, default=0, help="WebShop environment seed.")
    parser.add_argument("--train-start", type=int, default=500, help="First train goal index; lower indices are validation goals.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Number of train goal indices per rollout batch.")
    parser.add_argument("--total-rollouts", type=int, default=150, help="Number of train rollout batches to materialize.")
    parser.add_argument("--valid-size", type=int, default=100, help="Number of validation goals to materialize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.service_url:
        goal_count = _fetch_goal_count(args.service_url, args.timeout)
    elif args.num_goals is not None:
        goal_count = int(args.num_goals)
    else:
        raise ValueError("Either --service-url or --num-goals must be provided.")

    if goal_count <= 0:
        raise ValueError(f"goal_count must be positive, got {goal_count}")

    output_dir = Path(args.output_dir)
    train_rows, valid_rows, summary = _build_slime_webshop_rows(args, goal_count)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "valid.jsonl", valid_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
