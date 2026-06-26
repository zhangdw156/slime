#!/usr/bin/env python3
"""Prepare lightweight WebShop prompt JSONL files for slime.

The WebShop service owns the actual goals and instructions.  Slime only needs a
stable list of ``goal_idx`` values in sample metadata; the custom WebShop rollout
uses that metadata to reset the remote service to the requested goal.
"""

from __future__ import annotations

import argparse
import json
import random
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


def _positive_or_none(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def _split_indices(args: argparse.Namespace, goal_count: int) -> dict[str, list[int]]:
    indices = list(range(goal_count))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    train_size = _positive_or_none(args.train_size)
    valid_seen_size = _positive_or_none(args.valid_seen_size)
    valid_unseen_size = _positive_or_none(args.valid_unseen_size)

    if train_size is None:
        train_size = int(goal_count * args.train_ratio)
    if valid_seen_size is None:
        valid_seen_size = int(goal_count * args.valid_seen_ratio)
    if valid_unseen_size is None:
        valid_unseen_size = goal_count - train_size - valid_seen_size

    total = train_size + valid_seen_size + valid_unseen_size
    if total > goal_count:
        raise ValueError(f"requested split size {total} exceeds goal_count {goal_count}")

    train = indices[:train_size]
    valid_seen = indices[train_size : train_size + valid_seen_size]
    valid_unseen = indices[train_size + valid_seen_size : train_size + valid_seen_size + valid_unseen_size]
    return {
        "train": train,
        "valid_seen": valid_seen,
        "valid_unseen": valid_unseen,
    }


def _record(goal_idx: int, split: str) -> dict[str, Any]:
    return {
        "text": f"webshop goal {goal_idx}",
        "metadata": {
            "goal_idx": goal_idx,
            "split": split,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create WebShop goal-index JSONL files for slime GRPO.")
    parser.add_argument("--output-dir", required=True, help="Directory for train/valid JSONL files.")
    parser.add_argument("--service-url", default=None, help="Running WebShop service URL, e.g. http://127.0.0.1:3001")
    parser.add_argument("--num-goals", type=int, default=None, help="Goal count when --service-url is not available.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle goal indices before splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-seen-ratio", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=None, help="Exact train size; overrides --train-ratio.")
    parser.add_argument("--valid-seen-size", type=int, default=None, help="Exact valid_seen size; overrides ratio.")
    parser.add_argument("--valid-unseen-size", type=int, default=None, help="Exact valid_unseen size; defaults to remaining goals.")
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

    splits = _split_indices(args, goal_count)
    output_dir = Path(args.output_dir)
    summary = {}
    for split, indices in splits.items():
        rows = [_record(goal_idx, split) for goal_idx in indices]
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
        summary[split] = {"count": len(rows), "first_goal_idx": indices[0] if indices else None}

    _write_jsonl(output_dir / "all.jsonl", [_record(goal_idx, "all") for goal_idx in range(goal_count)])
    (output_dir / "summary.json").write_text(json.dumps({"goal_count": goal_count, "splits": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"goal_count": goal_count, "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
