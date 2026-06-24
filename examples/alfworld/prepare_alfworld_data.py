#!/usr/bin/env python3
"""Prepare ALFWorld game-file JSONL indices for slime training/evaluation.

Each output row is intentionally lightweight: slime reads `index` as the prompt
key, while `metadata.gamefile` tells `generate_with_alfworld.py` which game to
load for the environment episode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
DEFAULT_TASK_TYPES = set(TASK_TYPES.values())
SPLIT_TO_SUBDIR = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def iter_games(alfworld_data: Path, split: str, task_types: set[str]) -> Iterable[dict]:
    root = alfworld_data / "json_2.1.1" / SPLIT_TO_SUBDIR[split]
    if not root.exists():
        raise FileNotFoundError(f"ALFWorld split directory not found: {root}")

    for traj_path in sorted(root.rglob("traj_data.json")):
        task_root = traj_path.parent
        root_str = str(task_root)
        if "movable" in root_str or "Sliced" in root_str:
            continue

        traj_data = _read_json(traj_path)
        task_type = traj_data.get("task_type")
        if task_type not in task_types:
            continue

        gamefile = task_root / "game.tw-pddl"
        if not gamefile.exists():
            continue

        try:
            game_data = _read_json(gamefile)
        except json.JSONDecodeError:
            continue
        if not game_data.get("solvable", False):
            continue

        yield {
            "gamefile": str(gamefile),
            "split": split,
            "task_type": task_type,
            "task_root": str(task_root),
        }


def write_split(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for index, row in enumerate(rows):
            writer.write(json.dumps({"index": index, "metadata": row}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ALFWorld JSONL task indices for slime.")
    parser.add_argument("--alfworld-data", default=os.environ.get("ALFWORLD_DATA", "~/data/alfworld"))
    parser.add_argument("--local-dir", default="~/data/slime-alfworld")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=-1, help="<=0 keeps all train games")
    parser.add_argument("--valid-seen-size", type=int, default=-1, help="<=0 keeps all valid_seen games")
    parser.add_argument("--valid-unseen-size", type=int, default=-1, help="<=0 keeps all valid_unseen games")
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=sorted(DEFAULT_TASK_TYPES),
        help="ALFWorld task_type names to keep",
    )
    return parser.parse_args()


def _limit(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def main() -> None:
    args = parse_args()
    alfworld_data = Path(args.alfworld_data).expanduser()
    local_dir = Path(args.local_dir).expanduser()
    task_types = set(args.task_types)
    rng = random.Random(args.seed)

    split_limits = {
        "train": args.train_size,
        "valid_seen": args.valid_seen_size,
        "valid_unseen": args.valid_unseen_size,
    }
    output_names = {
        "train": "train_games.jsonl",
        "valid_seen": "valid_seen_games.jsonl",
        "valid_unseen": "valid_unseen_games.jsonl",
    }

    for split, limit in split_limits.items():
        rows = list(iter_games(alfworld_data, split, task_types))
        rows = _limit(rows, limit, rng)
        write_split(rows, local_dir / output_names[split])


if __name__ == "__main__":
    main()
