"""Prepare ScienceWorld task/variation JSONL indices for slime training/eval."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from scienceworld_env import ScienceWorldTextEpisode


def write_split(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for index, row in enumerate(rows):
            writer.write(json.dumps({"index": index, "metadata": row}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {output_path}")


def _limit(rows: list[dict], limit: int, rng: random.Random, *, shuffle: bool) -> list[dict]:
    rows = list(rows)
    if shuffle:
        rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ScienceWorld JSONL task/variation indices for slime.")
    parser.add_argument("--local-dir", default="~/data/slime-scienceworld")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simplification", default=os.environ.get("SCIENCEWORLD_SIMPLIFICATION", "easy"))
    parser.add_argument("--env-step-limit", type=int, default=100)
    parser.add_argument("--jar-path", default=os.environ.get("SCIENCEWORLD_JAR_PATH"))
    parser.add_argument("--train-size", type=int, default=-1, help="<=0 keeps all train variations")
    parser.add_argument("--eval-size", type=int, default=32, help="<=0 keeps all dev/eval variations")
    parser.add_argument("--test-size", type=int, default=32, help="<=0 keeps all test variations")
    parser.add_argument("--shuffle", action="store_true", help="shuffle before applying size limits")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional ScienceWorld task names or zero-based ids to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser()
    rng = random.Random(args.seed)

    split_rows = {"train": [], "eval": [], "test": []}
    with ScienceWorldTextEpisode(env_step_limit=args.env_step_limit, jar_path=args.jar_path) as episode:
        requested = set(args.tasks or [])
        requested_ids = {int(item) for item in requested if str(item).isdigit()}
        requested_names = requested - {str(item) for item in requested_ids}
        for task_id, task_name in enumerate(episode.task_names()):
            if requested and task_name not in requested_names and task_id not in requested_ids:
                continue
            splits = episode.variation_splits(task_name, args.simplification)
            for split, variation_indices in splits.items():
                for variation_idx in variation_indices:
                    split_rows[split].append(
                        {
                            "task_name": task_name,
                            "variation_idx": int(variation_idx),
                            "split": split,
                            "simplification": args.simplification,
                            "env_step_limit": args.env_step_limit,
                        }
                    )

    split_rows["train"] = _limit(split_rows["train"], args.train_size, rng, shuffle=args.shuffle)
    split_rows["eval"] = _limit(split_rows["eval"], args.eval_size, rng, shuffle=args.shuffle)
    split_rows["test"] = _limit(split_rows["test"], args.test_size, rng, shuffle=args.shuffle)

    write_split(split_rows["train"], local_dir / "train_tasks.jsonl")
    write_split(split_rows["eval"], local_dir / "eval_tasks.jsonl")
    write_split(split_rows["test"], local_dir / "test_tasks.jsonl")


if __name__ == "__main__":
    main()
