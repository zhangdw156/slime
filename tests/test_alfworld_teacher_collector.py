from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ALFWORLD_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "alfworld"
if str(ALFWORLD_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(ALFWORLD_EXAMPLE_DIR))

import collect_teacher_trajectories as collector  # noqa: E402


class _Progress:
    def __init__(self) -> None:
        self.count = 0
        self.postfix: dict[str, object] = {}

    def update(self, n: int = 1) -> None:
        self.count += n

    def set_postfix(self, **kwargs: object) -> None:
        self.postfix = kwargs


def _output_args(tmp_path: Path, *, resume: bool = False, overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path),
        trajectory_output=None,
        summary_output=None,
        resume=resume,
        overwrite=overwrite,
    )


@pytest.mark.unit
def test_parse_args_defaults_to_eight_samples_per_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SAMPLES_PER_TASK", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_teacher_trajectories.py",
            "--tokenizer-path",
            "teacher-tokenizer",
            "--task-file",
            str(tmp_path / "train_games.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    args = collector.parse_args()

    assert args.samples_per_task == 8


@pytest.mark.unit
def test_prepare_outputs_resumes_from_all_trajectory_ledger(tmp_path: Path) -> None:
    all_output = tmp_path / "all_trajectories.jsonl"
    all_output.write_text(json.dumps({"trajectory_id": "failed-attempt", "reward": 0.0}) + "\n", encoding="utf-8")

    _, _, seen_ids = collector._prepare_outputs(_output_args(tmp_path, resume=True))

    assert seen_ids == {"failed-attempt"}


@pytest.mark.unit
def test_jsonl_trajectory_writer_flushes_all_rows(tmp_path: Path) -> None:
    all_output = tmp_path / "all.jsonl"
    writer = collector.JsonlTrajectoryWriter(all_output, mode="w")
    try:
        writer.write({"trajectory_id": "failed", "reward": 0.0, "status": "failed"})
        assert [json.loads(line) for line in all_output.read_text(encoding="utf-8").splitlines()] == [{"trajectory_id": "failed", "reward": 0.0, "status": "failed"}]

        writer.write({"trajectory_id": "success", "reward": 1.0, "status": "success"})
        assert [row["trajectory_id"] for row in map(json.loads, all_output.read_text(encoding="utf-8").splitlines())] == [
            "failed",
            "success",
        ]
    finally:
        writer.close()


@pytest.mark.unit
def test_finish_and_write_trajectory_is_single_write_with_terminal_status(tmp_path: Path) -> None:
    sample = collector.SampleRow(row_id=0, sample_id=3, metadata={"split": "train", "gamefile": "game.tw-pddl"})
    task = collector.SamplingTask(sample=sample, sample_repeat_id=7)
    state = collector.TrajectoryState(
        task=task,
        worker=object(),
        seed=0,
        trajectory={
            "trajectory_id": collector._trajectory_id(task),
            "reward": 0.0,
            "done": True,
            "won": False,
            "aborted": False,
            "truncated": False,
            "invalid_action_count": 2,
            "steps": [{"step": 1}],
            "error": None,
        },
    )
    progress = _Progress()
    counters: Counter[str] = Counter()

    with collector.JsonlTrajectoryWriter(tmp_path / "all.jsonl", mode="w") as writer:
        collector._finish_and_write_trajectory(
            trajectory=state,
            elapsed_sec=1.25,
            writer=writer,
            progress=progress,
            counters=counters,
        )
        collector._finish_and_write_trajectory(
            trajectory=state,
            elapsed_sec=9.99,
            writer=writer,
            progress=progress,
            counters=counters,
        )

    rows = [json.loads(line) for line in (tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["terminal_reason"] == "env_done_without_reward"
    assert rows[0]["penalized_reward"] == -0.02
    assert progress.count == 1
    assert counters["completed"] == 1
    assert counters["status:failed"] == 1
