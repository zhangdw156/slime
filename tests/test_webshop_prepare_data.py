import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path("examples/webshop").resolve()))
from prepare_webshop_data import _build_slime_webshop_rows  # noqa: E402


NUM_GPUS = 0
SCRIPT = Path("examples/webshop/prepare_webshop_data.py")
LEGACY_PREFIX = "sd" + "ar"
ALGORITHM_PREFIX = "gr" + "po"


class WebShopPrepareDataTest(unittest.TestCase):
    def test_cli_is_slime_webshop_only(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=True,
            text=True,
            capture_output=True,
        )

        help_text = result.stdout.lower()
        self.assertIn("create slime webshop goal-index jsonl", help_text)
        self.assertIn("train.jsonl", help_text)
        self.assertIn("valid.jsonl", help_text)
        self.assertNotIn(LEGACY_PREFIX, help_text)
        self.assertNotIn(ALGORITHM_PREFIX, help_text)
        self.assertNotIn("valid_seen", help_text)
        self.assertNotIn("valid_unseen", help_text)
        self.assertNotIn("--train-ratio", help_text)
        self.assertNotIn("--shuffle", help_text)
        self.assertNotIn("--group-size", help_text)
        self.assertNotIn("--train-batch-size", help_text)
        self.assertNotIn("--total-rollouts", help_text)

    def test_has_no_legacy_seen_unseen_split_or_rollout_schedule_args(self):
        source = SCRIPT.read_text(encoding="utf-8").lower()

        self.assertNotIn(LEGACY_PREFIX, source)
        self.assertNotIn(ALGORITHM_PREFIX, source)
        self.assertNotIn("valid_seen", source)
        self.assertNotIn("valid_unseen", source)
        self.assertNotIn("_split_indices", source)
        self.assertNotIn("all.jsonl", source)
        self.assertNotIn("goal_pool", source)
        self.assertNotIn("total_rollouts", source)
        self.assertNotIn("train_batch_size", source)

    def test_default_schedule_uses_full_train_pool_and_first_100_goals_for_validation(self):
        args = Namespace(
            env_seed=0,
            train_start=500,
            valid_size=100,
        )

        train_rows, valid_rows, summary = _build_slime_webshop_rows(args, goal_count=6910)

        self.assertEqual(summary["schedule"]["valid_size"], 100)
        self.assertEqual(summary["splits"]["train"]["count"], 6410)
        self.assertEqual(summary["splits"]["valid"]["count"], 100)
        self.assertEqual([row["metadata"]["goal_idx"] for row in train_rows[:3]], [500, 501, 502])
        self.assertEqual([row["metadata"]["goal_idx"] for row in train_rows[-3:]], [6907, 6908, 6909])
        self.assertEqual([row["metadata"]["goal_idx"] for row in valid_rows], list(range(100)))
        self.assertEqual({row["metadata"]["goal_seed"] for row in train_rows}, {0})
        self.assertNotIn("rollout_id", train_rows[0]["metadata"])
        self.assertNotIn("env_id", train_rows[0]["metadata"])
        self.assertNotIn("total_rollouts", summary["schedule"])
        self.assertNotIn("train_batch_size", summary["schedule"])

    def test_cli_writes_full_train_pool_with_num_goals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--num-goals",
                    "6910",
                    "--output-dir",
                    tmpdir,
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            output_dir = Path(tmpdir)
            train_rows = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            valid_rows = (output_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(train_rows), 6410)
        self.assertEqual(len(valid_rows), 100)
        self.assertEqual(summary["splits"]["train"]["count"], 6410)

    def test_webshop_full_valid_eval_script_exists_and_uses_eval_only_path(self):
        script = Path("examples/webshop/eval_qwen2.5_3B_instruct_full_valid.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("--num-rollout 0", text)
        self.assertIn("--eval-interval 1", text)
        self.assertIn("--eval-prompt-data valid_full", text)
        self.assertIn("WEBSHOP_FULL_EVAL_TASK_DIR", text)
        self.assertIn("--valid-size 500", text)
        self.assertIn("CKPT_STEP", text)
        self.assertIn("--ckpt-step", text)
        self.assertNotIn("WEBSHOP_TASK_DIR", text)
        self.assertNotIn("--train-batch-size", text)
        self.assertNotIn("--total-rollouts", text)


if __name__ == "__main__":
    unittest.main()
