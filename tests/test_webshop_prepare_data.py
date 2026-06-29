import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path("examples/webshop").resolve()))
from prepare_webshop_data import _build_slime_webshop_rows  # noqa: E402


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

    def test_has_no_legacy_seen_unseen_split(self):
        source = SCRIPT.read_text(encoding="utf-8").lower()

        self.assertNotIn(LEGACY_PREFIX, source)
        self.assertNotIn(ALGORITHM_PREFIX, source)
        self.assertNotIn("valid_seen", source)
        self.assertNotIn("valid_unseen", source)
        self.assertNotIn("_split_indices", source)
        self.assertNotIn("all.jsonl", source)
        self.assertNotIn("goal_pool", source)

    def test_default_schedule_uses_first_100_goals_for_train_time_validation(self):
        args = Namespace(
            env_seed=0,
            train_start=500,
            train_batch_size=16,
            total_rollouts=150,
            valid_size=100,
        )

        _, valid_rows, summary = _build_slime_webshop_rows(args, goal_count=6910)

        self.assertEqual(summary["schedule"]["valid_size"], 100)
        self.assertEqual(summary["splits"]["valid"]["count"], 100)
        self.assertEqual([row["metadata"]["goal_idx"] for row in valid_rows], list(range(100)))


def test_webshop_full_valid_eval_script_exists_and_uses_eval_only_path():
    script = Path("examples/webshop/eval_qwen2.5_3B_instruct_full_valid.sh")
    text = script.read_text(encoding="utf-8")

    assert "--num-rollout 0" in text
    assert "--eval-interval 1" in text
    assert "--eval-prompt-data valid_full" in text
    assert "WEBSHOP_FULL_EVAL_TASK_DIR" in text
    assert "--valid-size 500" in text
    assert "CKPT_STEP" in text
    assert "--ckpt-step" in text
    assert "WEBSHOP_TASK_DIR" not in text


if __name__ == "__main__":
    unittest.main()
