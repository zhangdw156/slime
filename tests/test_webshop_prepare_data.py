import subprocess
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
