import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((ROOT / "examples/webshop").resolve()))


class _DynamicFilterOutput:
    def __init__(self, *args, **kwargs):
        pass


class _GenerateState:
    pass


class _Sample:
    class Status:
        ABORTED = "aborted"
        TRUNCATED = "truncated"
        COMPLETED = "completed"


async def _post(*args, **kwargs):
    raise AssertionError("HTTP post is not used by these helper tests")


sys.modules.setdefault(
    "slime.rollout.filter_hub.base_types",
    types.SimpleNamespace(DynamicFilterOutput=_DynamicFilterOutput),
)
sys.modules.setdefault("slime.rollout.sglang_rollout", types.SimpleNamespace(GenerateState=_GenerateState))
sys.modules.setdefault("slime.utils.http_utils", types.SimpleNamespace(post=_post))
sys.modules.setdefault("slime.utils.types", types.SimpleNamespace(Sample=_Sample))

import generate_with_webshop as webshop_generate  # noqa: E402


class WebShopGenerateRewardTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("WEBSHOP_REWARD_MODE", None)

    def test_dense_reward_mode_is_default_and_keeps_partial_reward(self):
        os.environ.pop("WEBSHOP_REWARD_MODE", None)

        self.assertEqual(webshop_generate._get_reward_mode(), "dense")
        self.assertEqual(
            webshop_generate._episode_reward_from_raw(raw_reward=0.35, done=True, reward_mode="dense"),
            3.5,
        )
        self.assertEqual(
            webshop_generate._episode_reward_from_raw(raw_reward=0.35, done=False, reward_mode="dense"),
            0.0,
        )

    def test_binary_reward_mode_preserves_full_success_gate(self):
        self.assertEqual(
            webshop_generate._episode_reward_from_raw(raw_reward=0.99, done=True, reward_mode="binary"),
            0.0,
        )
        self.assertEqual(
            webshop_generate._episode_reward_from_raw(raw_reward=1.0, done=True, reward_mode="binary"),
            10.0,
        )

    def test_service_action_uses_parser_normalization(self):
        available_actions = {"has_search_bar": False, "clickables": ["buy now"]}
        response = "<think>buy it</think><action>click[Buy  Now]</action>"

        parsed = webshop_generate.parse_action(response, available_actions)
        projected, valid_for_penalty, invalid_reason = webshop_generate._project_action_like_sdar(response)
        service_action = webshop_generate._service_action_from_projection(parsed.action, projected)

        self.assertTrue(valid_for_penalty)
        self.assertIsNone(invalid_reason)
        self.assertTrue(parsed.valid_admissible)
        self.assertEqual(projected, "click[buy  now]")
        self.assertEqual(service_action, "click[buy now]")

    def test_local_webshop_launcher_defaults_to_dense_and_checks_num_products(self):
        script = (ROOT / "examples/webshop/run_qwen2.5_3B_instruct_grpo.sh").read_text(encoding="utf-8")

        self.assertIn("WEBSHOP_REWARD_MODE=${WEBSHOP_REWARD_MODE:-dense}", script)
        self.assertIn("expected WebShop service num_products=1000", script)
        self.assertIn('"num_products": num_products', script)
        self.assertIn('\\"WEBSHOP_REWARD_MODE\\": \\"${WEBSHOP_REWARD_MODE}\\"', script)


if __name__ == "__main__":
    unittest.main()
