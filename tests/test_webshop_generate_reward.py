import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((ROOT / "examples/webshop").resolve()))

NUM_GPUS = 0


class _DynamicFilterOutput:
    def __init__(self, *args, **kwargs):
        self.keep = kwargs.get("keep", args[0] if args else None)
        self.reason = kwargs.get("reason")


class _GenerateState:
    pass


class _FakeTensor(list):
    def mean(self):
        return sum(self) / len(self)

    def std(self):
        mean = self.mean()
        if len(self) <= 1:
            return 0.0
        return (sum((value - mean) ** 2 for value in self) / (len(self) - 1)) ** 0.5

    def __sub__(self, other):
        return _FakeTensor([value - float(other) for value in self])

    def __truediv__(self, other):
        return _FakeTensor([value / float(other) for value in self])

    def tolist(self):
        return list(self)


def _fake_tensor(values, dtype=None):
    return _FakeTensor([float(value) for value in values])


class _Sample:
    class Status:
        ABORTED = "aborted"
        TRUNCATED = "truncated"
        COMPLETED = "completed"

    def get_reward_value(self, args):
        return self.reward


async def _post(*args, **kwargs):
    raise AssertionError("HTTP post is not used by these helper tests")


sys.modules.setdefault(
    "slime.rollout.filter_hub.base_types",
    types.SimpleNamespace(DynamicFilterOutput=_DynamicFilterOutput),
)
sys.modules.setdefault("slime.rollout.sglang_rollout", types.SimpleNamespace(GenerateState=_GenerateState))
sys.modules.setdefault("slime.utils.http_utils", types.SimpleNamespace(post=_post))
sys.modules.setdefault("slime.utils.types", types.SimpleNamespace(Sample=_Sample))
sys.modules.setdefault("torch", types.SimpleNamespace(tensor=_fake_tensor, float="float", float64="float64"))

import generate_with_webshop as webshop_generate  # noqa: E402


class WebShopGenerateRewardTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("WEBSHOP_REWARD_MODE", None)

    def _sample(
        self,
        index,
        *,
        raw_reward,
        final_reward,
        done,
        trajectory_steps=1,
        group_index=None,
        group_id=None,
        step_invalid_action_penalty=0.0,
    ):
        sample = _Sample()
        sample.index = index
        sample.group_index = index if group_index is None else group_index
        sample.group_id = index if group_id is None else group_id
        sample.reward = final_reward
        sample.metadata = {
            "raw_reward": raw_reward,
            "episode_reward": raw_reward if done else 0.0,
            "final_reward": final_reward,
            "done": done,
            "trajectory": [{} for _ in range(trajectory_steps)],
            "invalid_action_count": 0,
            "step_invalid_action_penalty": step_invalid_action_penalty,
        }
        return sample

    def test_webshop_summary_adds_paper_score_and_succ_without_changing_existing_metrics(self):
        metrics = webshop_generate._webshop_summary_from_samples(
            [
                self._sample(0, raw_reward=1.0, final_reward=1.0, done=True),
                self._sample(1, raw_reward=0.4, final_reward=0.4, done=True),
                self._sample(2, raw_reward=0.0, final_reward=0.0, done=False),
            ]
        )

        self.assertAlmostEqual(metrics["webshop/score"], (1.0 + 0.4 + 0.0) / 3)
        self.assertAlmostEqual(metrics["webshop/succ"], 1 / 3)
        self.assertAlmostEqual(metrics["webshop/raw_reward_mean"], metrics["webshop/score"])
        self.assertAlmostEqual(metrics["webshop/success_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["webshop/final_reward_mean"], (1.0 + 0.4 + 0.0) / 3)

    def test_dense_reward_mode_is_default_and_keeps_partial_reward(self):
        os.environ.pop("WEBSHOP_REWARD_MODE", None)

        self.assertEqual(webshop_generate._get_reward_mode(), "dense")
        self.assertEqual(
            webshop_generate._episode_reward_from_raw(raw_reward=0.35, done=True, reward_mode="dense"),
            0.35,
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
            1.0,
        )

    def test_grpo_normalizes_final_trajectory_reward_without_step_penalty_afterward(self):
        args = types.SimpleNamespace(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
        )
        samples = [
            self._sample(
                0,
                raw_reward=1.0,
                final_reward=0.99,
                done=True,
                group_index=0,
                group_id=10,
                step_invalid_action_penalty=0.01,
            ),
            self._sample(
                1,
                raw_reward=1.0,
                final_reward=0.99,
                done=True,
                group_index=0,
                group_id=10,
                step_invalid_action_penalty=0.0,
            ),
            self._sample(2, raw_reward=0.5, final_reward=0.5, done=True, group_index=0, group_id=11),
        ]

        raw_rewards, processed = webshop_generate.grpo_normalize_webshop_steps(args, samples)

        self.assertEqual(raw_rewards, [0.99, 0.99, 0.5])
        self.assertAlmostEqual(processed[0], 0.245)
        self.assertAlmostEqual(processed[1], 0.245)
        self.assertAlmostEqual(processed[2], -0.245)

    def test_dynamic_filter_uses_final_reward_after_invalid_penalty(self):
        args = types.SimpleNamespace()
        filtered = webshop_generate.check_episode_reward_nonzero_std(
            args,
            [
                [self._sample(0, raw_reward=1.0, final_reward=1.0, done=True)],
                [self._sample(1, raw_reward=1.0, final_reward=0.99, done=True)],
            ],
        )

        self.assertTrue(filtered.keep)

        zero_std = webshop_generate.check_episode_reward_nonzero_std(
            args,
            [
                [self._sample(0, raw_reward=1.0, final_reward=1.0, done=True)],
                [self._sample(1, raw_reward=1.0, final_reward=1.0, done=True)],
            ],
        )

        self.assertFalse(zero_std.keep)
        self.assertEqual(zero_std.reason, "zero_std_1.0")

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
