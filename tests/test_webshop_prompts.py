import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "examples/webshop").resolve()))

from prompts import build_observation_prompt, format_available_actions  # noqa: E402


class WebShopPromptTest(unittest.TestCase):
    def test_available_actions_match_sdar_placeholder_shape(self):
        formatted = format_available_actions(
            {
                "has_search_bar": True,
                "clickables": ["Next >", "Back to Search", "search", "Buy Now"],
            }
        )

        self.assertIn("'search[<your query>]',", formatted)
        self.assertIn("'click[next >]',", formatted)
        self.assertIn("'click[back to search]',", formatted)
        self.assertIn("'click[buy now]',", formatted)
        self.assertNotIn("search[keywords]", formatted)
        self.assertNotIn("click[search]", formatted)

    def test_prompt_uses_sdar_admissible_action_language_and_search_guidance(self):
        prompt, history_used = build_observation_prompt(
            instruction_text="find a navy straight leg pair of jeans in size 32w x 30l under 70 dollars",
            current_observation="WebShop search page",
            available_actions={"has_search_bar": True, "clickables": []},
        )

        self.assertEqual(history_used, 0)
        self.assertIn("Your admissible actions of the current situation are", prompt)
        self.assertIn("search[<your query>]", prompt)
        self.assertIn("short core product query", prompt)
        self.assertIn("Do not put color, size, price", prompt)
        self.assertIn("eventually click[buy now]", prompt)
        self.assertNotIn("search[keywords]", prompt)

    def test_history_prompt_reports_step_context(self):
        prompt, history_used = build_observation_prompt(
            instruction_text="find a bed frame",
            current_observation="Search results",
            available_actions={"has_search_bar": False, "clickables": ["Buy Now"]},
            history=[
                {"observation": "Search page", "action": "search[bed frame]"},
                {"observation": "Results page", "action": "click[item]"},
            ],
            history_length=1,
        )

        self.assertEqual(history_used, 1)
        self.assertIn("already taken 2 step(s)", prompt)
        self.assertIn("You are now at step 3", prompt)
        self.assertIn("most recent 1 observations", prompt)


if __name__ == "__main__":
    unittest.main()
