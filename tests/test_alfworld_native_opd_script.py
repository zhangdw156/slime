from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALFWORLD_EXAMPLE_DIR = REPO_ROOT / "examples" / "alfworld"
SCRIPT = ALFWORLD_EXAMPLE_DIR / "run_qwen2.5_0.5B_instruct_opd_from_3B.sh"
OPSD = ALFWORLD_EXAMPLE_DIR / "opsd.py"
NATIVE_OPD = ALFWORLD_EXAMPLE_DIR / "native_opd.py"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _opsd_tree() -> ast.Module:
    return ast.parse(OPSD.read_text(encoding="utf-8"))


def _native_opd_tree() -> ast.Module:
    return ast.parse(NATIVE_OPD.read_text(encoding="utf-8"))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


@pytest.mark.unit
def test_native_opd_0_5b_from_3b_script_uses_student_model_and_teacher_url() -> None:
    text = _script_text()

    assert "scripts/models/qwen2.5-0.5B.sh" in text
    assert "Qwen2.5-0.5B-Instruct_alfworld_native_opd_from_3B_slime" in text
    assert "TEACHER_URL" in text
    assert "--rm-url" in text
    assert "${TEACHER_URL}" in text
    assert "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}" in text
    assert "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}" in text
    assert "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}" in text


@pytest.mark.unit
def test_native_opd_0_5b_from_3b_script_selects_framework_native_opd() -> None:
    text = _script_text()

    assert "ALFWORLD_OPD_USE_NATIVE=1" in text
    assert "--use-opd" in text
    assert "--opd-type sglang" in text
    assert "--opd-kl-coef" in text
    assert "batched_rollout.generate_rollout" in text
    assert "generate_with_alfworld.zero_alfworld_rewards_for_opd" in text
    assert "ALFWORLD_OPSD_SKILLS_DIR" not in text
    assert "generate_with_alfworld.zero_alfworld_rewards_for_opsd" not in text
    assert "generate_with_alfworld.grpo_normalize_alfworld_steps" not in text


@pytest.mark.unit
def test_native_opd_helper_reuses_framework_helper_without_privileged_prompt() -> None:
    text = NATIVE_OPD.read_text(encoding="utf-8")
    tree = _native_opd_tree()
    native_enabled = _find_function(tree, "native_opd_enabled")
    native_annotate = _find_function(tree, "annotate_native_opd_teacher_log_probs")

    assert "ALFWORLD_OPD_USE_NATIVE" in ast.get_source_segment(text, native_enabled)
    native_source = ast.get_source_segment(text, native_annotate)
    assert "slime.rollout.on_policy_distillation" in native_source
    assert "reward_func" in native_source
    assert "post_process_rewards" in native_source
    assert "build_privileged_teacher_prompt" not in native_source
    assert "original_rewards" in native_source
    assert "sample.reward = reward" in native_source


@pytest.mark.unit
def test_opsd_dispatches_to_native_opd_helper_without_owning_native_logic() -> None:
    text = OPSD.read_text(encoding="utf-8")
    tree = _opsd_tree()
    dispatcher = _find_function(tree, "annotate_opsd_teacher_log_probs")

    assert "from native_opd import annotate_native_opd_teacher_log_probs, native_opd_enabled" in text
    assert "def native_opd_enabled" not in text
    assert "slime.rollout.on_policy_distillation" not in text
    dispatcher_source = ast.get_source_segment(text, dispatcher)
    assert "native_opd_enabled(args)" in dispatcher_source
    assert "annotate_native_opd_teacher_log_probs(args, samples)" in dispatcher_source
