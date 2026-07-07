from __future__ import annotations

import ast
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
ALFWORLD_EXAMPLE_DIR = REPO_ROOT / "examples" / "alfworld"
STUDENT_SCRIPT = ALFWORLD_EXAMPLE_DIR / "run_qwen2.5_0.5B_instruct_opd_from_3B.sh"
GRPO_OPSD_SCRIPT = ALFWORLD_EXAMPLE_DIR / "run_qwen2.5_3B_instruct_grpo_opsd.sh"
PURE_OPSD_SCRIPT = ALFWORLD_EXAMPLE_DIR / "run_qwen2.5_3B_instruct_opsd.sh"
ZOPD = ALFWORLD_EXAMPLE_DIR / "zopd.py"
ARGUMENTS = REPO_ROOT / "slime" / "utils" / "arguments.py"


def _script_text(path: Path = STUDENT_SCRIPT) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    return ast.get_source_segment(text, _find_function(_tree(path), name)) or ""


@pytest.mark.unit
def test_zopd_0_5b_from_3b_script_uses_student_model_and_external_teacher_url() -> None:
    text = _script_text()

    assert "scripts/models/qwen2.5-0.5B.sh" in text
    assert "TEACHER_URL" in text
    assert "--rm-url" in text
    assert "${TEACHER_URL}" in text
    assert "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}" in text
    assert "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}" in text
    assert "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}" in text


@pytest.mark.unit
def test_zopd_0_5b_from_3b_script_selects_normal_custom_opd_context() -> None:
    text = _script_text()

    assert "ALFWORLD_OPD_TEACHER_CONTEXT=${ALFWORLD_OPD_TEACHER_CONTEXT:-normal}" in text
    assert "ALFWORLD_OPD_TEACHER_SOURCE=${ALFWORLD_OPD_TEACHER_SOURCE:-external}" in text
    assert "--use-opd" in text
    assert "--opd-type zopd" in text
    assert "--opd-kl-coef" in text
    assert "batched_rollout.generate_rollout" in text
    assert "generate_with_alfworld.zero_alfworld_rewards_for_opd" in text
    assert "ALFWORLD_OPSD_SKILLS_DIR" not in text
    assert "ALFWORLD_OPD_USE_NATIVE" not in text
    assert "generate_with_alfworld.zero_alfworld_rewards_for_opsd" not in text
    assert "generate_with_alfworld.grpo_normalize_alfworld_steps" not in text


@pytest.mark.unit
def test_zopd_helper_contains_normal_and_privileged_contexts() -> None:
    text = ZOPD.read_text(encoding="utf-8")
    tree = _tree(ZOPD)

    for name in (
        "zopd_enabled",
        "zopd_teacher_context",
        "annotate_zopd_teacher_log_probs",
        "annotate_normal_teacher_log_probs",
        "annotate_privileged_teacher_log_probs",
        "build_privileged_teacher_prompt",
        "_score_privileged_teacher_log_probs",
        "_teacher_url",
        "ensure_zopd_teacher_log_probs",
    ):
        _find_function(tree, name)

    assert "ALFWORLD_OPD_USE_NATIVE" not in text
    assert "normal_opd_teacher_context_enabled" not in text
    assert "annotate_normal_opd_teacher_log_probs" not in text
    assert "annotate_opsd_teacher_log_probs" not in text
    assert "opsd_enabled" not in text

    assert 'getattr(args, "opd_type", None) == "zopd"' in _function_source(ZOPD, "zopd_enabled")

    teacher_context_source = _function_source(ZOPD, "zopd_teacher_context")
    assert "ALFWORLD_OPD_TEACHER_CONTEXT" in teacher_context_source
    assert "NORMAL_TEACHER_CONTEXT" in teacher_context_source
    assert "PRIVILEGED_TEACHER_CONTEXT" in teacher_context_source

    normal_source = _function_source(ZOPD, "annotate_normal_teacher_log_probs")
    assert "slime.rollout.on_policy_distillation" in normal_source
    assert "reward_func" in normal_source
    assert "post_process_rewards" in normal_source
    assert "build_privileged_teacher_prompt" not in normal_source
    assert "original_rewards" in normal_source
    assert "sample.reward = reward" in normal_source

    dispatcher_source = _function_source(ZOPD, "annotate_zopd_teacher_log_probs")
    assert "context == NORMAL_TEACHER_CONTEXT" in dispatcher_source
    assert "annotate_normal_teacher_log_probs(args, samples)" in dispatcher_source
    assert "annotate_privileged_teacher_log_probs(args, tokenizer, samples)" in dispatcher_source

    teacher_url_source = _function_source(ZOPD, "_teacher_url")
    assert "ALFWORLD_OPD_TEACHER_URL" in teacher_url_source
    assert "ALFWORLD_OPSD_TEACHER_URL" in teacher_url_source
    assert 'teacher_source == "external"' in teacher_url_source
    assert "_current_rollout_router_url(args)" in teacher_url_source


@pytest.mark.unit
def test_opsd_launchers_default_to_zopd_privileged_context() -> None:
    for script in (GRPO_OPSD_SCRIPT, PURE_OPSD_SCRIPT):
        text = _script_text(script)
        assert "OPSD_TYPE=${OPSD_TYPE:-zopd}" in text
        assert "ALFWORLD_OPD_TEACHER_CONTEXT=${ALFWORLD_OPD_TEACHER_CONTEXT:-privileged}" in text
        assert "--opd-type \"${OPSD_TYPE}\"" in text
        assert "OPSD_TYPE=self" not in text
        assert "compatibility alias" not in text


@pytest.mark.unit
def test_rollout_paths_import_zopd_only() -> None:
    expected_import = (
        "from zopd import annotate_zopd_teacher_log_probs, "
        "ensure_zopd_teacher_log_probs, zopd_enabled"
    )
    for path in (ALFWORLD_EXAMPLE_DIR / "batched_rollout.py", ALFWORLD_EXAMPLE_DIR / "generate_with_alfworld.py"):
        text = path.read_text(encoding="utf-8")
        assert expected_import in text
        assert "from opsd" not in text
        assert "annotate_opsd_teacher_log_probs" not in text
        assert "ensure_opsd_teacher_log_probs" not in text
        assert "opsd_enabled" not in text


@pytest.mark.unit
def test_opsd_file_removed_after_merge_into_zopd() -> None:
    assert not (ALFWORLD_EXAMPLE_DIR / "opsd.py").exists()


@pytest.mark.unit
def test_core_accepts_zopd_without_self_alias() -> None:
    text = ARGUMENTS.read_text(encoding="utf-8")

    assert 'choices=["sglang", "megatron", "zopd"]' in text
    assert 'choices=["sglang", "megatron", "zopd", "self"]' not in text
    assert "'self': Deprecated alias" not in text
    assert 'elif args.opd_type in {"sglang", "zopd"}' in text
    assert 'elif args.opd_type in {"sglang", "zopd", "self"}' not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
