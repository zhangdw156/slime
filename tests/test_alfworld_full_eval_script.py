from pathlib import Path

SCRIPT = Path("examples/alfworld/eval_qwen2.5_3B_instruct_full_valid.sh")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_full_valid_eval_script_exists_and_uses_eval_only_training_path():
    text = _script_text()

    assert "--num-rollout 0" in text
    assert "--eval-interval 1" in text
    assert "--eval-prompt-data" in text
    assert "batched_rollout.generate_rollout" in text
    assert "valid_seen_full" in text
    assert "valid_unseen_full" in text


def test_full_valid_eval_script_regenerates_separate_full_eval_indices():
    text = _script_text()

    assert "ALFWORLD_FULL_EVAL_TASK_DIR" in text
    assert "valid_seen_full_games.jsonl" in text
    assert "valid_unseen_full_games.jsonl" in text
    assert "iter_games" in text
    assert "write_split" in text
    assert "train_games.jsonl" not in text


def test_full_valid_eval_script_supports_checkpoint_step_selection_and_tracker_names():
    text = _script_text()

    assert "CKPT_STEP" in text
    assert "--ckpt-step" in text
    assert "iter_%07d" in text
    assert "SWANLAB_EXPERIMENT_NAME" in text
    assert "WANDB_GROUP" in text
    assert "TB_EXPERIMENT_NAME" in text
