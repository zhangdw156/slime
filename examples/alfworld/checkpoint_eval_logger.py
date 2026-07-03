"""Custom eval logger for checkpoint-sweep full-valid evaluation.

The default slime eval logger uses rollout_id-derived eval/step. For checkpoint
sweeps we want SwanLab's x-axis to be the checkpoint step, so this custom logger
sets eval/step from CHECKPOINT_EVAL_STEP and logs all eval metrics into the
current SwanLab/TensorBoard run.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import Any

from slime.utils import logging_utils

logger = logging.getLogger(__name__)


def _dict_add_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _add_default_dataset_metrics(
    args: Namespace, log_dict: dict[str, Any], key: str, dataset: dict[str, Any]
) -> None:
    """Keep slime's default per-dataset eval metrics while overriding only the step."""

    samples = dataset.get("samples")
    if samples is not None:
        # Import lazily: during real eval slime.ray.rollout is already loaded; lazy
        # import also keeps lightweight smoke tests from requiring Ray/SGLang.
        from slime.ray.rollout import compute_metrics_from_samples

        log_dict.update(
            _dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        )

    if getattr(args, "log_passrate", False):
        from slime.utils.metric_utils import compute_pass_rate

        rewards = dataset.get("rewards") or []
        log_dict.update(
            _dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=getattr(args, "n_samples_per_eval_prompt", 1),
                ),
                f"eval/{key}-",
            )
        )


def log_eval_at_checkpoint_step(
    rollout_id: int,
    args: Namespace,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Log eval metrics at checkpoint-step on the tracker x-axis.

    Return True to tell slime to skip its default eval logger.
    """

    raw_step = (
        os.environ.get("CHECKPOINT_EVAL_STEP")
        or os.environ.get("ALFWORLD_EVAL_CKPT_STEP")
        or getattr(args, "ckpt_step", None)
        or rollout_id
    )
    try:
        checkpoint_step = int(raw_step)
    except (TypeError, ValueError):
        checkpoint_step = int(rollout_id)

    log_dict: dict[str, Any] = dict(extra_metrics or {})
    for key, dataset in data.items():
        rewards = dataset.get("rewards") or []
        if rewards:
            log_dict[f"eval/{key}"] = _mean(rewards)
        truncated = dataset.get("truncated")
        if truncated is not None and len(truncated) > 0:
            log_dict[f"eval/{key}-truncated_ratio"] = sum(bool(x) for x in truncated) / len(truncated)
        _add_default_dataset_metrics(args, log_dict, key, dataset)

    # Keep both a scalar metric and the real tracker step for clarity.
    log_dict["eval/checkpoint_step"] = checkpoint_step
    log_dict["eval/step"] = checkpoint_step

    logging_utils.configure_logger()
    logger.info("checkpoint eval %s: %s", checkpoint_step, log_dict)
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
