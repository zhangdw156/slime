"""Custom eval logger for WebShop checkpoint-sweep full-valid evaluation.

The normal WebShop eval logger uses slime's rollout-derived eval/step.  For a
checkpoint sweep we need all full-valid scores in one tracker run and the x-axis
to be the checkpoint step.  This logger mirrors the WebShop eval summaries while
forcing eval/step from CHECKPOINT_EVAL_STEP / WEBSHOP_EVAL_CKPT_STEP.
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


def _checkpoint_step(rollout_id: int, args: Namespace) -> int:
    raw_step = (
        os.environ.get("CHECKPOINT_EVAL_STEP")
        or os.environ.get("WEBSHOP_EVAL_CKPT_STEP")
        or getattr(args, "ckpt_step", None)
        or rollout_id
    )
    try:
        return int(raw_step)
    except (TypeError, ValueError):
        return int(rollout_id)


def log_webshop_eval_at_checkpoint_step(
    rollout_id: int,
    args: Namespace,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Log WebShop eval metrics at checkpoint step and skip default logging."""

    # Lazy imports keep static smoke tests from requiring Ray/SGLang.
    from generate_with_webshop import _webshop_summary_from_samples
    from slime.ray.rollout import compute_metrics_from_samples
    from slime.utils.metric_utils import compute_pass_rate

    checkpoint_step = _checkpoint_step(rollout_id, args)
    log_dict: dict[str, Any] = dict(extra_metrics or {})

    for dataset_name, payload in data.items():
        rewards = payload.get("rewards") or []
        reward_values = [float(reward) for reward in rewards]
        log_dict[f"eval/{dataset_name}"] = _mean(reward_values)

        samples = payload.get("samples") or []
        if samples:
            log_dict.update(
                _dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{dataset_name}/")
            )
            for key, value in _webshop_summary_from_samples(samples).items():
                log_dict[f"eval/{dataset_name}/{key}"] = value

        truncated = payload.get("truncated")
        if truncated is not None:
            log_dict[f"eval/{dataset_name}-truncated_ratio"] = _mean([1.0 if item else 0.0 for item in truncated])

        if getattr(args, "log_passrate", False):
            log_dict.update(
                _dict_add_prefix(
                    compute_pass_rate(
                        flat_rewards=reward_values,
                        group_size=getattr(args, "n_samples_per_eval_prompt", 1),
                    ),
                    f"eval/{dataset_name}-",
                )
            )

    # Keep an explicit scalar too, but make the real tracker step the checkpoint.
    log_dict["eval/checkpoint_step"] = checkpoint_step
    log_dict["eval/step"] = checkpoint_step

    logging_utils.configure_logger()
    logger.info("webshop checkpoint eval %s: %s", checkpoint_step, log_dict)
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True
