"""Native slime OPD helpers for ALFWorld custom rollout.

ALFWorld uses ``batched_rollout.generate_rollout`` instead of slime's default
``sglang_rollout.generate_and_rm`` path, so the framework-native SGLang OPD
reward hooks are not invoked automatically.  This module bridges that gap by
calling ``slime.rollout.on_policy_distillation`` directly for ALFWorld step
samples, without adding the privileged-skill prompt used by OPSD.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_CONCURRENCY = 64


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, os.environ.get(name), default)
        return default


def native_opd_enabled(args: Any) -> bool:
    """Use slime's framework-native SGLang OPD teacher scoring for ALFWorld."""
    return bool(
        getattr(args, "use_opd", False)
        and getattr(args, "opd_type", None) == "sglang"
        and _str_to_bool(os.environ.get("ALFWORLD_OPD_USE_NATIVE"), default=False)
    )


async def annotate_native_opd_teacher_log_probs(args: Any, samples: list[Sample]) -> None:
    """Populate ``teacher_log_probs`` through slime's native SGLang OPD helper.

    The helper temporarily stores teacher scoring payloads in ``sample.reward``
    because ``post_process_rewards`` follows the standard slime reward hook
    contract.  It restores ALFWorld's scalar environment rewards afterward so
    rollout metrics and ALFWorld reward post-processing keep their existing
    semantics.
    """
    if not samples:
        return

    from slime.rollout.on_policy_distillation import post_process_rewards, reward_func

    concurrency = max(1, _get_int_env("ALFWORLD_NATIVE_OPD_TEACHER_CONCURRENCY", DEFAULT_TEACHER_CONCURRENCY))
    semaphore = asyncio.Semaphore(concurrency)

    async def _score(sample: Sample) -> Any:
        async with semaphore:
            return await reward_func(args, sample)

    original_rewards = [copy.deepcopy(sample.reward) for sample in samples]
    teacher_payloads = await asyncio.gather(*[_score(sample) for sample in samples])
    try:
        for sample, payload in zip(samples, teacher_payloads, strict=True):
            sample.reward = payload
        post_process_rewards(args, samples)
    finally:
        for sample, reward in zip(samples, original_rewards, strict=True):
            sample.reward = reward
