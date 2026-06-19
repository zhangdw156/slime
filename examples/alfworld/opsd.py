"""OPSD helpers for ALFWorld privileged-teacher log-prob scoring.

The helper mirrors the SDAR ALFWorld skill prompt construction: pick
privileged skill text from ``skill_mapping.json``, prepend it under the
``[Privileged Skill Information]`` header, then score the student response under
that teacher-conditioned prompt.  The resulting response-token log-probs are
stored on ``Sample.teacher_log_probs`` so slime's native ``--use-opd`` advantage
penalty can consume them.  ALFWorld's default OPSD path uses
``--opd-type self``: the current rollout SGLang router scores the fixed student
response under the privileged prompt with ``max_new_tokens=0``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
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


class SkillProvider:
    """Load ALFWorld privileged skills using the same contract as SDAR."""

    def __init__(self, skills_dir: str | os.PathLike[str], skill_all: bool = False) -> None:
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self.skill_all = skill_all
        self.skill_mapping = self._load_skill_mapping()
        self.skill_contents = self._load_skill_content()
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        self.task_keywords: dict[str, list[str]] = self.skill_mapping.get("task_keywords", {})
        if self.skill_all:
            self._all_skills_text = self._build_all_skills_text()

    def _load_skill_mapping(self) -> dict[str, Any]:
        mapping_path = self.skills_dir / "skill_mapping.json"
        with mapping_path.open() as f:
            return json.load(f)

    def _load_skill_content(self) -> dict[str, str]:
        contents = {}
        for skill_name, filename in self.skill_mapping["skill_files"].items():
            with (self.skills_dir / filename).open() as f:
                contents[skill_name] = f.read().strip()
        return contents

    def _build_all_skills_text(self) -> str:
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for skill_name, content in self.skill_contents.items():
            if skill_name != "general_skills":
                parts.append(content)
        return "\n\n".join(parts)

    def _get_skill_text(self, task_type: str | None) -> str:
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        if task_type:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info(self, gamefile: str) -> str:
        """Return general + task-specific skill text for an ALFWorld game path."""
        if self.skill_all:
            return self._all_skills_text
        matched_task = None
        for task_type in self.task_to_skill:
            if task_type in gamefile:
                matched_task = task_type
                break
        return self._get_skill_text(matched_task)

    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        """Infer task type from prompt text using SDAR's any-keyword behavior."""
        if self.skill_all:
            return self._all_skills_text
        text_lower = prompt_text.lower()
        matched_tasks = []
        for task_type, keywords in self.task_keywords.items():
            if keywords and any(keyword in text_lower for keyword in keywords):
                matched_tasks.append(task_type)

        if not matched_tasks:
            return self._get_skill_text(None)

        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for task_type in matched_tasks:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info_from_data_source(self, data_source: str, prompt_text: str) -> str:
        """Keep SDAR's lookup priority while falling back to prompt matching.

        ALFWorld rows normally carry ``metadata.gamefile``.  This method exists
        for compatibility with SDAR's teacher-batch construction order.
        """
        if self.skill_all:
            return self._all_skills_text
        if data_source:
            for task_type in self.task_to_skill:
                if task_type in data_source:
                    return self._get_skill_text(task_type)
        return self.get_privileged_info_from_prompt(prompt_text)


@lru_cache(maxsize=8)
def _cached_provider(skills_dir: str, skill_all: bool) -> SkillProvider:
    return SkillProvider(skills_dir=skills_dir, skill_all=skill_all)


def get_skill_provider() -> SkillProvider:
    skills_dir = os.environ.get("ALFWORLD_OPSD_SKILLS_DIR", str(DEFAULT_SKILLS_DIR))
    skill_all = _str_to_bool(os.environ.get("ALFWORLD_OPSD_SKILL_ALL"), default=False)
    return _cached_provider(str(Path(skills_dir).expanduser().resolve()), skill_all)


def opsd_enabled(args: Any) -> bool:
    return bool(getattr(args, "use_opd", False) and getattr(args, "opd_type", None) in {"self", "sglang"})


def _current_rollout_router_url(args: Any) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _teacher_url(args: Any) -> str:
    if getattr(args, "opd_type", None) == "self":
        return _current_rollout_router_url(args)

    explicit_url = os.environ.get("ALFWORLD_OPSD_TEACHER_URL") or getattr(args, "rm_url", None)
    if explicit_url:
        return explicit_url
    return _current_rollout_router_url(args)


def _router_headers(args: Any, sample: Sample) -> dict[str, str] | None:
    session_id = getattr(sample, "session_id", None)
    if session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": session_id}
    return None


def _max_teacher_prompt_tokens(args: Any, response_length: int) -> int | None:
    override = os.environ.get("ALFWORLD_OPSD_MAX_PROMPT_TOKENS")
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning("Invalid ALFWORLD_OPSD_MAX_PROMPT_TOKENS=%r; ignoring", override)

    rollout_max_prompt_len = getattr(args, "rollout_max_prompt_len", None)
    if rollout_max_prompt_len is not None:
        return int(rollout_max_prompt_len)

    rollout_max_context_len = getattr(args, "rollout_max_context_len", None)
    if rollout_max_context_len is not None:
        return max(1, int(rollout_max_context_len) - int(response_length))

    return None


def build_privileged_teacher_prompt(sample: Sample, provider: SkillProvider | None = None) -> str:
    """Construct SDAR-style privileged prompt for the sample's fixed response."""
    provider = provider or get_skill_provider()
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    prompt_text = str(sample.prompt)

    gamefile = metadata.get("gamefile")
    data_source = metadata.get("data_source")
    if gamefile is not None:
        skill_text = provider.get_privileged_info(str(gamefile))
    elif data_source is not None:
        skill_text = provider.get_privileged_info_from_data_source(str(data_source), prompt_text)
    else:
        skill_text = provider.get_privileged_info_from_prompt(prompt_text)

    skill_prefix = f"[Privileged Skill Information]\n{skill_text}\n\n"
    return skill_prefix + prompt_text


def _extract_response_log_probs(payload: dict[str, Any], response_length: int) -> list[float]:
    token_logprobs = payload.get("meta_info", {}).get("input_token_logprobs") or []
    # SGLang reports a placeholder/None log-prob for the first input token.  Match
    # slime's generic OPD helper by dropping that element before slicing the
    # generated response suffix.
    values = [item[0] for item in token_logprobs[1:]]
    response_log_probs = values[-response_length:] if response_length else []
    if len(response_log_probs) != response_length:
        raise ValueError(
            f"Teacher returned {len(response_log_probs)} response logprobs for response_length={response_length}."
        )
    if any(value is None for value in response_log_probs):
        raise ValueError("Teacher response logprobs contain None values.")
    return [float(value) for value in response_log_probs]


async def score_sample_teacher_log_probs(
    args: Any, tokenizer: Any, sample: Sample, provider: SkillProvider
) -> list[float]:
    teacher_prompt = build_privileged_teacher_prompt(sample, provider)
    teacher_prompt_tokens = tokenizer.encode(teacher_prompt, add_special_tokens=False)
    max_prompt_tokens = _max_teacher_prompt_tokens(args, sample.response_length)
    if max_prompt_tokens is not None and len(teacher_prompt_tokens) > max_prompt_tokens:
        teacher_prompt_tokens = teacher_prompt_tokens[-max_prompt_tokens:]

    response_tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
    payload = {
        "input_ids": teacher_prompt_tokens + response_tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    output = await post(_teacher_url(args), payload, headers=_router_headers(args, sample))
    return _extract_response_log_probs(output, sample.response_length)


async def annotate_opsd_teacher_log_probs(args: Any, tokenizer: Any, samples: list[Sample]) -> None:
    """Populate ``Sample.teacher_log_probs`` for train samples when OPSD is enabled."""
    if not opsd_enabled(args) or not samples:
        return

    provider = get_skill_provider()
    concurrency = max(1, _get_int_env("ALFWORLD_OPSD_TEACHER_CONCURRENCY", DEFAULT_TEACHER_CONCURRENCY))
    semaphore = asyncio.Semaphore(concurrency)

    async def _score(sample: Sample) -> list[float]:
        async with semaphore:
            return await score_sample_teacher_log_probs(args, tokenizer, sample, provider)

    teacher_log_probs = await asyncio.gather(*[_score(sample) for sample in samples])
    for sample, log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = log_probs


def ensure_opsd_teacher_log_probs(args: Any, samples: list[Sample]) -> None:
    """Fail early if an OPD run would reach training without teacher log-probs."""
    if not opsd_enabled(args):
        return
    for sample in samples:
        if sample.teacher_log_probs is not None:
            continue
        if sample.loss_mask is not None and not any(sample.loss_mask):
            sample.teacher_log_probs = [0.0] * sample.response_length
            continue
        raise ValueError(
            "ALFWorld OPSD requires teacher_log_probs for every non-empty train sample when "
            f"--use-opd --opd-type {getattr(args, 'opd_type', None)} is enabled."
        )
