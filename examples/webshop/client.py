"""Thin async client for a separately deployed WebShop service."""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from typing import Any

from slime.utils.http_utils import post

logger = logging.getLogger(__name__)

DEFAULT_WEBSHOP_SERVICE_URL = "http://127.0.0.1:3001"


def get_webshop_service_url() -> str:
    return os.environ.get("WEBSHOP_SERVICE_URL", DEFAULT_WEBSHOP_SERVICE_URL).rstrip("/")


async def reset_session(
    *,
    session_id: str,
    goal_idx: int | None = None,
    goal_seed: int | None = None,
    observation_mode: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"session_id": session_id}
    if goal_idx is not None:
        payload["goal_idx"] = int(goal_idx)
    if goal_seed is not None:
        payload["goal_seed"] = int(goal_seed)
    if observation_mode is not None:
        payload["observation_mode"] = observation_mode
    return await post(
        f"{get_webshop_service_url()}/v1/reset",
        payload,
        max_retries=10,
    )


async def step_session(*, session_id: str, action: str) -> dict[str, Any]:
    return await post(
        f"{get_webshop_service_url()}/v1/step",
        {"session_id": session_id, "action": action},
        max_retries=10,
    )


def _delete_url(url: str) -> None:
    request = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(request, timeout=10.0) as response:  # noqa: S310 - operator-provided service URL.
        response.read()


async def close_session(session_id: str) -> None:
    """Best-effort cleanup for server-side session state."""

    url = f"{get_webshop_service_url()}/v1/session/{session_id}"
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _delete_url, url)
    except Exception:
        logger.debug("Failed to close WebShop service session %s", session_id, exc_info=True)
