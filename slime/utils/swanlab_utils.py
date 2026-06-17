import importlib
import logging
import os
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx

try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency path
    swanlab = None

logger = logging.getLogger(__name__)

_OPEN_METRICS_MONITOR = None


def _require_swanlab():
    global swanlab
    if swanlab is None:
        try:
            swanlab = importlib.import_module("swanlab")
        except ImportError as exc:
            raise ImportError("swanlab is not installed. Please install it with: pip install swanlab") from exc
    return swanlab


def _is_offline_mode(args) -> bool:
    mode = getattr(args, "swanlab_mode", None) or os.environ.get("SWANLAB_MODE")
    return mode in {"offline", "local", "disabled"}


def _should_use_shared_parallel(args) -> bool:
    mode = getattr(args, "swanlab_mode", None) or os.environ.get("SWANLAB_MODE")
    return mode in {None, "cloud"}


def _maybe_login(args):
    if _is_offline_mode(args):
        return

    api_key = getattr(args, "swanlab_key", None)
    host = getattr(args, "swanlab_host", None)
    web_host = getattr(args, "swanlab_web_host", None)
    if api_key is None and host is None and web_host is None:
        return

    module = _require_swanlab()
    try:
        module.login(api_key=api_key, host=host, web_host=web_host, save=False)
    except TypeError:
        # Older SwanLab versions may not support save=.
        module.login(api_key=api_key, host=host, web_host=web_host)


def _generate_id() -> str:
    module = _require_swanlab()
    util = getattr(module, "util", None)
    if util is not None and hasattr(util, "generate_id"):
        return util.generate_id()
    return _default_run_name()


def _default_run_name() -> str:
    try:
        from slime.utils.external_utils.command_utils import create_run_id

        return create_run_id()
    except Exception:
        import uuid

        return f"run-{uuid.uuid4().hex[:8]}"



def _compute_config_for_logging(args):
    try:
        from .wandb_utils import _compute_config_for_logging as compute_wandb_config

        return compute_wandb_config(args)
    except Exception:
        output = deepcopy(args.__dict__)
        whitelist_env_vars = ["SLURM_JOB_ID"]
        output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
        return output


def _build_init_kwargs(args, primary: bool):
    if getattr(args, "swanlab_run_id", None) is None:
        args.swanlab_run_id = _generate_id()

    group = getattr(args, "swanlab_group", None) or getattr(args, "wandb_group", None)
    explicit_experiment_name = getattr(args, "swanlab_experiment_name", None)
    experiment_name = explicit_experiment_name or group or getattr(args, "swanlab_run_id", None) or _default_run_name()

    if getattr(args, "swanlab_random_suffix", True) and group:
        suffix = _generate_id()
        group = f"{group}_{suffix}"
        if explicit_experiment_name is None:
            experiment_name = group

    init_kwargs = {
        "project": getattr(args, "swanlab_project", None) or getattr(args, "wandb_project", None) or "slime",
        "workspace": getattr(args, "swanlab_workspace", None),
        "group": group,
        "experiment_name": experiment_name,
        "config": _compute_config_for_logging(args),
        "mode": getattr(args, "swanlab_mode", None),
        "id": getattr(args, "swanlab_run_id", None),
        "resume": "allow" if getattr(args, "swanlab_run_id", None) is not None else None,
        "reinit": True,
    }

    if _should_use_shared_parallel(args):
        init_kwargs["parallel"] = "shared"

    if swanlab_dir := getattr(args, "swanlab_dir", None):
        os.makedirs(swanlab_dir, exist_ok=True)
        init_kwargs["logdir"] = swanlab_dir
        logger.info("SwanLab logs will be stored in: %s", swanlab_dir)

    return {k: v for k, v in init_kwargs.items() if v is not None}


def init_swanlab_primary(args):
    if not getattr(args, "use_swanlab", False):
        args.swanlab_run_id = None
        return

    module = _require_swanlab()
    _maybe_login(args)

    init_kwargs = _build_init_kwargs(args, primary=True)
    run = module.init(**init_kwargs)
    active_run = run or getattr(module, "get_run", lambda: None)()
    if active_run is not None and getattr(active_run, "id", None) is not None:
        args.swanlab_run_id = active_run.id

    logger.info(
        "SwanLab initialized (primary). project=%s experiment_name=%s run_id=%s",
        init_kwargs.get("project"),
        init_kwargs.get("experiment_name"),
        getattr(args, "swanlab_run_id", None),
    )


def init_swanlab_secondary(args):
    if not getattr(args, "use_swanlab", False):
        return

    swanlab_run_id = getattr(args, "swanlab_run_id", None)
    if swanlab_run_id is None:
        return

    module = _require_swanlab()
    _maybe_login(args)

    init_kwargs = _build_init_kwargs(args, primary=False)
    init_kwargs["id"] = swanlab_run_id
    init_kwargs["resume"] = "allow"
    module.init(**init_kwargs)
    logger.info("SwanLab initialized (secondary), joined run_id=%s", swanlab_run_id)


def log_swanlab(args, metrics: dict[str, Any], step_key: str):
    if not getattr(args, "use_swanlab", False):
        return

    module = _require_swanlab()
    payload = {k: v for k, v in metrics.items() if k != step_key}
    if not payload:
        return

    step = metrics.get(step_key)
    if step is not None:
        try:
            step = int(step)
        except (TypeError, ValueError):
            pass
        module.log(payload, step=step)
    else:
        module.log(payload)


def finish_swanlab(args):
    global _OPEN_METRICS_MONITOR

    if not getattr(args, "use_swanlab", False) or swanlab is None:
        return

    try:
        if _OPEN_METRICS_MONITOR is not None:
            _OPEN_METRICS_MONITOR.stop()
            _OPEN_METRICS_MONITOR = None

        run = getattr(swanlab, "get_run", lambda: None)()
        if run is not None:
            swanlab.finish()
    except Exception:
        logger.exception("Failed to finish SwanLab run")


def reinit_swanlab_primary_with_open_metrics(args, router_addr):
    global _OPEN_METRICS_MONITOR

    if not getattr(args, "use_swanlab", False) or router_addr is None:
        return

    interval_s = int(getattr(args, "swanlab_open_metrics_interval", 0) or 0)
    if interval_s <= 0:
        return

    _require_swanlab()
    if _is_offline_mode(args):
        logger.info("SwanLab open metrics disabled in offline/local/disabled mode.")
        return

    if _OPEN_METRICS_MONITOR is not None:
        _OPEN_METRICS_MONITOR.stop()

    logger.info("Starting SwanLab open metrics monitor at %s.", router_addr)
    _OPEN_METRICS_MONITOR = _SwanlabOpenMetricsMonitor(
        args=args,
        router_addr=router_addr,
        interval_s=interval_s,
    ).start()


def _sanitize_metric_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name).strip("_")


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        metric_expr, value_text = parts[0], parts[1]
        try:
            value = float(value_text)
        except ValueError:
            continue

        if "{" in metric_expr:
            name, labels_text = metric_expr.split("{", 1)
            labels_text = labels_text.rstrip("}")
            label_parts = []
            for item in labels_text.split(","):
                if not item or "=" not in item:
                    continue
                key, raw_value = item.split("=", 1)
                clean_value = raw_value.strip().strip('"')
                label_parts.append(f"{_sanitize_metric_name(key)}_{_sanitize_metric_name(clean_value)}")
            metric_name = "_".join([name, *label_parts]) if label_parts else name
        else:
            metric_name = metric_expr

        metrics[_sanitize_metric_name(metric_name)] = value

    return metrics


@dataclass
class _SwanlabOpenMetricsMonitor:
    args: object
    router_addr: str
    interval_s: int

    def __post_init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._poll_step = 0

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="swanlab-open-metrics", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.interval_s)

    def poll_once(self):
        try:
            response = httpx.get(f"{self.router_addr}/engine_metrics", timeout=10.0)
            response.raise_for_status()
            metrics = _parse_prometheus_metrics(response.text)
            if not metrics:
                return

            payload = {f"sglang_engine/{key}": value for key, value in metrics.items()}
            swanlab.log(payload, step=self._poll_step)
            self._poll_step += 1
        except Exception:
            logger.exception("Failed to collect SwanLab open metrics from %s", self.router_addr)
