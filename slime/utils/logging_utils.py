import logging

import wandb

from . import swanlab_utils, wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        if getattr(args, "use_wandb", False):
            wandb_utils.init_wandb_primary(args, **kwargs)
        else:
            args.wandb_run_id = None
        if getattr(args, "use_swanlab", False):
            swanlab_utils.init_swanlab_primary(args)
        else:
            args.swanlab_run_id = None
    else:
        if getattr(args, "use_wandb", False):
            wandb_utils.init_wandb_secondary(args, **kwargs)
        if getattr(args, "use_swanlab", False):
            swanlab_utils.init_swanlab_secondary(args)


def update_tracking_open_metrics(args, router_addr):
    if getattr(args, "use_wandb", False):
        wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)
    if getattr(args, "use_swanlab", False):
        swanlab_utils.reinit_swanlab_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if getattr(args, "use_wandb", False):
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            logging.getLogger(__name__).exception("Failed to finish wandb run")

    if getattr(args, "use_swanlab", False):
        swanlab_utils.finish_swanlab(args)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if getattr(args, "use_wandb", False):
        wandb.log(metrics)

    if getattr(args, "use_swanlab", False):
        swanlab_utils.log_swanlab(args, metrics, step_key)

    if getattr(args, "use_tensorboard", False):
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
