import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

NUM_GPUS = 0


def _load_command_utils(monkeypatch):
    typer_utils_stub = ModuleType("slime.utils.external_utils.typer_utils")
    typer_utils_stub.dataclass_cli = lambda cls: cls
    misc_stub = ModuleType("slime.utils.misc")
    misc_stub.exec_command = lambda *args, **kwargs: "0"

    monkeypatch.setitem(sys.modules, "slime.utils.external_utils.typer_utils", typer_utils_stub)
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc_stub)

    path = Path(__file__).resolve().parents[2] / "slime" / "utils" / "external_utils" / "command_utils.py"
    spec = importlib.util.spec_from_file_location("_command_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_default_swanlab_args(monkeypatch, tmp_path):
    monkeypatch.setenv("SWANLAB_API_KEY", "test-key")

    U = _load_command_utils(monkeypatch)
    args = U.get_default_swanlab_args(str(tmp_path / "test_short.py"), run_name_prefix="prefix", run_id="run123")

    assert "--use-swanlab" in args
    assert "--swanlab-project slime-test_short" in args
    assert "--swanlab-group prefix_run123" in args
    assert "--swanlab-key 'test-key'" in args
    assert "--disable-swanlab-random-suffix" in args


def test_get_default_tracking_args_prefers_swanlab_when_key_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("SWANLAB_API_KEY", "swanlab-key")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")

    U = _load_command_utils(monkeypatch)
    args = U.get_default_tracking_args(str(tmp_path / "test_short.py"), run_id="run123")

    assert "--use-swanlab" in args
    assert "--swanlab-key 'swanlab-key'" in args
    assert "--use-wandb" not in args


def test_get_default_tracking_args_can_target_wandb(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")

    U = _load_command_utils(monkeypatch)
    args = U.get_default_tracking_args(
        str(tmp_path / "test_short.py"), run_name_prefix="prefix", run_id="run123", backend="wandb"
    )

    assert "--use-wandb" in args
    assert "--wandb-project slime-test_short" in args
    assert "--wandb-group prefix_run123" in args
    assert "--wandb-key 'wandb-key'" in args
    assert "--disable-wandb-random-suffix" in args


def test_logging_utils_dispatches_to_wandb_and_swanlab(monkeypatch):
    wandb_calls = []
    swanlab_calls = []

    wandb_stub = SimpleNamespace(run=SimpleNamespace(id="wandb-run"), log=lambda metrics: wandb_calls.append(metrics))
    tensorboard_stub = ModuleType("slime.utils.tensorboard_utils")
    tensorboard_stub._TensorboardAdapter = lambda args: SimpleNamespace(log=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)
    monkeypatch.setitem(sys.modules, "slime.utils.tensorboard_utils", tensorboard_stub)

    import slime.utils.logging_utils as logging_utils

    logging_utils = importlib.reload(logging_utils)
    monkeypatch.setattr(
        logging_utils,
        "swanlab_utils",
        SimpleNamespace(log_swanlab=lambda args, metrics, step_key: swanlab_calls.append((metrics, step_key))),
    )

    args = SimpleNamespace(use_wandb=True, use_swanlab=True, use_tensorboard=False)
    metrics = {"rollout/step": 3, "metric": 1.5}
    logging_utils.log(args, metrics, step_key="rollout/step")

    assert wandb_calls == [metrics]
    assert swanlab_calls == [(metrics, "rollout/step")]


def test_swanlab_primary_and_secondary_init_share_run_id(monkeypatch):
    import slime.utils.swanlab_utils as swanlab_utils

    init_calls = []
    login_calls = []
    swanlab_stub = SimpleNamespace(
        login=lambda **kwargs: login_calls.append(kwargs),
        init=lambda **kwargs: init_calls.append(kwargs) or SimpleNamespace(id=kwargs["id"]),
        get_run=lambda: SimpleNamespace(id="active-run"),
        util=SimpleNamespace(generate_id=lambda: "generated-id"),
    )
    monkeypatch.setattr(swanlab_utils, "swanlab", swanlab_stub)

    args = SimpleNamespace(
        use_swanlab=True,
        use_wandb=False,
        swanlab_mode="cloud",
        swanlab_key="secret",
        swanlab_host=None,
        swanlab_web_host=None,
        swanlab_workspace="workspace",
        swanlab_project="project",
        swanlab_group="group",
        swanlab_experiment_name=None,
        swanlab_dir=None,
        swanlab_random_suffix=False,
        swanlab_run_id=None,
    )

    swanlab_utils.init_swanlab_primary(args)
    assert args.swanlab_run_id == "generated-id"
    assert login_calls == [{"api_key": "secret", "host": None, "web_host": None, "save": False}]
    assert init_calls[0]["project"] == "project"
    assert init_calls[0]["workspace"] == "workspace"
    assert init_calls[0]["group"] == "group"
    assert init_calls[0]["experiment_name"] == "group"
    assert init_calls[0]["id"] == "generated-id"
    assert init_calls[0]["resume"] == "allow"
    assert init_calls[0]["parallel"] == "shared"

    swanlab_utils.init_swanlab_secondary(args)
    assert init_calls[1]["id"] == "generated-id"
    assert init_calls[1]["resume"] == "allow"


def test_swanlab_log_removes_step_and_uses_explicit_step(monkeypatch):
    import slime.utils.swanlab_utils as swanlab_utils

    logged = []
    monkeypatch.setattr(swanlab_utils, "swanlab", SimpleNamespace(log=lambda metrics, step=None: logged.append((metrics, step))))

    args = SimpleNamespace(use_swanlab=True)
    swanlab_utils.log_swanlab(args, {"rollout/step": 5, "metric": 2.5}, step_key="rollout/step")

    assert logged == [({"metric": 2.5}, 5)]


def test_swanlab_open_metrics_monitor_collects_and_logs(monkeypatch):
    import slime.utils.swanlab_utils as swanlab_utils

    logged = []
    monkeypatch.setattr(swanlab_utils, "swanlab", SimpleNamespace(log=lambda metrics, step=None: logged.append((metrics, step))))
    monkeypatch.setattr(
        swanlab_utils,
        "httpx",
        SimpleNamespace(
            get=lambda url, timeout=10.0: SimpleNamespace(
                raise_for_status=lambda: None,
                text='''# HELP sglang_requests_total requests
sglang_requests_total{engine="rollout-0"} 12
sglang_latency_seconds 0.5
''',
            )
        ),
    )

    monitor = swanlab_utils._SwanlabOpenMetricsMonitor(
        args=SimpleNamespace(), router_addr="http://127.0.0.1:8000", interval_s=1
    )
    monitor.poll_once()

    assert logged == [
        (
            {
                "sglang_engine/sglang_requests_total_engine_rollout_0": 12.0,
                "sglang_engine/sglang_latency_seconds": 0.5,
            },
            0,
        )
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
