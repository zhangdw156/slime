import dataclasses
import itertools
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, close_http_client, get_host_info, init_http_client, terminate_process
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.port_allocator import reserve_ports
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .rollout_validation import validate_server_group_gpu_indices
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# SGLang derives its internal gRPC port as ``ServerArgs.port + 10000``
# when SGLANG_GRPC_PORT is not set.  Keep automatically allocated public
# server ports below this ceiling so the derived gRPC port stays valid.
SGLANG_GRPC_PORT_OFFSET = 10000
SGLANG_MAX_SERVER_PORT = 65535 - SGLANG_GRPC_PORT_OFFSET


@dataclasses.dataclass
class RouterHandle:
    ip: str
    port: int
    prometheus_port: int | None = None
    process: multiprocessing.Process | None = None
    owned: bool = True

    def shutdown(self) -> None:
        if self.owned and self.process is not None:
            terminate_process(self.process)
            self.process = None


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpu_per_engine=num_gpu_per_engine,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
                    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    "SLIME_ENABLE_PROFILING": "true",
                }.items()
            }
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        try:
            if self.args.rollout_external:
                addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                    args=self.args, rollout_engines=rollout_engines
                )
            else:
                # base_port is ignored by the auto allocator, but keep the
                # calculation for compatibility with older helper signatures.
                base_port = max(port_cursors.values()) if port_cursors else 15000
                addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                    args=self.args,
                    rollout_engines=rollout_engines,
                    worker_type=self.worker_type,
                    num_gpus_per_engine=self.num_gpus_per_engine,
                    rank_offset=self.rank_offset,
                    base_port=base_port,
                )

            init_handles = [
                engine.init.remote(
                    **(addr_and_ports[rank]),
                    router_ip=self.router_ip,
                    router_port=self.router_port,
                )
                for rank, engine in rollout_engines
            ]
            return init_handles, port_cursors
        except Exception:
            for rank, engine in rollout_engines:
                try:
                    ray.get(engine.shutdown.remote(), timeout=10)
                except Exception:
                    pass
                try:
                    ray.kill(engine, no_restart=True)
                except Exception:
                    pass
                idx = rank - self.rank_offset
                if 0 <= idx < len(self.all_engines):
                    self.all_engines[idx] = None
            raise

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups.

        Used instead of ``resume_memory_occupation(tags=[WEIGHTS])`` so that
        CPU memory is not consumed by offloaded weight copies.
        """
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    router_handle: RouterHandle | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []

    def shutdown(self, timeout: float = 60.0) -> None:
        """Best-effort shutdown of SGLang engines, actors, and owned router."""

        actors = [engine for engine in self.all_engines if engine is not None]
        refs = []
        for engine in actors:
            try:
                refs.append(engine.shutdown.remote())
            except Exception as e:
                logger.warning(f"Failed to request SGLang engine shutdown: {e}")

        if refs:
            ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
            for ref in ready:
                try:
                    ray.get(ref)
                except Exception as e:
                    logger.warning(f"SGLang engine shutdown raised: {e}")
            for ref in pending:
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
                logger.warning("Timed out waiting for SGLang engine shutdown; killing actor.")

        for engine in actors:
            try:
                ray.kill(engine, no_restart=True)
            except Exception as e:
                logger.warning(f"Failed to kill SGLang engine actor: {e}")

        for group in self.server_groups:
            group.all_engines = [None for _ in group.all_engines]

        if self.router_handle is not None:
            try:
                self.router_handle.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown SGLang router {self.router_ip}:{self.router_port}: {e}")


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args
        self.servers: dict[str, RolloutServer] = {}
        self._health_monitors = []
        self._disposed = False

        try:
            data_source_cls = load_function(self.args.data_source_path)
            self.data_source = data_source_cls(args)

            self.generate_rollout = load_function(self.args.rollout_function_path)
            self.eval_generate_rollout = load_function(self.args.eval_function_path)
            self.custom_reward_post_process_func = None
            if self.args.custom_reward_post_process_path is not None:
                self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
            self.custom_convert_samples_to_train_data_func = None
            if self.args.custom_convert_samples_to_train_data_path is not None:
                self.custom_convert_samples_to_train_data_func = load_function(
                    self.args.custom_convert_samples_to_train_data_path
                )
            logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
            logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

            if self.args.debug_train_only:
                self.servers = {}
            else:
                init_http_client(args)
                self.servers = start_rollout_servers(args, pg)

            init_tracking(args, primary=False)
            self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
            self.rollout_id = -1

            if not self.args.debug_train_only and self.args.use_fault_tolerance:
                for srv in self.servers.values():
                    for group in srv.server_groups:
                        monitor = RolloutHealthMonitor(group, args)
                        monitor.start()
                        self._health_monitors.append(monitor)
                self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection
        except Exception:
            try:
                self.dispose()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup RolloutManager after init error: {cleanup_error}")
            raise

    def _get_metrics_router_addr(self) -> str | None:
        """Return the router address for scraping SGLang engine metrics.

        The sglang_router gateway exposes ``/engine_metrics`` on its main port,
        which aggregates Prometheus metrics from all backend sglang servers.
        Returns ``http://{ip}:{port}`` for the first server, or ``None`` when
        metrics are disabled or no servers are running.
        """
        srv = self.server
        if srv is None or srv.router_ip is None:
            return None
        return f"http://{srv.router_ip}:{srv.router_port}"

    def get_metrics_router_addr(self) -> str | None:
        """Public wrapper for remote calls from the driver process."""
        return self._get_metrics_router_addr()

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.server_groups[0].all_engines and self.server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        for monitor in getattr(self, "_health_monitors", []):
            try:
                monitor.stop()
            except Exception as e:
                logger.warning(f"Failed to stop rollout health monitor: {e}")

        for server in getattr(self, "servers", {}).values():
            try:
                server.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown rollout server {server.model_name}: {e}")

        try:
            close_http_client()
        except Exception as e:
            logger.warning(f"Failed to close rollout HTTP client: {e}")

        logging_utils.finish_tracking(self.args)

    @property
    def server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> RolloutServer | None:
        """Return the server with ``update_weights=True``.

        When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported).
        """
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates.

        Returns engines from the first model that has
        ``update_weights=True``.  Frozen models (reference, reward,
        etc.) are automatically excluded.
        """
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the group_id contract before flattening: any list[Sample]
            # encountered in the nested output must have group_id set on every
            # element. Default rollouts land at depth 1 and skip this validation;
            # compact / subagent paths that split one rollout into N samples must
            # set the same group_id on every sibling so the loss reducer counts
            # the group once instead of N times. Legacy rollout_id is accepted.
            _validate_group_id_annotated(data)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        # Group id (one per training aggregation unit). Default rollouts emit
        # one sample per group, so we fall back to the unique sample index.
        # Compact / subagent paths that emit multiple training samples per
        # group set ``Sample.group_id`` explicitly so all siblings share a
        # value; assigning legacy ``Sample.rollout_id`` still forwards here.
        group_ids = [sample.group_id if sample.group_id is not None else sample.index for sample in samples]

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "group_ids": group_ids,
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # Per-group aggregate, precomputed at the step level (where we can
        # see every sample of every group) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-group denominator even
        # when a group's samples land in different micro-batches (first-fit
        # packing can split a group across mbs):
        #
        #   ``group_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's group. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per group.
        group_id_list = train_data["group_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        group_total_mask: dict[int, int] = {}
        for group_id, ms in zip(group_id_list, mask_sums_per_sample, strict=True):
            group_total_mask[group_id] = group_total_mask.get(group_id, 0) + ms
        group_mask_sums = [group_total_mask[group_id] for group_id in group_id_list]
        train_data["group_mask_sums"] = group_mask_sums

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by group id (``samples[i].group_id``, falling back to
        ``samples[i].index``); each step holds exactly ``args.global_batch_size``
        groups so the training step count is fixed at
        ``rollout_batch_size * n_samples_per_prompt // global_batch_size``
        regardless of how many training samples each group produced.
        """
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=self.args.global_batch_size,
            group_indices=data["group_ids"],
        )

        # Package per-rank rollout_data
        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "group_ids",
                "group_mask_sums",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _validate_group_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``group_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In slime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-group),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``group_id`` (or legacy ``rollout_id``)
    and to share the same value, so the loss reducer counts the group once
    instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            group_ids = [s.group_id for s in node]
            missing = [i for i, group_id in enumerate(group_ids) if group_id is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but group_id is unset on "
                f"positions {missing}. Set Sample.group_id on every sibling so the loss "
                "reducer can aggregate them as one group instead of N."
            )
            assert (
                len(set(group_ids)) == 1
            ), f"Sibling samples from one compact rollout must share group_id; got {group_ids}."
        return
    for item in node:
        _validate_group_id_annotated(item, depth + 1)


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=None,
):
    """Reserve SGLang ports on the target Ray actor nodes.

    Ports are leased by each ``SGLangEngine`` actor so the reservation happens
    on the same node that will later bind the SGLang process.  The leases are
    released inside ``SGLangEngine.init`` immediately before process launch.
    """

    _ = base_port  # Kept for API compatibility with older callers.
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nodes_per_engine = max(1, _gpus_per_engine // args.num_gpus_per_node)
    dist_consecutive = 30 + args.sglang_dp_size

    addr_and_ports: dict[int, dict] = {}
    engine_by_rank = {rank: engine for rank, engine in rollout_engines}

    def reserve(
        engine,
        role: str,
        consecutive: int = 1,
        max_port: int = 65535,
    ) -> dict:
        return ray.get(
            engine.reserve_port.remote(
                role=role,
                consecutive=consecutive,
                max_port=max_port,
            )
        )

    def add_lease(rank: int, lease: dict) -> None:
        addr_and_ports.setdefault(rank, {}).setdefault("port_lease_tokens", []).append(lease["token"])

    for rank, engine in rollout_engines:
        addr_and_ports.setdefault(rank, {})

        server_lease = reserve(
            engine,
            f"sglang:{worker_type}:server",
            max_port=SGLANG_MAX_SERVER_PORT,
        )
        addr_and_ports[rank]["host"] = server_lease["host"]
        addr_and_ports[rank]["port"] = server_lease["port"]
        add_lease(rank, server_lease)

        nccl_lease = reserve(engine, f"sglang:{worker_type}:nccl")
        addr_and_ports[rank]["nccl_port"] = nccl_lease["port"]
        add_lease(rank, nccl_lease)

        if worker_type == "prefill":
            bootstrap_lease = reserve(engine, f"sglang:{worker_type}:bootstrap")
            addr_and_ports[rank]["disaggregation_bootstrap_port"] = bootstrap_lease["port"]
            add_lease(rank, bootstrap_lease)

    if nodes_per_engine > 1:
        group_start_ranks = sorted({rank_offset + ((rank - rank_offset) // nodes_per_engine) * nodes_per_engine for rank, _ in rollout_engines})
        for group_start in group_start_ranks:
            owner = engine_by_rank.get(group_start)
            assert owner is not None, f"First node rank {group_start} must be present for multi-node SGLang engine startup."
            dist_lease = reserve(owner, f"sglang:{worker_type}:dist_init", consecutive=dist_consecutive)
            dist_init_addr = f"{dist_lease['host']}:{dist_lease['port']}"
            add_lease(group_start, dist_lease)
            for i in range(nodes_per_engine):
                rank = group_start + i
                if rank in addr_and_ports:
                    addr_and_ports[rank]["dist_init_addr"] = dist_init_addr
    else:
        for rank, engine in rollout_engines:
            dist_lease = reserve(engine, f"sglang:{worker_type}:dist_init", consecutive=dist_consecutive)
            addr_and_ports[rank]["dist_init_addr"] = f"{dist_lease['host']}:{dist_lease['port']}"
            add_lease(rank, dist_lease)

    for i, _engine in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, {}


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> RouterHandle:
    """Start sglang_router and return its lifecycle handle."""

    if not force_new and args.sglang_router_ip is not None:
        return RouterHandle(ip=args.sglang_router_ip, port=args.sglang_router_port, owned=False)

    router_ip = _wrap_ipv6(get_host_info()[1])
    router_port_lease = None
    if force_new or args.sglang_router_port is None:
        router_port_lease = reserve_ports(router_ip, role="sglang:router")
        router_port = router_port_lease.port
    else:
        router_port = args.sglang_router_port

    prometheus_port_lease = reserve_ports(router_ip, role="sglang:router:prometheus")

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = prometheus_port_lease.port
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        # Disable circuit breaker to prevent RDMA transfer timeouts from
        # marking decode workers as dead. Timeouts are transient (PCIe
        # contention under high load) and do not indicate a dead server.
        router_args.disable_circuit_breaker = True

    # We will not use the health check from router.
    router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon

    # Release leased ports immediately before the router process binds them.
    if router_port_lease is not None:
        router_port_lease.release()
    prometheus_port_lease.release()

    process.start()
    # Wait 3 seconds
    time.sleep(3)
    if not process.is_alive():
        terminate_process(process)
        raise RuntimeError(f"SGLang router failed to start at {router_ip}:{router_port}")
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return RouterHandle(
        ip=router_ip,
        port=router_port,
        prometheus_port=router_args.prometheus_port,
        process=process,
        owned=True,
    )


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    started_groups: list[ServerGroup] = []
    router_handles: list[RouterHandle] = []
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    try:
        for model_idx, model_cfg in enumerate(config.models):
            model_cfg.resolve(args)

            has_pd = model_cfg.has_pd_disaggregation
            router_handle = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))
            router_handles.append(router_handle)
            router_ip, router_port = router_handle.ip, router_handle.port

            # Write back for backward compat (first model only).
            if model_idx == 0:
                args.sglang_router_ip = router_ip
                args.sglang_router_port = router_port

            server_groups: list[ServerGroup] = []
            port_cursors: dict[int, int] = {}

            has_epd = model_cfg.has_encoder_disaggregation

            def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
                nonlocal engine_offset, gpu_offset
                gpus_per_engine = group_cfg.num_gpus_per_engine
                num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
                num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

                group_abs_start = rollout_pg_offset + gpu_offset
                needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
                overrides = dict(group_cfg.overrides)
                if overrides_extra:
                    for k, v in overrides_extra.items():
                        overrides.setdefault(k, v)
                if args.offload_rollout and not needs_offload:
                    overrides.setdefault("enable_memory_saver", False)
                logger.info(
                    f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                    f"(abs={group_abs_start}): needs_offload={needs_offload}"
                )

                group = ServerGroup(
                    args=args,
                    pg=pg,
                    all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                    num_gpus_per_engine=gpus_per_engine,
                    num_new_engines=0,
                    worker_type=group_cfg.worker_type,
                    rank_offset=engine_offset,
                    gpu_offset=gpu_offset,
                    sglang_overrides=overrides,
                    needs_offload=needs_offload,
                    model_path=overrides.get("model_path", args.hf_checkpoint),
                    router_ip=router_ip,
                    router_port=router_port,
                )
                engine_offset += num_engines
                gpu_offset += group_cfg.num_gpus
                return group

            if has_epd:
                # --- Phase 1: start encoder groups, wait, collect URLs ---
                encoder_urls: list[str] = []
                for group_cfg in model_cfg.server_groups:
                    if group_cfg.worker_type != "encoder":
                        continue
                    group = _make_group(group_cfg, router_ip, router_port)
                    handles, port_cursors = group.start_engines(port_cursors)
                    if handles:
                        ray.get(handles)
                    urls = ray.get([e.get_url.remote() for e in group.engines])
                    encoder_urls.extend(u for u in urls if u is not None)
                    server_groups.append(group)
                    started_groups.append(group)

                logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

                # --- Phase 2: start non-encoder groups, injecting encoder URLs into
                # language-only LLM workers. Prefill groups use this for full EPD,
                # while regular groups allow encoder/LLM split without PD.
                non_encoder_handles: list = []
                for group_cfg in model_cfg.server_groups:
                    if group_cfg.worker_type == "encoder":
                        continue
                    overrides_extra = {}
                    if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                        overrides_extra["language_only"] = True
                        overrides_extra["encoder_urls"] = encoder_urls
                    group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                    handles, port_cursors = group.start_engines(port_cursors)
                    non_encoder_handles.extend(handles)
                    server_groups.append(group)
                    started_groups.append(group)

                if non_encoder_handles:
                    ray.get(non_encoder_handles)
            else:
                # No EPD — start all groups in one pass (original path).
                all_init_handles: list = []
                for group_cfg in model_cfg.server_groups:
                    group = _make_group(group_cfg, router_ip, router_port)
                    handles, port_cursors = group.start_engines(port_cursors)
                    all_init_handles.extend(handles)
                    server_groups.append(group)
                    started_groups.append(group)

                if all_init_handles:
                    ray.get(all_init_handles)

            servers[model_cfg.name] = RolloutServer(
                server_groups=server_groups,
                router_ip=router_ip,
                router_port=router_port,
                router_handle=router_handle,
                model_name=model_cfg.name,
                update_weights=model_cfg.update_weights,
            )

        # Expose per-model router info for custom rollout functions.
        args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

        return servers
    except Exception:
        for server in servers.values():
            try:
                server.shutdown()
            except Exception as e:
                logger.warning(f"Failed to cleanup started rollout server after startup error: {e}")

        if started_groups:
            try:
                RolloutServer(server_groups=started_groups).shutdown()
            except Exception as e:
                logger.warning(f"Failed to cleanup partially started SGLang groups after startup error: {e}")

        for router_handle in router_handles:
            try:
                router_handle.shutdown()
            except Exception as e:
                logger.warning(f"Failed to cleanup started router after startup error: {e}")
        raise


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
