from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCHED_ROLLOUT = REPO_ROOT / "examples" / "alfworld" / "batched_rollout.py"


def _tree() -> ast.Module:
    return ast.parse(BATCHED_ROLLOUT.read_text(encoding="utf-8"))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _find_function(node: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
            return child
    raise AssertionError(f"function {name} not found")


def _is_call_to(call: ast.Call, attr: str) -> bool:
    return isinstance(call.func, ast.Attribute) and call.func.attr == attr


@pytest.mark.unit
def test_alfworld_worker_pool_retires_ray_actor_processes() -> None:
    tree = _tree()
    pool = _find_class(tree, "AlfWorldWorkerPool")
    method_names = {node.name for node in pool.body if isinstance(node, ast.FunctionDef)}

    assert {"acquire", "release", "max_episodes_per_worker"}.issubset(method_names)
    assert "ALFWORLD_ENV_WORKER_MAX_EPISODES" in BATCHED_ROLLOUT.read_text(encoding="utf-8")

    release = _find_function(pool, "release")
    assert any(isinstance(node, ast.Call) and _is_call_to(node, "kill") for node in ast.walk(release))


@pytest.mark.unit
def test_run_batched_episodes_releases_workers_in_finally() -> None:
    tree = _tree()
    run_batched = _find_function(tree, "_run_batched_episodes")

    for node in ast.walk(run_batched):
        if not isinstance(node, ast.Try):
            continue
        for final_node in ast.walk(ast.Module(body=node.finalbody, type_ignores=[])):
            if not isinstance(final_node, ast.Call):
                continue
            func = final_node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "release"
                and isinstance(func.value, ast.Name)
                and func.value.id == "_WORKER_POOL"
            ):
                return

    raise AssertionError("_run_batched_episodes must release ALFWorld workers in a finally block")
