"""Baseline and oracle planners."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache

from models import BaselineSummary
from server.aqi import edge_metrics, wait_metrics
from server.city_graph import get_neighbors
from server.grader import weighted_cost
from server.tasks import get_task


@dataclass
class RolloutResult:
    path: list[str]
    cumulative_exposure: float
    cumulative_time_minutes: int
    steps_taken: int
    wait_steps: int
    reached_destination: bool


def _legal_moves(task_id: str, node_id: str) -> list[tuple[str, str]]:
    task = get_task(task_id)
    options: list[tuple[str, str]] = []
    for edge in get_neighbors(node_id):
        for mode in task.allowed_modes:
            if mode in edge["allowed_modes"]:
                options.append((edge["target_node_id"], mode))
    return options


def _distance_to_goal(task_id: str, node_id: str) -> float:
    from server.city_graph import node_lookup

    task = get_task(task_id)
    nodes = node_lookup()
    destination = nodes[task.destination_node_id]
    node = nodes[node_id]
    return abs(destination["x"] - node["x"]) + abs(destination["y"] - node["y"])


def _baseline_rollout(task_id: str) -> RolloutResult:
    task = get_task(task_id)
    current_node = task.start_node_id
    elapsed_minutes = 0
    steps_taken = 0
    cumulative_exposure = 0.0
    path = [current_node]

    while steps_taken < task.max_steps and current_node != task.destination_node_id:
        best_option: tuple[float, str, str] | None = None
        for target_node_id, mode in _legal_moves(task_id, current_node):
            metrics = edge_metrics(task, current_node, target_node_id, mode, elapsed_minutes)
            sort_cost = metrics["time_minutes"] + (metrics["exposure"] * 0.01)
            candidate = (sort_cost, target_node_id, mode)
            if best_option is None or candidate < best_option:
                best_option = candidate

        if best_option is None:
            break

        _, target_node_id, mode = best_option
        metrics = edge_metrics(task, current_node, target_node_id, mode, elapsed_minutes)
        current_node = target_node_id
        path.append(current_node)
        cumulative_exposure += metrics["exposure"]
        elapsed_minutes += int(metrics["time_minutes"])
        steps_taken += 1

    return RolloutResult(
        path=path,
        cumulative_exposure=round(cumulative_exposure, 2),
        cumulative_time_minutes=elapsed_minutes,
        steps_taken=steps_taken,
        wait_steps=0,
        reached_destination=current_node == task.destination_node_id,
    )


def _oracle_rollout(task_id: str) -> RolloutResult:
    task = get_task(task_id)
    pq: list[tuple[float, str, int, int, float, int, list[str]]] = [
        (0.0, task.start_node_id, 0, 0, 0.0, 0, [task.start_node_id])
    ]
    best: dict[tuple[str, int, int], float] = {(task.start_node_id, 0, 0): 0.0}

    while pq:
        cost, node_id, elapsed_minutes, steps_taken, exposure, wait_steps, path = heapq.heappop(pq)
        if node_id == task.destination_node_id:
            return RolloutResult(
                path=path,
                cumulative_exposure=round(exposure, 2),
                cumulative_time_minutes=elapsed_minutes,
                steps_taken=steps_taken,
                wait_steps=wait_steps,
                reached_destination=True,
            )
        if steps_taken >= task.max_steps:
            continue

        for target_node_id, mode in _legal_moves(task_id, node_id):
            metrics = edge_metrics(task, node_id, target_node_id, mode, elapsed_minutes)
            next_elapsed = elapsed_minutes + int(metrics["time_minutes"])
            next_exposure = round(exposure + metrics["exposure"], 2)
            next_cost = weighted_cost(
                task_id=task_id,
                cumulative_exposure=next_exposure,
                cumulative_time_minutes=next_elapsed,
                wait_steps=wait_steps,
                distance_remaining=_distance_to_goal(task_id, target_node_id),
                reached_destination=(target_node_id == task.destination_node_id),
            )["weighted_cost"]
            next_key = (target_node_id, next_elapsed, steps_taken + 1)
            if next_key not in best or next_cost < best[next_key]:
                best[next_key] = next_cost
                heapq.heappush(
                    pq,
                    (
                        next_cost,
                        target_node_id,
                        next_elapsed,
                        steps_taken + 1,
                        next_exposure,
                        wait_steps,
                        path + [target_node_id],
                    ),
                )

        if task.allow_wait:
            wait = wait_metrics(task, node_id, elapsed_minutes)
            next_elapsed = elapsed_minutes + int(wait["time_minutes"])
            next_exposure = round(exposure + wait["exposure"], 2)
            next_cost = weighted_cost(
                task_id=task_id,
                cumulative_exposure=next_exposure,
                cumulative_time_minutes=next_elapsed,
                wait_steps=wait_steps + 1,
                distance_remaining=_distance_to_goal(task_id, node_id),
                reached_destination=False,
            )["weighted_cost"]
            next_key = (node_id, next_elapsed, steps_taken + 1)
            if next_key not in best or next_cost < best[next_key]:
                best[next_key] = next_cost
                heapq.heappush(
                    pq,
                    (
                        next_cost,
                        node_id,
                        next_elapsed,
                        steps_taken + 1,
                        next_exposure,
                        wait_steps + 1,
                        path + [node_id],
                    ),
                )

    return _baseline_rollout(task_id)


@lru_cache(maxsize=16)
def get_baseline_summary(task_id: str) -> BaselineSummary:
    baseline = _baseline_rollout(task_id)
    oracle = _oracle_rollout(task_id)
    baseline_cost = weighted_cost(
        task_id=task_id,
        cumulative_exposure=baseline.cumulative_exposure,
        cumulative_time_minutes=baseline.cumulative_time_minutes,
        wait_steps=baseline.wait_steps,
        distance_remaining=_distance_to_goal(task_id, baseline.path[-1]),
        reached_destination=baseline.reached_destination,
    )["weighted_cost"]
    oracle_cost = weighted_cost(
        task_id=task_id,
        cumulative_exposure=oracle.cumulative_exposure,
        cumulative_time_minutes=oracle.cumulative_time_minutes,
        wait_steps=oracle.wait_steps,
        distance_remaining=_distance_to_goal(task_id, oracle.path[-1]),
        reached_destination=oracle.reached_destination,
    )["weighted_cost"]
    return BaselineSummary(
        task_id=task_id,
        baseline_cost=baseline_cost,
        oracle_cost=oracle_cost,
        baseline_exposure=baseline.cumulative_exposure,
        baseline_time_minutes=baseline.cumulative_time_minutes,
        oracle_exposure=oracle.cumulative_exposure,
        oracle_time_minutes=oracle.cumulative_time_minutes,
        baseline_path=baseline.path,
        oracle_path=oracle.path,
    )
