"""Deterministic grading utilities."""

from __future__ import annotations

from models import GradeRequest, GradeResponse
from tasks.tasks import get_task


def weighted_cost(
    task_id: str,
    cumulative_exposure: float,
    cumulative_time_minutes: int,
    wait_steps: int,
    distance_remaining: float,
    reached_destination: bool,
) -> dict[str, float]:
    task = get_task(task_id)
    exposure_component = round(task.exposure_weight * cumulative_exposure, 2)
    time_component = round(task.time_weight * cumulative_time_minutes, 2)
    failure_penalty = 0.0 if reached_destination else task.failure_penalty
    distance_component = 0.0 if reached_destination else round(distance_remaining * task.distance_penalty, 2)
    wait_component = round(wait_steps * task.wait_penalty, 2)
    total = round(
        exposure_component + time_component + failure_penalty + distance_component + wait_component,
        2,
    )
    return {
        "weighted_cost": total,
        "exposure_component": exposure_component,
        "time_component": time_component,
        "failure_penalty": failure_penalty,
        "distance_component": distance_component,
        "wait_component": wait_component,
    }


def normalize_score(agent_cost: float, baseline_cost: float, oracle_cost: float) -> float:
    if agent_cost <= oracle_cost:
        return 0.99
    denom = max(baseline_cost - oracle_cost, 1e-6)
    raw = (baseline_cost - agent_cost) / denom
    return round(max(0.01, min(0.99, raw)), 2)


def grade_request(
    request: GradeRequest,
    baseline_cost: float,
    oracle_cost: float,
) -> GradeResponse:
    cost_breakdown = weighted_cost(
        task_id=request.task_id,
        cumulative_exposure=request.cumulative_exposure,
        cumulative_time_minutes=request.cumulative_time_minutes,
        wait_steps=request.wait_steps,
        distance_remaining=request.distance_remaining,
        reached_destination=request.reached_destination,
    )
    score = normalize_score(
        agent_cost=cost_breakdown["weighted_cost"],
        baseline_cost=baseline_cost,
        oracle_cost=oracle_cost,
    )
    explanation = (
        f"Score {score:.2f} compares weighted cost {cost_breakdown['weighted_cost']:.2f} "
        f"against oracle {oracle_cost:.2f} and baseline {baseline_cost:.2f}."
    )
    return GradeResponse(
        task_id=request.task_id,
        score=score,
        weighted_cost=cost_breakdown["weighted_cost"],
        baseline_cost=baseline_cost,
        oracle_cost=oracle_cost,
        exposure_component=cost_breakdown["exposure_component"],
        time_component=cost_breakdown["time_component"],
        failure_penalty=cost_breakdown["failure_penalty"],
        distance_component=cost_breakdown["distance_component"],
        wait_component=cost_breakdown["wait_component"],
        explanation=explanation,
    )
