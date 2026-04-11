"""Deterministic grading utilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from models import GradeRequest, GradeResponse
from server.tasks import get_task

try:
    from openenv.core.env_server.types import Observation
except Exception:  # pragma: no cover
    Observation = Any


class GraderResult(BaseModel):
    score: float = Field(..., ge=0, le=1)
    reasoning: str = ""


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


def grade_task(
    request: GradeRequest | None = None,
    baseline_cost: float | None = None,
    oracle_cost: float | None = None,
) -> float:
    """Compatibility grader entrypoint with safe defaults for reflection checks."""
    if request is None or baseline_cost is None or oracle_cost is None:
        return 0.5
    return grade_request(request, baseline_cost, oracle_cost).score


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract(obs: Any, name: str, default: Any = None) -> Any:
    if obs is None:
        return default
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _score_from_observation(obs: Observation | None, label: str) -> GraderResult:
    if obs is None:
        return GraderResult(score=0.5, reasoning=f"{label}: no observation provided")

    for key in ("score", "normalized_score", "episode_score", "reward"):
        value = _coerce_float(_extract(obs, key), default=None)
        if value is not None:
            return GraderResult(
                score=max(0.01, min(0.99, value if 0.0 <= value <= 1.0 else 0.5)),
                reasoning=f"{label}: used {key}",
            )

    done = bool(_extract(obs, "done", False))
    reached = _extract(obs, "reached_destination", None)
    current_node_id = _extract(obs, "current_node_id", None)
    destination_node_id = _extract(obs, "destination_node_id", None)
    if reached is None and current_node_id is not None and destination_node_id is not None:
        reached = current_node_id == destination_node_id

    if done and reached:
        return GraderResult(score=0.99, reasoning=f"{label}: destination reached")
    if done:
        return GraderResult(score=0.01, reasoning=f"{label}: episode ended without destination")

    cumulative_cost = _coerce_float(_extract(obs, "cumulative_cost", None), default=None)
    if cumulative_cost is None:
        cumulative_cost = _coerce_float(_extract(obs, "weighted_cost", None), default=None)
    if cumulative_cost is not None:
        score = 1.0 / (1.0 + max(cumulative_cost, 0.0) / 1000.0)
        return GraderResult(
            score=max(0.01, min(0.99, round(score, 2))),
            reasoning=f"{label}: derived from cumulative cost",
        )

    return GraderResult(score=0.5, reasoning=f"{label}: fallback compatibility score")


def grade_task_1(obs: Observation | None = None) -> GraderResult:
    return _score_from_observation(obs, "easy_static_route")


def grade_task_2(obs: Observation | None = None) -> GraderResult:
    return _score_from_observation(obs, "medium_multimodal_route")


def grade_task_3(obs: Observation | None = None) -> GraderResult:
    return _score_from_observation(obs, "hard_dynamic_peak_route")


def get_grader(task_id: str):
    """Return the task-specific grader function by task id."""
    return {
        "easy_static_route": grade_task_1,
        "medium_multimodal_route": grade_task_2,
        "hard_dynamic_peak_route": grade_task_3,
    }.get(task_id, grade_task)
