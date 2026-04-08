"""Client for the Pollution Exposure Minimizer Environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ActionOption,
        GraphEdge,
        GraphNode,
        PollutionAction,
        PollutionObservation,
        PollutionState,
    )
except ImportError:  # pragma: no cover
    from models import (
        ActionOption,
        GraphEdge,
        GraphNode,
        PollutionAction,
        PollutionObservation,
        PollutionState,
    )


class PollutionExposureMinimizerEnv(
    EnvClient[PollutionAction, PollutionObservation, PollutionState]
):
    """Typed client for the pollution exposure benchmark."""

    def _step_payload(self, action: PollutionAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "target_node_id": action.target_node_id,
            "mode": action.mode,
            "rationale": action.rationale,
            "metadata": action.metadata,
        }

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[PollutionObservation]:
        obs_data = payload.get("observation", {})
        observation = PollutionObservation(
            task_id=obs_data.get("task_id", ""),
            task_name=obs_data.get("task_name", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            city_name=obs_data.get("city_name", "Delhi"),
            prompt=obs_data.get("prompt", ""),
            current_node_id=obs_data.get("current_node_id", ""),
            current_node_label=obs_data.get("current_node_label", ""),
            destination_node_id=obs_data.get("destination_node_id", ""),
            destination_node_label=obs_data.get("destination_node_label", ""),
            current_hour=obs_data.get("current_hour", 0.0),
            time_index=obs_data.get("time_index", 0),
            max_steps=obs_data.get("max_steps", 1),
            steps_remaining=obs_data.get("steps_remaining", 0),
            current_node_aqi=obs_data.get("current_node_aqi", 0.0),
            cumulative_exposure=obs_data.get("cumulative_exposure", 0.0),
            cumulative_time_minutes=obs_data.get("cumulative_time_minutes", 0),
            cumulative_cost=obs_data.get("cumulative_cost", 0.0),
            legal_actions=[
                ActionOption(**item) for item in obs_data.get("legal_actions", [])
            ],
            graph_nodes=[GraphNode(**item) for item in obs_data.get("graph_nodes", [])],
            graph_edges=[GraphEdge(**item) for item in obs_data.get("graph_edges", [])],
            final_summary=obs_data.get("final_summary"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> PollutionState:
        return PollutionState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            task_name=payload.get("task_name"),
            difficulty=payload.get("difficulty"),
            current_node_id=payload.get("current_node_id"),
            destination_node_id=payload.get("destination_node_id"),
            current_hour=payload.get("current_hour", 0.0),
            cumulative_exposure=payload.get("cumulative_exposure", 0.0),
            cumulative_time_minutes=payload.get("cumulative_time_minutes", 0),
            cumulative_cost=payload.get("cumulative_cost", 0.0),
            wait_steps=payload.get("wait_steps", 0),
            done=payload.get("done", False),
            episode_score=payload.get("episode_score"),
            last_action_summary=payload.get("last_action_summary"),
        )
