"""OpenEnv environment for pollution-aware urban commuting."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ActionOption,
        GradeRequest,
        GraphEdge,
        GraphNode,
        PollutionAction,
        PollutionObservation,
        PollutionState,
    )
except ImportError:  # pragma: no cover
    from models import (
        ActionOption,
        GradeRequest,
        GraphEdge,
        GraphNode,
        PollutionAction,
        PollutionObservation,
        PollutionState,
    )

from server.aqi import edge_metrics, hour_at_elapsed, node_aqi_map, node_snapshot, wait_metrics
from server.baseline import get_baseline_summary
from server.city_graph import get_neighbors, load_city_graph, node_lookup
from server.grader import grade_request, weighted_cost
from server.tasks import TASK_ORDER, get_task


class PollutionExposureMinimizerEnvironment(
    Environment[PollutionAction, PollutionObservation, PollutionState]
):
    """Deterministic pollution-aware routing environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._reset_count = 0
        self._graph = load_city_graph()
        self._nodes = node_lookup()
        self._state = PollutionState()
        self._task_id = TASK_ORDER[0]
        self._current_node_id = ""
        self._done = False

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> PollutionObservation:
        self._reset_rubric()
        requested_task_id = kwargs.get("task_id")
        if requested_task_id in TASK_ORDER:
            self._task_id = requested_task_id
        elif seed is not None:
            self._task_id = TASK_ORDER[seed % len(TASK_ORDER)]
        else:
            self._task_id = TASK_ORDER[self._reset_count % len(TASK_ORDER)]
        self._reset_count += 1

        task = get_task(self._task_id)
        self._current_node_id = task.start_node_id
        self._done = False
        self._state = PollutionState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            task_name=task.name,
            difficulty=task.difficulty,
            current_node_id=task.start_node_id,
            destination_node_id=task.destination_node_id,
            current_hour=float(task.start_hour),
            cumulative_exposure=0.0,
            cumulative_time_minutes=0,
            cumulative_cost=0.0,
            wait_steps=0,
            done=False,
            episode_score=None,
            last_action_summary="Environment reset.",
        )
        return self._build_observation(reward=0.0, done=False, final_summary=None)

    def step(
        self,
        action: PollutionAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> PollutionObservation:
        task = get_task(self._task_id)
        if self._state.task_id is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                final_summary="Episode already finished. Call reset() to start a new task.",
            )

        self._state.step_count += 1
        previous_distance = self._distance_to_goal(self._current_node_id)
        reward = 0.0
        summary = ""

        if action.action_type == "wait":
            if not task.allow_wait:
                reward = -2.0
                summary = "Wait is not allowed in this task."
            else:
                metrics = wait_metrics(task, self._current_node_id, self._state.cumulative_time_minutes)
                reward = self._apply_transition(
                    exposure=metrics["exposure"],
                    time_minutes=int(metrics["time_minutes"]),
                    reached_destination=False,
                    progress_bonus=0.0,
                )
                self._state.wait_steps += 1
                summary = f"Waited at {self._nodes[self._current_node_id]['label']} for {int(metrics['time_minutes'])} min."
        elif action.action_type == "move":
            if not action.target_node_id or not action.mode:
                reward = -2.5
                summary = "Move actions require both target_node_id and mode."
            else:
                edge = next(
                    (candidate for candidate in get_neighbors(self._current_node_id) if candidate["target_node_id"] == action.target_node_id),
                    None,
                )
                if edge is None:
                    reward = -2.5
                    summary = f"{action.target_node_id} is not adjacent to the current node."
                elif action.mode not in edge["allowed_modes"] or action.mode not in task.allowed_modes:
                    reward = -2.5
                    summary = f"Mode '{action.mode}' is not available for this move."
                else:
                    metrics = edge_metrics(
                        task,
                        self._current_node_id,
                        action.target_node_id,
                        action.mode,
                        self._state.cumulative_time_minutes,
                    )
                    self._current_node_id = action.target_node_id
                    reached_destination = self._current_node_id == task.destination_node_id
                    new_distance = self._distance_to_goal(self._current_node_id)
                    progress_bonus = max(0.0, previous_distance - new_distance) * 0.28
                    reward = self._apply_transition(
                        exposure=metrics["exposure"],
                        time_minutes=int(metrics["time_minutes"]),
                        reached_destination=reached_destination,
                        progress_bonus=progress_bonus,
                    )
                    summary = f"Moved to {self._nodes[self._current_node_id]['label']} via {action.mode}; segment exposure {metrics['exposure']:.2f}."
        else:
            reward = -2.5
            summary = f"Unknown action_type '{action.action_type}'."

        self._state.last_action_summary = summary
        if self._state.step_count >= task.max_steps and self._current_node_id != task.destination_node_id:
            self._done = True
            self._state.done = True

        final_summary = None
        if self._done:
            baseline = get_baseline_summary(self._task_id)
            grade = grade_request(
                request=self._grade_request_payload(),
                baseline_cost=baseline.baseline_cost,
                oracle_cost=baseline.oracle_cost,
            )
            self._state.episode_score = grade.score
            final_summary = f"Task finished with score {grade.score:.2f}. Weighted cost {grade.weighted_cost:.2f} vs baseline {grade.baseline_cost:.2f}."
            reward += grade.score

        return self._build_observation(
            reward=round(reward, 4),
            done=self._done,
            final_summary=final_summary,
        )

    @property
    def state(self) -> PollutionState:
        return self._state

    def _apply_transition(
        self,
        *,
        exposure: float,
        time_minutes: int,
        reached_destination: bool,
        progress_bonus: float,
    ) -> float:
        task = get_task(self._task_id)
        self._state.cumulative_exposure = round(self._state.cumulative_exposure + exposure, 2)
        self._state.cumulative_time_minutes += time_minutes
        self._state.current_hour = round(hour_at_elapsed(task, self._state.cumulative_time_minutes), 2)
        cost_parts = weighted_cost(
            task_id=self._task_id,
            cumulative_exposure=self._state.cumulative_exposure,
            cumulative_time_minutes=self._state.cumulative_time_minutes,
            wait_steps=self._state.wait_steps,
            distance_remaining=self._distance_to_goal(self._current_node_id),
            reached_destination=reached_destination,
        )
        self._state.cumulative_cost = cost_parts["weighted_cost"]
        self._state.current_node_id = self._current_node_id

        reward = progress_bonus - ((exposure * 0.01) + (time_minutes * task.time_weight * 0.06))
        if reached_destination:
            self._done = True
            self._state.done = True
            reward += task.arrival_bonus
        return reward

    def _distance_to_goal(self, node_id: str) -> float:
        task = get_task(self._task_id)
        destination = self._nodes[task.destination_node_id]
        node = self._nodes[node_id]
        return abs(destination["x"] - node["x"]) + abs(destination["y"] - node["y"])

    def _legal_actions(self) -> list[ActionOption]:
        task = get_task(self._task_id)
        actions: list[ActionOption] = []
        for edge in get_neighbors(self._current_node_id):
            for mode in task.allowed_modes:
                if mode not in edge["allowed_modes"]:
                    continue
                metrics = edge_metrics(
                    task,
                    self._current_node_id,
                    edge["target_node_id"],
                    mode,
                    self._state.cumulative_time_minutes,
                )
                target = self._nodes[edge["target_node_id"]]
                actions.append(
                    ActionOption(
                        action_type="move",
                        target_node_id=edge["target_node_id"],
                        target_label=target["label"],
                        mode=mode,
                        estimated_exposure=metrics["exposure"],
                        estimated_time_minutes=int(metrics["time_minutes"]),
                        description=f"Move to {target['label']} via {mode} on {edge['road_type']} with exposure {metrics['exposure']:.2f}.",
                    )
                )
        if task.allow_wait:
            wait = wait_metrics(task, self._current_node_id, self._state.cumulative_time_minutes)
            actions.append(
                ActionOption(
                    action_type="wait",
                    target_node_id=None,
                    target_label=None,
                    mode=None,
                    estimated_exposure=wait["exposure"],
                    estimated_time_minutes=int(wait["time_minutes"]),
                    description=f"Wait in place for {int(wait['time_minutes'])} min and absorb exposure {wait['exposure']:.2f}.",
                )
            )
        return actions

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        final_summary: str | None,
    ) -> PollutionObservation:
        task = get_task(self._task_id)
        current_node = self._nodes[self._current_node_id]
        destination_node = self._nodes[task.destination_node_id]
        current_aqi = node_aqi_map(self._task_id, self._state.cumulative_time_minutes)[self._current_node_id]
        legal_actions = [] if done else self._legal_actions()
        prompt = (
            "You are an urban commuting agent minimizing pollution exposure.\n"
            f"Task: {task.description}\n"
            f"Current location: {current_node['label']} ({self._current_node_id})\n"
            f"Destination: {destination_node['label']} ({task.destination_node_id})\n"
            f"Current hour: {self._state.current_hour:.2f}\n"
            f"Cumulative exposure: {self._state.cumulative_exposure:.2f}\n"
            f"Cumulative travel time: {self._state.cumulative_time_minutes} minutes\n"
            "Choose exactly one legal action."
        )
        return PollutionObservation(
            done=done,
            reward=reward,
            task_id=task.task_id,
            task_name=task.name,
            difficulty=task.difficulty,
            city_name=self._graph["city_name"],
            prompt=prompt,
            current_node_id=self._current_node_id,
            current_node_label=current_node["label"],
            destination_node_id=task.destination_node_id,
            destination_node_label=destination_node["label"],
            current_hour=self._state.current_hour,
            time_index=self._state.step_count,
            max_steps=task.max_steps,
            steps_remaining=max(0, task.max_steps - self._state.step_count),
            current_node_aqi=current_aqi,
            cumulative_exposure=self._state.cumulative_exposure,
            cumulative_time_minutes=self._state.cumulative_time_minutes,
            cumulative_cost=self._state.cumulative_cost,
            legal_actions=legal_actions,
            graph_nodes=[GraphNode(**node) for node in node_snapshot(task, self._state.cumulative_time_minutes)],
            graph_edges=[GraphEdge(**edge) for edge in self._graph["edges"]],
            final_summary=final_summary,
            metadata={
                "task_id": task.task_id,
                "task_name": task.name,
                "baseline": get_baseline_summary(task.task_id).model_dump(),
            },
        )

    def _grade_request_payload(self) -> GradeRequest:
        return GradeRequest(
            task_id=self._task_id,
            cumulative_exposure=self._state.cumulative_exposure,
            cumulative_time_minutes=self._state.cumulative_time_minutes,
            steps_taken=self._state.step_count,
            wait_steps=self._state.wait_steps,
            distance_remaining=self._distance_to_goal(self._current_node_id),
            reached_destination=self._current_node_id == get_task(self._task_id).destination_node_id,
        )
