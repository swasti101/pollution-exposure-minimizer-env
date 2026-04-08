"""Typed Pydantic models for the Pollution Exposure Minimizer Environment."""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


ModeType = Literal["walk", "bus", "metro"]
ActionType = Literal["move", "wait"]
DifficultyType = Literal["easy", "medium", "hard"]


class GraphNode(BaseModel):
    node_id: str = Field(..., description="Stable node identifier.")
    label: str = Field(..., description="Human-readable location label.")
    zone_type: str = Field(..., description="Land-use label for the node.")
    x: float = Field(..., description="Normalized x coordinate for plotting.")
    y: float = Field(..., description="Normalized y coordinate for plotting.")
    base_aqi: float = Field(..., ge=0, description="Base AQI-like value.")
    current_aqi: float = Field(..., ge=0, description="Current AQI-like value.")


class GraphEdge(BaseModel):
    source_node_id: str = Field(..., description="Start node id.")
    target_node_id: str = Field(..., description="End node id.")
    road_type: str = Field(..., description="Road or corridor category.")
    distance_km: float = Field(..., gt=0, description="Approximate segment distance.")
    allowed_modes: list[ModeType] = Field(default_factory=list)


class ActionOption(BaseModel):
    action_type: ActionType = Field(..., description="Whether the option moves or waits.")
    target_node_id: Optional[str] = Field(default=None, description="Target node id.")
    target_label: Optional[str] = Field(default=None, description="Human-readable target.")
    mode: Optional[ModeType] = Field(default=None, description="Transport mode.")
    estimated_exposure: float = Field(..., ge=0)
    estimated_time_minutes: int = Field(..., ge=0)
    description: str = Field(..., description="Plain-language summary.")


class TaskSummary(BaseModel):
    task_id: str
    name: str
    difficulty: DifficultyType
    description: str
    start_node_id: str
    destination_node_id: str
    allowed_modes: list[ModeType] = Field(default_factory=list)
    allow_wait: bool
    dynamic_aqi: bool
    start_hour: int = Field(..., ge=0, le=23)
    max_steps: int = Field(..., ge=1)


class BaselineSummary(BaseModel):
    task_id: str
    baseline_cost: float = Field(..., ge=0)
    oracle_cost: float = Field(..., ge=0)
    baseline_exposure: float = Field(..., ge=0)
    baseline_time_minutes: int = Field(..., ge=0)
    oracle_exposure: float = Field(..., ge=0)
    oracle_time_minutes: int = Field(..., ge=0)
    baseline_path: list[str] = Field(default_factory=list)
    oracle_path: list[str] = Field(default_factory=list)


class GradeRequest(BaseModel):
    task_id: str
    cumulative_exposure: float = Field(..., ge=0)
    cumulative_time_minutes: int = Field(..., ge=0)
    steps_taken: int = Field(..., ge=0)
    wait_steps: int = Field(..., ge=0)
    distance_remaining: float = Field(default=0.0, ge=0)
    reached_destination: bool


class GradeResponse(BaseModel):
    task_id: str
    score: float = Field(..., ge=0, le=1)
    weighted_cost: float = Field(..., ge=0)
    baseline_cost: float = Field(..., ge=0)
    oracle_cost: float = Field(..., ge=0)
    exposure_component: float = Field(..., ge=0)
    time_component: float = Field(..., ge=0)
    failure_penalty: float = Field(..., ge=0)
    distance_component: float = Field(..., ge=0)
    wait_component: float = Field(..., ge=0)
    explanation: str


class PollutionAction(Action):
    action_type: ActionType
    target_node_id: Optional[str] = None
    mode: Optional[ModeType] = None
    rationale: Optional[str] = None


class PollutionObservation(Observation):
    task_id: str
    task_name: str
    difficulty: DifficultyType
    city_name: str = "Delhi"
    prompt: str
    current_node_id: str
    current_node_label: str
    destination_node_id: str
    destination_node_label: str
    current_hour: float
    time_index: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    steps_remaining: int = Field(..., ge=0)
    current_node_aqi: float = Field(..., ge=0)
    cumulative_exposure: float = Field(..., ge=0)
    cumulative_time_minutes: int = Field(..., ge=0)
    cumulative_cost: float = Field(..., ge=0)
    legal_actions: list[ActionOption] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    final_summary: Optional[str] = None


class PollutionState(State):
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    difficulty: Optional[DifficultyType] = None
    current_node_id: Optional[str] = None
    destination_node_id: Optional[str] = None
    current_hour: float = 0.0
    cumulative_exposure: float = 0.0
    cumulative_time_minutes: int = 0
    cumulative_cost: float = 0.0
    wait_steps: int = 0
    done: bool = False
    episode_score: Optional[float] = None
    last_action_summary: Optional[str] = None
