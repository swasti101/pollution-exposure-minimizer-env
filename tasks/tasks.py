"""Task definitions for the Pollution Exposure Minimizer Environment."""

from __future__ import annotations

from dataclasses import dataclass

from models import TaskSummary


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    description: str
    start_node_id: str
    destination_node_id: str
    allowed_modes: tuple[str, ...]
    allow_wait: bool
    dynamic_aqi: bool
    start_hour: int
    max_steps: int
    exposure_weight: float
    time_weight: float
    arrival_bonus: float
    failure_penalty: float
    distance_penalty: float
    wait_penalty: float


TASKS: dict[str, TaskConfig] = {
    "easy_static_route": TaskConfig(
        task_id="easy_static_route",
        name="Easy Static AQI Route",
        difficulty="easy",
        description="Walk from North Campus to Nehru Place while minimizing pollution exposure on a static AQI map.",
        start_node_id="north_campus",
        destination_node_id="nehru_place",
        allowed_modes=("walk",),
        allow_wait=False,
        dynamic_aqi=False,
        start_hour=9,
        max_steps=8,
        exposure_weight=1.0,
        time_weight=0.18,
        arrival_bonus=8.0,
        failure_penalty=600.0,
        distance_penalty=180.0,
        wait_penalty=10.0,
    ),
    "medium_multimodal_route": TaskConfig(
        task_id="medium_multimodal_route",
        name="Medium Multimodal Southbound Commute",
        difficulty="medium",
        description="Commute from Karol Bagh to Saket using walk, bus, and metro while balancing pollution and travel time across several southbound route choices.",
        start_node_id="karol_bagh",
        destination_node_id="saket",
        allowed_modes=("walk", "bus", "metro"),
        allow_wait=False,
        dynamic_aqi=False,
        start_hour=8,
        max_steps=9,
        exposure_weight=1.0,
        time_weight=0.32,
        arrival_bonus=10.0,
        failure_penalty=900.0,
        distance_penalty=240.0,
        wait_penalty=18.0,
    ),
    "hard_dynamic_peak_route": TaskConfig(
        task_id="hard_dynamic_peak_route",
        name="Hard Dynamic Peak Routing",
        difficulty="hard",
        description="Travel from Civil Lines to Okhla during peak conditions with dynamic AQI, diffusion, transport modes, and optional waiting.",
        start_node_id="civil_lines",
        destination_node_id="okhla_phase_2",
        allowed_modes=("walk", "bus", "metro"),
        allow_wait=True,
        dynamic_aqi=True,
        start_hour=8,
        max_steps=9,
        exposure_weight=1.0,
        time_weight=0.42,
        arrival_bonus=12.0,
        failure_penalty=1800.0,
        distance_penalty=420.0,
        wait_penalty=36.0,
    ),
    "bonus_dynamic_cross_city_route": TaskConfig(
        task_id="bonus_dynamic_cross_city_route",
        name="Bonus Dynamic Cross-City Route",
        difficulty="hard",
        description="Travel from Karol Bagh to Okhla Phase II during peak conditions with dynamic AQI, multimodal choices, and optional waiting.",
        start_node_id="karol_bagh",
        destination_node_id="okhla_phase_2",
        allowed_modes=("walk", "bus", "metro"),
        allow_wait=True,
        dynamic_aqi=True,
        start_hour=8,
        max_steps=10,
        exposure_weight=1.0,
        time_weight=0.38,
        arrival_bonus=12.0,
        failure_penalty=1750.0,
        distance_penalty=405.0,
        wait_penalty=34.0,
    ),
}


TASK_ORDER = tuple(TASKS.keys())


def get_task(task_id: str) -> TaskConfig:
    return TASKS[task_id]


def list_task_summaries() -> list[TaskSummary]:
    return [
        TaskSummary(
            task_id=task.task_id,
            name=task.name,
            difficulty=task.difficulty,
            description=task.description,
            start_node_id=task.start_node_id,
            destination_node_id=task.destination_node_id,
            allowed_modes=list(task.allowed_modes),
            allow_wait=task.allow_wait,
            dynamic_aqi=task.dynamic_aqi,
            start_hour=task.start_hour,
            max_steps=task.max_steps,
        )
        for task in TASKS.values()
    ]
