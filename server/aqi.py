"""Deterministic AQI and travel-time simulation."""

from __future__ import annotations

from functools import lru_cache

from server.city_graph import get_edge, get_neighbors, load_city_graph, node_lookup
from server.tasks import TaskConfig


ZONE_OFFSETS = {
    "industrial": 36.0,
    "commercial": 12.0,
    "transit_hub": 18.0,
    "arterial": 24.0,
    "mixed_use": 8.0,
    "park": -16.0,
    "park_edge": -10.0,
    "residential": 0.0,
}

ROAD_EXPOSURE_FACTORS = {
    "local": 1.0,
    "arterial": 1.18,
    "ring_road": 1.3,
    "industrial_connector": 1.34,
    "metro_corridor": 0.82,
}

MODE_EXPOSURE_FACTORS = {
    "walk": 1.0,
    "bus": 0.78,
    "metro": 0.42,
}

MODE_SPEED_KMPH = {
    "walk": 4.8,
    "bus": 20.0,
    "metro": 33.0,
}


def hour_at_elapsed(task: TaskConfig, elapsed_minutes: int) -> float:
    return (task.start_hour + (elapsed_minutes / 60.0)) % 24.0


def traffic_multiplier(road_type: str, hour: float, dynamic_aqi: bool) -> float:
    if not dynamic_aqi or road_type == "metro_corridor":
        return 1.0
    if 8.0 <= hour < 10.0:
        peak = 1.28
    elif 10.0 <= hour < 12.0:
        peak = 1.12
    elif 17.0 <= hour < 19.0:
        peak = 1.24
    else:
        peak = 1.0
    road_boost = {
        "local": 1.02,
        "arterial": 1.08,
        "ring_road": 1.14,
        "industrial_connector": 1.12,
        "metro_corridor": 1.0,
    }.get(road_type, 1.0)
    return peak * road_boost


def congestion_penalty(road_type: str, mode: str, hour: float, dynamic_aqi: bool) -> float:
    if mode == "metro":
        return 1.0
    multiplier = traffic_multiplier(road_type, hour, dynamic_aqi)
    if mode == "bus":
        return multiplier
    return 1.0 + ((multiplier - 1.0) * 0.55)


@lru_cache(maxsize=256)
def node_aqi_map(task_id: str, elapsed_minutes: int) -> dict[str, float]:
    from server.tasks import get_task

    task = get_task(task_id)
    nodes = node_lookup()
    base_values: dict[str, float] = {}
    for node_id, node in nodes.items():
        zone_offset = ZONE_OFFSETS.get(node["zone_type"], 0.0)
        hour = hour_at_elapsed(task, elapsed_minutes)
        traffic_load = node["traffic_sensitivity"] * 22.0
        dynamic_offset = 0.0
        if task.dynamic_aqi:
            if 8.0 <= hour < 10.0:
                dynamic_offset = traffic_load
            elif 10.0 <= hour < 12.0:
                dynamic_offset = traffic_load * 0.55
            else:
                dynamic_offset = traffic_load * 0.15
        base_values[node_id] = max(35.0, node["base_aqi"] + zone_offset + dynamic_offset)

    diffused: dict[str, float] = {}
    for node_id in nodes:
        neighbors = get_neighbors(node_id)
        if neighbors:
            neighbor_avg = sum(base_values[edge["target_node_id"]] for edge in neighbors) / len(neighbors)
        else:
            neighbor_avg = base_values[node_id]
        diffused[node_id] = round((0.84 * base_values[node_id]) + (0.16 * neighbor_avg), 2)
    return diffused


def edge_metrics(
    task: TaskConfig,
    source_node_id: str,
    target_node_id: str,
    mode: str,
    elapsed_minutes: int,
) -> dict[str, float]:
    edge = get_edge(source_node_id, target_node_id)
    if edge is None:
        raise ValueError(f"No edge from {source_node_id} to {target_node_id}.")
    if mode not in edge["allowed_modes"] or mode not in task.allowed_modes:
        raise ValueError(f"Mode '{mode}' is not allowed on edge {source_node_id}->{target_node_id}.")

    aqi_map = node_aqi_map(task.task_id, elapsed_minutes)
    edge_aqi = (aqi_map[source_node_id] + aqi_map[target_node_id]) / 2.0
    hour = hour_at_elapsed(task, elapsed_minutes)
    exposure = edge_aqi
    exposure *= edge["distance_km"]
    exposure *= ROAD_EXPOSURE_FACTORS.get(edge["road_type"], 1.0)
    exposure *= MODE_EXPOSURE_FACTORS[mode]
    exposure *= traffic_multiplier(edge["road_type"], hour, task.dynamic_aqi)
    exposure = round(exposure, 2)

    speed = MODE_SPEED_KMPH[mode] / congestion_penalty(edge["road_type"], mode, hour, task.dynamic_aqi)
    travel_time = max(1, int(round((edge["distance_km"] / speed) * 60.0)))

    return {
        "exposure": exposure,
        "time_minutes": float(travel_time),
        "edge_aqi": round(edge_aqi, 2),
    }


def wait_metrics(task: TaskConfig, node_id: str, elapsed_minutes: int) -> dict[str, float]:
    local_aqi = node_aqi_map(task.task_id, elapsed_minutes)[node_id]
    return {
        "exposure": round(local_aqi * 0.42, 2),
        "time_minutes": 5.0,
        "edge_aqi": round(local_aqi, 2),
    }


def node_snapshot(task: TaskConfig, elapsed_minutes: int) -> list[dict[str, float | str]]:
    graph = load_city_graph()
    aqi_map = node_aqi_map(task.task_id, elapsed_minutes)
    return [
        {
            "node_id": node["node_id"],
            "label": node["label"],
            "zone_type": node["zone_type"],
            "x": node["x"],
            "y": node["y"],
            "base_aqi": node["base_aqi"],
            "current_aqi": aqi_map[node["node_id"]],
        }
        for node in graph["nodes"]
    ]
