"""Utilities for loading and querying the Delhi-inspired city graph."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_city_graph() -> dict[str, Any]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "pollution_city_graph.json"
    with data_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def node_lookup() -> dict[str, dict[str, Any]]:
    return {node["node_id"]: node for node in load_city_graph()["nodes"]}


def edge_list() -> list[dict[str, Any]]:
    return load_city_graph()["edges"]


def get_neighbors(node_id: str) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for edge in edge_list():
        if edge["source_node_id"] == node_id:
            neighbors.append(edge)
        elif edge["target_node_id"] == node_id:
            reverse_edge = dict(edge)
            reverse_edge["source_node_id"] = edge["target_node_id"]
            reverse_edge["target_node_id"] = edge["source_node_id"]
            neighbors.append(reverse_edge)
    return neighbors


def get_edge(source_node_id: str, target_node_id: str) -> dict[str, Any] | None:
    for edge in get_neighbors(source_node_id):
        if edge["target_node_id"] == target_node_id:
            return edge
    return None

