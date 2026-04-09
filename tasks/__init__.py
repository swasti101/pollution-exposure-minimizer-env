"""Task and grading compatibility package.

This package mirrors the scaffold-style project structure expected by some
OpenEnv validators while re-exporting the implementation that lives under
``server/``.
"""

from server.baseline import get_baseline_summary
from server.grader import grade_request, normalize_score, weighted_cost
from server.tasks import TASKS, TASK_ORDER, TaskConfig, get_task, list_task_summaries

__all__ = [
    "TASKS",
    "TASK_ORDER",
    "TaskConfig",
    "get_task",
    "list_task_summaries",
    "weighted_cost",
    "normalize_score",
    "grade_request",
    "get_baseline_summary",
]
