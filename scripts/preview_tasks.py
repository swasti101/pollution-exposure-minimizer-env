"""Quick preview helper for task configurations and baseline references."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.baseline import get_baseline_summary
from server.tasks import list_task_summaries


def main() -> None:
    for task in list_task_summaries():
        baseline = get_baseline_summary(task.task_id)
        print(
            f"{task.task_id}: baseline={baseline.baseline_cost:.2f}, "
            f"oracle={baseline.oracle_cost:.2f}"
        )


if __name__ == "__main__":
    main()
