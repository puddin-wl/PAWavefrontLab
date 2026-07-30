"""步骤一：生成清晰物体、固定系统像差和基准模糊图。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.static_simulation.config import SETTINGS  # noqa: E402
from workflows.static_simulation.workflow import prepare_ground_truth  # noqa: E402


if __name__ == "__main__":
    prepare_ground_truth(SETTINGS)
