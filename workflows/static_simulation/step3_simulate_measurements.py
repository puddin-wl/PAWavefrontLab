"""步骤三：直接从清晰物体、固定系统像差和 SLM 相位生成测量。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.static_simulation.config import SETTINGS  # noqa: E402
from workflows.static_simulation.workflow import (  # noqa: E402
    simulate_modulated_measurements,
)


if __name__ == "__main__":
    simulate_modulated_measurements(SETTINGS)
