"""步骤五：将网络恢复结果与清晰物体和系统像差真值比较。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.static_simulation.config import SETTINGS  # noqa: E402
from workflows.static_simulation.workflow import evaluate_reconstruction  # noqa: E402


if __name__ == "__main__":
    evaluate_reconstruction(SETTINGS)
