"""步骤四：用调制测量和已知 SLM 相位恢复物体与系统像差。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.static_simulation.config import SETTINGS  # noqa: E402
from workflows.static_simulation.workflow import reconstruct_static_scene  # noqa: E402


if __name__ == "__main__":
    reconstruct_static_scene(SETTINGS)
