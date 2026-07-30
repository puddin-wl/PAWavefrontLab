"""步骤二：生成 50 张已知的随机 SLM 调制相位。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.static_simulation.config import SETTINGS  # noqa: E402
from workflows.static_simulation.workflow import generate_slm_patterns  # noqa: E402


if __name__ == "__main__":
    generate_slm_patterns(SETTINGS)
