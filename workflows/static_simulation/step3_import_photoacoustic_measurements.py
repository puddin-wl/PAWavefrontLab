"""真实数据版步骤三：三维光声 TIFF 减 2048、置零并沿第 0 维投影。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.static_simulation.config import SETTINGS  # noqa: E402
from workflows.static_simulation.workflow import (  # noqa: E402
    import_photoacoustic_measurements,
)


if __name__ == "__main__":
    import_photoacoustic_measurements(SETTINGS)
