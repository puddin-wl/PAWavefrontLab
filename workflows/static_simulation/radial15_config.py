"""Configuration for the radial-order-15 closed-loop simulation."""

import math
from pathlib import Path

from workflows.static_simulation.workflow import SimulationSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_NAME = "test_static_radial15_both_50"
BASELINE_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "test_static_zernike_50"
    / "evaluation"
    / "reconstruction_report.json"
)


SETTINGS = SimulationSettings(
    project_root=PROJECT_ROOT,
    input_image=PROJECT_ROOT / "data" / "test.tif",
    data_dir=PROJECT_ROOT / "data" / SCENE_NAME,
    result_root=PROJECT_ROOT,
    scene_name=SCENE_NAME,
    input_mode="image",
    size=256,
    aperture_height=None,
    num_frames=50,
    system_noll_start=4,
    system_noll_end=136,
    system_num_modes=136,
    system_disabled_noll_indices=(),
    system_sigma=0.6 * math.sqrt(12.0 / 133.0),
    system_seed=20260730,
    slm_num_modes=136,
    slm_disabled_noll_indices=(2, 3),
    slm_sigma=1.22,
    slm_seed=20260731,
    phase_sign=-1,
    noise_std=0.0,
    noise_seed=20260732,
    simulation_batch_size=8,
    generation_device="cuda",
    training_device="cuda",
    training_epochs=1000,
    training_batch_size=8,
    network_zernike_features=136,
    phase_layers=4,
    initial_learning_rate=1e-3,
    final_learning_rate=1e-3,
    visualization_frequency=1000,
    overwrite=False,
    silence_tqdm=True,
)
