import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from dataio import BatchDataset
from networks import model_from_checkpoint
from workflows.reconstruction.reconstruct_neuws import build_parser
from workflows.real_experiment.run_pipeline import _build_training_command


ROOT = Path(__file__).resolve().parents[1]


def test_original_is_the_default_model_mode():
    assert build_parser().parse_args([]).model_mode == "original"


def _write_dataset(directory: Path) -> None:
    directory.mkdir()
    rng = np.random.default_rng(4)
    for index in range(1, 3):
        sio.savemat(
            directory / f"SLM_sim{index}.mat",
            {"proj_sim": rng.normal(0, 0.1, (16, 16)).astype(np.float32)},
        )
        sio.savemat(
            directory / f"SLM_raw{index}.mat",
            {"imsdata": rng.random((16, 16), dtype=np.float32)},
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "size": 16,
                "aperture_height": 16,
                "phase_sign": -1,
                "measurement_max": 1.0,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("model_mode", ["original", "object_mmes", "dual_mmes"])
def test_all_model_modes_train_save_and_reload(tmp_path, model_mode):
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    _write_dataset(data_dir)
    command = [
        sys.executable,
        str(ROOT / "workflows" / "reconstruction" / "reconstruct_neuws.py"),
        "--root_dir",
        str(run_dir),
        "--data_dir",
        str(data_dir),
        "--scene_name",
        model_mode,
        "--model_mode",
        model_mode,
        "--num_epochs",
        "1",
        "--batch_size",
        "2",
        "--phs_layers",
        "1",
        "--zernike_features",
        "4",
        "--static_phase",
        "--vis_freq",
        "0",
        "--device",
        "cpu",
        "--silence_tqdm",
        "--mmes_tau",
        "2",
        "--object_mmes_rank1",
        "16",
        "--object_mmes_rank2",
        "4",
        "--object_mmes_rank3",
        "16",
        "--aberration_mmes_rank1",
        "16",
        "--aberration_mmes_rank2",
        "4",
        "--aberration_mmes_rank3",
        "16",
        "--mmes_noise_std",
        "0",
        "--mmes_chunk_size",
        "0",
        "--mmes_checkpoint_chunks",
        "false",
    ]
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)

    final_dir = run_dir / "vis" / model_mode / "final"
    for filename in (
        "final_aberration.mat",
        "training_summary.json",
        "model_final.pt",
    ):
        assert (final_dir / filename).is_file()
    aberration = sio.loadmat(final_dir / "final_aberration.mat")
    assert {
        "phase",
        "field",
        "phase_parameter",
        "complex_phase",
        "amplitude_parameter",
        "field_amplitude",
    } <= set(aberration)
    if model_mode != "dual_mmes":
        assert np.allclose(aberration["field"], np.exp(1j * aberration["phase"]), atol=1e-6)
    else:
        assert np.allclose(aberration["phase"], aberration["phase_parameter"])
        assert np.allclose(
            aberration["field"],
            aberration["amplitude_parameter"]
            * np.exp(1j * aberration["phase_parameter"]),
            atol=1e-6,
        )
        assert np.allclose(
            aberration["complex_phase"], np.angle(aberration["field"]), atol=1e-6
        )
        assert np.allclose(
            aberration["field_amplitude"], np.abs(aberration["field"]), atol=1e-6
        )

    summary = json.loads((final_dir / "training_summary.json").read_text())
    assert summary["model_mode"] == model_mode
    assert summary["early_stopping_metric"] == "measurement_mse"
    for name in (
        "final_measurement_mse",
        "final_object_ae_mse",
        "final_aberration_ae_mse",
        "final_total_loss",
    ):
        assert np.isfinite(summary[name])

    saved = torch.load(final_dir / "model_final.pt", map_location="cpu", weights_only=True)
    restored = model_from_checkpoint(saved)
    dataset = BatchDataset(data_dir)
    slm = torch.stack([dataset[0][0], dataset[1][0]])
    with torch.no_grad():
        prediction = restored(slm, torch.tensor([-0.5, 0.5]))[0]
    assert prediction.shape == (2, 16, 16)


def test_pipeline_forwards_mmes_training_configuration(tmp_path):
    dataset_dir = tmp_path / "data"
    config = {
        "training": {
            "model_mode": "dual_mmes",
            "object_ae_weight": 0.5,
            "aberration_ae_weight": 0.25,
            "mmes_tau": 3,
            "mmes_checkpoint_chunks": False,
            "amplitude_offset": 0.75,
        }
    }
    command = _build_training_command(config, dataset_dir, "scene")
    assert command[command.index("--model_mode") + 1] == "dual_mmes"
    assert command[command.index("--object_ae_weight") + 1] == "0.5"
    assert command[command.index("--mmes_checkpoint_chunks") + 1] == "False"
    assert command[command.index("--amplitude_offset") + 1] == "0.75"
