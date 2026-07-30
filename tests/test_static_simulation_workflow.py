import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from dataset import BatchDataset
from optics import simulate_measurements, slm_complex_field
from workflows.static_simulation.workflow import (
    SimulationSettings,
    evaluate_reconstruction,
    generate_slm_patterns,
    prepare_ground_truth,
    reconstruct_static_scene,
    sample_system_aberration_coefficients,
    simulate_modulated_measurements,
)


ROOT = Path(__file__).resolve().parents[1]


def make_settings(root: Path, name: str = "workflow") -> SimulationSettings:
    image_path = root / "object.png"
    if not image_path.exists():
        image = np.arange(18 * 26, dtype=np.uint16).reshape(18, 26)
        Image.fromarray(image).save(image_path)
    return SimulationSettings(
        project_root=ROOT,
        input_image=image_path,
        data_dir=root / f"{name}_data",
        result_root=root / f"{name}_results",
        scene_name=name,
        size=16,
        aperture_height=None,
        num_frames=3,
        system_noll_start=4,
        system_noll_end=15,
        system_sigma=0.6,
        system_seed=20260730,
        slm_num_modes=15,
        slm_sigma=5.0,
        slm_seed=20260731,
        phase_sign=-1,
        noise_std=0.0,
        noise_seed=20260732,
        simulation_batch_size=2,
        generation_device="cpu",
        training_device="cpu",
        training_epochs=3,
        training_batch_size=2,
        phase_layers=1,
        visualization_frequency=0,
        overwrite=False,
        silence_tqdm=True,
    )


class StaticSimulationWorkflowTests(unittest.TestCase):
    def test_system_aberration_is_reproducible_and_limited_to_selected_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = make_settings(Path(temporary))
            first = sample_system_aberration_coefficients(settings)
            second = sample_system_aberration_coefficients(settings)
            self.assertTrue(np.array_equal(first, second))
            self.assertTrue(np.all(first[:3] == 0))
            self.assertTrue(np.all(first[15:] == 0))
            self.assertTrue(np.any(first[3:15] != 0))

    def test_measurements_use_clear_object_and_combined_pupil_directly(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = make_settings(Path(temporary))
            prepare_ground_truth(settings)
            generate_slm_patterns(settings)
            simulate_modulated_measurements(settings)
            truth = sio.loadmat(settings.data_dir / "ground_truth.mat")
            clear_object = torch.from_numpy(truth["clear_object"].astype(np.float32))
            baseline = torch.from_numpy(
                truth["baseline_aberrated_measurement"].astype(np.float32)
            )
            system_field = torch.from_numpy(
                truth["system_aberration_field"].astype(np.complex64)
            )
            phase = torch.from_numpy(
                sio.loadmat(settings.data_dir / "SLM_sim1.mat")["proj_sim"].astype(
                    np.float32
                )
            ).unsqueeze(0)
            slm = slm_complex_field(phase, settings.size, settings.phase_sign)
            expected, _ = simulate_measurements(clear_object, system_field, slm)
            cascaded, _ = simulate_measurements(baseline, system_field, slm)
            saved = sio.loadmat(settings.data_dir / "SLM_raw1.mat")["imsdata"]
            self.assertTrue(np.allclose(saved, expected[0, 0].numpy(), atol=1e-6))
            self.assertGreater(float(np.max(np.abs(saved - cascaded[0, 0].numpy()))), 1e-5)
            dataset = BatchDataset(settings.data_dir)
            self.assertEqual(len(dataset), settings.num_frames)
            self.assertEqual(dataset.phase_sign, -1)
            manifest = json.loads((settings.data_dir / "manifest.json").read_text())
            self.assertTrue(manifest["completed_steps"]["measurements"])

    def test_all_five_stages_produce_finite_reconstruction_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = make_settings(Path(temporary), "closed_loop")
            prepare_ground_truth(settings)
            generate_slm_patterns(settings)
            simulate_modulated_measurements(settings)
            final_dir = reconstruct_static_scene(settings)
            report = evaluate_reconstruction(settings)
            reconstructed_object = np.load(
                final_dir / "reconstructed_object.npy", allow_pickle=False
            )
            reconstructed_phase = np.load(
                final_dir / "reconstructed_aberration_phase.npy", allow_pickle=False
            )
            network_object = np.load(
                final_dir / "reconstructed_object_network_units.npy", allow_pickle=False
            )
            manifest = json.loads((settings.data_dir / "manifest.json").read_text())
            self.assertTrue(np.isfinite(reconstructed_object).all())
            self.assertTrue(np.isfinite(reconstructed_phase).all())
            self.assertTrue(
                np.allclose(
                    reconstructed_object,
                    np.clip(network_object * manifest["measurement_max"], 0, 1),
                )
            )
            self.assertTrue((settings.report_dir / "reconstruction_comparison.png").is_file())
            self.assertTrue((settings.report_dir / "training_loss.png").is_file())
            training = report["training"]
            self.assertEqual(training["epochs"], settings.training_epochs)
            self.assertLessEqual(training["minimum_loss"], training["initial_loss"])
            self.assertTrue(
                np.isfinite(
                    report["reconstructed_system_aberration"][
                        "primary_piston_tip_tilt_removed"
                    ]["rmse_rad"]
                )
            )


if __name__ == "__main__":
    unittest.main()
