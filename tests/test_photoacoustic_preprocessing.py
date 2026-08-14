import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile
from PIL import Image

from dataset import BatchDataset
from preprocessing.photoacoustic import (
    preprocess_photoacoustic_volume,
    subtract_photoacoustic_baseline,
)
from workflows.static_simulation.workflow import (
    SimulationSettings,
    generate_slm_patterns,
    import_photoacoustic_measurements,
    prepare_ground_truth,
)


ROOT = Path(__file__).resolve().parents[1]


def make_settings(root: Path) -> SimulationSettings:
    image_path = root / "object.png"
    Image.fromarray(np.arange(20 * 24, dtype=np.uint16).reshape(20, 24)).save(image_path)
    return SimulationSettings(
        project_root=ROOT,
        input_image=image_path,
        data_dir=root / "data",
        result_root=root / "results",
        scene_name="photoacoustic_test",
        size=16,
        num_frames=3,
        generation_device="cpu",
        training_device="cpu",
        training_epochs=1,
        training_batch_size=2,
        phase_layers=1,
        simulation_batch_size=2,
        visualization_frequency=0,
        raw_measurement_dir=root / "raw_measurements",
        silence_tqdm=True,
    )


class PhotoacousticPreprocessingTests(unittest.TestCase):
    def test_old_baseline_clipping_and_axis_zero_projection_without_512_layers(self):
        volume = np.array(
            [
                [[2047, 2048], [2050, 2030]],
                [[2049, 2055], [2048, 2100]],
                [[2048, 2050], [2060, 2040]],
            ],
            dtype=np.uint16,
        )
        corrected = subtract_photoacoustic_baseline(volume)
        expected_projection = np.array([[1, 7], [12, 52]], dtype=np.float32)
        self.assertEqual(corrected.dtype, np.float32)
        self.assertGreaterEqual(float(corrected.min()), 0.0)
        self.assertTrue(
            np.array_equal(preprocess_photoacoustic_volume(volume), expected_projection)
        )

    def test_step_one_accepts_a_three_dimensional_photoacoustic_tiff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(root)
            signal = np.arange(12 * 14, dtype=np.uint16).reshape(12, 14)
            volume = np.stack(
                [
                    np.full_like(signal, 2040),
                    2048 + signal,
                    2048 + signal // 2,
                ]
            )
            volume_path = root / "clear_volume.tif"
            tifffile.imwrite(volume_path, volume, photometric="minisblack")
            settings = replace(settings, input_image=volume_path, input_mode="auto")
            manifest = prepare_ground_truth(settings)
            projection = np.load(
                settings.reference_dir / "photoacoustic_projection_raw.npy",
                allow_pickle=False,
            )
            self.assertEqual(projection.shape, (12, 14))
            self.assertTrue(np.array_equal(projection, signal.astype(np.float32)))
            self.assertEqual(manifest["input_preprocessing"]["projection_axis"], 0)
            self.assertIsNone(manifest["input_preprocessing"]["layer_count_required"])

    def test_real_measurement_import_preserves_cross_frame_intensity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(root)
            prepare_ground_truth(settings)
            generate_slm_patterns(settings)
            raw_dir = settings.raw_measurement_dir
            assert raw_dir is not None
            raw_dir.mkdir()
            expected = []
            grid = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
            for frame in range(1, settings.num_frames + 1):
                signal = grid + frame * 100
                volume = np.stack(
                    [
                        np.full_like(signal, 2040),
                        2048 + signal // 2,
                        2048 + signal,
                        np.full_like(signal, 2048),
                    ]
                )
                tifffile.imwrite(
                    raw_dir / f"capture_{frame}.tif",
                    volume,
                    photometric="minisblack",
                )
                expected.append(signal.astype(np.float32))
            manifest = import_photoacoustic_measurements(settings)
            measurements = np.load(settings.data_dir / "measurements.npy")
            self.assertTrue(np.array_equal(measurements, np.stack(expected)))
            self.assertTrue(
                np.array_equal(
                    sio.loadmat(settings.data_dir / "SLM_raw1.mat")["imsdata"],
                    expected[0],
                )
            )
            self.assertEqual(manifest["measurement_max"], float(expected[-1].max()))
            self.assertEqual(
                manifest["photoacoustic_preprocessing"]["preview_normalization"],
                "one_dataset_wide_maximum",
            )
            dataset = BatchDataset(settings.data_dir)
            _, first_frame, _ = dataset[0]
            self.assertAlmostEqual(
                float(first_frame.max()),
                float(expected[0].max() / expected[-1].max()),
                places=6,
            )
            saved_manifest = json.loads(
                (settings.data_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(saved_manifest["completed_steps"]["measurements"])


if __name__ == "__main__":
    unittest.main()
