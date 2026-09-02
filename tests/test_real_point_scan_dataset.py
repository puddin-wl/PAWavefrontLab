import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io as sio
import tifffile

from dataset import BatchDataset
from tools.evaluate_real_reconstruction import evaluate
from tools.prepare_real_point_scan_dataset import prepare


class RealPointScanDatasetTests(unittest.TestCase):
    def test_origin_only_evaluation_does_not_require_defocus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            reference_dir = data_dir / "reference"
            final_dir = root / "final"
            output_dir = root / "evaluation"
            reference_dir.mkdir(parents=True)
            final_dir.mkdir()
            image = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
            phase = np.zeros((16, 16), dtype=np.float32)
            np.save(reference_dir / "clear_object.npy", image)
            (data_dir / "manifest.json").write_text(
                json.dumps({"origin_only": True}), encoding="utf-8"
            )
            sio.savemat(
                final_dir / "final_I_est_network_units.mat", {"image": image}
            )
            sio.savemat(
                final_dir / "final_aberration.mat",
                {"phase": phase, "field": np.exp(1j * phase)},
            )
            (final_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "num_epochs": 2,
                        "requested_num_epochs": 2,
                        "batch_size": 1,
                        "phase_layers": 1,
                        "elapsed_seconds": 1.0,
                        "loss_history": [0.1, 0.05],
                        "device": "cpu",
                        "early_stopping": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            result = evaluate(
                argparse.Namespace(
                    data_dir=str(data_dir),
                    reconstruction_dir=str(final_dir),
                    output_dir=str(output_dir),
                    scene_name="origin_only_evaluation",
                )
            )
            report = json.loads((result / "reconstruction_report.json").read_text())
            self.assertTrue(report["origin_only"])
            self.assertNotIn("defocus_baseline", report)
            self.assertTrue(report["selected_comparison"]["registration_accepted"])

    def test_samples_only_ignores_non_sample_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            phases = root / "phases"
            output = root / "prepared"
            source.mkdir()
            phases.mkdir()
            for index in (1, 2):
                (source / f"S{index}_test_PA1.bin").write_bytes(b"sample")
                sio.savemat(
                    phases / f"SLM_sim{index}.mat",
                    {"proj_sim": np.zeros((600, 600), dtype=np.float32)},
                )
            (source / "NO_SLM_test_PA1.bin").write_bytes(b"ignored")

            def fake_projection(path, destination, **_kwargs):
                index = 1 if path.name.startswith("S1_") else 2
                image = np.full((600, 600), index, dtype=np.float32)
                tifffile.imwrite(destination, image.astype(np.uint16))
                return image

            args = argparse.Namespace(
                source_dir=str(source),
                output_dir=str(output),
                phase_dir=str(phases),
                scene_name="samples_only_test",
                num_frames=2,
                baseline=2048.0,
                chunk_mb=1,
                samples_only=True,
            )
            with patch(
                "tools.prepare_real_point_scan_dataset._convert_projection",
                side_effect=fake_projection,
            ):
                prepare(args)

            manifest = json.loads((output / "manifest.json").read_text())
            dataset = BatchDataset(output)
            self.assertEqual(len(dataset), 2)
            self.assertTrue(manifest["samples_only"])
            self.assertFalse(manifest["completed_steps"]["ground_truth"])
            self.assertEqual(
                manifest["ignored_non_sample_files"], ["NO_SLM_test_PA1.bin"]
            )
            self.assertNotIn("ground_truth", manifest)
            self.assertFalse((output / "ground_truth.mat").exists())
            self.assertTrue((output / "quality_control.png").is_file())

    def test_origin_only_uses_origin_without_requiring_defocus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            reference_source = root / "reference_raw"
            phases = root / "phases"
            output = root / "prepared"
            source.mkdir()
            reference_source.mkdir()
            phases.mkdir()
            origin_path = reference_source / "origin_test_PA1.bin"
            origin_path.write_bytes(b"origin")
            for index in (1, 2):
                (source / f"d{index}_test_PA1.bin").write_bytes(b"sample")
                sio.savemat(
                    phases / f"SLM_sim{index}.mat",
                    {"proj_sim": np.zeros((600, 600), dtype=np.float32)},
                )

            def fake_projection(path, destination, **_kwargs):
                if path.name.startswith("origin_"):
                    image = np.arange(600 * 600, dtype=np.float32).reshape(600, 600)
                else:
                    index = 1 if path.name.startswith("d1_") else 2
                    image = np.full((600, 600), index, dtype=np.float32)
                tifffile.imwrite(destination, image.astype(np.uint32))
                return image

            args = argparse.Namespace(
                source_dir=str(source),
                output_dir=str(output),
                phase_dir=str(phases),
                scene_name="origin_only_test",
                num_frames=2,
                baseline=2048.0,
                chunk_mb=1,
                samples_only=False,
                origin_only=True,
                sample_prefix="d",
                origin_source=str(origin_path),
            )
            with patch(
                "tools.prepare_real_point_scan_dataset._convert_projection",
                side_effect=fake_projection,
            ):
                prepare(args)

            manifest = json.loads((output / "manifest.json").read_text())
            ground_truth = sio.loadmat(output / "ground_truth.mat")
            self.assertEqual(manifest["workflow"], "real_point_scan_neuws_origin_only")
            self.assertEqual(manifest["sample_prefix"], "d")
            self.assertTrue(manifest["origin_only"])
            self.assertFalse(manifest["samples_only"])
            self.assertNotIn("defocus_baseline", manifest)
            self.assertNotIn("defocus_projection", ground_truth)
            self.assertEqual(manifest["ignored_non_sample_files"], [])
            self.assertTrue((output / "reference" / "origin_projection.npy").is_file())
            self.assertTrue((output / "measurement_mip_tiff" / "d1_mip.tif").is_file())


if __name__ == "__main__":
    unittest.main()
