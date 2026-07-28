import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from dataset import BatchDataset


ROOT = Path(__file__).resolve().parents[1]


class ClosedLoopTests(unittest.TestCase):
    def _generate(self, directory: Path, mode: str = "zernike", seed: int = 4):
        image = np.arange(20 * 28, dtype=np.uint16).reshape(20, 28)
        image_path = directory.parent / f"object-{mode}-{seed}.png"
        Image.fromarray(image).save(image_path)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "generate_neuws_data.py"),
                "simulate",
                "--input-image", str(image_path),
                "--output-dir", str(directory),
                "--size", "16",
                "--num-frames", "3",
                "--aberration-mode", mode,
                "--seed", str(seed),
                "--device", "cpu",
            ],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_generator_loader_round_trip_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            different, gaussian = root / "different", root / "gaussian"
            self._generate(first)
            self._generate(second)
            self._generate(different, seed=5)
            self._generate(gaussian, mode="complex-gaussian")
            self.assertTrue(np.array_equal(np.load(first / "slm_patterns.npy"), np.load(second / "slm_patterns.npy")))
            self.assertTrue(np.array_equal(
                sio.loadmat(first / "SLM_raw1.mat")["imsdata"],
                sio.loadmat(second / "SLM_raw1.mat")["imsdata"],
            ))
            self.assertTrue(np.array_equal(
                sio.loadmat(first / "ground_truth.mat")["aberration_field"],
                sio.loadmat(second / "ground_truth.mat")["aberration_field"],
            ))
            self.assertFalse(np.array_equal(
                sio.loadmat(first / "SLM_raw1.mat")["imsdata"],
                sio.loadmat(different / "SLM_raw1.mat")["imsdata"],
            ))
            self.assertEqual(sio.loadmat(first / "SLM_sim1.mat")["proj_sim"].shape, (10, 16))
            self.assertEqual(sio.loadmat(first / "SLM_raw1.mat")["imsdata"].shape, (16, 16))
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["phase_sign"], -1)
            dataset = BatchDataset(first)
            slm, measurement, _ = dataset[0]
            phase = torch.from_numpy(sio.loadmat(first / "SLM_sim1.mat")["proj_sim"])
            self.assertTrue(torch.allclose(slm[:, 3:13], torch.exp(-1j * phase)))
            self.assertEqual(tuple(measurement.shape), (16, 16))
            gaussian_field = sio.loadmat(gaussian / "ground_truth.mat")["aberration_field"]
            self.assertTrue(np.isfinite(gaussian_field).all())

    def test_static_training_decreases_loss_without_per_frame_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, run_dir = root / "data", root / "run"
            self._generate(data_dir, seed=9)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "recon_exp_data.py"),
                    "--root_dir", str(run_dir),
                    "--data_dir", str(data_dir),
                    "--scene_name", "test",
                    "--num_epochs", "3",
                    "--batch_size", "2",
                    "--phs_layers", "1",
                    "--vis_freq", "0",
                    "--static_phase",
                    "--device", "cpu",
                    "--silence_tqdm",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            final_dir = run_dir / "vis" / "test" / "final"
            summary = json.loads((final_dir / "training_summary.json").read_text())
            self.assertLess(summary["loss_history"][-1], summary["loss_history"][0])
            self.assertTrue((final_dir / "final_I_est.mat").is_file())
            self.assertTrue((final_dir / "final_aberration.mat").is_file())
            self.assertFalse((final_dir / "per_frame").exists())

    def test_loader_rejects_non_contiguous_numbering(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            self._generate(data_dir)
            (data_dir / "SLM_raw2.mat").unlink()
            (data_dir / "SLM_sim2.mat").unlink()
            with self.assertRaisesRegex(ValueError, "continuous"):
                BatchDataset(data_dir)


if __name__ == "__main__":
    unittest.main()
