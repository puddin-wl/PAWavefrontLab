import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from optics import zernike_basis_numpy
from tools.generate_dual_grid_slm import (
    _sample_coefficients,
    _synthesize_phases_low_memory,
    generate,
)


class DualGridSlmTests(unittest.TestCase):
    def test_radial15_coefficients_are_reproducible_and_disable_tip_tilt(self):
        first = _sample_coefficients(3, 136, 1.22, 20260731, (2, 3))
        second = _sample_coefficients(3, 136, 1.22, 20260731, (2, 3))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (3, 136))
        self.assertTrue(np.all(first[:, 1:3] == 0))
        self.assertTrue(np.any(first[:, 135] != 0))

    def test_truncated_low_order_coefficients_obey_limit_and_disabled_terms(self):
        first = _sample_coefficients(
            50, 15, 0.25, 20260731, (1, 2, 3, 4), coefficient_limit=0.5
        )
        second = _sample_coefficients(
            50, 15, 0.25, 20260731, (1, 2, 3, 4), coefficient_limit=0.5
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[:, :4] == 0))
        self.assertTrue(np.any(first[:, 4:] != 0))
        self.assertLessEqual(float(np.abs(first).max()), 0.5)

    def test_low_memory_synthesis_matches_full_basis(self):
        coefficients = _sample_coefficients(2, 136, 1.22, 7, (2, 3))
        expected = np.einsum(
            "fm,mhw->fhw",
            coefficients,
            zernike_basis_numpy(136, 16),
            optimize=True,
        ).astype(np.float32)
        actual = _synthesize_phases_low_memory(coefficients, 16, block_size=1)
        self.assertTrue(np.allclose(actual, expected, rtol=2e-5, atol=2e-5))

    def test_generate_writes_paired_model_and_hardware_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dual"
            args = argparse.Namespace(
                output_dir=str(output),
                num_frames=2,
                hardware_size=32,
                model_size=16,
                num_modes=136,
                disabled_noll_indices=[2, 3],
                sigma=1.22,
                seed=20260731,
                render_block_size=1,
            )
            generate(args)
            model = sio.loadmat(output / "model_16/mat/SLM_sim1.mat")["proj_sim"]
            hardware = sio.loadmat(output / "hardware_32/mat/SLM_hw1.mat")["phase_hw"]
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(model.shape, (16, 16))
            self.assertEqual(hardware.shape, (32, 32))
            self.assertTrue(np.isfinite(model).all())
            self.assertTrue(np.isfinite(hardware).all())
            self.assertEqual(manifest["noll_indices"][-1], 136)
            self.assertEqual(manifest["disabled_noll_indices"], [2, 3])
            self.assertTrue(manifest["same_coefficients_on_both_grids"])
            self.assertTrue(manifest["defocus_included"])
            self.assertTrue(manifest["piston_included"])

    def test_generate_scales_saved_coefficients_and_extends_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.npy"
            base = np.arange(8, dtype=np.float32).reshape(2, 4)
            np.save(base_path, base)
            output = root / "extended"
            args = argparse.Namespace(
                output_dir=str(output),
                num_frames=2,
                hardware_size=16,
                model_size=16,
                num_modes=6,
                disabled_noll_indices=[1, 2, 3],
                sigma=1.0,
                seed=1,
                base_coefficients=str(base_path),
                coefficient_scale=0.25,
                extension_sigma=0.1,
                extension_seed=9,
                render_block_size=1,
            )
            generate(args)
            coefficients = np.load(output / "slm_coefficients.npy")
            manifest = json.loads((output / "manifest.json").read_text())
            np.testing.assert_array_equal(coefficients[:, 3], base[:, 3] * 0.25)
            self.assertTrue(np.all(coefficients[:, :3] == 0))
            self.assertTrue(np.any(coefficients[:, 4:] != 0))
            self.assertEqual(manifest["base_mode_count"], 4)
            self.assertEqual(manifest["base_coefficient_scale"], 0.25)
            self.assertEqual(manifest["extension_noll_range"], [5, 6])
            self.assertEqual(manifest["extension_seed"], 9)
            self.assertFalse(manifest["piston_included"])


if __name__ == "__main__":
    unittest.main()
