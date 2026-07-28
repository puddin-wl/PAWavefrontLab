import os
import unittest

import numpy as np
import torch
from aotools.functions import zernIndex

from evaluation import evaluate_images, evaluate_phases
from optics import (
    aperture_mask,
    default_aperture_height,
    make_static_aberration,
    paper_aperture_height,
    sample_slm_coefficients,
    synthesize_slm_patterns,
    validate_geometry,
    zernike_basis_numpy,
    zernike_basis_torch,
)
from utils import compute_zernike_basis


class OpticsTests(unittest.TestCase):
    def test_noll_mapping_one_through_fifteen(self):
        expected = [
            [0, 0], [1, 1], [1, -1], [2, 0], [2, -2], [2, 2],
            [3, -1], [3, 1], [3, -3], [3, 3], [4, 0], [4, 2],
            [4, -2], [4, 4], [4, -4],
        ]
        self.assertEqual([zernIndex(index) for index in range(1, 16)], expected)

    def test_paper_geometry_and_even_validation(self):
        self.assertEqual(default_aperture_height(256), 256)
        self.assertEqual(default_aperture_height(1000), 1000)
        self.assertEqual(paper_aperture_height(256), 144)
        self.assertEqual(paper_aperture_height(1000), 562)
        self.assertEqual(validate_geometry(32).aperture_height, 32)
        with self.assertRaises(ValueError):
            validate_geometry(255)
        with self.assertRaises(ValueError):
            validate_geometry(32, 34)

    def test_network_and_generator_share_basis(self):
        shared = zernike_basis_torch(15, 32)
        network = compute_zernike_basis(15, (32, 32))
        self.assertTrue(torch.equal(shared, network))
        pupil = np.any(zernike_basis_numpy(1, 32) != 0, axis=0)
        self.assertAlmostEqual(float(np.mean(shared[0].numpy()[pupil] ** 2)), 1.0, delta=0.03)

    def test_seeded_patterns_are_reproducible(self):
        basis = zernike_basis_numpy(15, 32)
        first = sample_slm_coefficients(3, seed=10)
        second = sample_slm_coefficients(3, seed=10)
        different = sample_slm_coefficients(3, seed=11)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, different))
        patterns = synthesize_slm_patterns(first, basis, 18)
        self.assertEqual(patterns.shape, (3, 18, 32))
        self.assertTrue(np.isfinite(patterns).all())

    def test_aberration_modes_are_finite_and_normalized(self):
        zernike, coefficients = make_static_aberration("zernike", 32, 18, seed=2)
        gaussian, no_coefficients = make_static_aberration("complex-gaussian", 32, 18, seed=2)
        mask = aperture_mask(32, 18).bool()
        self.assertEqual(coefficients.shape, (28,))
        self.assertEqual(no_coefficients.shape, (0,))
        self.assertTrue(torch.isfinite(zernike).all())
        self.assertTrue(torch.isfinite(gaussian).all())
        self.assertAlmostEqual(float(zernike.abs().square()[mask].mean()), 1.0, places=6)
        self.assertAlmostEqual(float(gaussian.abs().square()[mask].mean()), 1.0, places=5)


class EvaluationTests(unittest.TestCase):
    def test_identity_image_metrics(self):
        image = np.linspace(0, 1, 32 * 32, dtype=np.float64).reshape(32, 32)
        result, _ = evaluate_images(image, image)
        self.assertTrue(np.isinf(result["raw"]["psnr_db"]))
        self.assertAlmostEqual(result["raw"]["ssim"], 1.0, places=12)

    def test_known_integer_shift_is_recovered(self):
        reference = np.zeros((32, 32), dtype=np.float64)
        reference[8:20, 6:17] = np.arange(12 * 11).reshape(12, 11) / (12 * 11)
        estimate = np.zeros_like(reference)
        estimate[10:22, 9:20] = reference[8:20, 6:17]
        result, _ = evaluate_images(reference, estimate, register=True)
        self.assertEqual(result["registered"]["shift_yx_pixels"], [-2, -3])
        self.assertTrue(np.isinf(result["registered"]["psnr_db"]))
        self.assertAlmostEqual(result["registered"]["ssim"], 1.0, places=12)

    def test_piston_tip_tilt_are_removed(self):
        size = 32
        mask = aperture_mask(size, 18).numpy().astype(bool)
        reference = np.zeros((size, size), dtype=np.float64)
        basis = zernike_basis_numpy(3, size)
        estimate = np.einsum("m,mhw->hw", np.array([0.7, 0.25, -0.2]), basis)
        result, primary, _ = evaluate_phases(reference, estimate, mask)
        self.assertLess(result["primary_piston_tip_tilt_removed"]["rmse_rad"], 1e-5)
        self.assertLess(float(np.sqrt(np.mean(primary[mask] ** 2))), 1e-5)


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("NEUWS_RUN_LARGE_CUDA") == "1",
    "Set NEUWS_RUN_LARGE_CUDA=1 to run the 1000x1000 CUDA acceptance test.",
)
class LargeCudaTests(unittest.TestCase):
    def test_1000_forward_backward(self):
        from networks import StaticDiffuseNet

        size = 1000
        network = StaticDiffuseNet(size, size, phs_layers=1, bsize=1, static_phase=True).cuda()
        phase = torch.zeros((1, default_aperture_height(size), size), device="cuda")
        from optics import slm_complex_field

        slm = slm_complex_field(phase, size).unsqueeze(1)
        output, _, _, _, _ = network(slm, torch.tensor([-0.5], device="cuda"))
        output.mean().backward()
        self.assertTrue(torch.isfinite(output).all())
