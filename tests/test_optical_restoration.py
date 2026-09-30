import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile

from tools.evaluate_optical_restoration import run


class OpticalCorrectionConsistencyTests(unittest.TestCase):
    @staticmethod
    def _origin(size: int = 64) -> np.ndarray:
        y, x = np.mgrid[:size, :size]
        return np.asarray(
            np.exp(-((y - 18) ** 2 + (x - 21) ** 2) / 40.0)
            + 0.7 * np.exp(-((y - 45) ** 2 + (x - 43) ** 2) / 65.0)
            + 0.2 * np.exp(-((y - 29) ** 2 + (x - 51) ** 2) / 25.0),
            dtype=np.float32,
        )

    @staticmethod
    def _translate(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
        translated = np.zeros_like(image)
        translated[dy:, dx:] = image[: image.shape[0] - dy, : image.shape[1] - dx]
        return translated

    def _prepare_case(
        self, root: Path, *, shift: tuple[int, int]
    ) -> tuple[argparse.Namespace, Path, Path]:
        data_dir = root / "data"
        reference_dir = data_dir / "reference"
        reconstruction_dir = root / "reconstruction"
        output_dir = root / "evaluation"
        reference_dir.mkdir(parents=True)
        reconstruction_dir.mkdir()

        origin = self._origin()
        corrected = self._translate(origin, *shift)
        computational = np.asarray(
            origin + 0.03 * np.linspace(0.0, 1.0, origin.shape[1])[None, :],
            dtype=np.float32,
        )
        np.save(reference_dir / "clear_object.npy", origin)
        (data_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        sio.savemat(
            reconstruction_dir / "final_I_est_network_units.mat",
            {"image": computational},
        )
        corrected_tiff = root / "re.tif"
        tifffile.imwrite(corrected_tiff, corrected)

        args = argparse.Namespace(
            data_dir=str(data_dir),
            reconstruction_dir=str(reconstruction_dir),
            corrected_tiff=str(corrected_tiff),
            output_dir=str(output_dir),
            scene_name="optical_correction_test",
            preprocessing_description="synthetic test projection",
            max_registration_shift=3,
        )
        return args, data_dir, output_dir

    def test_origin_and_corrected_acquisition_accept_reasonable_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, data_dir, output_dir = self._prepare_case(
                Path(temporary), shift=(2, 1)
            )
            run(args)

            report = json.loads(
                (output_dir / "final_validation_report.json").read_text()
            )
            selected = report["origin_corrected_similarity"]["selected"]
            comparisons = report["origin_corrected_similarity"]["all_comparisons"]
            self.assertEqual(selected["mode"], "registered_overlap")
            self.assertTrue(selected["registration_accepted"])
            self.assertEqual(selected["psnr_db"], comparisons["registered"]["psnr_db"])
            self.assertEqual(selected["ssim"], comparisons["registered"]["ssim"])
            self.assertTrue(
                (output_dir / "registered_optically_corrected.npy").is_file()
            )
            registered_origin = np.load(
                output_dir / "registered_origin_for_optically_corrected.npy"
            )
            registered_corrected = np.load(
                output_dir / "registered_optically_corrected.npy"
            )
            difference = np.load(
                output_dir / "origin_corrected_absolute_difference.npy"
            )
            np.testing.assert_allclose(
                difference, np.abs(registered_origin - registered_corrected)
            )
            self.assertTrue((output_dir / "final_validation_comparison.png").is_file())
            self.assertEqual(
                report["optical_quality_metrics"]["status"],
                "not_computed",
            )
            self.assertIn(
                "uncorrected",
                report["physical_semantics"]["origin_baseline"],
            )
            self.assertEqual(
                report["origin_baseline_source"]["legacy_filename"],
                "clear_object.npy",
            )
            self.assertNotIn("defocus", json.dumps(report).lower())
            manifest = json.loads((data_dir / "manifest.json").read_text())
            self.assertEqual(
                manifest["optical_correction_consistency"][
                    "selected_similarity_mode"
                ],
                "registered_overlap",
            )

    def test_origin_and_corrected_acquisition_reject_large_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, output_dir = self._prepare_case(
                Path(temporary), shift=(11, 8)
            )
            run(args)

            report = json.loads(
                (output_dir / "final_validation_report.json").read_text()
            )
            similarity = report["origin_corrected_similarity"]
            selected = similarity["selected"]
            self.assertEqual(selected["mode"], "same_coordinates")
            self.assertFalse(selected["registration_accepted"])
            self.assertEqual(
                selected["psnr_db"], similarity["all_comparisons"]["raw"]["psnr_db"]
            )
            self.assertEqual(
                selected["ssim"], similarity["all_comparisons"]["raw"]["ssim"]
            )
            self.assertIsNotNone(similarity["registration_rejection_reason"])
            self.assertFalse(
                (output_dir / "registered_optically_corrected.npy").exists()
            )
            difference = np.load(
                output_dir / "origin_corrected_absolute_difference.npy"
            )
            self.assertEqual(difference.shape, (64, 64))

    def test_missing_corrected_acquisition_has_explicit_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "evaluation"
            args = argparse.Namespace(
                data_dir=str(root / "data"),
                reconstruction_dir=str(root / "reconstruction"),
                corrected_tiff=str(root / "missing_re.tif"),
                output_dir=str(output_dir),
                scene_name="missing_corrected_test",
                preprocessing_description="synthetic test projection",
                max_registration_shift=3,
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "optically corrected acquisition 不存在"
            ):
                run(args)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
