import argparse
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.export_ppt_assets import export


class ExportPptAssetsTests(unittest.TestCase):
    def test_exports_corrected_acquisition_assets_without_defocus_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            reference_dir = data_dir / "reference"
            reconstruction_dir = root / "reconstruction"
            validation_dir = root / "validation"
            correction_dir = root / "correction"
            output_dir = root / "ppt"
            for directory in (
                reference_dir,
                reconstruction_dir,
                validation_dir,
                correction_dir,
            ):
                directory.mkdir(parents=True)

            y, x = np.mgrid[:16, :16]
            origin = np.asarray(x + y + 1, dtype=np.float32)
            corrected = np.asarray(1.1 * x + 0.9 * y + 2, dtype=np.float32)
            np.save(reference_dir / "clear_object.npy", origin)
            np.save(reference_dir / "optically_corrected_projection.npy", corrected)
            np.save(
                reconstruction_dir / "reconstructed_object_normalized.npy",
                np.flipud(origin),
            )
            np.save(
                reconstruction_dir / "reconstructed_aberration_phase.npy",
                np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(16, 16),
            )
            (reconstruction_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "loss_history": [1.0, 0.5, 0.25],
                        "num_epochs": 3,
                        "early_stopping": {"best_epoch": 3},
                    }
                ),
                encoding="utf-8",
            )
            np.save(
                correction_dir / "SLM_final_correction_1080.npy",
                np.linspace(0, 2 * np.pi, 256, dtype=np.float32).reshape(16, 16),
            )
            plt.imsave(
                correction_dir / "SLM_final_correction_preview.png",
                origin,
                cmap="gray",
            )
            plt.imsave(
                validation_dir / "final_validation_comparison.png",
                corrected,
                cmap="gray",
            )
            report = {
                "origin_corrected_similarity": {
                    "selected": {
                        "mode": "same_coordinates",
                        "psnr_db": 20.5,
                        "ssim": 0.81,
                        "registration_accepted": False,
                        "proposed_shift_yx_pixels": [12, 1],
                    }
                },
                "optical_quality_metrics": {
                    "status": "not_computed",
                    "reason": "no validated ROI",
                },
            }
            (validation_dir / "final_validation_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            result = export(
                argparse.Namespace(
                    data_dir=str(data_dir),
                    reconstruction_dir=str(reconstruction_dir),
                    validation_dir=str(validation_dir),
                    correction_dir=str(correction_dir),
                    output_dir=str(output_dir),
                )
            )

            self.assertEqual(result, output_dir)
            self.assertTrue((output_dir / "01_origin_baseline.png").is_file())
            self.assertTrue(
                (output_dir / "03_optically_corrected_acquisition.png").is_file()
            )
            self.assertTrue(
                (output_dir / "07_origin_corrected_similarity.png").is_file()
            )
            self.assertFalse((output_dir / "02_defocus_baseline.png").exists())
            readme = (output_dir / "PPT素材说明.md").read_text(encoding="utf-8")
            self.assertIn("不是 clear ground truth", readme)
            self.assertIn("不能据此声称光学校正质量提高", readme)
            self.assertIn("not_computed", readme)
            self.assertIn("same_coordinates", readme)
            self.assertNotIn("Defocus", readme)


if __name__ == "__main__":
    unittest.main()
