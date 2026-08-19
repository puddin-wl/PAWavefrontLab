"""Run the complete radial-order-15 simulation, reconstruction and comparison."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks import StaticDiffuseNet  # noqa: E402
from optics import slm_complex_field  # noqa: E402
from workflows.static_simulation.radial15_config import (  # noqa: E402
    BASELINE_REPORT,
    SETTINGS,
)
from workflows.static_simulation.workflow import (  # noqa: E402
    compare_reconstruction_with_baseline,
    evaluate_reconstruction,
    generate_slm_patterns,
    prepare_ground_truth,
    reconstruct_static_scene,
    simulate_modulated_measurements,
)


def select_cuda_batch_size(settings, candidates=(8, 4, 2, 1)):
    """Select the first batch size that completes one representative optimizer step."""
    if settings.training_device != "cuda":
        raise ValueError("径向 15 阶正式仿真要求 training_device='cuda'。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法运行径向 15 阶正式仿真。")
    device = torch.device("cuda")
    attempts = []
    for batch_size in candidates:
        network = phase = slm = output = loss = None
        image_optimizer = phase_optimizer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            network = StaticDiffuseNet(
                settings.size,
                settings.size,
                phs_layers=settings.phase_layers,
                bsize=batch_size,
                static_phase=True,
                zernike_features=settings.network_zernike_features,
            ).to(device)
            image_optimizer = torch.optim.Adam(
                network.g_im.parameters(), lr=settings.initial_learning_rate
            )
            phase_optimizer = torch.optim.Adam(
                network.g_g.parameters(), lr=settings.initial_learning_rate
            )
            phase = torch.zeros(
                (batch_size, settings.size, settings.size), device=device
            )
            slm = slm_complex_field(phase, settings.size, settings.phase_sign).unsqueeze(1)
            output, _, _, _, _ = network(
                slm, torch.zeros(batch_size, device=device)
            )
            loss = output.mean()
            loss.backward()
            image_optimizer.step()
            phase_optimizer.step()
            record = {
                "batch_size": batch_size,
                "status": "passed",
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
            attempts.append(record)
            return replace(settings, training_batch_size=batch_size), {
                "device_name": torch.cuda.get_device_name(device),
                "pytorch_version": torch.__version__,
                "pytorch_cuda_version": torch.version.cuda,
                "network_zernike_features": settings.network_zernike_features,
                "image_size": settings.size,
                "attempts": attempts,
                "selected_batch_size": batch_size,
            }
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "out_of_memory",
                    "message": str(error),
                }
            )
        finally:
            del loss, output, slm, phase, image_optimizer, phase_optimizer, network
            torch.cuda.empty_cache()
    raise RuntimeError(f"batch=8、4、2、1 均显存不足：{attempts}")


def main() -> None:
    settings, smoke_report = select_cuda_batch_size(SETTINGS)
    print(json.dumps(smoke_report, indent=2, ensure_ascii=False), flush=True)
    prepare_ground_truth(settings)
    generate_slm_patterns(settings)
    simulate_modulated_measurements(settings)
    reconstruct_static_scene(settings)
    report = evaluate_reconstruction(settings)
    comparison = compare_reconstruction_with_baseline(settings, BASELINE_REPORT)
    (settings.report_dir / "cuda_smoke_test.json").write_text(
        json.dumps(smoke_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
