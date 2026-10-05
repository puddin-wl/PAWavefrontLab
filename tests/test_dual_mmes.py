import io

import pytest
import torch

from networks import DualMMES, model_from_checkpoint


def _options(seed=0):
    return dict(
        tau=2,
        ranks=[16, 4, 16],
        noise_std=0,
        seed=seed,
        init_scale=0.1,
        chunk_size=0,
        checkpoint_chunks=False,
    )


@pytest.mark.parametrize("static_phase", [True, False])
def test_dual_mmes_independent_branches_optics_gradients_and_batch_one(static_phase):
    model = DualMMES(
        16,
        16,
        bsize=2,
        num_frames=3,
        static_phase=static_phase,
        object_kwargs=_options(0),
        aberration_kwargs=_options(1),
        amplitude_offset=1.0,
        phs_layers=1,
        zernike_features=4,
    )
    assert not ({id(p) for p in model.g_im.parameters()} & {id(p) for p in model.g_g.parameters()})
    model.eval()
    slm = torch.exp(-1j * torch.randn(2, 1, 16, 16))
    time = torch.tensor([0.5, -0.5])
    prediction, kernel, field, phase, obj = model(slm, time)
    components = model.g_g(
        None if static_phase else torch.tensor([2, 0])
    )
    if static_phase:
        components = components.expand(2, -1, -1, -1)
    expected_amplitude = 1.0 + components[:, 0:1]
    assert torch.allclose(field, expected_amplitude * torch.exp(1j * components[:, 1:2]))
    assert torch.allclose(phase, components[:, 1:2])
    assert torch.allclose(kernel.sum((-2, -1)), torch.ones(2, 1), atol=1e-6)
    assert prediction.shape == (2, 16, 16)
    assert obj.shape == (1, 1, 16, 16)

    target = torch.rand_like(prediction)
    object_gradient, aberration_gradient = torch.autograd.grad(
        (prediction - target).square().mean(), [model.g_im.z, model.g_g.z]
    )
    assert torch.isfinite(object_gradient).all() and torch.count_nonzero(object_gradient)
    assert torch.isfinite(aberration_gradient).all()
    assert torch.count_nonzero(aberration_gradient[:, 0])
    assert torch.count_nonzero(aberration_gradient[:, 1])

    one_prediction = model(slm[:1], time[:1])[0]
    assert one_prediction.shape == (1, 16, 16)


def test_dynamic_dual_mmes_rejects_non_acquired_time():
    model = DualMMES(
        8,
        8,
        num_frames=3,
        static_phase=False,
        object_kwargs=_options(),
        aberration_kwargs=_options(1),
        zernike_features=4,
    )
    with pytest.raises(ValueError, match="discrete"):
        model.frame_indices(torch.tensor([0.1]))


def test_dual_mmes_checkpoint_reload():
    config = {
        "model_mode": "dual_mmes",
        "width": 8,
        "psf_size": 8,
        "num_frames": 3,
        "batch_size": 2,
        "phs_layers": 1,
        "static_phase": True,
        "zernike_features": 4,
        "object_mmes_config": _options(0),
        "aberration_mmes_config": _options(1),
        "amplitude_offset": 1.0,
    }
    model = DualMMES(
        8,
        8,
        bsize=2,
        num_frames=3,
        static_phase=True,
        object_kwargs=_options(0),
        aberration_kwargs=_options(1),
        phs_layers=1,
        zernike_features=4,
    ).eval()
    buffer = io.BytesIO()
    torch.save(
        {"model_config": config, "model_state_dict": model.state_dict()}, buffer
    )
    buffer.seek(0)
    restored = model_from_checkpoint(torch.load(buffer, weights_only=True))
    slm = torch.exp(1j * torch.randn(2, 1, 8, 8))
    time = torch.tensor([-0.5, 0.5])
    with torch.no_grad():
        for before, after in zip(model(slm, time), restored(slm, time)):
            assert torch.allclose(before, after, atol=1e-6)
