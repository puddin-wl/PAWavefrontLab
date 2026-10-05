import copy

import numpy as np
import pytest
import torch

from networks import MMESObject, PatchEmbedding, StaticDiffuseMMES, StaticDiffuseNet


@pytest.mark.parametrize("tau", [1, 2, 4])
def test_patch_embedding_matches_reflect_reference_and_inverse(tau):
    image = torch.randn(2, 1, 8, 9, dtype=torch.float64, requires_grad=True)
    embedding = PatchEmbedding(tau).double()
    padding = tau - 1
    source = np.pad(
        image.detach().numpy(),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="reflect",
    )
    height, width = 8 + padding, 9 + padding
    expected = np.stack(
        [
            source[:, 0, row : row + height, column : column + width]
            for row in range(tau)
            for column in range(tau)
        ],
        axis=1,
    )
    actual = embedding(image)
    assert np.allclose(actual.detach().numpy(), expected, atol=1e-12)
    assert torch.allclose(embedding.inverse(actual), image, atol=1e-12)
    gradient = torch.autograd.grad(embedding.inverse(actual).sum(), image)[0]
    assert torch.allclose(gradient, torch.ones_like(image), atol=1e-12)


def test_patch_inverse_uses_overlap_reconstruction_for_arbitrary_patches():
    tau = 3
    embedding = PatchEmbedding(tau).double()
    patches = torch.randn(1, tau * tau, 8, 9, dtype=torch.float64)
    padding = tau - 1
    reference = np.zeros((1, 1, 8 + padding, 9 + padding))
    for row in range(tau):
        for column in range(tau):
            reference[:, 0, row : row + 8, column : column + 9] += patches[
                :, row * tau + column
            ].numpy()
    reference = reference[:, :, padding:8, padding:9] / (tau * tau)
    assert np.allclose(embedding.inverse(patches).numpy(), reference, atol=1e-12)


@pytest.mark.parametrize("channels", [1, 2])
def test_mmes_autoencoder_loss_noise_eval_and_chunk_equivalence(channels):
    options = dict(
        tau=2,
        ranks=[16, 4, 16],
        noise_std=0.02,
        seed=7,
        init_scale=0.1,
        channels=channels,
    )
    chunked = MMESObject(8, chunk_size=11, checkpoint_chunks=True, **options)
    unchunked = MMESObject(8, chunk_size=0, checkpoint_chunks=False, **options)
    unchunked.load_state_dict(chunked.state_dict())
    chunked.eval()
    unchunked.eval()
    actual, ae_loss = chunked.forward_with_loss()
    expected, expected_loss = unchunked.forward_with_loss()
    assert torch.allclose(actual, expected, atol=1e-7)
    assert torch.allclose(ae_loss, expected_loss, atol=1e-8)
    assert torch.equal(chunked(), chunked())

    latent = chunked.z
    batch, channel_count, height, width = latent.shape
    embedded = chunked.embedding(latent.reshape(batch * channel_count, 1, height, width))
    patch_channels = embedded.shape[1]
    clean = embedded.reshape(batch, channel_count * patch_channels, *embedded.shape[-2:])
    decoded = chunked.autoencoder(clean.permute(0, 2, 3, 1).reshape(-1, clean.shape[1]))
    decoded = decoded.reshape(batch, *clean.shape[-2:], clean.shape[1]).permute(0, 3, 1, 2)
    assert torch.allclose(ae_loss, (decoded - clean).square().mean(), atol=1e-8)


def test_multiframe_selection_routes_gradient_only_to_selected_latents():
    model = MMESObject(
        8,
        tau=2,
        ranks=[16, 4, 16],
        noise_std=0,
        channels=2,
        num_frames=3,
        chunk_size=13,
    )
    field, ae_loss = model.forward_with_loss(frame_indices=torch.tensor([2, 0]))
    (field.square().mean() + ae_loss).backward()
    assert torch.count_nonzero(model.z.grad[0])
    assert not torch.count_nonzero(model.z.grad[1])
    assert torch.count_nonzero(model.z.grad[2])


@pytest.mark.parametrize("static_phase", [True, False])
def test_object_mmes_preserves_original_aberration_branch_and_optics(static_phase):
    shared = dict(
        width=16,
        PSF_size=16,
        phs_layers=1,
        bsize=2,
        static_phase=static_phase,
        zernike_features=4,
    )
    torch.manual_seed(12)
    baseline = StaticDiffuseNet(**shared)
    torch.manual_seed(12)
    model = StaticDiffuseMMES(
        **shared,
        mmes_kwargs=dict(
            tau=2,
            ranks=[16, 4, 16],
            noise_std=0,
            chunk_size=0,
        ),
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(baseline.g_g.parameters(), model.g_g.parameters())
    )
    baseline.g_im = copy.deepcopy(model.g_im)
    baseline.eval()
    model.eval()
    slm = torch.exp(1j * torch.randn(2, 1, 16, 16))
    time = torch.tensor([-0.5, 0.5])
    actual_outputs = model(slm, time)
    expected_outputs = baseline(slm, time)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert torch.allclose(actual, expected, atol=1e-6)
    actual_outputs[0].square().mean().backward()
    expected_outputs[0].square().mean().backward()
    for actual, expected in zip(model.g_g.parameters(), baseline.g_g.parameters()):
        assert torch.allclose(actual.grad, expected.grad, atol=1e-7)
