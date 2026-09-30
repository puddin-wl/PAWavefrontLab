"""NeuWS: independent MMES object and amplitude/phase aberration branches.

两个 MMES 均优化可学习 Z 和自编码器；光学前向沿用原代码。
训练 loss = measurement MSE + weighted object AE + weighted aberration AE.
"""
from pathlib import Path
import json
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import tqdm

from dataset import BatchDataset
from networks_DualMMES import build_model


# ======================== CONFIG ========================
ROOT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = 'data/Zernike_SLM_data'   # Absolute paths also work.
SCENE_NAME = 'my_scene'
MODEL_MODE = 'object_mmes'            # 'dual_mmes', 'object_mmes', or 'original'.
STATIC_PHASE = True
DYNAMIC_SCENE = False               # This version generates ONE static object.
NUM_T = 100
HEIGHT=256
WIDTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 1000
PHS_LAYERS = 4                     # Used only by the original aberration comparator.
IM_PREFIX = 'SLM_raw'
MAX_INTENSITY = 0
ZERO_FREQ = -1
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Keep both learning rates constant initially, as in the uploaded baseline.
IM_LR = 1e-3
PH_LR = 1e-3
IM_FINAL_LR = 1e-3
PH_FINAL_LR = 1e-3
VIS_FREQ = 50                     # Iterations; 0 disables intermediate plots.
SAVE_PER_FRAME = True               # MAT files for time-dependent aberration.
SILENCE_TQDM = False

# MEAN losses: separate, adjustable starting weights, not paper-sum lambdas.
OBJECT_AE_WEIGHT = 1.0
ABERRATION_AE_WEIGHT = 1.0
AMPLITUDE_OFFSET = 1.0             # a = offset + decoded residual; still allows negative a.
OBJECT_MMES_CONFIG = dict(
    tau=4, ranks=[512,16,512], noise_std=0.01, seed=0, init_scale=0.1,
    chunk_size=4096, checkpoint_chunks=True,
)
ABERRATION_MMES_CONFIG = dict(
    tau=4, ranks=[512,16,512], noise_std=0.01, seed=1, init_scale=0.1,
    chunk_size=4096, checkpoint_chunks=True,
)
# Aberration input/output patch dimension is 2*tau**2, with a shared bottleneck.
# STATIC_PHASE=False uses frame-specific Z slices with one shared aberration AE.
# ========================================================


def _time_coordinate(indices):
    return indices.float() / (NUM_T - 1) - 0.5 if NUM_T > 1 else indices.float() * 0 - 0.5


def clean_preview(function):
    def wrapped(net, *args, **kwargs):
        was_training = net.training
        net.eval()
        try:
            return function(net, *args, **kwargs)
        finally:
            net.train(was_training)
    return wrapped


@clean_preview
@torch.no_grad()
def save_preview(net, x, measurement, t, destination):
    """Visualize the same time and SLM pattern as the measured image."""
    prediction, kernel, g, phase, obj = net(x[:1], t[:1])
    arrays = [measurement[0], prediction.reshape(1, WIDTH, WIDTH)[0],
              obj[0, 0], g[0, 0].abs(), torch.angle(g[0, 0]), kernel[0, 0]]
    titles = ['Measurement', 'Prediction', 'Object', '|g|', 'arg(g) [rad]', 'PSF']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (ax, array, title) in enumerate(zip(axes.flat, arrays, titles)):
        kwargs = dict(cmap='gray')
        if i in (0, 1):
            kwargs.update(vmin=0, vmax=1)
        if i == 4:
            kwargs.update(cmap='twilight', vmin=-np.pi, vmax=np.pi)
        artist = ax.imshow(array.detach().cpu().numpy(), **kwargs)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(artist, ax=ax, shrink=0.7)
    fig.suptitle(f't = {t[0].item():.4f}; object display uses its own value range')
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def main():
    if DYNAMIC_SCENE:
        raise ValueError('This MMES branch models one static object. Set DYNAMIC_SCENE=False.')
    if MODEL_MODE not in ('dual_mmes','object_mmes','original'):
        raise ValueError('Unknown MODEL_MODE.')
    for weight in (OBJECT_AE_WEIGHT, ABERRATION_AE_WEIGHT):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('AE weights must be finite and nonnegative.')
    if min(NUM_T, NUM_EPOCHS, BATCH_SIZE) < 1:
        raise ValueError('NUM_T, NUM_EPOCHS and BATCH_SIZE must be positive.')
    # Uploaded loader and pupil calibration are fixed at 144x256 -> 256x256.
    #if WIDTH != 256:
    #    raise ValueError('The original experimental loader requires WIDTH=256.')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False

    data_dir = Path(DATA_DIR).expanduser()
    if not data_dir.is_absolute():
        data_dir = Path(ROOT_DIR) / data_dir
    branch_tag = MODEL_MODE
    out_dir = Path(ROOT_DIR) / 'vis' / f'{SCENE_NAME}_{branch_tag}'
    final_dir = out_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    config_names = (
        'ROOT_DIR DATA_DIR SCENE_NAME MODEL_MODE STATIC_PHASE DYNAMIC_SCENE '
        'NUM_T WIDTH BATCH_SIZE NUM_EPOCHS PHS_LAYERS IM_PREFIX MAX_INTENSITY '
        'ZERO_FREQ SEED DEVICE IM_LR PH_LR IM_FINAL_LR PH_FINAL_LR VIS_FREQ '
        'SAVE_PER_FRAME SILENCE_TQDM OBJECT_MMES_CONFIG ABERRATION_MMES_CONFIG '
        'OBJECT_AE_WEIGHT ABERRATION_AE_WEIGHT AMPLITUDE_OFFSET'
    ).split()
    config = {key: str(globals()[key]) if isinstance(globals()[key], Path) else globals()[key]
              for key in config_names}
    print(f'Device: {DEVICE}; mode: {MODEL_MODE}; output: {out_dir}')
    print('Loss: measurement MSE + separately weighted object/aberration AE; no TV.')

    dset = BatchDataset(str(data_dir), num=NUM_T, im_prefix=IM_PREFIX,
                        max_intensity=MAX_INTENSITY, zero_freq=ZERO_FREQ,WIDTH=WIDTH,HEIGHT=HEIGHT)
    if len(dset) != NUM_T:
        raise ValueError(f'Loaded {len(dset)} of {NUM_T} frames. Fix missing/invalid files before training.')
    x_all = torch.stack(dset.xs).to(DEVICE)
    y_all = torch.stack(dset.ys).to(DEVICE)
    if x_all.shape != (NUM_T, 1, WIDTH, WIDTH) or y_all.shape != (NUM_T, WIDTH, WIDTH):
        raise ValueError(f'Unexpected data shapes: x={tuple(x_all.shape)}, y={tuple(y_all.shape)}')
    if not torch.isfinite(x_all).all() or not torch.isfinite(y_all).all():
        raise ValueError('Nonfinite data; check MAT values and positive normalization intensity.')

    net = build_model(config)
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    net = net.to(DEVICE)
    net.train()
    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=IM_LR)
    ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=PH_LR)
    im_sche = torch.optim.lr_scheduler.CosineAnnealingLR(im_opt, NUM_EPOCHS, eta_min=IM_FINAL_LR)
    ph_sche = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, NUM_EPOCHS, eta_min=PH_FINAL_LR)
    print('Object trainable parameters:', sum(p.numel() for p in net.g_im.parameters() if p.requires_grad))
    print('Aberration trainable parameters:', sum(p.numel() for p in net.g_g.parameters() if p.requires_grad))

    history = []
    component_history = []
    total_it = 0
    started = time.time()
    # Independent sampler keeps batch order identical between MMES/original runs.
    order_rng = torch.Generator().manual_seed(SEED + 100)
    epoch_bar = tqdm.trange(NUM_EPOCHS, disable=SILENCE_TQDM)
    for epoch in epoch_bar:
        indices = torch.randperm(NUM_T, generator=order_rng).to(DEVICE)
        running_mse = 0.0
        running_obj_ae = 0.0
        running_ab_ae = 0.0
        running_total = 0.0
        for start in range(0, NUM_T, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            t = _time_coordinate(idx)
            im_opt.zero_grad(set_to_none=True)
            ph_opt.zero_grad(set_to_none=True)
            prediction, _, _, _, _ = net(x_all[idx], t)
            prediction = prediction.reshape(len(idx), WIDTH, WIDTH)
            mse = F.mse_loss(prediction, y_all[idx])
            obj_ae = net.g_im.pop_ae_loss() if MODEL_MODE != 'original' else mse.new_zeros(())
            ab_ae = net.g_g.pop_ae_loss() if MODEL_MODE == 'dual_mmes' else mse.new_zeros(())
            loss = mse + OBJECT_AE_WEIGHT*obj_ae + ABERRATION_AE_WEIGHT*ab_ae
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, iteration {total_it}.')
            loss.backward()
            ph_opt.step()
            im_opt.step()
            running_mse += mse.item() * len(idx)
            running_obj_ae += obj_ae.item() * len(idx)
            running_ab_ae += ab_ae.item() * len(idx)
            running_total += loss.item() * len(idx)
            total_it += 1
            if VIS_FREQ > 0 and total_it % VIS_FREQ == 0:
                save_preview(net, x_all[idx], y_all[idx], t, out_dir / f'it_{total_it:06d}.png')
                # Save actual floats as well as the display, without retaining a graph.
                with torch.no_grad():
                    if MODEL_MODE != 'original':
                        obj = net.g_im.forward_with_loss(add_noise=False)[0][0, 0].cpu().numpy()
                    else:
                        obj = net.g_im()[0, 0].cpu().numpy()
                sio.savemat(out_dir / f'object_it_{total_it:06d}.mat', {'object': obj})
        history.append(running_mse / NUM_T)
        component_history.append([running_mse / NUM_T, running_obj_ae / NUM_T, running_ab_ae / NUM_T, running_total / NUM_T])
        epoch_bar.set_postfix(MSE=f'{history[-1]:.4e}', O_AE=f'{running_obj_ae / NUM_T:.3e}', G_AE=f'{running_ab_ae / NUM_T:.3e}')
        im_sche.step()
        ph_sche.step()

    # Deterministic evaluation without patch noise; no output clipping in the model.
    net.eval()
    with torch.no_grad():
        final_mse = 0.0
        final_obj_ae = 0.0
        final_ab_ae = 0.0
        for start in range(0, NUM_T, BATCH_SIZE):
            idx = torch.arange(start, min(start + BATCH_SIZE, NUM_T), device=DEVICE)
            prediction, _, _, _, _ = net(x_all[idx], _time_coordinate(idx))
            final_mse += F.mse_loss(prediction.reshape(len(idx), WIDTH, WIDTH), y_all[idx]).item() * len(idx)
            if MODEL_MODE != 'original':
                final_obj_ae += net.g_im.pop_ae_loss().item() * len(idx)
            if MODEL_MODE == 'dual_mmes':
                final_ab_ae += net.g_g.pop_ae_loss().item() * len(idx)
        final_mse /= NUM_T
        final_obj_ae /= NUM_T
        final_ab_ae /= NUM_T
        final_total = final_mse + OBJECT_AE_WEIGHT*final_obj_ae + ABERRATION_AE_WEIGHT*final_ab_ae
        t0 = torch.tensor([-0.5], device=DEVICE)
        obj, g, phase = net.get_estimates(t0)
        obj_np = obj[0, 0].cpu().numpy()
        sio.savemat(final_dir / 'final_reconstruction.mat', {
            'object': obj_np,
            'g_real_t0': g[0, 0].real.cpu().numpy(),
            'g_imag_t0': g[0, 0].imag.cpu().numpy(),
            'amplitude_parameter_t0': (g[0,0]*torch.exp(-1j*phase[0,0])).real.cpu().numpy(),
            'field_amplitude_t0': g[0, 0].abs().cpu().numpy(),
            'phase_parameter_t0': phase[0, 0].cpu().numpy(),
            'complex_phase_t0': torch.angle(g[0, 0]).cpu().numpy(),
            'pupil': dset.a_slm.numpy(),
            'measurement_scale': float(dset.max_intensity),
            'final_mse': final_mse,
            'final_object_ae_mse': final_obj_ae,
            'final_aberration_ae_mse': final_ab_ae,
            'final_total': final_total,
            'object_ae_weight': OBJECT_AE_WEIGHT,
            'aberration_ae_weight': ABERRATION_AE_WEIGHT,
        })
        if not STATIC_PHASE and SAVE_PER_FRAME:
            phase_dir = final_dir / 'per_frame'
            phase_dir.mkdir(exist_ok=True)
            for frame in range(NUM_T):
                _, g, phase = net.get_estimates(_time_coordinate(torch.tensor([frame], device=DEVICE)))
                sio.savemat(phase_dir / f'phase_{frame:04d}.mat', {
                    'g_real': g[0, 0].real.cpu().numpy(),
                    'g_imag': g[0, 0].imag.cpu().numpy(),
                    'amplitude_parameter': (g[0,0]*torch.exp(-1j*phase[0,0])).real.cpu().numpy(),
                    'field_amplitude': g[0, 0].abs().cpu().numpy(),
                    'phase_parameter': phase[0, 0].cpu().numpy(),
                    'complex_phase': torch.angle(g[0, 0]).cpu().numpy(),
                })
    imageio.imwrite(final_dir / 'final_I_est.png', (np.clip(obj_np, 0, 1) * 255).round().astype(np.uint8))
    np.savetxt(out_dir / 'loss.csv', np.column_stack((np.arange(1, NUM_EPOCHS + 1), np.asarray(component_history))),
               delimiter=',', header='epoch,measurement_mse,object_ae_mse,aberration_ae_mse,total', comments='')
    save_preview(net, x_all, y_all, torch.tensor([-0.5], device=DEVICE), final_dir / 'summary.png')
    torch.save({
        'model_state_dict': net.state_dict(),  # Includes trainable g_im.z and fixed embedding filters for MMES.
        'object_optimizer': im_opt.state_dict(), 'phase_optimizer': ph_opt.state_dict(),
        'object_scheduler': im_sche.state_dict(), 'phase_scheduler': ph_sche.state_dict(),
        'config': config, 'epochs_completed': NUM_EPOCHS,
        'loss_history': history, 'loss_components_history': component_history,
        'final_mse': final_mse, 'final_object_ae_mse': final_obj_ae,
        'final_aberration_ae_mse': final_ab_ae, 'final_total': final_total,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'measurement_scale': float(dset.max_intensity),
        'batch_order_rng_state': order_rng.get_state(),
    }, final_dir / 'model_final.pt')
    print(f'Finished in {time.time() - started:.1f}s. Final measurement MSE: {final_mse:.6g}')
    print(f'Object range [{obj_np.min():.6g}, {obj_np.max():.6g}]; MAT stores unclipped values.')
    print(f'Results: {final_dir}')
    return net


if __name__ == '__main__':
    main()
