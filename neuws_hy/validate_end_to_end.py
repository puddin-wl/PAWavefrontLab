"""Synthetic CPU integration check, not a reconstruction quality benchmark.

Run: python validate_end_to_end.py
Creates temporary MAT data and removes the temporary outputs afterwards.
"""
from pathlib import Path
import tempfile
import numpy as np
import scipy.io as sio
import torch
import recon_dual_mmes as runner
from networks_DualMMES import model_from_checkpoint
from utils import fft_2xPad_Conv2D


def run():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    with tempfile.TemporaryDirectory(prefix='neuws_object_mlp_') as tmp:
        scratch = Path(tmp)
        data = scratch / 'data'
        data.mkdir()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing='ij')
        obj = (0.8 * torch.exp(-8 * (xx**2 + yy**2)))[None, None]
        g = (1 + 0.1 * torch.sin(3 * xx)) * torch.exp(1j * (0.2 * torch.cos(6 * xx) + 0.3 * torch.sin(4 * yy)))
        pupil = torch.zeros(256, 256)
        pupil[56:200] = 1
        for index in range(3):
            phase = (index + 1) * (0.4 * torch.sin(6 * xx) + 0.3 * torch.cos(5 * yy))
            x = pupil * torch.exp(-1j * phase)
            psf = torch.fft.fftshift(torch.fft.fft2(x * g, norm='forward')).abs().square()
            psf /= psf.sum()
            measurement = fft_2xPad_Conv2D(obj, psf.flip(0, 1)[None, None])[0, 0]
            sio.savemat(data / f'SLM_sim{index+1}.mat', {'proj_sim': phase[56:200].numpy()})
            sio.savemat(data / f'SLM_raw{index+1}.mat', {'imsdata': measurement.numpy()})
        runner.ROOT_DIR = scratch
        runner.DATA_DIR = str(data)
        runner.NUM_T = 3
        runner.NUM_EPOCHS = 1
        runner.BATCH_SIZE = 2
        runner.DEVICE = 'cpu'
        runner.VIS_FREQ = 2
        runner.SILENCE_TQDM = True
        runner.SCENE_NAME = 'synthetic'
        for mode,static in (('dual_mmes',True),('dual_mmes',False),('object_mmes',True),('original',True)):
            runner.MODEL_MODE = mode
            runner.STATIC_PHASE = static
            runner.SCENE_NAME = f'synthetic_{mode}_{static}'
            model = runner.main()
            final = scratch/'vis'/f'{runner.SCENE_NAME}_{mode}'/'final'
            result = sio.loadmat(final/'final_reconstruction.mat')
            assert result['object'].shape == (256,256)
            assert np.isfinite(result['object']).all()
            complex_field = result['g_real_t0']+1j*result['g_imag_t0']
            assert np.allclose(result['field_amplitude_t0'],np.abs(complex_field),atol=1e-6)
            assert np.allclose(result['amplitude_parameter_t0']*np.exp(1j*result['phase_parameter_t0']),complex_field,atol=1e-6)
            assert (final/'summary.png').is_file() and (final/'final_I_est.png').is_file()
            assert (final.parent/'it_000002.png').is_file()
            saved = torch.load(final/'model_final.pt',map_location='cpu',weights_only=True)
            state = saved['model_state_dict']
            assert ('g_im.z' in state)==(mode!='original')
            assert ('g_g.z' in state)==(mode=='dual_mmes')
            assert saved['config']['MODEL_MODE']==mode
            assert saved['object_optimizer']['state'] and saved['phase_optimizer']['state']
            expected = saved['final_mse']+runner.OBJECT_AE_WEIGHT*saved['final_object_ae_mse']+runner.ABERRATION_AE_WEIGHT*saved['final_aberration_ae_mse']
            assert np.isclose(expected,saved['final_total'])
            assert np.isclose(result['final_total'].item(),expected)
            log = np.genfromtxt(final.parent/'loss.csv',delimiter=',',names=True)
            assert set(log.dtype.names)=={'epoch','measurement_mse','object_ae_mse','aberration_ae_mse','total'}
            assert np.allclose(log['total'],log['measurement_mse']+runner.OBJECT_AE_WEIGHT*log['object_ae_mse']+runner.ABERRATION_AE_WEIGHT*log['aberration_ae_mse'])
            restored = model_from_checkpoint(saved)
            with torch.no_grad():
                for frame in range(3):
                    t = torch.tensor([frame/2-0.5])
                    ref = model.get_estimates(t)
                    for before,after in zip(ref,restored.get_estimates(t)):
                        assert torch.allclose(before,after,atol=1e-6)
                    if not static:
                        perframe = sio.loadmat(final/'per_frame'/f'phase_{frame:04d}.mat')
                        assert np.allclose(perframe['g_real'],ref[1][0,0].real.numpy(),atol=1e-6)
                if mode=='dual_mmes':
                    _,obj_ae = model.g_im.forward_with_loss(add_noise=False)
                    _,ab_ae = model.g_g.forward_with_loss(add_noise=False,frame_indices=None if static else torch.arange(3))
                    assert np.isclose(saved['final_object_ae_mse'],obj_ae.item(),atol=1e-7)
                    assert np.isclose(saved['final_aberration_ae_mse'],ab_ae.item(),atol=1e-7)
            print(f'PASS {mode}, static={static}: full-size training, batch 2+1, logs/MAT/PNG, final AE averaging and reload')
            del model,restored,saved,state


if __name__=='__main__':
    run()
