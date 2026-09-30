"""Independent MMES object and pupil-field branches with original NeuWS optics."""
import math
import torch
from networks import StaticDiffuseNet
from networks_MMES import MMESObject, StaticDiffuseMMES


class DualMMES(StaticDiffuseNet):
    """Static object; static or per-measurement amplitude/phase aberration.

    g_im has a single-channel Z and its own AE.
    g_g has a two-channel Z and a separate AE: [amplitude residual, phase].
    For dynamic aberrations Z has one slice per acquired frame; AE weights
    are shared across frames. No temporal interpolation/prior is assumed.
    """
    def __init__(self, width, PSF_size, bsize=8, static_phase=True, num_frames=100,
                 object_kwargs=None, aberration_kwargs=None, amplitude_offset=1.0,
                 phs_layers=4, use_FFT=True):
        if width != PSF_size:
            raise ValueError('This integration requires PSF_size == width.')
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError('num_frames must be a positive integer.')
        if not math.isfinite(amplitude_offset):
            raise ValueError('amplitude_offset must be finite.')
        super().__init__(width, PSF_size, phs_layers=phs_layers, use_FFT=use_FFT,
                         bsize=bsize, static_phase=static_phase)
        obj = dict(object_kwargs or {})
        ab = dict(aberration_kwargs or {})
        if any(key in opts for opts in (obj,ab) for key in ('channels','num_frames')):
            raise ValueError('channels/num_frames are managed by DualMMES, not branch configs.')
        obj.setdefault('seed',0)
        ab.setdefault('seed',1)
        self.g_im = MMESObject(width, channels=1, num_frames=1, **obj)
        self.g_g = MMESObject(width, channels=2, num_frames=1 if static_phase else num_frames, **ab)
        self.num_frames = num_frames
        self.amplitude_offset = float(amplitude_offset)
        # The original coordinate features are no longer used in this model.
        del self.basis
        del self.t_grid

    def frame_indices(self, t):
        if t.ndim != 1 or t.numel() < 1 or not torch.isfinite(t).all():
            raise ValueError('t must be a nonempty finite 1D vector.')
        if self.num_frames == 1:
            if not torch.allclose(t, torch.full_like(t,-0.5), atol=1e-6, rtol=0):
                raise ValueError('A single-frame dynamic model expects t=-0.5.')
            return None
        positions = (t+0.5)*(self.num_frames-1)
        indices = positions.round().long()
        if ((indices<0)|(indices>=self.num_frames)).any() or not torch.allclose(
                positions,indices.to(positions.dtype),atol=1e-4,rtol=0):
            raise ValueError('Dynamic MMES only supports the discrete acquired frame times.')
        return indices

    def get_estimates(self, t):
        if t.ndim != 1 or t.numel() < 1:
            raise ValueError('t must be a nonempty 1D tensor.')
        obj = self.g_im()
        if self.static_phase:
            fields = self.g_g().expand(t.numel(),-1,-1,-1)
        else:
            fields = self.g_g(frame_indices=self.frame_indices(t))
        amplitude = self.amplitude_offset + fields[:,0:1]
        phase = fields[:,1:2]  # Radians, no modulo/sigmoid/tanh in the forward.
        g = amplitude * torch.exp(1j*phase)
        return obj, g, phase

    def forward(self, x_batch, t):
        if x_batch.shape[0] != t.numel():
            raise ValueError('SLM batch and t must have equal lengths.')
        prediction, kernel, g, phase, obj = super().forward(x_batch,t)
        return prediction.reshape(x_batch.shape[0],*obj.shape[-2:]),kernel,g,phase,obj


def build_model(config):
    mode = config['MODEL_MODE']
    shared = dict(width=config['WIDTH'], PSF_size=config['WIDTH'],
                  phs_layers=config['PHS_LAYERS'], use_FFT=True,
                  bsize=config['BATCH_SIZE'], static_phase=config['STATIC_PHASE'])
    if mode == 'dual_mmes':
        return DualMMES(**shared, num_frames=config['NUM_T'],
                        object_kwargs=config['OBJECT_MMES_CONFIG'],
                        aberration_kwargs=config['ABERRATION_MMES_CONFIG'],
                        amplitude_offset=config['AMPLITUDE_OFFSET'])
    if mode == 'object_mmes':
        return StaticDiffuseMMES(**shared, mmes_kwargs=config['OBJECT_MMES_CONFIG'])
    if mode == 'original':
        return StaticDiffuseNet(**shared)
    raise ValueError('MODEL_MODE must be dual_mmes, object_mmes, or original.')


def model_from_checkpoint(saved, device='cpu'):
    model = build_model(saved['config'])
    model.load_state_dict(saved['model_state_dict'])
    return model.to(device).eval()
