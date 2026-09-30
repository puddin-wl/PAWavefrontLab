"""PyTorch MMES object + unchanged NeuWS amplitude/phase aberration MLP.

H/Hinv reproduce the user's TensorFlow one-hot filters, REFLECT padding,
transpose convolution, crop, and tau**2 scaling. All tensors here are NCHW.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from networks import StaticDiffuseNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class PatchEmbedding(nn.Module):
    def __init__(self, tau):
        super().__init__()
        if type(tau) is not int or tau < 1:
            raise ValueError('tau must be a positive integer.')
        self.tau = tau
        # Channel ti*tau+tj extracts the pixel at offset (ti,tj), as in TF.
        self.register_buffer('filters', torch.eye(tau*tau).reshape(tau*tau, 1, tau, tau))

    def forward(self, z):
        if z.ndim != 4 or z.shape[1] != 1:
            raise ValueError('Expected single-channel NCHW image [B,1,H,W].')
        p = self.tau - 1
        if min(z.shape[-2:]) <= p:
            raise ValueError('REFLECT padding requires image dimensions >= tau.')
        padded = F.pad(z, (p,p,p,p), mode='reflect') if p else z
        return F.conv2d(padded, self.filters)

    def inverse(self, patches):
        if patches.ndim != 4 or patches.shape[1] != self.tau**2:
            raise ValueError('Expected patches [B,tau**2,H+tau-1,W+tau-1].')
        p = self.tau - 1
        hh, ww = patches.shape[-2:]
        if min(hh, ww) <= p:
            raise ValueError('Embedded spatial dimensions are too small.')
        accumulated = F.conv_transpose2d(patches, self.filters)
        return accumulated[..., p:hh, p:ww] / (self.tau**2)


class MMESObject(nn.Module):
    """Trainable image Z -> shared patch MLP autoencoder -> reconstructed object.

    Three hidden layers use LeakyReLU(0.2), matching tf.nn.leaky_relu's
    default. The last layer is LINEAR, with no clipping or positivity transform.
    Both Z and AE weights are optimized. Optional channels are jointly decoded;
    optional frames use separate Z slices and one shared AE. One noise draw is shared by data and
    AE losses in a forward call. eval() disables noise. AE target is H(Z),
    without detach, matching joint optimization of Z and AE in the paper.
    """
    def __init__(self, width, tau=4, ranks=None, noise_std=0.01,
                 seed=0, init_scale=0.1, chunk_size=4096,
                 checkpoint_chunks=True, channels=1, num_frames=1):
        super().__init__()
        if type(width) is not int or width < tau:
            raise ValueError('width must be an integer >= tau.')
        self.embedding = PatchEmbedding(tau)
        if type(channels) is not int or channels < 1 or type(num_frames) is not int or num_frames < 1:
            raise ValueError('channels and num_frames must be positive integers.')
        self.channels, self.num_frames = channels, num_frames
        d = channels*tau*tau
        ranks = [32*d, d, 32*d] if ranks is None else list(ranks)
        if len(ranks) != 3 or any(type(n) is not int or n < 1 for n in ranks):
            raise ValueError('ranks must contain exactly three positive integers.')
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError('noise_std must be finite and >= 0.')
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError('init_scale must be finite and > 0.')
        if type(chunk_size) is not int or chunk_size < 0:
            raise ValueError('chunk_size must be a nonnegative integer; 0 disables chunking.')
        self.width, self.noise_std = width, noise_std
        self.chunk_size, self.checkpoint_chunks = chunk_size, checkpoint_chunks
        self.last_ae_loss = None
        # Local RNG preserves original aberration initialization and global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.z = nn.Parameter(torch.rand(num_frames, channels, width, width) * init_scale)
            dims = [d, *ranks, d]
            layers = []
            for index in range(4):
                linear = nn.Linear(dims[index], dims[index+1])
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                if index < 3:
                    layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.autoencoder = nn.Sequential(*layers)

    def decode_vectors(self, vectors):
        size = self.chunk_size or len(vectors)
        outputs = []
        for part in vectors.split(size):
            if self.checkpoint_chunks and torch.is_grad_enabled() and self.training:
                out = checkpoint(self.autoencoder, part, use_reentrant=False)
            else:
                out = self.autoencoder(part)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def forward_with_loss(self, add_noise=None, frame_indices=None):
        if add_noise is None:
            add_noise = self.training
        if self.num_frames == 1:
            if frame_indices is not None:
                raise ValueError('Single-frame fields do not take frame_indices.')
            z = self.z
        else:
            if frame_indices is None or frame_indices.ndim != 1 or frame_indices.numel() == 0:
                raise ValueError('Time-varying MMES requires nonempty 1D frame indices.')
            z = self.z.index_select(0, frame_indices.to(device=self.z.device, dtype=torch.long))
        b, c, height, width = z.shape
        # Preserve row-major offsets within each channel, then concatenate channels.
        embedded = self.embedding(z.reshape(b*c,1,height,width))
        _, dd, hh, ww = embedded.shape
        clean = embedded.reshape(b,c*dd,hh,ww)
        noisy = clean + self.noise_std * torch.randn_like(clean) if add_noise and self.noise_std else clean
        d = c*dd
        vectors = noisy.permute(0,2,3,1).reshape(-1,d)
        decoded = self.decode_vectors(vectors).reshape(b,hh,ww,d).permute(0,3,1,2).contiguous()
        ae_loss = F.mse_loss(decoded, clean)
        field = self.embedding.inverse(decoded.reshape(b*c,dd,hh,ww))
        return field.reshape(b,c,height,width), ae_loss

    def forward(self, frame_indices=None):
        obj, self.last_ae_loss = self.forward_with_loss(frame_indices=frame_indices)
        return obj

    def pop_ae_loss(self):
        if self.last_ae_loss is None:
            raise RuntimeError('Call the optical forward before collecting its AE loss.')
        loss = self.last_ae_loss
        self.last_ae_loss = None
        return loss


class StaticDiffuseMMES(StaticDiffuseNet):
    """Replace only g_im. Original get_estimates and optical forward are inherited."""
    def __init__(self, width, PSF_size, phs_layers=4, use_FFT=True, bsize=8,
                 use_pe=False, static_phase=True, mmes_kwargs=None):
        super().__init__(width, PSF_size, phs_layers=phs_layers, use_FFT=use_FFT,
                         bsize=bsize, use_pe=use_pe, static_phase=static_phase)
        self.g_im = MMESObject(width=width, **dict(mmes_kwargs or {}))

    def forward(self, x_batch, t):
        y, kernel, g, phase, obj = super().forward(x_batch, t)
        return y.reshape(x_batch.shape[0], *obj.shape[-2:]), kernel, g, phase, obj


def model_from_checkpoint(saved, device='cpu'):
    c = saved['config']
    args = dict(width=c['WIDTH'], PSF_size=c['WIDTH'], phs_layers=c['PHS_LAYERS'],
                bsize=c['BATCH_SIZE'], static_phase=c['STATIC_PHASE'], use_FFT=True)
    if c['OBJECT_BRANCH'] == 'mmes':
        net = StaticDiffuseMMES(**args, mmes_kwargs=c['MMES_CONFIG'])
    elif c['OBJECT_BRANCH'] == 'original':
        net = StaticDiffuseNet(**args)
    else:
        raise ValueError('Unknown OBJECT_BRANCH in checkpoint.')
    net.load_state_dict(saved['model_state_dict'])
    return net.to(device).eval()
