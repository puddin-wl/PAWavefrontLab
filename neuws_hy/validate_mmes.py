"""Structural/numerical tests against an independent NumPy patch reference.

These verify the supplied TF algorithm's math, not a TensorFlow runtime run.
"""
import copy
import numpy as np
import torch
from networks import StaticDiffuseNet
from networks_MMES import PatchEmbedding, MMESObject, StaticDiffuseMMES


def run():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    for tau in (1,2,4):
        x = torch.randn(2,1,8,9,dtype=torch.float64,requires_grad=True)
        emb = PatchEmbedding(tau).double()
        p = tau-1
        source = np.pad(x.detach().numpy(), ((0,0),(0,0),(p,p),(p,p)), mode='reflect')
        hh, ww = 8+p, 9+p
        expected = np.stack([source[:,0,i:i+hh,j:j+ww] for i in range(tau) for j in range(tau)], axis=1)
        actual = emb(x)
        assert np.allclose(actual.detach().numpy(),expected,atol=1e-12)
        assert torch.allclose(emb.inverse(actual),x,atol=1e-12)
        # Arbitrary inconsistent patches also use the exact TF crop and overlap rule.
        q = torch.randn_like(actual)
        ref = np.zeros((2,1,hh+p,ww+p))
        for i in range(tau):
            for j in range(tau):
                ref[:,0,i:i+hh,j:j+ww] += q[:,i*tau+j].numpy()
        ref = ref[:,:,p:hh,p:ww] / (tau*tau)
        assert np.allclose(emb.inverse(q).numpy(),ref,atol=1e-12)
        grad = torch.autograd.grad(emb.inverse(actual).sum(),x)[0]
        assert torch.allclose(grad,torch.ones_like(x),atol=1e-12)
        assert not list(emb.parameters())
    print('PASS H/Hinv: row-major patch ordering, REFLECT edges, overlap, tau=1/2/4, identity gradient')

    opts=dict(tau=4,ranks=[32,8,32],noise_std=0.02,chunk_size=128)
    obj=MMESObject(16,**opts)
    clean=obj.embedding(obj.z)
    obj.eval()
    output,ae=obj.forward_with_loss()
    ref=obj.autoencoder(clean.permute(0,2,3,1).reshape(-1,16))
    ref=ref.reshape(1,19,19,16).permute(0,3,1,2).contiguous()
    assert torch.allclose(output,obj.embedding.inverse(ref),atol=1e-7)
    assert torch.allclose(ae,(ref-clean).square().mean(),atol=1e-8)
    assert isinstance(obj.autoencoder[-1],torch.nn.Linear)
    assert all(m.negative_slope==0.2 for m in obj.modules() if isinstance(m,torch.nn.LeakyReLU))
    # Inject identical noise to confirm SAME noisy decode feeds both loss terms.
    obj.train()
    torch.manual_seed(21)
    out,ae=obj.forward_with_loss()
    torch.manual_seed(21)
    noisy=clean+0.02*torch.randn_like(clean)
    decoded=obj.autoencoder(noisy.permute(0,2,3,1).reshape(-1,16))
    decoded=decoded.reshape(1,19,19,16).permute(0,3,1,2).contiguous()
    assert torch.allclose(out,obj.embedding.inverse(decoded),atol=1e-7)
    assert torch.allclose(ae,(decoded-clean).square().mean(),atol=1e-8)
    (out.square().mean()+ae).backward()
    assert obj.z.grad is not None and torch.count_nonzero(obj.z.grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.autoencoder.parameters())
    obj.eval()
    assert torch.equal(obj(),obj())
    print('PASS AE: same noisy decode, attached clean target, chunk/checkpoint path, trainable Z, deterministic eval')

    kwargs=dict(width=64,PSF_size=64,phs_layers=4,bsize=2,static_phase=False)
    torch.manual_seed(12)
    baseline=StaticDiffuseNet(**kwargs)
    torch.manual_seed(12)
    model=StaticDiffuseMMES(**kwargs,mmes_kwargs=opts)
    assert all(torch.equal(a,b) for a,b in zip(baseline.g_g.parameters(),model.g_g.parameters()))
    baseline.g_im=copy.deepcopy(model.g_im)
    baseline.eval();model.eval()
    x=torch.exp(1j*torch.randn(2,1,64,64))
    t=torch.tensor([-0.5,0.5])
    outputs=model(x,t); refs=baseline(x,t)
    for a,b in zip(outputs,refs):
        assert torch.allclose(a,b,atol=1e-6)
    outputs[0].square().mean().backward()
    refs[0].square().mean().backward()
    for a,b in zip(model.g_g.parameters(),baseline.g_g.parameters()):
        assert torch.allclose(a.grad,b.grad,atol=1e-7)
    model.g_im.pop_ae_loss()
    model.train()
    opt_im=torch.optim.Adam(model.g_im.parameters(),lr=1e-3)
    opt_g=torch.optim.Adam(model.g_g.parameters(),lr=1e-3)
    old_z=model.g_im.z.detach().clone()
    old_g=next(model.g_g.parameters()).detach().clone()
    for batch in (2,1):
        opt_im.zero_grad(set_to_none=True);opt_g.zero_grad(set_to_none=True)
        y,k,*_=model(x[:batch],t[:batch])
        assert y.shape==(batch,64,64)
        assert torch.allclose(k.sum((-2,-1)),torch.ones(batch,1),atol=1e-5)
        loss=(y-torch.rand_like(y)).square().mean()+model.g_im.pop_ae_loss()
        loss.backward()
        for p in [*model.g_im.parameters(),*model.g_g.parameters()]:
            assert p.grad is not None and torch.isfinite(p.grad).all()
        opt_im.step();opt_g.step()
    assert not torch.equal(old_z,model.g_im.z)
    assert not torch.equal(old_g,next(model.g_g.parameters()))
    print('PASS original aberration weights, forward AND gradients; normalized PSF; joint updates batches 2+1')


if __name__=='__main__':
    run()
