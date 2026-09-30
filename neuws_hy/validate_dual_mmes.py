"""Dual-MMES gradient/physics and per-frame latent indexing checks (CPU)."""
import io
import torch
from networks_MMES import MMESObject
from networks_DualMMES import DualMMES, model_from_checkpoint
from utils import fft_2xPad_Conv2D


def run():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    opts=dict(tau=2,ranks=[16,4,16],noise_std=0.01,chunk_size=127)
    field=MMESObject(16,channels=2,num_frames=3,**opts)
    field.eval()
    ids=torch.tensor([2,0])
    actual,ae=field.forward_with_loss(frame_indices=ids)
    z=field.z.index_select(0,ids)
    # Channel concatenation explicitly checked against separate single-channel H.
    clean=torch.cat([field.embedding(z[:,c:c+1]) for c in range(2)],dim=1)
    decoded=field.autoencoder(clean.permute(0,2,3,1).reshape(-1,8))
    decoded=decoded.reshape(2,17,17,8).permute(0,3,1,2).contiguous()
    expected=torch.cat([field.embedding.inverse(decoded[:,c*4:(c+1)*4]) for c in range(2)],dim=1)
    assert torch.allclose(actual,expected,atol=1e-7)
    assert torch.allclose(ae,(decoded-clean).square().mean(),atol=1e-8)
    ae.backward()
    assert torch.count_nonzero(field.z.grad[0]) and torch.count_nonzero(field.z.grad[2])
    assert torch.count_nonzero(field.z.grad[1])==0
    print('PASS multichannel patch ordering, overlap reconstruction, selected latent frames and AE target')

    for static in (True,False):
        model=DualMMES(32,32,bsize=2,num_frames=3,static_phase=static,
                       object_kwargs=opts,aberration_kwargs=dict(opts,seed=1))
        assert not hasattr(model,'basis') and not hasattr(model,'t_grid')
        assert not ({id(p) for p in model.g_im.parameters()} & {id(p) for p in model.g_g.parameters()})
        model.eval()
        x=torch.exp(-1j*torch.randn(2,1,32,32))
        x[:,:,:4]=0
        t=torch.tensor([0.5,-0.5])
        y,k,g,phase,obj=model(x,t)
        fields=model.g_g(None if static else torch.tensor([2,0]))
        if static:fields=fields.expand(2,-1,-1,-1)
        assert torch.allclose(g,(1+fields[:,0:1])*torch.exp(1j*fields[:,1:2]))
        kernel=torch.fft.fftshift(torch.fft.fft2(g*x,norm='forward'),dim=(-2,-1)).abs().square()
        kernel=kernel/kernel.sum((-2,-1),keepdim=True)
        kernel=kernel.flip(2).flip(3)
        expected_y=fft_2xPad_Conv2D(obj,kernel).reshape(2,32,32)
        assert torch.allclose(y,expected_y,atol=1e-7)
        assert torch.allclose(k.sum((-2,-1)),torch.ones(2,1),atol=1e-6)
        # Data loss alone must reach BOTH field channels, not just AE regularization.
        target=torch.rand_like(y)
        grads=torch.autograd.grad((y-target).square().mean(),[model.g_im.z,model.g_g.z])
        assert all(torch.isfinite(a).all() and torch.count_nonzero(a) for a in grads)
        assert all(torch.count_nonzero(grads[1][:,c]) for c in range(2))
        if not static:
            single=model.get_estimates(torch.tensor([0.5]))[1]
            assert torch.allclose(single,g[:1],atol=1e-7)
            try:model.frame_indices(torch.tensor([0.1]))
            except ValueError:pass
            else:raise AssertionError('Off-grid time must not silently select another frame')
        opt_im=torch.optim.Adam(model.g_im.parameters(),lr=1e-3)
        opt_g=torch.optim.Adam(model.g_g.parameters(),lr=1e-3)
        old_im=model.g_im.z.detach().clone()
        old_g=model.g_g.z.detach().clone()
        model.train()
        for batch in (2,1):
            opt_im.zero_grad(set_to_none=True)
            opt_g.zero_grad(set_to_none=True)
            pred,*_=model(x[:batch],t[:batch])
            loss=(pred-target[:batch]).square().mean()+model.g_im.pop_ae_loss()+model.g_g.pop_ae_loss()
            assert torch.isfinite(loss)
            loss.backward()
            for branch in (model.g_im,model.g_g):
                for parameter in branch.parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            opt_im.step();opt_g.step()
        assert not torch.equal(model.g_im.z,old_im)
        assert not torch.equal(model.g_g.z,old_g)
        config=dict(MODEL_MODE='dual_mmes',WIDTH=32,NUM_T=3,BATCH_SIZE=2,PHS_LAYERS=4,
                    STATIC_PHASE=static,OBJECT_MMES_CONFIG=opts,
                    ABERRATION_MMES_CONFIG=dict(opts,seed=1),AMPLITUDE_OFFSET=1.0)
        buf=io.BytesIO()
        torch.save(dict(config=config,model_state_dict=model.state_dict()),buf)
        buf.seek(0)
        restored=model_from_checkpoint(torch.load(buf,weights_only=True))
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(x,t)[0],model(x,t)[0])
            for a,b in zip(model(x,t),restored(x,t)):
                assert torch.allclose(a,b,atol=1e-6)
        print(f'PASS static={static}: independent branches, original optics, data-only gradients, updates, batch=1 and reload')


if __name__=='__main__':
    run()
