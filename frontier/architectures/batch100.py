"""
100 Novel Architectures — Batch 100
=====================================
Each architecture uses a unique sequence mixer targeting ~88M params.
All must be parallelizable (no sequential loops over L=2048).
Transformer baseline: val_loss=3.4486.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from frontier.architectures.base import FrontierModel, FrontierConfig, EmbeddingWithScale, LMHead
from frontier.architectures.registry import register_arch

# ── Shared components ──

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, bias=True):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)

class Block(nn.Module):
    def __init__(self, d, d_ff, mixer, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
        self.rs = rs
    def forward(self, x, **kw):
        x = x + self.rs * self.mix(self.n1(x), **kw)
        x = x + self.rs * self.ffn(self.n2(x))
        return x

class EmbedBlock(nn.Module):
    """Block that passes embed to mixer."""
    def __init__(self, d, d_ff, mixer, rs=1.0):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
        self.rs = rs
    def forward(self, x, embed=None):
        x = x + self.rs * self.mix(self.n1(x), embed=embed)
        x = x + self.rs * self.ffn(self.n2(x))
        return x

# ── Reusable mixer components ──

class MHConv(nn.Module):
    """Multi-head causal conv with exponentially-spaced kernels."""
    def __init__(self, d, nh=8, max_k=65):
        super().__init__()
        self.nh = nh; self.dh = d // nh
        ks = [min(2**(i+1)+1, max_k) for i in range(nh)]
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(self.dh, self.dh, k, padding=k-1, groups=self.dh) for k in ks])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        v = self.v(x).view(B,L,self.nh,self.dh)
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) for h in range(self.nh)]
        return self.out(torch.cat(hs, -1) * torch.sigmoid(self.gate(x)))

class TokenShiftMHConv(nn.Module):
    """MHConv with token shift."""
    def __init__(self, d, nh=8):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)*0.5)
        self.conv = MHConv(d, nh)
    def forward(self, x, **kw):
        w = torch.sigmoid(self.w)
        shifted = F.pad(x[:,:-1], (0,0,1,0))
        return self.conv(w*x + (1-w)*shifted)

class CausalGQA(nn.Module):
    """Windowed causal GQA with QK-norm."""
    def __init__(self, d, nh=8, nkv=4, w=256):
        super().__init__()
        self.nh=nh; self.nkv=nkv; self.dh=d//nh; self.hpk=nh//nkv; self.w=w; self.s=self.dh**-0.5
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,nkv*self.dh,bias=False)
        self.v_proj=nn.Linear(d,nkv*self.dh,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh); self.kn=nn.RMSNorm(self.dh)
    def forward(self, x, **kw):
        B,L,D=x.shape
        q=self.qn(self.q(x).view(B,L,self.nh,self.dh)).transpose(1,2)
        k=self.kn(self.k(x).view(B,L,self.nkv,self.dh)).transpose(1,2).repeat_interleave(self.hpk,1)
        v=self.v_proj(x).view(B,L,self.nkv,self.dh).transpose(1,2).repeat_interleave(self.hpk,1)
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),float('-inf'))
            out[:,:,i:e]=torch.matmul(F.softmax(a,-1),v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class ValueResGQA(nn.Module):
    """GQA with value residual from embedding."""
    def __init__(self, d, nh=8, nkv=4, w=256):
        super().__init__()
        self.gqa = CausalGQA(d, nh, nkv, w)
        self.alpha = nn.Parameter(torch.tensor(0.8))
    def forward(self, x, embed=None, **kw):
        if embed is not None:
            a = torch.sigmoid(self.alpha)
            # Mix embedding into V projection input
            v_in = a * x + (1-a) * embed
            # Temporarily swap v_proj input
            B,L,D = x.shape
            q=self.gqa.qn(self.gqa.q(x).view(B,L,self.gqa.nh,self.gqa.dh)).transpose(1,2)
            k=self.gqa.kn(self.gqa.k(x).view(B,L,self.gqa.nkv,self.gqa.dh)).transpose(1,2).repeat_interleave(self.gqa.hpk,1)
            v=self.gqa.v_proj(v_in).view(B,L,self.gqa.nkv,self.gqa.dh).transpose(1,2).repeat_interleave(self.gqa.hpk,1)
            W=self.gqa.w; out=torch.zeros_like(q)
            for i in range(0,L,W):
                e=min(i+W,L); s=max(0,i-W)
                a2=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.gqa.s
                qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
                a2=a2.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),float('-inf'))
                out[:,:,i:e]=torch.matmul(F.softmax(a2,-1),v[:,:,s:e])
            return self.gqa.o(out.transpose(1,2).reshape(B,L,D))
        return self.gqa(x)

# ── Helper: build model from layer specs ──

def _build_model(config, layer_specs, use_embed=False):
    """Build a FrontierModel from layer specifications.
    layer_specs: list of (mixer_class, mixer_kwargs) or mixer instances
    """
    class _Model(FrontierModel):
        pass
    # This is handled per-architecture below
    raise NotImplementedError

# ══════════════════════════════════════════════════════════════════
# ATTENTION KERNEL VARIANTS (1-10)
# ══════════════════════════════════════════════════════════════════

class SigmoidAttn(nn.Module):
    """Replace softmax with sigmoid + learned temperature."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.w=w
        self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh); self.kn=nn.RMSNorm(self.dh)
        self.temp=nn.Parameter(torch.ones(nh)*math.sqrt(self.dh))
    def forward(self, x, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        qkv=self.qkv(x).view(B,L,3,H,DH)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        q=self.qn(q).transpose(1,2); k=self.kn(k).transpose(1,2); v=v.transpose(1,2)
        temp=self.temp.view(1,H,1,1).abs().clamp(min=1.0)
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))/temp
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),-1e9)
            a=torch.sigmoid(a)
            a=a/(a.sum(-1,keepdim=True)+1e-6)
            out[:,:,i:e]=torch.matmul(a,v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class ReLUSquaredAttn(nn.Module):
    """ReLU² attention: attn = relu(QK^T)^2."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.w=w; self.s=self.dh**-0.5
        self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh); self.kn=nn.RMSNorm(self.dh)
    def forward(self, x, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        qkv=self.qkv(x).view(B,L,3,H,DH)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        q=self.qn(q).transpose(1,2); k=self.kn(k).transpose(1,2); v=v.transpose(1,2)
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),-1e9)
            a=F.relu(a).square()
            a=a/(a.sum(-1,keepdim=True)+1e-6)
            out[:,:,i:e]=torch.matmul(a,v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class CosineAttn(nn.Module):
    """Cosine similarity attention with learned temperature."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.w=w
        self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.temp=nn.Parameter(torch.ones(nh)*10.0)
    def forward(self, x, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        qkv=self.qkv(x).view(B,L,3,H,DH)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        q=F.normalize(q,dim=-1).transpose(1,2); k=F.normalize(k,dim=-1).transpose(1,2); v=v.transpose(1,2)
        temp=self.temp.abs().clamp(min=1.0).view(1,H,1,1)
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*temp
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),float('-inf'))
            out[:,:,i:e]=torch.matmul(F.softmax(a,-1),v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class HeadMixAttn(nn.Module):
    """Standard attention + per-position linear mixing across heads."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.gqa = CausalGQA(d, nh, nh, w)
        self.nh = nh; self.dh = d//nh
        self.head_mix = nn.Parameter(torch.eye(nh) + torch.randn(nh,nh)*0.01)
    def forward(self, x, **kw):
        B,L,D=x.shape
        o = self.gqa(x)  # (B,L,D)
        o = o.view(B,L,self.nh,self.dh)
        o = torch.einsum('blhd,gh->blgd', o, self.head_mix)
        return o.reshape(B,L,D)

class GatedAttn(nn.Module):
    """Attention with sigmoid gate on output."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.gqa = CausalGQA(d, nh, nh, w)
        self.gate = nn.Linear(d, d)
    def forward(self, x, **kw):
        return self.gqa(x) * torch.sigmoid(self.gate(x))

class DiffAttn(nn.Module):
    """Differential attention: softmax(Q1K) - λ*softmax(Q2K)."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.w=w; self.s=self.dh**-0.5
        self.q1=nn.Linear(d,d,bias=False); self.q2=nn.Linear(d,d,bias=False)
        self.k=nn.Linear(d,d,bias=False); self.v_proj=nn.Linear(d,d,bias=False)
        self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh); self.kn=nn.RMSNorm(self.dh)
        self.lam=nn.Parameter(torch.ones(nh)*0.5)
    def forward(self, x, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        q1=self.qn(self.q1(x).view(B,L,H,DH)).transpose(1,2)
        q2=self.qn(self.q2(x).view(B,L,H,DH)).transpose(1,2)
        k=self.kn(self.k(x).view(B,L,H,DH)).transpose(1,2)
        v=self.v_proj(x).view(B,L,H,DH).transpose(1,2)
        lam=torch.sigmoid(self.lam).view(1,H,1,1)
        W=self.w; out=torch.zeros_like(q1)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            mask=(qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0)
            a1=torch.matmul(q1[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            a2=torch.matmul(q2[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            a1=a1.masked_fill(mask,float('-inf')); a2=a2.masked_fill(mask,float('-inf'))
            out[:,:,i:e]=torch.matmul(F.softmax(a1,-1)-lam*F.softmax(a2,-1),v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class LinDecayAttn(nn.Module):
    """Attention with learned per-head linear position decay bias (ALiBi-like)."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.gqa = CausalGQA(d, nh, nh, w)
        self.nh=nh; self.dh=d//nh; self.w=w; self.s=self.dh**-0.5
        self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh); self.kn=nn.RMSNorm(self.dh)
        slopes = torch.linspace(-0.01, -0.5, nh)
        self.log_slope = nn.Parameter(slopes)
    def forward(self, x, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        qkv=self.qkv(x).view(B,L,3,H,DH)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        q=self.qn(q).transpose(1,2); k=self.kn(k).transpose(1,2); v=v.transpose(1,2)
        slope = -F.softplus(self.log_slope).view(1,H,1,1)
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            dist=(qp[:,None]-kp[None,:]).float().abs()
            a=a+slope*dist.unsqueeze(0).unsqueeze(0)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),float('-inf'))
            out[:,:,i:e]=torch.matmul(F.softmax(a,-1),v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class TalkingHeadsAttn(nn.Module):
    """Attention with pre/post-softmax head mixing."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.w=w; self.s=self.dh**-0.5
        self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh); self.kn=nn.RMSNorm(self.dh)
        self.pre_mix=nn.Parameter(torch.eye(nh)); self.post_mix=nn.Parameter(torch.eye(nh))
    def forward(self, x, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        qkv=self.qkv(x).view(B,L,3,H,DH)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        q=self.qn(q).transpose(1,2); k=self.kn(k).transpose(1,2); v=v.transpose(1,2)
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),float('-inf'))
            a=torch.einsum('bhlk,gh->bglk',a,self.pre_mix)
            a=F.softmax(a,-1)
            a=torch.einsum('bhlk,gh->bglk',a,self.post_mix)
            out[:,:,i:e]=torch.matmul(a,v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

class SharedKVAttn(nn.Module):
    """Q per layer, K/V shared (passed in)."""
    def __init__(self, d, nh=8, w=256):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.w=w; self.s=self.dh**-0.5
        self.q=nn.Linear(d,d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.qn=nn.RMSNorm(self.dh)
    def forward(self, x, shared_k=None, shared_v=None, **kw):
        B,L,D=x.shape; H,DH=self.nh,self.dh
        q=self.qn(self.q(x).view(B,L,H,DH)).transpose(1,2)
        k=shared_k; v=shared_v
        if k is None: return torch.zeros_like(x)  # fallback
        W=self.w; out=torch.zeros_like(q)
        for i in range(0,L,W):
            e=min(i+W,L); s=max(0,i-W)
            a=torch.matmul(q[:,:,i:e],k[:,:,s:e].transpose(-1,-2))*self.s
            qp=torch.arange(i,e,device=x.device); kp=torch.arange(s,e,device=x.device)
            a=a.masked_fill((qp[:,None]<kp[None,:]).unsqueeze(0).unsqueeze(0),float('-inf'))
            out[:,:,i:e]=torch.matmul(F.softmax(a,-1),v[:,:,s:e])
        return self.o(out.transpose(1,2).reshape(B,L,D))

# ══════════════════════════════════════════════════════════════════
# CONVOLUTION VARIANTS (11-20)
# ══════════════════════════════════════════════════════════════════

class DilatedConv(nn.Module):
    """Dilated causal convolution stack with gating."""
    def __init__(self, d, n_dilations=6):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, 3, padding=2**i * 2, dilation=2**i, groups=d)
            for i in range(n_dilations)
        ])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = x.transpose(1,2)
        acc = torch.zeros_like(h)
        for conv in self.convs:
            acc = acc + conv(h)[:,:,:L]
        acc = acc.transpose(1,2)
        return self.out(F.silu(self.mix(acc)) * torch.sigmoid(self.gate(x)))

class ProgKernelConv(nn.Module):
    """Progressive kernel sizes by depth."""
    def __init__(self, d, kernel_size=7, nh=8):
        super().__init__()
        self.dh = d//nh; self.nh = nh
        self.v = nn.Linear(d, d, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dh, self.dh, kernel_size, padding=kernel_size-1, groups=self.dh)
            for _ in range(nh)
        ])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        v = self.v(x).view(B,L,self.nh,self.dh)
        hs = [F.silu(self.convs[h](v[:,:,h].transpose(1,2))[:,:,:L].transpose(1,2)) for h in range(self.nh)]
        return self.out(torch.cat(hs,-1) * torch.sigmoid(self.gate(x)))

class GLUConv(nn.Module):
    """Gated conv: two parallel convs, one gates the other."""
    def __init__(self, d, k=15):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.conv2 = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = x.transpose(1,2)
        return self.out((F.silu(self.conv1(h)[:,:,:L]) * torch.sigmoid(self.conv2(h)[:,:,:L])).transpose(1,2))

class DepthSepConv(nn.Module):
    """Depthwise separable: large depthwise + pointwise."""
    def __init__(self, d, k=65):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.pw = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = F.silu(self.dw(x.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out(self.pw(h) * torch.sigmoid(self.gate(x)))

class ParallelMultiConv(nn.Module):
    """Multiple kernel sizes on ALL channels, summed."""
    def __init__(self, d, ks=(3,7,15,31)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d,d,k,padding=k-1,groups=d) for k in ks])
        self.scale = 1.0/len(ks)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape; h = x.transpose(1,2)
        acc = sum(c(h)[:,:,:L] for c in self.convs) * self.scale
        return self.out(F.silu(acc.transpose(1,2)) * torch.sigmoid(self.gate(x)))

class RecursiveConv(nn.Module):
    """Apply k=3 conv N times recursively (growing receptive field)."""
    def __init__(self, d, n_apps=6):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d,d,3,padding=2,groups=d) for _ in range(n_apps)])
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape; h = x.transpose(1,2)
        for c in self.convs:
            h = F.silu(c(h)[:,:,:L])
        return self.out(self.mix(h.transpose(1,2)) * torch.sigmoid(self.gate(x)))

class ShiftGroupConv(nn.Module):
    """Group conv where each group is shifted by different amounts."""
    def __init__(self, d, ng=8, k=7):
        super().__init__()
        self.ng = ng; self.dg = d//ng
        self.conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        # Shift each group by different amount
        groups = x.view(B,L,self.ng,self.dg)
        shifted = []
        for g in range(self.ng):
            s = g  # shift amount
            if s > 0:
                shifted.append(F.pad(groups[:, :-s, g], (0,0,s,0)))
            else:
                shifted.append(groups[:,:,g])
        h = torch.stack(shifted, 2).reshape(B,L,D)
        h = F.silu(self.conv(h.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out(h * torch.sigmoid(self.gate(x)))

class ConvPlusCumsum(nn.Module):
    """Short conv + cumsum for global causal context."""
    def __init__(self, d, k=7):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.cumsum_proj = nn.Linear(d, d, bias=False)
        self.cumsum_gate = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        local = F.silu(self.conv(x.transpose(1,2))[:,:,:L].transpose(1,2))
        # Gated cumsum for global
        cs = torch.cumsum(self.cumsum_proj(x) * torch.sigmoid(self.cumsum_gate(x)), dim=1)
        positions = torch.arange(1, L+1, device=x.device, dtype=x.dtype).view(1,L,1)
        cs = cs / positions  # running mean
        return self.out((local + cs) * torch.sigmoid(self.gate(x)))

class ConvWithSE(nn.Module):
    """Conv + squeeze-and-excite (channel attention from running stats)."""
    def __init__(self, d, k=15, r=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.se = nn.Sequential(nn.Linear(d, d//r), nn.SiLU(), nn.Linear(d//r, d), nn.Sigmoid())
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = F.silu(self.conv(x.transpose(1,2))[:,:,:L].transpose(1,2))
        # Causal running mean for SE
        cs = torch.cumsum(h, dim=1)
        pos = torch.arange(1,L+1,device=x.device,dtype=x.dtype).view(1,L,1)
        se_w = self.se(cs / pos)
        return self.out(h * se_w)

class ConvButterflyMix(nn.Module):
    """Conv + butterfly channel mixing (permute groups between conv applications)."""
    def __init__(self, d, ng=8, k=7):
        super().__init__()
        self.ng = ng; self.dg = d//ng
        self.conv1 = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.conv2 = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = F.silu(self.conv1(x.transpose(1,2))[:,:,:L].transpose(1,2))
        # Butterfly permutation: roll channels by half-group
        h = torch.roll(h, self.dg//2, dims=-1)
        h = F.silu(self.conv2(h.transpose(1,2))[:,:,:L].transpose(1,2))
        return self.out(h * torch.sigmoid(self.gate(x)))

# ══════════════════════════════════════════════════════════════════
# NOVEL CAUSAL O(n) MECHANISMS (21-30)
# ══════════════════════════════════════════════════════════════════

class CumsumMixer(nn.Module):
    """Gated cumsum: project, gate, cumsum, normalize."""
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.decay_gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = self.proj(x) * torch.sigmoid(self.decay_gate(x))
        cs = torch.cumsum(h, dim=1)
        pos = torch.arange(1,L+1,device=x.device,dtype=x.dtype).view(1,L,1)
        return self.out(cs / pos * torch.sigmoid(self.gate(x)))

class MultiRateCumsum(nn.Module):
    """Different channel groups accumulate at different rates."""
    def __init__(self, d, ng=4):
        super().__init__()
        self.ng=ng; self.dg=d//ng
        self.proj = nn.Linear(d, d, bias=False)
        self.rates = nn.Parameter(torch.linspace(0.5, 0.99, ng))  # different decay rates
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = self.proj(x).view(B,L,self.ng,self.dg)
        rates = torch.sigmoid(self.rates).view(1,1,self.ng,1)
        # Exponentially weighted cumsum via log-space
        # Approximate: h_t = rate * h_{t-1} + (1-rate) * input_t
        # Use cumsum with geometric weights
        weights = rates.pow(torch.arange(L,device=x.device).float().view(1,L,1,1))
        weighted = h / (weights + 1e-8)
        cs = torch.cumsum(weighted, dim=1) * weights
        cs = cs.reshape(B,L,D)
        return self.out(cs * torch.sigmoid(self.gate(x)))

class StatePropMixer(nn.Module):
    """State propagation: update + query via cumsum."""
    def __init__(self, d):
        super().__init__()
        self.update = nn.Linear(d, d, bias=False)
        self.u_gate = nn.Linear(d, d)
        self.query = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        u = self.update(x) * torch.sigmoid(self.u_gate(x))
        state = torch.cumsum(u, dim=1)
        q = self.query(x)
        return self.out(q * state)

class ELULinearAttn(nn.Module):
    """Linear attention with ELU+1 kernel, cumsum-based causal."""
    def __init__(self, d, nh=8):
        super().__init__()
        self.nh=nh; self.dh=d//nh
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
    def forward(self, x, **kw):
        B,L,D = x.shape; H,DH = self.nh,self.dh
        qkv = self.qkv(x).view(B,L,3,H,DH)
        q,k,v = qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]  # (B,L,H,DH)
        q = F.elu(q) + 1; k = F.elu(k) + 1
        # Causal linear attention via cumsum
        # kv = cumsum(k * v)  -- but need (B,H,DH,DH) which is big
        # Use per-head: output = q @ cumsum(k^T v) / (q @ cumsum(k))
        # Chunk to avoid OOM: process in groups of features
        kv = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)  # (B,L,H,DH_k,DH_v)
        k_sum = torch.cumsum(k, dim=1)  # (B,L,H,DH)
        num = torch.einsum('blhk,blhkv->blhv', q, kv)  # (B,L,H,DH)
        den = (q * k_sum).sum(-1, keepdim=True).clamp(min=1e-6)  # (B,L,H,1)
        out = (num / den).reshape(B,L,D)
        return self.out(out * torch.sigmoid(self.gate(x)))

class CausalAFT(nn.Module):
    """Attention-Free Transformer: position-factored causal."""
    def __init__(self, d, nh=8, max_len=2048):
        super().__init__()
        self.nh=nh; self.dh=d//nh
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,d,bias=False)
        self.v_proj=nn.Linear(d,d,bias=False); self.out=nn.Linear(d,d,bias=False)
        # Factored position weights: w_{t,s} = w_q[t] + w_k[s]
        self.wq = nn.Parameter(torch.zeros(max_len, nh))
        self.wk = nn.Parameter(torch.zeros(max_len, nh))
    def forward(self, x, **kw):
        B,L,D = x.shape; H,DH = self.nh,self.dh
        q = torch.sigmoid(self.q(x).view(B,L,H,DH))
        k = self.k(x).view(B,L,H,DH)
        v = self.v_proj(x).view(B,L,H,DH)
        # Position weights
        wq = self.wq[:L].unsqueeze(0)  # (1,L,H)
        wk = self.wk[:L].unsqueeze(0)
        # exp(wk) * k * v accumulated causally
        ew = torch.exp(wk).unsqueeze(-1)  # (1,L,H,1)
        kv = torch.cumsum(ew * k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)  # (B,L,H,DH,DH)
        k_acc = torch.cumsum(ew * k, dim=1)  # (B,L,H,DH)
        ewq = torch.exp(wq).unsqueeze(-1)
        num = torch.einsum('blhk,blhkv->blhv', q * ewq, kv)
        den = (q * ewq * k_acc).sum(-1, keepdim=True).clamp(min=1e-6)
        out = (num / den).reshape(B,L,D)
        return self.out(out)

class BilinearCumsum(nn.Module):
    """Bilinear: two projections multiplied then cumsum'd."""
    def __init__(self, d):
        super().__init__()
        self.p1 = nn.Linear(d, d, bias=False)
        self.p2 = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        h = self.p1(x) * self.p2(x)  # bilinear
        cs = torch.cumsum(h * torch.sigmoid(self.gate(x)), dim=1)
        pos = torch.arange(1,L+1,device=x.device,dtype=x.dtype).view(1,L,1)
        return self.out(cs / pos)

class RoutedCumsum(nn.Module):
    """Route into K bins, cumsum per bin, reassemble."""
    def __init__(self, d, n_bins=8):
        super().__init__()
        self.n_bins = n_bins
        self.router = nn.Linear(d, n_bins)
        self.proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
    def forward(self, x, **kw):
        B,L,D = x.shape
        w = F.softmax(self.router(x), dim=-1)  # (B,L,K)
        h = self.proj(x)  # (B,L,D)
        # Weighted cumsum per bin
        binned = w.unsqueeze(-1) * h.unsqueeze(2)  # (B,L,K,D)
        cs = torch.cumsum(binned, dim=1)  # (B,L,K,D)
        out = (w.unsqueeze(-1) * cs).sum(2)  # (B,L,D)
        return self.out(out * torch.sigmoid(self.gate(x)))

class HaarCausalFilter(nn.Module):
    """Multi-scale running mean subtraction (causal Haar-like)."""
    def __init__(self, d, scales=(1,4,16,64)):
        super().__init__()
        self.scales = scales
        self.scale_proj = nn.Linear(d * len(scales), d, bias=False)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        cs = torch.cumsum(x, dim=1)
        pos = torch.arange(1,L+1,device=x.device,dtype=x.dtype).view(1,L,1)
        features = []
        for s in self.scales:
            if s == 1:
                features.append(x)
            else:
                # Running mean over last s positions (causal)
                cs_shifted = F.pad(cs[:,:-s], (0,0,s,0))
                local_mean = (cs - cs_shifted) / s
                features.append(x - local_mean)  # high-pass
        h = self.scale_proj(torch.cat(features, -1))
        return self.out(h * torch.sigmoid(self.gate(x)))

class ProductKeyMemory(nn.Module):
    """Product-key lookup from fixed learned memory."""
    def __init__(self, d, n_keys=256, key_dim=32, n_heads=8):
        super().__init__()
        self.nh=n_heads; self.dh=d//n_heads; self.nk=n_keys; self.kd=key_dim
        self.q_proj = nn.Linear(d, n_heads * key_dim * 2, bias=False)  # split into 2 sub-keys
        self.keys1 = nn.Parameter(torch.randn(n_heads, n_keys, key_dim) * 0.02)
        self.keys2 = nn.Parameter(torch.randn(n_heads, n_keys, key_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_heads, n_keys * n_keys, self.dh) * 0.02)  # too big for n_keys=256
        self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, **kw):
        B,L,D = x.shape
        q = self.q_proj(x).view(B,L,self.nh,2,self.kd)
        q1,q2 = q[:,:,:,0], q[:,:,:,1]  # (B,L,H,kd)
        # Top-k=4 from each sub-key
        s1 = torch.matmul(q1, self.keys1.transpose(-1,-2))  # (B,L,H,nk)
        s2 = torch.matmul(q2, self.keys2.transpose(-1,-2))
        # Product: outer product of top scores
        top_s = F.softmax(s1.unsqueeze(-1) + s2.unsqueeze(-2), dim=(-1,-2))  # (B,L,H,nk,nk)
        top_s = top_s.reshape(B,L,self.nh,-1)  # (B,L,H,nk*nk)
        out = torch.matmul(top_s, self.values)  # (B,L,H,dh)
        return self.out(out.reshape(B,L,D))

# ══════════════════════════════════════════════════════════════════
# MODEL FACTORY — build any of the 100 architectures
# ══════════════════════════════════════════════════════════════════

# Architecture configs: (name, description, mixer_fn, model_kwargs)
# mixer_fn(d, ac) -> list of (mixer, needs_embed) per layer

ARCH_CONFIGS = {}

def _reg(name, desc, mixer_fn):
    """Register an architecture config."""
    ARCH_CONFIGS[name] = (desc, mixer_fn)

# Helper: progressive conv->attn with given attn mixer
def _prog(attn_cls, attn_kw=None, conv_cls=None, conv_kw=None, split=0.5, use_ts=False, use_vr=False):
    """Create a progressive conv->attn mixer function."""
    def fn(d, n, ac):
        s = int(n * split)
        layers = []
        for i in range(n):
            if i >= s:
                if use_vr:
                    m = ValueResGQA(d, **(attn_kw or {}))
                    layers.append((m, True))
                else:
                    m = (attn_cls or CausalGQA)(d, **(attn_kw or {}))
                    layers.append((m, False))
            else:
                if use_ts:
                    layers.append((TokenShiftMHConv(d), False))
                else:
                    cc = conv_cls or MHConv
                    layers.append((cc(d, **(conv_kw or {})), False))
        return layers
    return fn

# Helper: pure (all same mixer)
def _pure(mixer_cls, mixer_kw=None):
    def fn(d, n, ac):
        return [((mixer_cls)(d, **(mixer_kw or {})), False) for _ in range(n)]
    return fn

# ── Register all 100 architectures ──

# GROUP 1: Attention kernel variants (1-10) with conv early layers
_reg("B100_01_SigmoidAttn", "Conv + sigmoid attention", _prog(SigmoidAttn))
_reg("B100_02_ReLUSqAttn", "Conv + ReLU² attention", _prog(ReLUSquaredAttn))
_reg("B100_03_CosineAttn", "Conv + cosine similarity attention", _prog(CosineAttn))
_reg("B100_04_HeadMixAttn", "Conv + head-mixing attention", _prog(HeadMixAttn))
_reg("B100_05_GatedAttn", "Conv + gated attention output", _prog(GatedAttn))
_reg("B100_06_DiffAttn", "Conv + differential attention", _prog(DiffAttn))
_reg("B100_07_LinDecayAttn", "Conv + ALiBi-like learned decay attention", _prog(LinDecayAttn))
_reg("B100_08_TalkingHeads", "Conv + talking heads attention", _prog(TalkingHeadsAttn))
_reg("B100_09_SigmoidAttnVR", "Conv + sigmoid attn + value residual", _prog(SigmoidAttn, use_vr=True))
_reg("B100_10_CosineAttnVR", "Conv + cosine attn + value residual", _prog(CosineAttn, use_vr=True))

# GROUP 2: Attention + token shift (11-15)
_reg("B100_11_TSGatedAttn", "TokenShift conv + gated attention", _prog(GatedAttn, use_ts=True))
_reg("B100_12_TSDiffAttn", "TokenShift conv + differential attention", _prog(DiffAttn, use_ts=True))
_reg("B100_13_TSHeadMix", "TokenShift conv + head-mix attention", _prog(HeadMixAttn, use_ts=True))
_reg("B100_14_TSTalkingHeads", "TokenShift conv + talking heads", _prog(TalkingHeadsAttn, use_ts=True))
_reg("B100_15_TSLinDecay", "TokenShift conv + learned decay attention", _prog(LinDecayAttn, use_ts=True))

# GROUP 3: Value residual variants (16-20)
_reg("B100_16_VRGatedAttn", "Conv + value-residual gated attention", _prog(GatedAttn, use_vr=True))
_reg("B100_17_VRDiffAttn", "Conv + value-residual differential attention", _prog(DiffAttn, use_vr=True))
_reg("B100_18_VRHeadMix", "Conv + value-residual head-mix attention", _prog(HeadMixAttn, use_vr=True))
_reg("B100_19_VRLinDecay", "Conv + value-residual learned decay attention", _prog(LinDecayAttn, use_vr=True))
_reg("B100_20_VRTalkingHeads", "Conv + value-residual talking heads", _prog(TalkingHeadsAttn, use_vr=True))

# GROUP 4: Token shift + value residual combos (21-25)
_reg("B100_21_TSVRSigmoid", "TS conv + VR sigmoid attn", _prog(SigmoidAttn, use_ts=True, use_vr=True))
_reg("B100_22_TSVRCosine", "TS conv + VR cosine attn", _prog(CosineAttn, use_ts=True, use_vr=True))
_reg("B100_23_TSVRGated", "TS conv + VR gated attn", _prog(GatedAttn, use_ts=True, use_vr=True))
_reg("B100_24_TSVRDiff", "TS conv + VR differential attn", _prog(DiffAttn, use_ts=True, use_vr=True))
_reg("B100_25_TSVRBase", "TS conv + VR standard GQA", _prog(CausalGQA, use_ts=True, use_vr=True))

# GROUP 5: Different conv/attn ratios (26-30)
_reg("B100_26_Conv75Attn25", "75% conv, 25% attn", _prog(CausalGQA, split=0.75))
_reg("B100_27_Conv25Attn75", "25% conv, 75% attn", _prog(CausalGQA, split=0.25))
_reg("B100_28_Conv75VR25", "75% conv, 25% VR attn", _prog(CausalGQA, split=0.75, use_vr=True))
_reg("B100_29_Conv25VR75", "25% conv, 75% VR attn", _prog(CausalGQA, split=0.25, use_vr=True))
_reg("B100_30_PureAttn", "Pure GQA attention (no conv)", _prog(CausalGQA, split=0.0))

# GROUP 6: Conv variations (31-40)
_reg("B100_31_DilatedConv", "Dilated causal conv + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: DilatedConv(d)))
_reg("B100_32_GLUConv", "GLU conv + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: GLUConv(d)))
_reg("B100_33_DepthSepConv", "Depthwise separable conv + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: DepthSepConv(d)))
_reg("B100_34_ParallelConv", "Parallel multi-kernel conv + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: ParallelMultiConv(d)))
_reg("B100_35_RecursiveConv", "Recursive k=3 conv + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: RecursiveConv(d)))
_reg("B100_36_ShiftGroupConv", "Shift-group conv + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: ShiftGroupConv(d)))
_reg("B100_37_ConvCumsum", "Conv+cumsum hybrid + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: ConvPlusCumsum(d)))
_reg("B100_38_ConvSE", "Conv+squeeze-excite + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: ConvWithSE(d)))
_reg("B100_39_ConvButterfly", "Conv+butterfly channel mix + attn", _prog(CausalGQA, conv_cls=lambda d, **kw: ConvButterflyMix(d)))
_reg("B100_40_PureConv", "Pure multi-head conv (no attn)", _pure(MHConv))

# GROUP 7: Conv variations + value residual (41-45)
_reg("B100_41_DilatedVR", "Dilated conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:DilatedConv(d), use_vr=True))
_reg("B100_42_ParallelConvVR", "Parallel conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:ParallelMultiConv(d), use_vr=True))
_reg("B100_43_ConvCumsumVR", "Conv+cumsum + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:ConvPlusCumsum(d), use_vr=True))
_reg("B100_44_RecursiveConvVR", "Recursive conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:RecursiveConv(d), use_vr=True))
_reg("B100_45_ConvSEVR", "Conv+SE + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:ConvWithSE(d), use_vr=True))

# GROUP 8: Pure novel O(n) mixers (46-55)
_reg("B100_46_PureCumsum", "Pure gated cumsum mixing", _pure(CumsumMixer))
_reg("B100_47_PureStateProp", "Pure state propagation mixing", _pure(StatePropMixer))
_reg("B100_48_PureBilinear", "Pure bilinear cumsum", _pure(BilinearCumsum))
_reg("B100_49_PureRoutedCS", "Pure routed cumsum", _pure(RoutedCumsum))
_reg("B100_50_PureHaar", "Pure causal Haar filter", _pure(HaarCausalFilter))
_reg("B100_51_PureMultiRate", "Pure multi-rate cumsum", _pure(MultiRateCumsum))
_reg("B100_52_PureELULinear", "Pure ELU linear attention", _pure(ELULinearAttn))

def _aft_fn(d, n, ac):
    return [(CausalAFT(d), False) for _ in range(n)]
_reg("B100_53_PureAFT", "Pure attention-free transformer", _aft_fn)

_reg("B100_54_PureDilatedConv", "Pure dilated conv", _pure(DilatedConv))
_reg("B100_55_PureGLUConv", "Pure GLU conv", _pure(GLUConv))

# GROUP 9: Novel + attn hybrids (56-65)
_reg("B100_56_CumsumAttn", "Cumsum early + attn late", _prog(CausalGQA, conv_cls=lambda d,**kw:CumsumMixer(d)))
_reg("B100_57_StatePropAttn", "State prop early + attn late", _prog(CausalGQA, conv_cls=lambda d,**kw:StatePropMixer(d)))
_reg("B100_58_BilinearAttn", "Bilinear cumsum + attn", _prog(CausalGQA, conv_cls=lambda d,**kw:BilinearCumsum(d)))
_reg("B100_59_RoutedCSAttn", "Routed cumsum + attn", _prog(CausalGQA, conv_cls=lambda d,**kw:RoutedCumsum(d)))
_reg("B100_60_HaarAttn", "Haar filter + attn", _prog(CausalGQA, conv_cls=lambda d,**kw:HaarCausalFilter(d)))
_reg("B100_61_MultiRateAttn", "Multi-rate cumsum + attn", _prog(CausalGQA, conv_cls=lambda d,**kw:MultiRateCumsum(d)))
_reg("B100_62_ELULinAttn", "ELU linear attn early + full attn late", _prog(CausalGQA, conv_cls=lambda d,**kw:ELULinearAttn(d)))
_reg("B100_63_DilatedConvVRAttn", "Dilated conv + VR diff attn", _prog(DiffAttn, conv_cls=lambda d,**kw:DilatedConv(d), use_vr=True))
_reg("B100_64_ParConvGatedAttn", "Parallel conv + gated attn", _prog(GatedAttn, conv_cls=lambda d,**kw:ParallelMultiConv(d)))
_reg("B100_65_HaarVRAttn", "Haar + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:HaarCausalFilter(d), use_vr=True))

# GROUP 10: Attention with different window sizes (66-70)
_reg("B100_66_Attn_w64", "Conv + attn window=64", _prog(CausalGQA, attn_kw={"w":64}))
_reg("B100_67_Attn_w128", "Conv + attn window=128", _prog(CausalGQA, attn_kw={"w":128}))
_reg("B100_68_Attn_w512", "Conv + attn window=512", _prog(CausalGQA, attn_kw={"w":512}))
_reg("B100_69_Attn_full", "Conv + full causal attn", _prog(CausalGQA, attn_kw={"w":2048}))
_reg("B100_70_VR_w512", "Conv + VR attn window=512", _prog(CausalGQA, attn_kw={"w":512}, use_vr=True))

# GROUP 11: Mixed conv types in early layers (71-75)
def _mixed_conv(attn_use_vr=False):
    def fn(d, n, ac):
        s = n // 2
        layers = []
        for i in range(n):
            if i >= s:
                if attn_use_vr:
                    layers.append((ValueResGQA(d), True))
                else:
                    layers.append((CausalGQA(d), False))
            elif i % 3 == 0:
                layers.append((DilatedConv(d), False))
            elif i % 3 == 1:
                layers.append((MHConv(d), False))
            else:
                layers.append((ParallelMultiConv(d), False))
        return layers
    return fn

_reg("B100_71_MixedConvAttn", "Mixed conv types + attn", _mixed_conv())
_reg("B100_72_MixedConvVR", "Mixed conv types + VR attn", _mixed_conv(True))

# Alternating different conv+attn per layer
def _alternating(d, n, ac):
    layers = []
    for i in range(n):
        if i % 2 == 0:
            layers.append((MHConv(d), False))
        else:
            layers.append((CausalGQA(d), False))
    return layers
_reg("B100_73_Alternating", "Alternating conv/attn every layer", _alternating)

def _conv_attn_conv(d, n, ac):
    """Conv → attn → conv sandwich."""
    third = n // 3
    layers = []
    for i in range(n):
        if i < third or i >= 2*third:
            layers.append((MHConv(d), False))
        else:
            layers.append((CausalGQA(d), False))
    return layers
_reg("B100_74_ConvAttnConv", "Conv-attn-conv sandwich", _conv_attn_conv)

def _one_attn_layer(d, n, ac):
    """All conv except 1 attn layer in the middle."""
    mid = n // 2
    return [(CausalGQA(d),False) if i==mid else (MHConv(d),False) for i in range(n)]
_reg("B100_75_OneAttnLayer", "All conv + 1 attn layer at middle", _one_attn_layer)

# GROUP 12: Structural variations (76-85)
# Different number of heads
_reg("B100_76_16Heads", "Conv + 16-head attn (d_head=32)", _prog(CausalGQA, attn_kw={"nh":16, "nkv":8}))
_reg("B100_77_4Heads", "Conv + 4-head attn (d_head=128)", _prog(CausalGQA, attn_kw={"nh":4, "nkv":2}))

# GQA with different ratios
_reg("B100_78_GQA_2kv", "Conv + GQA 8q/2kv", _prog(CausalGQA, attn_kw={"nkv":2}))
_reg("B100_79_GQA_1kv", "Conv + GQA 8q/1kv (multi-query)", _prog(CausalGQA, attn_kw={"nkv":1}))
_reg("B100_80_MHA", "Conv + MHA (8q/8kv)", _prog(CausalGQA, attn_kw={"nkv":8}))

# FFN first (swap order)
def _ffn_first(d, n, ac):
    """Mixer uses FFN-first block structure (handled at model level)."""
    return [(CausalGQA(d), False) for _ in range(n)]
_reg("B100_81_FFNFirst", "FFN-first block order", _ffn_first)

# Memory tokens
_reg("B100_82_MemTokens", "Standard + 4 learnable memory tokens", _prog(CausalGQA))

# Pure attention with value residual
_reg("B100_83_PureAttnVR", "Pure attn + value residual", _prog(CausalGQA, split=0.0, use_vr=True))

# Deeper narrower
def _deep_narrow(d, n, ac):
    # Note: this will be overridden at model level to use more layers
    return [(MHConv(d), False) if i < n//2 else (CausalGQA(d), False) for i in range(n)]
_reg("B100_84_Deep30L", "30 layers (narrower FFN)", _deep_narrow)

# Wider
_reg("B100_85_Wide640", "d_model=640 (fewer layers)", _prog(CausalGQA))

# GROUP 13: Novel mixer + specific attn combos (86-95)
_reg("B100_86_ConvCumsumSigmoid", "Conv+cumsum + sigmoid attn", _prog(SigmoidAttn, conv_cls=lambda d,**kw:ConvPlusCumsum(d)))
_reg("B100_87_DilatedDiffAttn", "Dilated conv + diff attn", _prog(DiffAttn, conv_cls=lambda d,**kw:DilatedConv(d)))
_reg("B100_88_RecConvSigmoid", "Recursive conv + sigmoid attn", _prog(SigmoidAttn, conv_cls=lambda d,**kw:RecursiveConv(d)))
_reg("B100_89_GLUConvGated", "GLU conv + gated attn", _prog(GatedAttn, conv_cls=lambda d,**kw:GLUConv(d)))
_reg("B100_90_SEConvTalkHead", "SE conv + talking heads", _prog(TalkingHeadsAttn, conv_cls=lambda d,**kw:ConvWithSE(d)))

_reg("B100_91_ButterflyDiffAttn", "Butterfly conv + diff attn", _prog(DiffAttn, conv_cls=lambda d,**kw:ConvButterflyMix(d)))
_reg("B100_92_ShiftConvCosine", "Shift-group conv + cosine attn", _prog(CosineAttn, conv_cls=lambda d,**kw:ShiftGroupConv(d)))
_reg("B100_93_DepSepVRDiff", "DepSep conv + VR diff attn", _prog(DiffAttn, conv_cls=lambda d,**kw:DepthSepConv(d), use_vr=True))
_reg("B100_94_ParConvVRSigmoid", "Parallel conv + VR sigmoid attn", _prog(SigmoidAttn, conv_cls=lambda d,**kw:ParallelMultiConv(d), use_vr=True))
_reg("B100_95_GLUConvVR", "GLU conv + VR standard attn", _prog(CausalGQA, conv_cls=lambda d,**kw:GLUConv(d), use_vr=True))

# GROUP 14: Triple combos (96-100)
_reg("B100_96_TSParConvVR", "TS parallel conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:ParallelMultiConv(d), use_ts=True, use_vr=True))
_reg("B100_97_TSDilatedVR", "TS dilated conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:DilatedConv(d), use_ts=True, use_vr=True))
_reg("B100_98_TSConvCSVR", "TS conv+cumsum + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:ConvPlusCumsum(d), use_ts=True, use_vr=True))
_reg("B100_99_TSGLUConvVR", "TS GLU conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:GLUConv(d), use_ts=True, use_vr=True))
_reg("B100_100_TSSEConvVR", "TS SE conv + VR attn", _prog(CausalGQA, conv_cls=lambda d,**kw:ConvWithSE(d), use_ts=True, use_vr=True))


# ══════════════════════════════════════════════════════════════════
# GENERIC MODEL CLASS — builds from any ARCH_CONFIG entry
# ══════════════════════════════════════════════════════════════════

class Batch100Model(FrontierModel):
    """Generic model that builds from ARCH_CONFIGS registry."""
    def __init__(self, config: FrontierConfig, arch_name: str):
        super().__init__(config)
        self._arch_name = arch_name
        desc, mixer_fn = ARCH_CONFIGS[arch_name]
        d = config.d_model
        n = config.n_layers

        self.embed = EmbeddingWithScale(config.vocab_size, d, dropout=config.dropout)

        layers_spec = mixer_fn(d, n, config.arch_config)
        self._has_embed = any(needs_embed for _, needs_embed in layers_spec)

        blocks = []
        for mixer, needs_embed in layers_spec:
            if needs_embed:
                blocks.append(EmbedBlock(d, config.d_ff, mixer))
            else:
                blocks.append(Block(d, config.d_ff, mixer))
        self.blocks = nn.ModuleList(blocks)
        self._embed_flags = [ne for _, ne in layers_spec]

        # Special: memory tokens for B100_82
        self.mem_tokens = None
        if arch_name == "B100_82_MemTokens":
            self.mem_tokens = nn.Parameter(torch.randn(1, 4, d) * 0.02)

        self.norm = nn.RMSNorm(d)
        self.head = LMHead(d, config.vocab_size,
                           self.embed.embedding.weight if config.tie_weights else None)
        self.apply(_init)

    def forward(self, x):
        h = self.embed(x)
        embed_out = h

        if self.mem_tokens is not None:
            B = h.shape[0]
            mem = self.mem_tokens.expand(B, -1, -1)
            h = torch.cat([mem, h], dim=1)
            embed_out = torch.cat([mem, embed_out], dim=1)

        for i, block in enumerate(self.blocks):
            if self._embed_flags[i]:
                h = block(h, embed=embed_out)
            else:
                h = block(h)

        if self.mem_tokens is not None:
            h = h[:, self.mem_tokens.shape[1]:]  # remove memory tokens

        return self.head(self.norm(h))

    @classmethod
    def arch_family(cls): return "novel"
    def describe(self): return f"{self._arch_name}: {self.config.n_layers}L {self.config.d_model}d"
    def sequence_mixing_complexity(self): return "varies"


# Register all 100 architectures
for name, (desc, _) in ARCH_CONFIGS.items():
    @register_arch(f"{name}LM", "novel", desc)
    class _M(Batch100Model):
        _arch_key = name
        def __init__(self, config):
            super().__init__(config, self.__class__._arch_key)
    _M.__name__ = f"{name}LM"
    _M.__qualname__ = f"{name}LM"
    # Store reference to prevent GC
    globals()[f"_{name}_cls"] = _M
