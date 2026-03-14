import torch
import torch.nn.functional as F
import math
import random

# coeffs for polar express
# not pre_computed, same as modded-nanoGPT
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile()
def zeropower_polar_express(G: torch.Tensor, steps: int = 5):
    """Polar express as replacement for Newton-Schulz iteration"""
    assert G.ndim >= 2

    X = G.bfloat16()
    transpose_needed = G.size(-2) > G.size(-1)
    if transpose_needed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    # 1. Use polar express for the first 5 steps (as long as coefficients exist)
    pe_steps = min(steps, len(coeffs_list))
    for a, b, c in coeffs_list[:pe_steps]:
        A = X @ X.mT
        A2 = A @ A
        B = b * A + c * A2
        X = a * X + B @ X

    # 2. Use Newton-Schulz for any remaining steps
    for _ in range(steps - pe_steps):
        A = X @ X.mT
        X = 0.5 * (3.0 * torch.eye(A.size(-1), device=A.device, dtype=A.dtype) - A) @ X

    if transpose_needed:
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz

    Extended with variant flags for novel optimizer ablation experiments (Gen9).

    Variant flags:
      post_momentum     : ortho(grad) first, then apply momentum to orthogonalized updates
      grad_centralize   : subtract row mean from gradient before orthogonalization
      half_ortho        : interpolate raw normalized grad and ortho grad (alpha=half_ortho, 0=raw, 1=full ortho)
      cautious          : zero update elements where sign(update) != sign(original grad)
      frob_scale        : scale update by Frobenius norm of grad / sqrt(m*n) instead of aspect ratio
      double_ortho      : apply polar express twice in sequence
      sign_mix          : add sign_mix * sign(orig_grad) to ortho update (then renormalize)
      row_norm          : L2-normalize each row of grad before polar express
      col_norm          : L2-normalize each column of grad before polar express
      ema_ortho         : maintain a separate EMA buffer on orthogonalized updates
      adaptive_ns       : use more polar express steps for larger parameter matrices
      trust_region      : clip update magnitude so ||delta|| <= trust_region * ||param||
      update_clip       : clip orthogonalized update Frobenius norm to this value (0=disabled)
      stochastic_ortho  : skip orthogonalization with this probability (0=always ortho)
      warm_momentum     : ramp momentum linearly from 0.5 to target over warm_momentum_steps steps
      rms_norm_grad     : divide gradient by its RMS before orthogonalization
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 # Novel variant flags
                 post_momentum=False,
                 grad_centralize=False,
                 half_ortho=0.0,
                 cautious=False,
                 frob_scale=False,
                 double_ortho=False,
                 sign_mix=0.0,
                 row_norm=False,
                 col_norm=False,
                 ema_ortho=False,
                 ema_ortho_momentum=0.9,
                 adaptive_ns=False,
                 trust_region=0.0,
                 update_clip=0.0,
                 stochastic_ortho=0.0,
                 warm_momentum=False,
                 warm_momentum_steps=100,
                 rms_norm_grad=False,
                 ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            post_momentum=post_momentum, grad_centralize=grad_centralize,
            half_ortho=half_ortho, cautious=cautious, frob_scale=frob_scale,
            double_ortho=double_ortho, sign_mix=sign_mix, row_norm=row_norm,
            col_norm=col_norm, ema_ortho=ema_ortho, ema_ortho_momentum=ema_ortho_momentum,
            adaptive_ns=adaptive_ns, trust_region=trust_region, update_clip=update_clip,
            stochastic_ortho=stochastic_ortho, warm_momentum=warm_momentum,
            warm_momentum_steps=warm_momentum_steps, rms_norm_grad=rms_norm_grad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["step"] = 0

                state["step"] += 1
                buf = state["momentum_buffer"]

                # === warm_momentum: linearly ramp momentum from 0.5 to target ===
                if group["warm_momentum"]:
                    t = state["step"]
                    T = max(1, group["warm_momentum_steps"])
                    eff_momentum = 0.5 + (momentum - 0.5) * min(1.0, t / T)
                else:
                    eff_momentum = momentum

                # Store original grad for cautious/frob_scale/sign_mix
                need_orig = group["cautious"] or group["frob_scale"] or group["sign_mix"] > 0
                g_orig = g.clone() if need_orig else None

                # === grad_centralize: subtract row mean (reduces inter-neuron coupling) ===
                if group["grad_centralize"] and g.ndim >= 2:
                    g = g - g.mean(dim=-1, keepdim=True)

                # === rms_norm_grad: normalize by gradient RMS ===
                if group["rms_norm_grad"]:
                    rms = g.square().mean().sqrt() + 1e-7
                    g = g / rms

                # === row_norm: L2-normalize each row before polar express ===
                if group["row_norm"] and g.ndim >= 2:
                    g = g / (g.norm(dim=-1, keepdim=True) + 1e-7)

                # === col_norm: L2-normalize each column before polar express ===
                if group["col_norm"] and g.ndim >= 2:
                    g = g / (g.norm(dim=-2, keepdim=True) + 1e-7)

                # Determine actual NS steps (adaptive or fixed)
                if group["adaptive_ns"] and g.ndim >= 2:
                    m, n = g.shape[-2], g.shape[-1]
                    size = max(m, n)
                    actual_steps = min(8, max(3, ns_steps + max(0, int(math.log2(size / 512 + 1)))))
                else:
                    actual_steps = ns_steps

                if not group["post_momentum"]:
                    # Standard flow: momentum → ortho
                    buf.lerp_(g, 1 - eff_momentum)
                    g = g.lerp_(buf, eff_momentum) if group["nesterov"] else buf.clone()

                    # === stochastic_ortho: randomly skip orthogonalization ===
                    if group["stochastic_ortho"] > 0 and random.random() < group["stochastic_ortho"]:
                        # Skip ortho: just normalize to preserve scale
                        g_norm = g.norm()
                        g_ortho = g / (g_norm + 1e-7)
                    else:
                        g_ortho = zeropower_polar_express(g, steps=actual_steps)
                        if group["double_ortho"]:
                            g_ortho = zeropower_polar_express(g_ortho, steps=actual_steps)

                    # === half_ortho: blend between raw normalized grad and fully ortho grad ===
                    if group["half_ortho"] > 0 and g.ndim >= 2:
                        alpha = group["half_ortho"]
                        g_raw_norm = g / (g.norm() + 1e-7)
                        g_ortho = (1.0 - alpha) * g_raw_norm + alpha * g_ortho

                    # === sign_mix: add fraction of sign(orig_grad) to ortho update ===
                    if group["sign_mix"] > 0 and g_orig is not None:
                        g_ortho = g_ortho + group["sign_mix"] * g_orig.sign()
                        g_ortho_norm = g_ortho.norm()
                        if g_ortho_norm > 0:
                            g_ortho = g_ortho / g_ortho_norm

                    # === ema_ortho: EMA buffer on orthogonalized updates ===
                    if group["ema_ortho"]:
                        if "ema_ortho_buffer" not in state:
                            state["ema_ortho_buffer"] = torch.zeros_like(g_ortho)
                        ema_buf = state["ema_ortho_buffer"]
                        ema_buf.lerp_(g_ortho, 1 - group["ema_ortho_momentum"])
                        g_ortho = ema_buf.clone()

                    g = g_ortho

                else:
                    # post_momentum: ortho first, then apply momentum to orthogonalized grad
                    g_ortho = zeropower_polar_express(g, steps=actual_steps)
                    if group["double_ortho"]:
                        g_ortho = zeropower_polar_express(g_ortho, steps=actual_steps)

                    buf.lerp_(g_ortho, 1 - eff_momentum)
                    g = g_ortho.lerp_(buf, eff_momentum) if group["nesterov"] else buf.clone()

                # === cautious: mask updates where update and orig grad have opposite sign ===
                if group["cautious"] and g_orig is not None:
                    mask = (g * g_orig).gt(0).float()
                    mask_mean = mask.mean()
                    if mask_mean > 0:
                        g = g * mask / (mask_mean + 1e-7)

                # === Compute update scale factor ===
                if group["frob_scale"] and g_orig is not None and p.ndim >= 2:
                    m, n = p.shape[-2], p.shape[-1]
                    frob = g_orig.norm(p='fro')
                    scale = (frob / ((m * n) ** 0.5 + 1e-7)).clamp(min=1e-7)
                else:
                    scale = max(1, p.size(-2) / p.size(-1)) ** 0.5

                # === update_clip: clip orthogonalized update by Frobenius norm ===
                if group["update_clip"] > 0 and g.ndim >= 2:
                    g_norm = g.norm(p='fro')
                    if g_norm > group["update_clip"]:
                        g = g * group["update_clip"] / g_norm

                delta = g.view_as(p).to(p.dtype) * (lr * scale)

                # === trust_region: clip delta so ||delta|| <= tau * ||param|| ===
                if group["trust_region"] > 0:
                    p_norm = p.norm()
                    delta_norm = delta.norm()
                    max_delta = group["trust_region"] * p_norm
                    if delta_norm > max_delta and delta_norm > 0:
                        delta = delta * max_delta / delta_norm

                p.add_(-delta)
