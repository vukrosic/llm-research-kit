Hello 👋

Here are 19+ structured ablation experiments you can implement to accelerate your LLM / Transformer training. 

I ran 38 experiments on an 88M parameter LLM, training each configuration for 10M tokens. Below are the quantitative results comparing a standard Squared ReLU baseline against SwiGLU implementations and novel gating mechanisms.

---

### 📊 All 38 Experiments Ranked
*(Ranked by Validation Loss against a Squared ReLU Baseline)*

| Rank | Experiment | Val Loss | Delta % | Implementation Details |
|:-----|:-----------|:---------|:--------|:-----------------------|
| | **🟢 IDEAS TO TRY (Beat Baseline)** | | | |
| 1 | **swiglu_sandwich** | 4.6682 | -2.20% | SwiGLU with RMSNorm applied to the gated activations prior to down-projection: `RMSNorm(SiLU(xW_g) * xW_u)W_d`. |
| 2 | **scispace_tripleprojglu** | 4.7180 | -1.16% | Three independent projections instead of two: `(SiLU(xW_g) * xW_u * Sigmoid(xW_a))W_d`, where W_a is an auxiliary projection. |
| 3 | **swiglu_full** | 4.7226 | -1.06% | SwiGLU with hidden dimension exactly equal to `d_ff` (giving 50% more parameters than the standard 2/3 ratio). |
| 4 | **scispace_hardswishglu** | 4.7255 | -1.00% | Replaces the SiLU gate activation with Hardswish: `(Hardswish(xW_g) * xW_u)W_d`. |
| 5 | **scispace_celuglu** | 4.7309 | -0.89% | Replaces the SiLU gate activation with CELU: `(CELU(xW_g) * xW_u)W_d`. |
| 6 | **swiglu** | 4.7347 | -0.81% | Standard SwiGLU (hidden size `2/3 * d_ff`): `(SiLU(xW_g) * xW_u)W_d`. |
| 7 | **swiglu_bias** | 4.7352 | -0.80% | Standard SwiGLU where all linear projections (`W_g`, `W_u`, `W_d`) include learnable bias vectors. |
| 8 | **swiglu_wide** | 4.7375 | -0.75% | SwiGLU with hidden size `8/3 * d_model` (standard LLaMA configuration when baseline `d_ff = 4 * d_model`). |
| 9 | **swiglu_parallel** | 4.7396 | -0.71% | FFN is evaluated in parallel with Attention: `x = x + Attn(x) + FFN(x)` instead of sequentially. |
| 10 | **scispace_eluglu** | 4.7443 | -0.61% | Replaces the SiLU gate activation with ELU: `(ELU(xW_g) * xW_u)W_d`. |
| 11 | **scispace_moelite** | 4.7443 | -0.61% | Mixture-of-Experts routing with 2 small experts (hidden size `1/3 * d_ff`) soft-routed via `softmax(xW_router)`. |
| 12 | **geglu** | 4.7444 | -0.61% | Replaces the SiLU gate activation with GELU: `(GELU(xW_g) * xW_u)W_d`. |
| 13 | **scispace_composite** | 4.7457 | -0.58% | Hybrid gate combining SiLU and GELU: `((0.5 * SiLU(xW_g) + 0.5 * GELU(xW_g)) * xW_u)W_d`. |
| 14 | **swiglu_3q** | 4.7486 | -0.52% | SwiGLU with hidden dimension set to `3/4 * d_ff`. |
| 15 | **scispace_scalegate** | 4.7502 | -0.48% | Learnable per-dimension scale parameter `S` applied after the gate: `((SiLU(xW_g) * S) * xW_u)W_d`. |
| 16 | **scispace_singleprojglu** | 4.7569 | -0.34% | Gate and Value projections share a single projection matrix chunked in half (identical formulation to shared_gate). |
| 17 | **reglu** | 4.7613 | -0.25% | Replaces the SiLU gate activation with ReLU: `(ReLU(xW_g) * xW_u)W_d`. |
| 18 | **swiglu_residual** | 4.7632 | -0.21% | Adds an internal linear residual connection: `(SiLU(xW_g) * xW_u)W_d + xW_aux`. |
| 19 | **swiglu_narrow** | 4.7711 | -0.05% | SwiGLU with hidden dimension set to `1/2 * d_ff` to reduce parameter count. |
| 20 | **baseline** | 4.7733 | +0.00% | Standard Feed-Forward Network with Squared ReLU activation: `(ReLU(xW_in)²)W_out`. |
| | **🔴 PROBABLY SKIP (Worse than Baseline)** | | | |
| 21 | **swiglu_shared_gate** | 4.7773 | +0.08% | Gate and Value projections share a single projection matrix chunked in half: `W_gu = xW; g, u = chunk(W_gu); (SiLU(g) * u)W_d`. |
| 22 | **scispace_singlu** | 4.7852 | +0.25% | Replaces the SiLU gate activation with Sine: `(Sin(xW_g) * xW_u)W_d`. |
| 23 | **scispace_leakyglu** | 4.7931 | +0.41% | Replaces the SiLU gate activation with LeakyReLU (negative slope 0.01): `(LeakyReLU(xW_g) * xW_u)W_d`. |
| 24 | **scispace_asymglu** | 4.7936 | +0.43% | Asymmetric activations: `(SiLU(xW_g) * GELU(xW_u))W_d`. |
| 25 | **swiglu_depth_init** | 4.8017 | +0.59% | Standard SwiGLU, but model weights are scaled down proportional to layer depth during initialization. |
| 26 | **scispace_softplusglu** | 4.8038 | +0.64% | Replaces the SiLU gate activation with Softplus: `(Softplus(xW_g) * xW_u)W_d`. |
| 27 | **scispace_tanhglu** | 4.8042 | +0.65% | Replaces the SiLU gate activation with Tanh: `(Tanh(xW_g) * xW_u)W_d`. |
| 28 | **scispace_postgatelu** | 4.8050 | +0.66% | Gate is applied after the down-projection: `(SiLU(xW_u)W_d) * Sigmoid(xW_g)`. |
| 29 | **scispace_topkglu** | 4.8079 | +0.72% | Gate is sparsely activated; only top-K values per token pass (where K = 25% of hidden dims): `(TopK(SiLU(xW_g)) * xW_u)W_d`. |
| 30 | **swiglu_mqa** | 4.8205 | +0.99% | Standard SwiGLU, but the preceding attention block uses Multi-Query Attention (`n_kv_heads=1`). |
| 31 | **scispace_sincosglu** | 4.8360 | +1.31% | Dual periodic gating: `((Sin(xW_g) + Cos(xW_g)) * xW_u)W_d`. |
| 32 | **swiglu_deep** | 4.8474 | +1.55% | Sequential identical SwiGLU blocks with inner residuals: `x' = x + SwiGLU_1(x); x'' = x' + SwiGLU_2(x')`. |
| 33 | **scispace_sigmoidglu** | 4.8528 | +1.66% | Replaces the SiLU gate activation with Sigmoid (Classic DAU GLU): `(Sigmoid(xW_g) * xW_u)W_d`. |
| 34 | **scispace_pregatelu** | 4.8614 | +1.84% | Gate is applied to input before value projection: `(SiLU(xW_g) * x)W_u W_d`. |
| 35 | **swiglu_dual_gate** | 4.8659 | +1.94% | Independent dual SiLU gating paths: `(SiLU(xW_g1) * SiLU(xW_g2) * xW_u)W_d`. |
| 36 | **scispace_prenormdown** | 4.8727 | +2.08% | Applies standard LayerNorm right before the down-projection: `LayerNorm(SiLU(xW_g) * xW_u)W_d`. |
| 37 | **scispace_laplaceglu** | 4.9130 | +2.93% | Gating function uses double-exponential Laplace decay: `(exp(-abs(xW_g)) * xW_u)W_d`. |
| 38 | **swiglu_swiglu** | 4.9342 | +3.37% | Two sequential SwiGLU sub-blocks (no inner residual): `SwiGLU_2(SwiGLU_1(x))` where each sub-block projects to `1/2 * hidden`. |
