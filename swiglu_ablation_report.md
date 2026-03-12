Hello 👋

here are 19+ experiments to run on your LLM / transformer to accelerate the training.

I ran 38 experiments on an 88M param LLM, each time training for 10M tokens.

Here are best performing ideas for my setup

Thank you SciSpace for a lot of great ideas (HardSwish, CELU, triple-dense-layer structures and more).

---

### 📊 All 38 Experiments Ranked (10M Tokens)
*(Ranked by Validation Loss against a Squared ReLU Baseline)*

| Rank | Experiment | Val Loss | Delta % | Mathematical / Architectural Description |
|:-----|:-----------|:---------|:--------|:-------------|
| | **🟢 IDEAS TO TRY (Beat Baseline)** | | | |
| 1 | **swiglu_sandwich** | 4.6682 | -2.20% | Applies an RMSNorm layer immediately after the element-wise multiplication of the gate and value branches, before the output projection. |
| 2 | **scispace_tripleprojglu** | 4.7180 | -1.16% | Computes the element-wise product of three independent linear projections (rather than the standard two) prior to the final output projection. |
| 3 | **swiglu_full** | 4.7226 | -1.06% | Increases the intermediate hidden dimension of the SwiGLU block to match standard wider FFN expansion ratios. |
| 4 | **scispace_hardswishglu** | 4.7255 | -1.00% | Replaces the SiLU gating activation with HardSwish (a piece-wise linear approximation of Swish). |
| 5 | **scispace_celuglu** | 4.7309 | -0.89% | Replaces the SiLU gating activation with CELU (Continuously Differentiable Exponential Linear Unit). |
| 6 | **swiglu** | 4.7347 | -0.81% | Standard SwiGLU formulation: `(SiLU(xW_gate) * xW_val) W_out` without biases. |
| 7 | **swiglu_bias** | 4.7352 | -0.80% | Standard SwiGLU formulation, but includes bias terms in all linear projection layers. |
| 8 | **swiglu_wide** | 4.7375 | -0.75% | Increases the intermediate hidden dimension factor, scaled specifically for stability. |
| 9 | **swiglu_parallel** | 4.7396 | -0.71% | Computes the FFN output in parallel with the Attention block instead of sequentially (`x + Attention(x) + FFN(x)`). |
| 10 | **scispace_eluglu** | 4.7443 | -0.61% | Replaces the SiLU gating activation with the standard ELU activation. |
| 11 | **scispace_moelite** | 4.7443 | -0.61% | Replaces the continuous standard FFN parameterization with a small-scale, discrete Mixture-of-Experts routing mechanism. |
| 12 | **geglu** | 4.7444 | -0.61% | Replaces the SiLU gating activation with the GELU activation. |
| 13 | **scispace_composite** | 4.7457 | -0.58% | Uses a composite activation function (combining SiLU and Tanh) on the gating branch. |
| 14 | **swiglu_3q** | 4.7486 | -0.52% | Scales the intermediate hidden dimension to precisely 3/4 of the standard FFN expanded width. |
| 15 | **scispace_scalegate** | 4.7502 | -0.48% | Introduces a learned scalar parameter that multiplies the output of the gating branch. |
| 16 | **scispace_singleprojglu** | 4.7569 | -0.34% | Tied weights: uses the identical weight matrix for both the gating and value branches (`xW_gate == xW_val`). |
| 17 | **reglu** | 4.7613 | -0.25% | Replaces the SiLU gating activation with the ReLU activation. |
| 18 | **swiglu_residual** | 4.7632 | -0.21% | Adds an internal linear residual connection that bypasses the non-linear gating mechanism within the FFN. |
| 19 | **swiglu_narrow** | 4.7711 | -0.05% | Reduces the intermediate hidden dimension compared to the standard SwiGLU width expansion factor. |
| 20 | **baseline** | 4.7733 | +0.00% | Standard FFN using a squared ReLU activation function (`ReLU(x)**2`). |
| | **🔴 PROBABLY SKIP (Worse than Baseline)** | | | |
| 21 | **swiglu_shared_gate** | 4.7773 | +0.08% | Tied weights between gating and value projections (identical concept to singleprojglu approach). |
| 22 | **scispace_singlu** | 4.7852 | +0.25% | Replaces the continuous monotonic activation with a periodic Sine activation function. |
| 23 | **scispace_leakyglu** | 4.7931 | +0.41% | Replaces the SiLU gating activation with LeakyReLU. |
| 24 | **scispace_asymglu** | 4.7936 | +0.43% | Applies distinct activation functions respectively to the gating and the value branches before element-wise multiplication. |
| 25 | **swiglu_depth_init** | 4.8017 | +0.59% | Scales the standard deviation of the linear projection weight initializations inversely proportional to the layer depth. |
| 26 | **scispace_softplusglu** | 4.8038 | +0.64% | Replaces the SiLU gating activation with the Softplus activation function. |
| 27 | **scispace_tanhglu** | 4.8042 | +0.65% | Replaces the SiLU gating activation with the Tanh activation function. |
| 28 | **scispace_postgatelu** | 4.8050 | +0.66% | Applies layer normalization strictly to the output signal following the element-wise gating product. |
| 29 | **scispace_topkglu** | 4.8079 | +0.72% | Applies a deterministic Top-K magnitude hard-sparsity mask to the gating branch activations prior to multiplication. |
| 30 | **swiglu_mqa** | 4.8205 | +0.99% | Applies an architectural weight-sharing scheme to the FFN projections analogous to Multi-Query Attention grouped parameters. |
| 31 | **scispace_sincosglu** | 4.8360 | +1.31% | Applies parallel dual projections using Sine for the gating branch and Cosine for the value branch. |
| 32 | **swiglu_deep** | 4.8474 | +1.55% | Replaces the single standard SwiGLU block with two sequentially stacked, reduced-parameter SwiGLU blocks. |
| 33 | **scispace_sigmoidglu** | 4.8528 | +1.66% | Replaces the SiLU gating activation with a standard Sigmoid activation function. |
| 34 | **scispace_pregatelu** | 4.8614 | +1.84% | Applies normalization explicitly to the input vector of the gating branch prior to the non-linear activation function. |
| 35 | **swiglu_dual_gate** | 4.8659 | +1.94% | Employs two independent gating streams with SiLU activations, combining both via element-wise multiplication with the value stream. |
| 36 | **scispace_prenormdown** | 4.8727 | +2.08% | Applies an additional normalization layer immediately preceding the final down-projection weight matrix. |
| 37 | **scispace_laplaceglu** | 4.9130 | +2.93% | Replaces the SiLU gating activation with the cumulative distribution function of a Laplace distribution. |
| 38 | **swiglu_swiglu** | 4.9342 | +3.37% | Stacks two identical, full-capacity SwiGLU blocks sequentially within a single residual FFN layer block. |

---

### 💡 Key Takeaways from the Top Performers

- **Extra Normalization ("Sandwich Norm")**: Introducing an additional RMSNorm layer inside the SwiGLU block—specifically placed immediately after the element-wise product of the gate and value branches, but before the final linear projection—yielded the lowest validation loss (-2.20%).
- **Triple Projections**: Formulating the block with three independent linear projections combined via element-wise multiplication (`scispace_tripleprojglu`, -1.16%), rather than the standard two (gate and value), improves performance by increasing the representation capacity of the gating interaction.
- **Adjusting FFN Width**: Expanding the intermediate hidden dimensionality (`swiglu_full`, `swiglu_wide`) decreases validation loss by ~0.7% to ~1.06%. This directly trades increased parameter count for representation capacity.
- **Alternative Gating Activations**: Replacing the standard SiLU activation in the gating mechanism with alternatives such as HardSwish (`scispace_hardswishglu`, -1.00%) or Continuously Differentiable ELU (`scispace_celuglu`, -0.89%) can lower validation loss without increasing parameter count.
