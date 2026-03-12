Hello 👋

here are 19+ experiments to run on your LLM / transformer to accelerate the training.

I ran 38 experiments on an 88M param LLM, each time training for 10M tokens.

---

### 📊 All 38 Experiments Ranked (10M Tokens)
*(Ranked by Validation Loss against a Squared ReLU Baseline)*

| Rank | Experiment | Val Loss | Delta % | Idea Concept |
|:-----|:-----------|:---------|:--------|:-------------|
| | **🟢 IDEAS TO TRY (Beat Baseline)** | | | |
| 1 | **swiglu_sandwich** | 4.6682 | -2.20% | SwiGLU with extra RMSNorm inside the block. |
| 2 | **scispace_tripleprojglu** | 4.7180 | -1.16% | Uses three complete sets of weights for maximum expressive power. |
| 3 | **swiglu_full** | 4.7226 | -1.06% | A wider SwiGLU block that uses more parameters to capture deeper patterns. |
| 4 | **scispace_hardswishglu** | 4.7255 | -1.00% | A faster, linear approximation of Swish/SiLU for mobile-friendly efficiency. |
| 5 | **scispace_celuglu** | 4.7309 | -0.89% | A continuously differentiable version of ELU for smoother gradients. |
| 6 | **swiglu** | 4.7347 | -0.81% | The classic SiLU-gated linear unit. The industry standard for many LLMs today. |
| 7 | **swiglu_bias** | 4.7352 | -0.80% | Standard SwiGLU but adds biases to all linear projections. |
| 8 | **swiglu_wide** | 4.7375 | -0.75% | Similar to full, but optimized for stability. Currently one of our top performers. |
| 9 | **swiglu_parallel** | 4.7396 | -0.71% | Running the FFN in parallel with Attention (PaLM style). Speeds up training steps! |
| 10 | **scispace_eluglu** | 4.7443 | -0.61% | Uses the ELU activation for gating. Great for handling negative values smoothly. 🧪 |
| 11 | **scispace_moelite** | 4.7443 | -0.61% | A tiny version of Mixture-of-Experts. Routes information dynamically. 🚦 |
| 12 | **geglu** | 4.7444 | -0.61% | GELU-gated linear unit. Smoother than ReLU, used in models like PaLM. |
| 13 | **scispace_composite** | 4.7457 | -0.58% | A hybrid gate that combines the strengths of SiLU and Tanh. |
| 14 | **swiglu_3q** | 4.7486 | -0.52% | Three-quarter width hidden state, a solid middle ground. |
| 15 | **scispace_scalegate** | 4.7502 | -0.48% | Lets the model learn exactly how much to scale each gate output. |
| 16 | **scispace_singleprojglu** | 4.7569 | -0.34% | Shares weights between gate and value branches to save 33% parameters! |
| 17 | **reglu** | 4.7613 | -0.25% | ReLU-gated linear unit. Simple, sparse, and surprisingly effective. |
| 18 | **swiglu_residual** | 4.7632 | -0.21% | Adds an internal linear residual connection straight through the FFN. |
| 19 | **swiglu_narrow** | 4.7711 | -0.05% | A restricted width version that is parameter-efficient but struggles to learn complex representations. |
| 20 | **baseline** | 4.7733 | +0.00% | *Squared ReLU Baseline* |
| | **🔴 PROBABLY SKIP (Worse than Baseline)** | | | |
| 21 | **swiglu_shared_gate** | 4.7773 | +0.08% | Shares weights between the gating and value branch for parameter savings. |
| 22 | **scispace_singlu** | 4.7852 | +0.25% | Gating with a Sine wave! Sounds crazy, but it helps the model learn periodic patterns. |
| 23 | **scispace_leakyglu** | 4.7931 | +0.41% | A leaky version of the gate so neurons never fully die out. |
| 24 | **scispace_asymglu** | 4.7936 | +0.43% | Mixes two different activation types to get the best of both worlds. |
| 25 | **swiglu_depth_init** | 4.8017 | +0.59% | Scales the initialization weights based on layer depth. |
| 26 | **scispace_softplusglu** | 4.8038 | +0.64% | Softplus gating for a smoother, always-positive activation. |
| 27 | **scispace_tanhglu** | 4.8042 | +0.65% | Gating with Tanh. Old school but very stable for deep networks. |
| 28 | **scispace_postgatelu** | 4.8050 | +0.66% | Normalizes the signal AFTER the product for super clean outputs. |
| 29 | **scispace_topkglu** | 4.8079 | +0.72% | Only the strongest neurons fire (Top-K sparsity). Very efficient for large models! 🧠 |
| 30 | **swiglu_mqa** | 4.8205 | +0.99% | A SwiGLU variant tailored for Multi-Query Attention. |
| 31 | **scispace_sincosglu** | 4.8360 | +1.31% | Uses both Sin and Cos projections to capture complex frequency data. |
| 32 | **swiglu_deep** | 4.8474 | +1.55% | Stacking multiple SwiGLU sub-blocks sequentially. Looks good on paper, but optimizations get difficult. |
| 33 | **scispace_sigmoidglu** | 4.8528 | +1.66% | Standard Sigmoid gating, similar to classic GLU. |
| 34 | **scispace_pregatelu** | 4.8614 | +1.84% | Normalizes the signal BEFORE the gate to keep things from getting too wild. |
| 35 | **swiglu_dual_gate** | 4.8659 | +1.94% | Double the gates, double the fun! Uses two independent SiLU streams. |
| 36 | **scispace_prenormdown** | 4.8727 | +2.08% | Adds an extra layer of stability right before the final projection. |
| 37 | **scispace_laplaceglu** | 4.9130 | +2.93% | Biology-inspired gating based on the Laplace distribution. |
| 38 | **swiglu_swiglu** | 4.9342 | +3.37% | Two SwiGLU blocks stacked back-to-back inside one FFN layer. |

---

### 💡 Key Takeaways from the Top Performers

- **Extra Normalization ("Sandwich Norm")**: Try adding an extra layer of RMSNorm *inside* the SwiGLU block. In my test, **`swiglu_sandwich`** performed best overall (-2.20%). It significantly stabilizes the gradients by normalizing the signal right after the gating product.
- **Triple Projections**: Instead of the standard two projections (Gate and Value) that are then multiplied, try decoupling the output projection completely. **`scispace_tripleprojglu`** (-1.16%) uses three independent dense layers for maximum expressive power.
- **Adjusting FFN Width**: Play with the hidden size dimension. Going wider (**`swiglu_full`** or **`swiglu_wide`**) gave ~0.7% to ~1% improvement. Giving the FFN more parameter capacity usually helps capture deeper patterns quicker, assuming you have the compute budget.
- **Computationally Efficient Activations**: Swap out the SiLU gate for something friendlier to hardware. Both **`scispace_hardswishglu`** (-1.00%) and **`scispace_celuglu`** (Continuously Differentiable ELU, -0.89%) outperformed the classic SiLU gate. They flow gradients better and run faster on certain hardware accelerators.

Hope this sparks some ideas for your next ablation study! Let me know what FFN setups you've had success with. 📉✨
