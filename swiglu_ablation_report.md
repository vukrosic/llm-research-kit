# SwiGLU vs Squared ReLU: 10M Token Training Progress

### 📊 Current Progress: **26/38** experiments completed!

## 🚀 Friendly Recap for the Squad
Hey friends! I am running a massive ablation study to find the absolute best way to build Transformers. We are currently testing **38 different architectures** at 10 million tokens each. Here is a breakdown of what we have found so far:

- **Rank 1: swiglu_sandwich** (-2.20% vs Baseline) - SwiGLU with extra RMSNorm inside the block. It prevents values from exploding and really helps convergence! 🥪
- **Rank 2: swiglu_full** (-1.06% vs Baseline) - A wider SwiGLU block that uses more parameters to capture deeper patterns. Heavy but powerful! 🚀
- **Rank 3: scispace_hardswishglu** (-1.00% vs Baseline) - A faster, linear approximation of Swish/SiLU for mobile-friendly efficiency.
- **Rank 4: scispace_celuglu** (-0.89% vs Baseline) - A continuously differentiable version of ELU for smoother gradients.
- **Rank 5: swiglu** (-0.81% vs Baseline) - The classic SiLU-gated linear unit. The industry standard for many LLMs today.

## 🏆 Full Leaderboard
| Rank | Experiment | Val Loss | Val Acc | Delta % | Status |
|:-----|:-----------|:---------|:--------|:--------|:-------|
| 1 | swiglu_sandwich |   4.6682 |  0.2617 |   -2.20% | ✅ Complete |
| 2 | swiglu_full |   4.7226 |  0.2548 |   -1.06% | ✅ Complete |
| 3 | scispace_hardswishglu |   4.7255 |  0.2555 |   -1.00% | ✅ Complete |
| 4 | scispace_celuglu |   4.7309 |  0.2539 |   -0.89% | ✅ Complete |
| 5 | swiglu |   4.7347 |  0.2541 |   -0.81% | ✅ Complete |
| 6 | swiglu_bias |   4.7352 |  0.2536 |   -0.80% | ✅ Complete |
| 7 | swiglu_wide |   4.7375 |  0.2537 |   -0.75% | ✅ Complete |
| 8 | swiglu_parallel |   4.7396 |  0.2534 |   -0.71% | ✅ Complete |
| 9 | scispace_eluglu |   4.7443 |  0.2522 |   -0.61% | ✅ Complete |
| 10 | geglu |   4.7444 |  0.2532 |   -0.61% | ✅ Complete |
| 11 | swiglu_3q |   4.7486 |  0.2518 |   -0.52% | ✅ Complete |
| 12 | reglu |   4.7613 |  0.2507 |   -0.25% | ✅ Complete |
| 13 | swiglu_residual |   4.7632 |  0.2515 |   -0.21% | ✅ Complete |
| 14 | swiglu_narrow |   4.7711 |  0.2502 |   -0.05% | ✅ Complete |
| 15 | **baseline** |   4.7733 |  0.2492 |   +0.00% | ✅ Complete |
| 16 | swiglu_shared_gate |   4.7773 |  0.2492 |   +0.08% | ✅ Complete |
| 17 | scispace_singlu |   4.7852 |  0.2506 |   +0.25% | ✅ Complete |
| 18 | swiglu_depth_init |   4.8017 |  0.2494 |   +0.59% | ✅ Complete |
| 19 | scispace_softplusglu |   4.8038 |  0.2468 |   +0.64% | ✅ Complete |
| 20 | scispace_tanhglu |   4.8042 |  0.2479 |   +0.65% | ✅ Complete |
| 21 | swiglu_mqa |   4.8205 |  0.2450 |   +0.99% | ✅ Complete |
| 22 | swiglu_deep |   4.8474 |  0.2446 |   +1.55% | ✅ Complete |
| 23 | scispace_sigmoidglu |   4.8528 |  0.2445 |   +1.66% | ✅ Complete |
| 24 | swiglu_dual_gate |   4.8659 |  0.2424 |   +1.94% | ✅ Complete |
| 25 | scispace_laplaceglu |   4.9130 |  0.2433 |   +2.93% | ✅ Complete |
| 26 | swiglu_swiglu |   4.9342 |  0.2386 |   +3.37% | ✅ Complete |
| - | scispace_sincosglu   |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_singleprojglu |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_tripleprojglu |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_pregatelu   |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_postgatelu  |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_topkglu     |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_leakyglu    |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_asymglu     |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_prenormdown |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_scalegate   |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_composite   |        - |       - |       - | ⏳ Running/Pending |
| - | scispace_moelite     |        - |       - |       - | ⏳ Running/Pending |


## 🧪 The Full Lab Notebook (Detailed Definitions)
- **baseline**: The standard Squared ReLU activation we use as our baseline. Tried and true, but can we beat it?
- **swiglu**: The classic SiLU-gated linear unit. The industry standard for many LLMs today.
- **swiglu_narrow**: Experimental variant.
- **swiglu_3q**: Experimental variant.
- **swiglu_full**: A wider SwiGLU block that uses more parameters to capture deeper patterns. Heavy but powerful! 🚀
- **swiglu_wide**: Similar to full, but optimized for stability. Currently one of our top performers.
- **geglu**: GELU-gated linear unit. Smoother than ReLU, used in models like PaLM.
- **reglu**: ReLU-gated linear unit. Simple, sparse, and surprisingly effective.
- **swiglu_dual_gate**: Double the gates, double the fun! Uses two independent SiLU streams.
- **swiglu_residual**: Experimental variant.
- **swiglu_shared_gate**: Experimental variant.
- **swiglu_deep**: Experimental variant.
- **swiglu_swiglu**: Two SwiGLU blocks stacked back-to-back inside one FFN layer.
- **swiglu_bias**: Experimental variant.
- **swiglu_parallel**: Running the FFN in parallel with Attention (PaLM style). Speeds up training steps!
- **swiglu_sandwich**: SwiGLU with extra RMSNorm inside the block. It prevents values from exploding and really helps convergence! 🥪
- **swiglu_mqa**: Experimental variant.
- **swiglu_depth_init**: Experimental variant.
- **scispace_singlu**: Gating with a Sine wave! Sounds crazy, but it helps the model learn periodic patterns.
- **scispace_tanhglu**: Gating with Tanh. Old school but very stable for deep networks.
- **scispace_sigmoidglu**: Standard Sigmoid gating, similar to classic GLU.
- **scispace_softplusglu**: Softplus gating for a smoother, always-positive activation.
- **scispace_eluglu**: Uses the ELU activation for gating. Great for handling negative values smoothly. 🧪
- **scispace_celuglu**: A continuously differentiable version of ELU for smoother gradients.
- **scispace_hardswishglu**: A faster, linear approximation of Swish/SiLU for mobile-friendly efficiency.
- **scispace_laplaceglu**: Biology-inspired gating based on the Laplace distribution.
- **scispace_sincosglu**: Uses both Sin and Cos projections to capture complex frequency data.
- **scispace_singleprojglu**: Shares weights between gate and value branches to save 33% parameters!
- **scispace_tripleprojglu**: Uses three complete sets of weights for maximum expressive power.
- **scispace_pregatelu**: Normalizes the signal BEFORE the gate to keep things from getting too wild.
- **scispace_postgatelu**: Normalizes the signal AFTER the product for super clean outputs.
- **scispace_topkglu**: Only the strongest neurons fire (Top-K sparsity). Very efficient for large models! 🧠
- **scispace_leakyglu**: A leaky version of the gate so neurons never fully die out.
- **scispace_asymglu**: Mixes two different activation types to get the best of both worlds.
- **scispace_prenormdown**: Adds an extra layer of stability right before the final projection.
- **scispace_scalegate**: Lets the model learn exactly how much to scale each gate output.
- **scispace_composite**: A hybrid gate that combines the strengths of SiLU and Tanh.
- **scispace_moelite**: A tiny version of Mixture-of-Experts. Routes information dynamically. 🚦
