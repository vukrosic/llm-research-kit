# QK Normalization Ablations at 100M Tokens

## Abstract
We evaluate QK-normalization variants in a controlled decoder-only Transformer setup trained on the same dataset and optimizer settings. The primary comparison uses 100M-token runs (`6104` steps), with two short 15M-token sanity runs for failure-mode checks. At 100M tokens, **K-only normalization** achieves the best final validation loss (`3.7345`), improving over the base QK-norm configuration by `-0.0224` loss points (`-0.60%`). **QK bias** is second-best in final loss (`3.7439`) and best in trajectory-averaged loss (AUC-normalized), but with higher training time. Shared norm and per-head scaling are nearly tied with base and do not improve final quality at this scale.

## 1. Setup

### 1.1 Model and data
- Architecture: decoder-only Transformer (`d_model=512`, `n_layers=22`, `n_heads=8`, `n_kv_heads=4`, `d_ff=2048`, `d_k=64`, `max_seq_len=2048`, `vocab_size=49152`).
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`.
- Dataset path: `./processed_data/pretrain_1B`.
- Split size observed in logs: `439,453` train sequences, `48,829` val sequences.

### 1.2 Optimization and schedule
- Batch size: `8`, grad accumulation: `1`.
- Optimizers: Muon + AdamW (`muon_lr=0.024`, `adamw_lr=0.006`, momentum `0.95`).
- Schedule: constant LR, no warmup (`warmup_ratio=0.0`).
- Regularization: weight decay `0.2`, dropout `0.0`, grad clip `1.0`.
- Precision: AMP enabled.

### 1.3 Ablations
- `qk_norm`: base QK RMS normalization before RoPE.
- `k_only_norm`: normalize keys only.
- `qk_bias`: add learnable bias to normalized Q and K.
- `shared_norm`: share one norm module for Q and K.
- `per_head_scaling`: learn per-head attention temperature.
- `no_qk_norm` and `rope_then_norm`: short 15M-token baselines for failure-mode checks.

## 2. Mathematical framing

Standard attention logits per head are:

$$
z_{ij}^{(h)}=\frac{q_i^{(h)}\cdot k_j^{(h)}}{\sqrt{d_k}}, \quad
\alpha_{ij}^{(h)}=\mathrm{softmax}_j(z_{ij}^{(h)}).
$$

RMSNorm on a vector \(x\in\mathbb{R}^{d_k}\):

$$
\mathrm{RMSNorm}(x)=\gamma\odot \frac{x}{\sqrt{\frac{1}{d_k}\sum_{m=1}^{d_k}x_m^2+\epsilon}}.
$$

A generic normalized-QK formulation used by the variants:

$$
\tilde{q}_i^{(h)}=R_{\theta_i}\big(N_q(q_i^{(h)})+b_q^{(h)}\big), \quad
\tilde{k}_j^{(h)}=R_{\theta_j}\big(N_k(k_j^{(h)})+b_k^{(h)}\big),
$$
$$
z_{ij}^{(h)}=\alpha_h\frac{\tilde{q}_i^{(h)}\cdot\tilde{k}_j^{(h)}}{\sqrt{d_k}},
$$

where:
- \(N_q, N_k\) are either RMSNorm or identity (K-only sets \(N_q=I\)).
- \(b_q, b_k\) are zero unless using QK bias.
- \(\alpha_h=1\) except in per-head scaling.
- \(R_{\theta}\) is RoPE.

### Why normalization helps
If unnormalized components are approximately independent with variances \(\sigma_q^2,\sigma_k^2\), then:

$$
\mathrm{Var}(q\cdot k)\approx d_k \sigma_q^2\sigma_k^2.
$$

So logit scale can drift as activations grow, causing overly sharp softmax and low-entropy attention. QK normalization controls this scale by stabilizing query/key norms before dot products, reducing logit variance drift during training.

## 3. Results

### 3.1 Main trajectories

![All runs validation loss](plots/paper_fig1_all_runs_val_loss.png)

Figure 1 overlays all available runs. Solid lines are 100M-token runs; dashed lines are 15M-token runs.

![100M zoomed late-stage validation loss](plots/paper_fig2_100M_zoom.png)

Figure 2 zooms into late training (`step >= 1000`) to expose small but consistent differences.

### 3.2 Final metrics at 100M tokens

![100M final validation loss ranking](plots/paper_fig3_100M_final_val_loss_bar.png)

| Variant | Final Val Loss | Delta vs QK-Norm Base | Final PPL | Final Val Acc | Active Train Time (min) |
|---|---:|---:|---:|---:|---:|
| K-only Norm | 3.7345 | -0.0224 (-0.60%) | 41.865 | 0.3478 | 14.87 |
| QK Bias | 3.7439 | -0.0130 (-0.35%) | 42.264 | 0.3465 | 16.87 |
| QK Norm (Base) | 3.7569 | 0.0000 (0.00%) | 42.815 | 0.3455 | 15.29 |
| Shared Norm | 3.7579 | +0.0010 (+0.03%) | 42.858 | 0.3447 | 15.27 |
| Per-Head Scaling | 3.7588 | +0.0019 (+0.05%) | 42.895 | 0.3453 | 15.35 |

### 3.3 Relative dynamics and efficiency

![Delta vs base](plots/paper_fig4_100M_delta_vs_base.png)

Figure 4 shows loss differences relative to QK-norm base. QK bias is best through step `2000`, then K-only becomes best from step `3000` onward.

![Efficiency vs quality](plots/paper_fig5_100M_efficiency.png)

Figure 5 shows a quality/time tradeoff. QK bias improves loss but takes ~`10.36%` more active training time than base. K-only is both faster (`-2.70%`) and better.

![AUC-normalized trajectory ranking](plots/paper_fig6_100M_auc_bar.png)

Figure 6 ranks mean trajectory loss (AUC-normalized). QK bias is best by this metric (`4.3029`), with K-only nearly tied (`4.3037`), consistent with QK bias’s stronger early-phase behavior.

### 3.4 Short-run (15M) dropped ablations

![15M short-run comparison](plots/paper_fig7_15M_shortrun_bar.png)

At 15M tokens, `rope_then_norm` (4.4209) is better than `no_qk_norm` (4.5175), but both are materially worse than the 100M family and are not competitive long-run candidates from these results alone.

## 4. Interpretation

- **Best final quality at 100M**: K-only normalization.
- **Best early trajectory / AUC**: QK bias.
- **No gain at this scale**: per-head scaling and shared norm.
- **Cross-over pattern**: QK bias leads earlier; K-only leads later.

This supports a useful split:
- If the budget is fixed on final quality, K-only is the strongest current setting.
- If earlier quality per step matters (e.g., short-horizon selection), QK bias is attractive despite higher compute.

## 5. Limitations

- Single run per variant (no multi-seed confidence intervals).
- Uneven token budgets across all experiments (100M main set, 15M short set).
- Results are specific to this architecture, optimizer stack, and dataset shard.

## 6. Reproducibility

Generate all figures and summary table from checkpoints:

```bash
python generate_paper_plots.py
```

Primary artifact table source:
- `plots/paper_results_summary.csv`

