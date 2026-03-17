# Frontier Architecture Leaderboard

## 5-Minute Training Budget (Primary Benchmark)

**Transformer baseline**: val_loss = 3.784 (5 min, separate training infrastructure)

| Rank | Architecture | Family | val_loss | Δ vs transformer | Params | tok/s | Batch |
|------|-------------|--------|----------|------------------|--------|-------|-------|
| 1 | V10_09 VR@10 alpha=0.8 | conv+attn | **3.665** | **-0.119 (-3.1%)** | 124M | 35,500 | V10 |
| 2 | V9_13 d640 VR@10 | conv+attn | 3.668 | -0.116 | 124M | 35,681 | V9 |
| 3 | V11_08 Control | conv+attn | 3.676 | -0.108 | 124M | 35,630 | V11 |
| 4 | V11_04 2VR+ConvVR | conv+attn | 3.680 | -0.104 | 124M | 34,536 | V11 |
| 5 | V12_09 WideFFN@attn | conv+attn | 3.683 | -0.101 | 129M | 34,761 | V12 |
| 6 | V8_13 d640 VR end | conv+attn | 3.684 | -0.100 | 124M | 35,842 | V8 |
| 7 | V12_08 TwoVR@5,10 | conv+attn | 3.686 | -0.098 | 124M | 34,565 | V12 |
| 8 | V8_01 VR mid d=512 | conv+attn | 3.698 | -0.086 | 107M | 35,072 | V8 |
| 9 | V12_01 MultiQuery4Q | conv+attn | 3.706 | -0.078 | 124M | 33,633 | V12 |
| 10 | V7_08 VR end d=512 | conv+attn | 3.713 | -0.071 | 107M | 34,867 | V7 |
| — | **Transformer baseline** | transformer | **3.784** | **0.000** | ~107M | ~35,000 | — |
| 11 | V6_06 SingleHead | conv+attn | 3.768 | -0.016 | 107M | 34,685 | V6 |
| 12 | V2_04 GatedMHConv | pure conv | 3.915 | +0.131 | 107M | 34,000 | V2 |

### Key
- **conv+attn**: GatedMHConv layers + SingleHead attention with value residual
- **pure conv**: GatedMHConv only (no attention)
- All results at 5-minute training budget on same GPU, same data pipeline
- Results within 0.002 of each other are ties (noise floor)

## 12M Token Training Budget (Legacy)

| Rank | Architecture | val_loss | Δ vs transformer | Params |
|------|-------------|----------|------------------|--------|
| 1 | novel_value_residual | 3.449 | +0.000 | 107M |
| 2 | novel_conv_ts_qknorm | 3.558 | +0.109 | 107M |
| 3 | novel_differential_gqa | 3.567 | +0.118 | 110M |

## History

Last updated: 2026-03-17
- V10-V12 results added (5-min benchmark)
- Comprehensive leaderboard reorganized by training budget
