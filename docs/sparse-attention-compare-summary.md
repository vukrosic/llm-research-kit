# Sparse Attention Comparison Summary

## Baseline vs Sparse

| Run | Params | Tokens | Progress | Val loss | Val acc | Active time | Avg tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5m dense` | `6,652,800` | `8,011,776 / 8,000,000` | `100%` | `5.4153` | `0.1997` | `61.7s` | `129.9k` |
| `5m sparse` | `6,685,824` | `8,011,776 / 8,000,000` | `100%` | `5.8466` | `0.1616` | `81.6s` | `98.2k` |
| `25m dense` | `25,366,272` | `25,001,984 / 25,000,000` | `100%` | `4.3581` | `0.2897` | `176.1s` | `142.0k` |
| `25m sparse` | `25,735,296` | `25,001,984 / 25,000,000` | `100%` | `5.2053` | `0.1964` | `491.4s` | `50.9k` |

## What To Notice

- The sparse runs are slower than the dense baselines.
- The `5m` sparse run also ends at a worse validation loss, so this is not just a speed issue.
- The `25m` sparse run finished normally, but it is still slower and ended worse on validation loss than dense.
- The parameter difference is small compared with the architecture change: the sparse `5m` model adds about `33k` parameters over dense.

## Plot

- Step/time comparison plot: `plots/attention_compare/5m_dense_vs_sparse_minimax.png`
- Step/token/time comparison plot: `plots/attention_compare/25m_dense_vs_sparse_minimax.png`
