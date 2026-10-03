# Final ETH/UCY Benchmark

Formal held-out results do not reproduce the clean-validation social gain: EMT-SR vs EMT increases ADE by 3.0380% and FDE by 3.3809%, with 7/15 wins in each metric. The largest degradation is on HOTEL; ZARA2 also degrades. These frozen results are reported unchanged, with no retuning or repeat test.

Formal held-out test, K=1, 8 observed / 12 predicted steps at 2.5 Hz, world meters. All four source scenes used in full. No internal validation remains in final training.

Protocol SHA256: `e9240c2f042282e370f007ff9310c39a21fcf747e3b93a0f96317960f9f28914`. All 60 models finished before test access. Exactly 60 logged one-pass evaluations; no test-based model/checkpoint/epoch selection.

## Main table

| Held-out | ETT ADE / FDE (m) | EMT ADE / FDE (m) | ETT-SR ADE / FDE (m) | EMT-SR ADE / FDE (m) |
|---|---|---|---|---|
| ETH | 1.047690 ± 0.012950 / 2.054902 ± 0.027432 | 1.079633 ± 0.011883 / 2.149197 ± 0.019547 | 1.089215 ± 0.048887 / 2.112959 ± 0.102565 | 1.064610 ± 0.016549 / 2.086750 ± 0.035675 |
| HOTEL | 0.549299 ± 0.019380 / 1.091970 ± 0.028694 | 0.529726 ± 0.024724 / 1.077778 ± 0.063254 | 0.499369 ± 0.020872 / 1.010440 ± 0.050409 | 0.641549 ± 0.065610 / 1.332837 ± 0.135580 |
| UNIV | 0.607979 ± 0.001339 / 1.281375 ± 0.005537 | 0.586250 ± 0.007647 / 1.251840 ± 0.011682 | 0.602789 ± 0.006017 / 1.278637 ± 0.013809 | 0.584229 ± 0.001378 / 1.254103 ± 0.001683 |
| ZARA1 | 0.485153 ± 0.062594 / 1.013721 ± 0.095029 | 0.454267 ± 0.041245 / 0.977951 ± 0.087291 | 0.425836 ± 0.007396 / 0.909743 ± 0.008944 | 0.433379 ± 0.010134 / 0.937519 ± 0.018118 |
| ZARA2 | 0.354709 ± 0.009346 / 0.761085 ± 0.012712 | 0.351370 ± 0.004855 / 0.748822 ± 0.005429 | 0.363600 ± 0.017918 / 0.784603 ± 0.050861 | 0.368658 ± 0.016278 / 0.804185 ± 0.036597 |
| Average | 0.608966 / 1.240611 | 0.600249 / 1.241118 | 0.596162 / 1.219276 | 0.618485 / 1.283079 |

Per fold: three-seed mean ± sample SD. Average: mean over seeds within fold, then equal weight over five folds. Average has no pooled-window weighting and no fabricated aggregate SD.

## Comparisons

| Comparison (treated − control) | ΔADE (m) | ΔFDE (m) | ADE % | FDE % | ADE wins /15 | FDE wins /15 |
|---|---|---|---|---|---|---|
| EMT vs ETT | -0.008717 | +0.000507 | -1.4314 | +0.0409 | 8 | 9 |
| EMT-SR vs EMT | +0.018236 | +0.041961 | +3.0380 | +3.3809 | 7 | 7 |
| EMT-SR vs ETT-SR | +0.022323 | +0.063803 | +3.7444 | +5.2328 | 6 | 5 |

Negative differences favor the treated model. Paired wins compare the same fold and seed. No new GO/STOP or significance tests are applied.

## Frozen epochs

| Fold | ETT | EMT | ETT-SR | EMT-SR |
|---|---|---|---|---|
| ETH | 38 | 10 | 10 | 14 |
| HOTEL | 11 | 14 | 11 | 15 |
| UNIV | 21 | 12 | 13 | 4 |
| ZARA1 | 30 | 12 | 34 | 19 |
| ZARA2 | 27 | 14 | 30 | 16 |

Epochs are the median of the three prior validation best_epoch values for each fold/model. Fixed AdamW lr=1e-3, weight decay=1e-4, batch 128, SmoothL1 beta=1, clip=5. No scheduler: the old validation-driven plateau monitor does not exist in full-source training. The last frozen epoch is saved, without early stopping.

## Runtime

| Model | Total params | Trainable | B=1 ms | B=128 ms | Mean train peak MiB | Max train peak MiB | Mean B=128 inference peak MiB |
|---|---|---|---|---|---|---|---|
| ETT | 616344 | 616344 | 0.795524 | 0.836066 | 58.249 | 58.249 | 16.164 |
| EMT | 370968 | 370968 | 1.448930 | 1.562408 | 67.354 | 67.354 | 21.062 |
| ETT-SR | 722736 | 106392 | 2.679340 | 4.677900 | 162.916 | 193.698 | 49.315 |
| EMT-SR | 477360 | 106392 | 4.034381 | 13.653183 | 162.253 | 193.535 | 101.611 |

Runtime uses exclusive GPU synchronized wall time, 20 warmups +100 full uncached forwards, batch 1/128, N=8 for SR. Diagnostic attention export is excluded. Peak memory is per-process PyTorch allocated memory; social training includes frozen context caches. Training times include four-process GPU contention. Runtime does not by itself support a general speed claim.

## Audit and artifacts

Historical files unchanged: 932. NaN/Inf=false. Maximum pre-clip gradient norm: 8.348210. All social backbones frozen/eval and initial predictions equal their corresponding final base.

All saved predictions recompute ADE/FDE exactly. Prediction NPZs include full IDs/frames/ground truth and social neighbors/attention/residual/base outputs. Visualization candidates are saved without drawing figures.

Validation EMT-ZR remains a separate, explicitly labeled capacity/social-information ablation. It is not mixed into this formal test table. Formal EMT-SR vs EMT combines capacity and social effects.

Full 30-item Chinese handoff: brain_report.md. Test log: data_audit/test_access_log.json. No tuning, additional models, statistical tests or plots follow this benchmark.
