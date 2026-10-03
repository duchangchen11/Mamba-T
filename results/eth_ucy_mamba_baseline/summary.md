# ETH/UCY matched baseline validation

Internal decision: **STOP**. Source-scene validation only; held-out test accessed: **false**.

Five-fold equal-weight means:

- ETT: ADE 0.445898 m; FDE 0.942888 m.
- EMT: ADE 0.446307 m; FDE 0.944740 m.
- EMT relative changes: ADE +0.092%; FDE +0.196%.
- EMT paired wins: ADE 7/15; FDE 6/15; both 6/15.

| Fold | ETT ADE | EMT ADE | ΔADE | ETT FDE | EMT FDE | ΔFDE |
|---|---|---|---|---|---|---|
| ETH | 0.44989 ± 0.00138 | 0.45554 ± 0.00225 | +0.00566 | 0.95944 ± 0.00274 | 0.97091 ± 0.00804 | +0.01148 |
| HOTEL | 0.46504 ± 0.00185 | 0.46693 ± 0.00056 | +0.00189 | 0.98825 ± 0.00510 | 0.99363 ± 0.00387 | +0.00538 |
| UNIV | 0.33321 ± 0.00277 | 0.33305 ± 0.00171 | -0.00016 | 0.68253 ± 0.00688 | 0.68357 ± 0.00622 | +0.00104 |
| ZARA1 | 0.46781 ± 0.00264 | 0.46545 ± 0.00142 | -0.00236 | 0.99747 ± 0.00689 | 0.99545 ± 0.00276 | -0.00202 |
| ZARA2 | 0.51354 ± 0.00272 | 0.51055 ± 0.00335 | -0.00299 | 1.08675 ± 0.00404 | 1.08014 ± 0.00276 | -0.00661 |

Mean ± sample SD across three seeds; Δ = EMT − ETT. Negative favors EMT.

| Fold | Seed | ETT ADE | EMT ADE | ΔADE | ETT FDE | EMT FDE | ΔFDE |
|---|---|---|---|---|---|---|---|
| eth | 42 | 0.449600 | 0.454713 | 0.005113 | 0.958002 | 0.964892 | 0.006889 |
| eth | 123 | 0.451385 | 0.453825 | 0.002440 | 0.962596 | 0.967809 | 0.005213 |
| eth | 2024 | 0.448674 | 0.458093 | 0.009419 | 0.957712 | 0.980044 | 0.022332 |
| hotel | 42 | 0.465835 | 0.466940 | 0.001105 | 0.988899 | 0.992438 | 0.003539 |
| hotel | 123 | 0.462927 | 0.466371 | 0.003444 | 0.982849 | 0.990488 | 0.007639 |
| hotel | 2024 | 0.466359 | 0.467493 | 0.001135 | 0.992991 | 0.997949 | 0.004958 |
| univ | 42 | 0.334469 | 0.334149 | -0.000320 | 0.687584 | 0.677880 | -0.009704 |
| univ | 123 | 0.335133 | 0.333929 | -0.001204 | 0.685318 | 0.690215 | 0.004897 |
| univ | 2024 | 0.330035 | 0.331085 | 0.001049 | 0.674688 | 0.682616 | 0.007928 |
| zara1 | 42 | 0.466944 | 0.465207 | -0.001737 | 0.994244 | 0.992948 | -0.001296 |
| zara1 | 123 | 0.470773 | 0.464167 | -0.006606 | 1.005388 | 0.995000 | -0.010388 |
| zara1 | 2024 | 0.465719 | 0.466981 | 0.001262 | 0.992787 | 0.998411 | 0.005624 |
| zara2 | 42 | 0.512778 | 0.512331 | -0.000447 | 1.086908 | 1.083310 | -0.003598 |
| zara2 | 123 | 0.516562 | 0.512635 | -0.003927 | 1.090714 | 1.078843 | -0.011871 |
| zara2 | 2024 | 0.511275 | 0.506680 | -0.004595 | 1.082635 | 1.078264 | -0.004371 |

## Data audit

| Scene | Raw pedestrians | Eligible pedestrians | Windows |
|---|---|---|---|
| eth | 360 | 44 | 364 |
| hotel | 389 | 122 | 1197 |
| univ | 849 | 722 | 24334 |
| zara1 | 148 | 142 | 2356 |
| zara2 | 204 | 189 | 5910 |

| Fold | Train pedestrians | Validation pedestrians | Train windows | Validation windows |
|---|---|---|---|---|
| eth | 1015 | 160 | 29831 | 3966 |
| hotel | 947 | 150 | 29086 | 3878 |
| univ | 428 | 69 | 8631 | 1196 |
| zara1 | 930 | 147 | 28077 | 3728 |
| zara2 | 880 | 150 | 24891 | 3360 |

## Runtime

Maximum pre-clip gradient norm: 1.636078. NaN/Inf: false.

| Fold | Model | Seed | Parameters | Best epoch | Seconds | Peak allocated MiB | Single latency ms | Batch 128 latency ms |
|---|---|---|---|---|---|---|---|---|
| eth | ETT | 42 | 616344 | 63 | 418.08 | 58.25 | 0.793 | 0.836 |
| eth | EMT | 42 | 370968 | 19 | 178.66 | 67.35 | 1.427 | 1.540 |
| eth | ETT | 123 | 616344 | 38 | 275.82 | 58.25 | 0.774 | 0.814 |
| eth | EMT | 123 | 370968 | 6 | 110.03 | 67.35 | 1.430 | 1.535 |
| eth | ETT | 2024 | 616344 | 31 | 236.60 | 58.25 | 0.775 | 0.815 |
| eth | EMT | 2024 | 370968 | 10 | 130.53 | 67.35 | 1.428 | 1.545 |
| hotel | ETT | 42 | 616344 | 11 | 115.78 | 58.62 | 0.775 | 0.822 |
| hotel | EMT | 42 | 370968 | 11 | 112.56 | 67.35 | 1.430 | 1.546 |
| hotel | ETT | 123 | 616344 | 19 | 158.04 | 58.62 | 0.775 | 0.815 |
| hotel | EMT | 123 | 370968 | 14 | 145.17 | 67.35 | 1.427 | 1.540 |
| hotel | ETT | 2024 | 616344 | 7 | 101.14 | 58.25 | 0.776 | 0.814 |
| hotel | EMT | 2024 | 370968 | 20 | 177.99 | 67.35 | 1.431 | 1.556 |
| univ | ETT | 42 | 616344 | 21 | 52.16 | 58.25 | 0.786 | 0.823 |
| univ | EMT | 42 | 370968 | 12 | 47.32 | 67.35 | 1.429 | 1.543 |
| univ | ETT | 123 | 616344 | 36 | 84.78 | 58.62 | 0.768 | 0.818 |
| univ | EMT | 123 | 370968 | 12 | 47.07 | 67.35 | 1.427 | 1.545 |
| univ | ETT | 2024 | 616344 | 21 | 54.89 | 58.62 | 0.776 | 0.817 |
| univ | EMT | 2024 | 370968 | 34 | 79.82 | 67.35 | 1.426 | 1.553 |
| zara1 | ETT | 42 | 616344 | 30 | 230.11 | 58.25 | 0.774 | 0.816 |
| zara1 | EMT | 42 | 370968 | 7 | 111.93 | 67.35 | 1.428 | 1.545 |
| zara1 | ETT | 123 | 616344 | 22 | 183.76 | 58.62 | 0.775 | 0.817 |
| zara1 | EMT | 123 | 370968 | 12 | 134.44 | 67.35 | 1.436 | 1.550 |
| zara1 | ETT | 2024 | 616344 | 36 | 254.61 | 58.62 | 0.775 | 0.816 |
| zara1 | EMT | 2024 | 370968 | 15 | 148.18 | 67.35 | 1.425 | 1.553 |
| zara2 | ETT | 42 | 616344 | 27 | 177.47 | 58.25 | 1.289 | 0.834 |
| zara2 | EMT | 42 | 370968 | 15 | 120.86 | 67.35 | 1.422 | 1.545 |
| zara2 | ETT | 123 | 616344 | 8 | 91.42 | 58.25 | 0.774 | 0.814 |
| zara2 | EMT | 123 | 370968 | 14 | 108.84 | 67.35 | 1.426 | 1.544 |
| zara2 | ETT | 2024 | 616344 | 53 | 280.35 | 58.25 | 0.771 | 0.818 |
| zara2 | EMT | 2024 | 370968 | 10 | 77.22 | 67.35 | 1.431 | 1.539 |

Latency uses 20 warmup forwards and 100 synchronized eval forwards; GPU memory is peak allocated memory during training/validation.

## Reproduction

Environment: Python 3.10.21; Torch 2.5.1+cu124; CUDA runtime 12.4; NVIDIA GeForce RTX 3080; mamba-ssm 2.2.6.post3.

Data source: [Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/datasets/raw/all_data). SHA256 and exact URLs: `data_audit/data_provenance.json`. Raw provenance: `/home/lrj/Mamba-T/data/raw/eth_ucy/DATA_PROVENANCE.md`.

UNIV uses students001 and students003 only. Pedestrian IDs include the recording name. All eligible contiguous track windows use stride 1. No interpolation or future-derived input. Hash threshold gives approximately 85/15 tracks, not an exact count quota.

GO requires >=2% ADE improvement and >=10 ADE paired wins, or >=3% FDE improvement and >=10 FDE paired wins; neither metric may worsen >10% on any fold. Validation selection/decision is exploratory and does not establish held-out benchmark superiority.

No hyperparameter search or stage 2 is initiated.

Final verification: 23 pytest tests passed; all required py_compile checks passed. Full requested 30-item handoff: `brain_report.md`.
