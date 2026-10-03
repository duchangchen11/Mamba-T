# Clean Social Validation on ETH/UCY

**CLEAN SOCIAL: GO**. Source-scene validation only; heldout_test_accessed=false.

Train neighbors must belong to the same scene’s frozen train target-pedestrian set, the same recording, and be visible in all eight observed frames. Selection excludes the target and ranks by last-observation distance. N=8, no radius. Validation uses all observation-visible neighbors and is identical to the previous stage. Audit-only split labels do not enter the model.

| Model | Five-fold equal-weight ADE (m) | FDE (m) |
|---|---|---|
| Frozen EMT-ZR | 0.438699 | 0.937226 |
| Old EMT-SR | 0.427041 | 0.900124 |
| Clean EMT-SR | 0.431781 | 0.911596 |
| Clean ETT-SR | 0.437414 | 0.922414 |

## Core comparisons

| Comparison (treated minus control) | ΔADE (m) | ΔFDE (m) | ADE change % | FDE change % | ADE wins /15 | FDE wins /15 | Both wins /15 |
|---|---|---|---|---|---|---|---|
| Clean EMT-SR vs Frozen EMT-ZR | -0.006919 | -0.025630 | -1.5771 | -2.7347 | 13 | 15 | 13 |
| Clean EMT-SR vs Clean ETT-SR | -0.005633 | -0.010818 | -1.2878 | -1.1728 | 13 | 11 | 11 |
| Clean EMT-SR vs Old EMT-SR | +0.004740 | +0.011472 | +1.1099 | +1.2745 | 1 | 2 | 0 |

>=1.5% ADE or >=2.5% FDE improvement, with >=10/15 wins in the same metric. >=2% ADE AND >=3% FDE improvement, with >=12/15 paired runs improving both metrics. Negative changes favor Clean EMT-SR.

## Old social gain vs clean gain

| Metric | Old improvement vs ZR % | Clean improvement vs ZR % | Gain reduction (percentage points) |
|---|---|---|---|
| ADE | 2.657518 | 1.577077 | +1.080441 |
| FDE | 3.958708 | 2.734695 | +1.224013 |

## Fifteen paired validation runs

| Fold | Seed | Frozen EMT-ZR ADE/FDE | Old EMT-SR ADE/FDE | Clean EMT-SR ADE/FDE | Clean ETT-SR ADE/FDE |
|---|---|---|---|---|---|
| ETH | 42 | 0.445040 / 0.957707 | 0.432966 / 0.918784 | 0.439763 / 0.938716 | 0.449696 / 0.956784 |
| ETH | 123 | 0.445291 / 0.958745 | 0.428796 / 0.912342 | 0.435447 / 0.927684 | 0.448608 / 0.955920 |
| ETH | 2024 | 0.446200 / 0.960200 | 0.437899 / 0.932835 | 0.438540 / 0.932252 | 0.443267 / 0.941065 |
| HOTEL | 42 | 0.456481 / 0.983586 | 0.446106 / 0.945085 | 0.451360 / 0.962969 | 0.453560 / 0.962556 |
| HOTEL | 123 | 0.458979 / 0.984695 | 0.452531 / 0.964680 | 0.452768 / 0.959166 | 0.457323 / 0.973211 |
| HOTEL | 2024 | 0.459309 / 0.985053 | 0.450900 / 0.953596 | 0.456088 / 0.966086 | 0.453467 / 0.959984 |
| UNIV | 42 | 0.320634 / 0.668953 | 0.309888 / 0.627137 | 0.321701 / 0.661187 | 0.322895 / 0.657340 |
| UNIV | 123 | 0.327342 / 0.680384 | 0.312540 / 0.631715 | 0.315885 / 0.643238 | 0.327633 / 0.662985 |
| UNIV | 2024 | 0.326429 / 0.680644 | 0.312510 / 0.640312 | 0.328231 / 0.672572 | 0.316411 / 0.639963 |
| ZARA1 | 42 | 0.459855 / 0.990959 | 0.447312 / 0.954971 | 0.449297 / 0.956604 | 0.456356 / 0.969042 |
| ZARA1 | 123 | 0.461424 / 0.993034 | 0.453733 / 0.967735 | 0.453720 / 0.967997 | 0.456679 / 0.970203 |
| ZARA1 | 2024 | 0.462053 / 0.993511 | 0.449836 / 0.957492 | 0.450857 / 0.959019 | 0.463738 / 0.988361 |
| ZARA2 | 42 | 0.503518 / 1.073864 | 0.486268 / 1.025999 | 0.493618 / 1.039260 | 0.503216 / 1.064375 |
| ZARA2 | 123 | 0.502985 / 1.070762 | 0.492020 / 1.032302 | 0.495706 / 1.040296 | 0.498625 / 1.053820 |
| ZARA2 | 2024 | 0.504953 / 1.076295 | 0.492307 / 1.036876 | 0.493732 / 1.046892 | 0.509733 / 1.080600 |

## Fold mean ± sample SD

| Fold | Model | ADE (m) | FDE (m) |
|---|---|---|---|
| ETH | Frozen EMT-ZR | 0.445510 ± 0.000610 | 0.958884 ± 0.001252 |
| ETH | Old EMT-SR | 0.433220 ± 0.004557 | 0.921320 ± 0.010480 |
| ETH | Clean EMT-SR | 0.437917 ± 0.002225 | 0.932884 ± 0.005543 |
| ETH | Clean ETT-SR | 0.447191 ± 0.003441 | 0.951256 ± 0.008837 |
| HOTEL | Frozen EMT-ZR | 0.458256 ± 0.001547 | 0.984445 ± 0.000765 |
| HOTEL | Old EMT-SR | 0.449846 ± 0.003340 | 0.954454 ± 0.009826 |
| HOTEL | Clean EMT-SR | 0.453405 ± 0.002428 | 0.962740 ± 0.003466 |
| HOTEL | Clean ETT-SR | 0.454783 ± 0.002200 | 0.965250 ± 0.007013 |
| UNIV | Frozen EMT-ZR | 0.324801 ± 0.003638 | 0.676660 ± 0.006676 |
| UNIV | Old EMT-SR | 0.311646 ± 0.001522 | 0.633055 ± 0.006689 |
| UNIV | Clean EMT-SR | 0.321939 ± 0.006176 | 0.658999 ± 0.014789 |
| UNIV | Clean ETT-SR | 0.322313 ± 0.005634 | 0.653429 ± 0.011999 |
| ZARA1 | Frozen EMT-ZR | 0.461111 ± 0.001132 | 0.992502 ± 0.001357 |
| ZARA1 | Old EMT-SR | 0.450294 ± 0.003235 | 0.960066 ± 0.006760 |
| ZARA1 | Clean EMT-SR | 0.451291 ± 0.002243 | 0.961207 ± 0.006004 |
| ZARA1 | Clean ETT-SR | 0.458924 ± 0.004172 | 0.975869 ± 0.010835 |
| ZARA2 | Frozen EMT-ZR | 0.503818 ± 0.001018 | 1.073640 ± 0.002773 |
| ZARA2 | Old EMT-SR | 0.490199 ± 0.003407 | 1.031726 ± 0.005461 |
| ZARA2 | Clean EMT-SR | 0.494352 ± 0.001174 | 1.042149 ± 0.004140 |
| ZARA2 | Clean ETT-SR | 0.503858 ± 0.005582 | 1.066265 ± 0.013490 |

## Neighbor counts and split audit

Training cross-split neighbor violations: **0**. Every effective training neighbor belongs to the frozen train target set. Validation neighbor train/val/other proportions are audit-only.

| Fold | Split | Samples | Mean | Median | p90 | Zero ratio | Old mean | 0 ratio | 1–2 ratio | 3–4 ratio | 5+ ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ETH | train | 29831 | 7.363447 | 8.000000 | 8.000000 | 0.005397 | 7.524019 | 0.005397 | 0.030673 | 0.050887 | 0.913043 |
| ETH | val | 3966 | 7.505043 | 8.000000 | 8.000000 | 0.000504 | 7.505043 | 0.000504 | 0.014372 | 0.052950 | 0.932173 |
| HOTEL | train | 29086 | 7.425531 | 8.000000 | 8.000000 | 0.006257 | 7.563158 | 0.006257 | 0.025717 | 0.047446 | 0.920580 |
| HOTEL | val | 3878 | 7.580970 | 8.000000 | 8.000000 | 0.002579 | 7.580969 | 0.002579 | 0.013667 | 0.033265 | 0.950490 |
| UNIV | train | 8631 | 5.575484 | 6.000000 | 8.000000 | 0.033368 | 6.204611 | 0.033368 | 0.119337 | 0.180396 | 0.666898 |
| UNIV | val | 1196 | 6.170569 | 7.000000 | 8.000000 | 0.008361 | 6.170568 | 0.008361 | 0.054348 | 0.188127 | 0.749164 |
| ZARA1 | train | 28077 | 7.512555 | 8.000000 | 8.000000 | 0.009153 | 7.669445 | 0.009153 | 0.020729 | 0.031556 | 0.938562 |
| ZARA1 | val | 3728 | 7.655311 | 8.000000 | 8.000000 | 0.002146 | 7.655311 | 0.002146 | 0.005365 | 0.036212 | 0.956277 |
| ZARA2 | train | 24891 | 7.459644 | 8.000000 | 8.000000 | 0.010606 | 7.586115 | 0.010606 | 0.033948 | 0.035635 | 0.919810 |
| ZARA2 | val | 3360 | 7.496726 | 8.000000 | 8.000000 | 0.002976 | 7.496726 | 0.002976 | 0.019345 | 0.059821 | 0.917857 |

| Validation fold | Train-side neighbor ratio | Val-side ratio | Other/non-target-eligible ratio |
|---|---|---|---|
| ETH | 0.887351 | 0.101226 | 0.011423 |
| HOTEL | 0.886017 | 0.098745 | 0.015239 |
| UNIV | 0.872764 | 0.078726 | 0.048509 |
| ZARA1 | 0.878692 | 0.102526 | 0.018781 |
| ZARA2 | 0.871412 | 0.108341 | 0.020247 |

## Neighbor distance audit (meters)

| Fold | Split | Effective neighbors | Mean | Median | p25 | p75 | p90 | Max |
|---|---|---|---|---|---|---|---|---|
| ETH | train | 219659 | 2.639031 | 2.299964 | 1.385210 | 3.400167 | 4.819707 | 15.621048 |
| ETH | val | 29765 | 2.406644 | 2.007545 | 1.183572 | 3.049554 | 4.622089 | 14.915202 |
| HOTEL | train | 215979 | 2.601206 | 2.280909 | 1.375005 | 3.361189 | 4.722827 | 19.710497 |
| HOTEL | val | 29399 | 2.381826 | 1.994570 | 1.184474 | 3.012261 | 4.541745 | 18.464260 |
| UNIV | train | 48122 | 3.626231 | 3.140336 | 1.716001 | 4.944393 | 7.117655 | 19.710497 |
| UNIV | val | 7380 | 3.352270 | 2.782239 | 1.149727 | 4.769857 | 6.759285 | 18.464260 |
| ZARA1 | train | 210930 | 2.590503 | 2.276830 | 1.375571 | 3.341004 | 4.680995 | 19.710497 |
| ZARA1 | val | 28539 | 2.372789 | 1.989472 | 1.182324 | 2.992076 | 4.482831 | 18.464260 |
| ZARA2 | train | 185678 | 2.501876 | 2.222128 | 1.360404 | 3.228673 | 4.444272 | 19.710497 |
| ZARA2 | val | 25189 | 2.281705 | 1.942004 | 1.201831 | 2.861852 | 4.180534 | 18.464260 |

| Fold | Split | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Rank 6 | Rank 7 | Rank 8 |
|---|---|---|---|---|---|---|---|---|---|
| ETH | train | 1.045944 | 1.726526 | 2.243180 | 2.648274 | 3.027583 | 3.329257 | 3.635633 | 3.882352 |
| ETH | val | 0.876483 | 1.528692 | 1.976781 | 2.462924 | 2.751644 | 3.090208 | 3.376117 | 3.553501 |
| HOTEL | train | 1.022023 | 1.696831 | 2.193155 | 2.592052 | 2.961087 | 3.269066 | 3.594612 | 3.859996 |
| HOTEL | val | 0.867460 | 1.483933 | 1.896877 | 2.371232 | 2.735101 | 3.073322 | 3.370707 | 3.558850 |
| UNIV | train | 1.482697 | 2.586403 | 3.353953 | 3.915117 | 4.509279 | 4.919177 | 5.443426 | 5.855838 |
| UNIV | val | 1.118545 | 2.095562 | 2.807066 | 3.714306 | 4.059958 | 4.769478 | 5.352823 | 5.607489 |
| ZARA1 | train | 1.008842 | 1.644995 | 2.145660 | 2.556315 | 2.962870 | 3.275607 | 3.588721 | 3.846938 |
| ZARA1 | val | 0.837477 | 1.475067 | 1.901481 | 2.385392 | 2.686005 | 3.076159 | 3.354823 | 3.528335 |
| ZARA2 | train | 1.027095 | 1.670161 | 2.109701 | 2.467135 | 2.810710 | 3.121749 | 3.407568 | 3.656898 |
| ZARA2 | val | 0.871312 | 1.551575 | 2.013133 | 2.341810 | 2.563792 | 2.803705 | 3.079011 | 3.301141 |

Distances are observations at the last observed frame, ranked per sample. No distance threshold is applied or tuned.

## Validation neighbor-count stratification

Validation groups are identical for all controls. Average each fold over seeds, then equally weight folds with observations. Empty groups use null. Per-fold/per-seed sample counts are in comparison.json.

| Model | Neighbor group | Samples per seed across five validation folds | ADE (m) | FDE (m) |
|---|---|---|---|---|
| Frozen EMT-ZR | 0 | 40 | 0.815972 | 1.468171 |
| Frozen EMT-ZR | 1-2 | 260 | 0.508356 | 1.087837 |
| Frozen EMT-ZR | 3-4 | 900 | 0.452469 | 0.926344 |
| Frozen EMT-ZR | 5+ | 14928 | 0.432473 | 0.928101 |
| Old EMT-SR | 0 | 40 | 0.805210 | 1.463054 |
| Old EMT-SR | 1-2 | 260 | 0.516541 | 1.022711 |
| Old EMT-SR | 3-4 | 900 | 0.404306 | 0.802939 |
| Old EMT-SR | 5+ | 14928 | 0.423610 | 0.899371 |
| Clean EMT-SR | 0 | 40 | 0.800030 | 1.453463 |
| Clean EMT-SR | 1-2 | 260 | 0.519798 | 1.057457 |
| Clean EMT-SR | 3-4 | 900 | 0.433503 | 0.869296 |
| Clean EMT-SR | 5+ | 14928 | 0.426544 | 0.905873 |
| Clean ETT-SR | 0 | 40 | 0.756052 | 1.408943 |
| Clean ETT-SR | 1-2 | 260 | 0.508927 | 1.023612 |
| Clean ETT-SR | 3-4 | 900 | 0.426558 | 0.852065 |
| Clean ETT-SR | 5+ | 14928 | 0.432774 | 0.918299 |

## Frozen models, protocol and runtime

All 671 frozen files unchanged, including both old results directories, checkpoints, old social data, manifests and model source. No gate experiments. EMT-ZR is reused with metrics and checkpoint SHA256 in data_audit/emt_zr_references.json. All new initial outputs equal their frozen base (max_abs_diff <1e-7); all backbone state SHA256 values match after training.

Maximum pre-clip gradient norm: 4.470529. NaN/Inf: false.

The model and hyperparameters match stage two: 50 epochs, batch 128, AdamW 1e-3/1e-4, SmoothL1 beta=1, clip 5, ReduceLROnPlateau factor .5/patience 4, early stopping 8, checkpoint minimum source validation ADE. Only train-neighbor eligibility changes.

| Model | Total params | Trainable | Mean training peak allocated MiB | Mean uncached latency B=1,N=8 ms | B=128,N=8 ms |
|---|---|---|---|---|---|
| Clean EMT-SR | 477360 | 106392 | 161.944 | 4.005303 | 13.695677 |
| Clean ETT-SR | 722736 | 106392 | 162.929 | 2.633542 | 4.735020 |

Training uses frozen eval context caches; memory includes resident caches, and wall times include parallel GPU contention. Latency is measured after training, on the complete uncached model, including shared temporal encoding; 20 warmups +100 synchronized eval/no_grad forwards; batch 1/128, zero/eight neighbors. No data transfer.

No held-out metrics, backbone unfreezing, radius/N/layer/lr/loss changes, gate, or other new models are initiated.

Verification: 59 pytest tests passed; py_compile passed. Full 20-item Chinese handoff: brain_report.md.
