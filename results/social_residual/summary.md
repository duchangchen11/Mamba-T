# ETH/UCY Frozen Social Residual Validation

**SOCIAL: GO. GATE: STOP.** Source-scene validation only; heldout_test_accessed=false.

Baselines reuse stage-one results and checkpoints. All temporal encoders and decoders remain frozen and in eval mode. No backbone fine-tuning, radius, neighbor-count search or held-out evaluation.

| Model | Equal-weight ADE (m) | Equal-weight FDE (m) |
|---|---|---|
| ETT | 0.445898 | 0.942888 |
| EMT | 0.446307 | 0.944740 |
| EMT-ZR | 0.438699 | 0.937226 |
| EMT-SR | 0.427041 | 0.900124 |
| EMT-GSR | 0.427938 | 0.901125 |
| ETT-SR | 0.436744 | 0.920476 |

## Core comparisons

| Comparison | ΔADE % | ΔFDE % | ADE wins /15 | FDE wins /15 | Decision |
|---|---|---|---|---|---|
| EMT-SR vs EMT-ZR | -2.6575 | -3.9587 | 15 | 15 | GO |
| EMT-GSR vs EMT-SR | +0.2100 | +0.1112 | 7 | 8 | STOP |
| EMT-SR vs ETT-SR | -2.2218 | -2.2111 | 15 | 15 | descriptive |
| EMT-ZR vs EMT | -1.7045 | -0.7954 | 15 | 15 | descriptive |

Change = (treated / control − 1) × 100; negative favors treated. The threshold and paired-win count must hold for the same metric.

>=1.5% ADE or >=2.5% FDE improvement; >= 10/15 wins must hold for the same qualifying metric. >=0.75% ADE or >=1.25% FDE improvement; >= 9/15 wins must hold for the same qualifying metric.

## Fifteen paired validation runs

| Fold | Seed | ETT ADE/FDE | EMT ADE/FDE | EMT-ZR ADE/FDE | EMT-SR ADE/FDE | EMT-GSR ADE/FDE | ETT-SR ADE/FDE |
|---|---|---|---|---|---|---|---|
| ETH | 42 | 0.449600 / 0.958002 | 0.454713 / 0.964892 | 0.445040 / 0.957707 | 0.432966 / 0.918784 | 0.438061 / 0.926526 | 0.449124 / 0.956049 |
| ETH | 123 | 0.451385 / 0.962596 | 0.453825 / 0.967809 | 0.445291 / 0.958745 | 0.428796 / 0.912342 | 0.433191 / 0.925534 | 0.449148 / 0.956222 |
| ETH | 2024 | 0.448674 / 0.957712 | 0.458093 / 0.980044 | 0.446200 / 0.960200 | 0.437899 / 0.932835 | 0.437701 / 0.930991 | 0.445359 / 0.947045 |
| HOTEL | 42 | 0.465835 / 0.988899 | 0.466940 / 0.992438 | 0.456481 / 0.983586 | 0.446106 / 0.945085 | 0.449921 / 0.956534 | 0.452665 / 0.961678 |
| HOTEL | 123 | 0.462927 / 0.982849 | 0.466371 / 0.990488 | 0.458979 / 0.984695 | 0.452531 / 0.964680 | 0.447498 / 0.949809 | 0.458275 / 0.976539 |
| HOTEL | 2024 | 0.466359 / 0.992991 | 0.467493 / 0.997949 | 0.459309 / 0.985053 | 0.450900 / 0.953596 | 0.455137 / 0.964148 | 0.458010 / 0.974344 |
| UNIV | 42 | 0.334469 / 0.687584 | 0.334149 / 0.677880 | 0.320634 / 0.668953 | 0.309888 / 0.627137 | 0.309875 / 0.627911 | 0.315889 / 0.636633 |
| UNIV | 123 | 0.335133 / 0.685318 | 0.333929 / 0.690215 | 0.327342 / 0.680384 | 0.312540 / 0.631715 | 0.311202 / 0.633285 | 0.316890 / 0.635089 |
| UNIV | 2024 | 0.330035 / 0.674688 | 0.331085 / 0.682616 | 0.326429 / 0.680644 | 0.312510 / 0.640312 | 0.314099 / 0.637284 | 0.318017 / 0.642530 |
| ZARA1 | 42 | 0.466944 / 0.994244 | 0.465207 / 0.992948 | 0.459855 / 0.990959 | 0.447312 / 0.954971 | 0.447495 / 0.950233 | 0.459304 / 0.977664 |
| ZARA1 | 123 | 0.470773 / 1.005388 | 0.464167 / 0.995000 | 0.461424 / 0.993034 | 0.453733 / 0.967735 | 0.454196 / 0.965160 | 0.459831 / 0.976531 |
| ZARA1 | 2024 | 0.465719 / 0.992787 | 0.466981 / 0.998411 | 0.462053 / 0.993511 | 0.449836 / 0.957492 | 0.446635 / 0.947894 | 0.459548 / 0.980102 |
| ZARA2 | 42 | 0.512778 / 1.086908 | 0.512331 / 1.083310 | 0.503518 / 1.073864 | 0.486268 / 1.025999 | 0.493954 / 1.039356 | 0.504454 / 1.066209 |
| ZARA2 | 123 | 0.516562 / 1.090714 | 0.512635 / 1.078843 | 0.502985 / 1.070762 | 0.492020 / 1.032302 | 0.491172 / 1.028321 | 0.494760 / 1.039675 |
| ZARA2 | 2024 | 0.511275 / 1.082635 | 0.506680 / 1.078264 | 0.504953 / 1.076295 | 0.492307 / 1.036876 | 0.488925 / 1.033891 | 0.509891 / 1.080836 |

## Fold means and sample SD

| Fold | Model | ADE mean ± sample SD | FDE mean ± sample SD |
|---|---|---|---|
| ETH | ETT | 0.449886 ± 0.001378 | 0.959437 ± 0.002740 |
| ETH | EMT | 0.455544 ± 0.002252 | 0.970915 ± 0.008039 |
| ETH | EMT-ZR | 0.445510 ± 0.000610 | 0.958884 ± 0.001252 |
| ETH | EMT-SR | 0.433220 ± 0.004557 | 0.921320 ± 0.010480 |
| ETH | EMT-GSR | 0.436318 ± 0.002714 | 0.927684 ± 0.002907 |
| ETH | ETT-SR | 0.447877 ± 0.002181 | 0.953105 ± 0.005249 |
| HOTEL | ETT | 0.465040 ± 0.001849 | 0.988246 ± 0.005102 |
| HOTEL | EMT | 0.466935 ± 0.000561 | 0.993625 ± 0.003869 |
| HOTEL | EMT-ZR | 0.458256 ± 0.001547 | 0.984445 ± 0.000765 |
| HOTEL | EMT-SR | 0.449846 ± 0.003340 | 0.954454 ± 0.009826 |
| HOTEL | EMT-GSR | 0.450852 ± 0.003904 | 0.956830 ± 0.007174 |
| HOTEL | ETT-SR | 0.456317 ± 0.003165 | 0.970854 ± 0.008022 |
| UNIV | ETT | 0.333213 ± 0.002772 | 0.682530 ± 0.006885 |
| UNIV | EMT | 0.333054 ± 0.001709 | 0.683570 ± 0.006223 |
| UNIV | EMT-ZR | 0.324801 ± 0.003638 | 0.676660 ± 0.006676 |
| UNIV | EMT-SR | 0.311646 ± 0.001522 | 0.633055 ± 0.006689 |
| UNIV | EMT-GSR | 0.311725 ± 0.002160 | 0.632827 ± 0.004703 |
| UNIV | ETT-SR | 0.316932 ± 0.001065 | 0.638084 ± 0.003927 |
| ZARA1 | ETT | 0.467812 ± 0.002636 | 0.997473 ± 0.006893 |
| ZARA1 | EMT | 0.465452 ± 0.001423 | 0.995453 ± 0.002760 |
| ZARA1 | EMT-ZR | 0.461111 ± 0.001132 | 0.992502 ± 0.001357 |
| ZARA1 | EMT-SR | 0.450294 ± 0.003235 | 0.960066 ± 0.006760 |
| ZARA1 | EMT-GSR | 0.449442 ± 0.004139 | 0.954429 ± 0.009367 |
| ZARA1 | ETT-SR | 0.459561 ± 0.000264 | 0.978099 ± 0.001825 |
| ZARA2 | ETT | 0.513538 ± 0.002724 | 1.086752 ± 0.004042 |
| ZARA2 | EMT | 0.510549 ± 0.003354 | 1.080139 ± 0.002761 |
| ZARA2 | EMT-ZR | 0.503818 ± 0.001018 | 1.073640 ± 0.002773 |
| ZARA2 | EMT-SR | 0.490199 ± 0.003407 | 1.031726 ± 0.005461 |
| ZARA2 | EMT-GSR | 0.491351 ± 0.002519 | 1.033856 ± 0.005518 |
| ZARA2 | ETT-SR | 0.503035 ± 0.007665 | 1.062240 ± 0.020866 |

## Neighbor audit

Total social windows: 34161; N=8, no radius. Neighbors must occur at all eight observed frames in the same recording. Original target windows, labels, order and split manifests are unchanged.

Candidates may be any observed pedestrian in the same source recording, including tracks assigned to the other target split; only their eight current observation coordinates are used, never their future labels.

| Fold | Split | Samples | Pedestrians | Mean neighbors | Median | p90 | Zero-neighbor ratio |
|---|---|---|---|---|---|---|---|
| ETH | train | 29831 | 1015 | 7.524019 | 8.0 | 8.0 | 0.002615 |
| ETH | val | 3966 | 160 | 7.505043 | 8.0 | 8.0 | 0.000504 |
| HOTEL | train | 29086 | 947 | 7.563158 | 8.0 | 8.0 | 0.002097 |
| HOTEL | val | 3878 | 150 | 7.580970 | 8.0 | 8.0 | 0.002579 |
| UNIV | train | 8631 | 428 | 6.204611 | 7.0 | 8.0 | 0.012629 |
| UNIV | val | 1196 | 69 | 6.170569 | 7.0 | 8.0 | 0.008361 |
| ZARA1 | train | 28077 | 930 | 7.669445 | 8.0 | 8.0 | 0.003419 |
| ZARA1 | val | 3728 | 147 | 7.655311 | 8.0 | 8.0 | 0.002146 |
| ZARA2 | train | 24891 | 880 | 7.586115 | 8.0 | 8.0 | 0.003696 |
| ZARA2 | val | 3360 | 150 | 7.496726 | 8.0 | 8.0 | 0.002976 |

## Gate statistics at best validation checkpoint

| Fold | Seed | Mean | Population SD | p10 | p50 | p90 | Zero-neighbor mean | Has-neighbor mean |
|---|---|---|---|---|---|---|---|---|
| ETH | 42 | 0.407382 | 0.206688 | 0.105844 | 0.424612 | 0.671021 | 0.326738 | 0.407423 |
| ETH | 123 | 0.373764 | 0.123917 | 0.213649 | 0.377320 | 0.528397 | 0.248115 | 0.373828 |
| ETH | 2024 | 0.313730 | 0.156746 | 0.092179 | 0.312873 | 0.516722 | 0.293350 | 0.313740 |
| HOTEL | 42 | 0.538150 | 0.218799 | 0.209860 | 0.574420 | 0.797988 | 0.523190 | 0.538189 |
| HOTEL | 123 | 0.464968 | 0.159990 | 0.210477 | 0.515122 | 0.625490 | 0.418674 | 0.465088 |
| HOTEL | 2024 | 0.391702 | 0.130678 | 0.211981 | 0.404343 | 0.548007 | 0.355926 | 0.391794 |
| UNIV | 42 | 0.332775 | 0.329998 | 0.014952 | 0.198058 | 0.844169 | 0.658412 | 0.330030 |
| UNIV | 123 | 0.305847 | 0.310532 | 0.036179 | 0.138064 | 0.799134 | 0.663738 | 0.302830 |
| UNIV | 2024 | 0.324517 | 0.314164 | 0.016154 | 0.199456 | 0.822878 | 0.589259 | 0.322284 |
| ZARA1 | 42 | 0.581627 | 0.207894 | 0.260747 | 0.646469 | 0.795750 | 0.542588 | 0.581711 |
| ZARA1 | 123 | 0.429920 | 0.131576 | 0.240865 | 0.453816 | 0.580249 | 0.453961 | 0.429869 |
| ZARA1 | 2024 | 0.390077 | 0.169096 | 0.149223 | 0.406344 | 0.600185 | 0.513274 | 0.389812 |
| ZARA2 | 42 | 0.495349 | 0.201848 | 0.184244 | 0.527606 | 0.732579 | 0.420256 | 0.495573 |
| ZARA2 | 123 | 0.534920 | 0.183864 | 0.266801 | 0.561269 | 0.744236 | 0.470188 | 0.535114 |
| ZARA2 | 2024 | 0.454023 | 0.209282 | 0.150664 | 0.468056 | 0.724578 | 0.416893 | 0.454133 |

Scalar gate values describe the learned model; they do not independently establish interaction necessity or causal interpretation. Zero-neighbor validation groups have only 2–10 samples per fold.

## Neighbor-count group ADE/FDE

Each fold is averaged over seeds, then folds are equally weighted. Per-run group sample counts and metrics are in comparison.json; groups with no samples use null, not fabricated zero errors.

| Model | Neighbor count | ADE (m) | FDE (m) |
|---|---|---|---|
| ETT | 0 | 0.814862 | 1.481624 |
| ETT | 1-2 | 0.549030 | 1.161666 |
| ETT | 3-4 | 0.457578 | 0.929112 |
| ETT | 5+ | 0.438325 | 0.930701 |
| EMT | 0 | 0.825780 | 1.493892 |
| EMT | 1-2 | 0.532711 | 1.142470 |
| EMT | 3-4 | 0.462013 | 0.937035 |
| EMT | 5+ | 0.439479 | 0.934416 |
| EMT-ZR | 0 | 0.815972 | 1.468171 |
| EMT-ZR | 1-2 | 0.508356 | 1.087837 |
| EMT-ZR | 3-4 | 0.452469 | 0.926344 |
| EMT-ZR | 5+ | 0.432473 | 0.928101 |
| EMT-SR | 0 | 0.805210 | 1.463054 |
| EMT-SR | 1-2 | 0.516541 | 1.022711 |
| EMT-SR | 3-4 | 0.404306 | 0.802939 |
| EMT-SR | 5+ | 0.423610 | 0.899371 |
| EMT-GSR | 0 | 0.787922 | 1.450959 |
| EMT-GSR | 1-2 | 0.514134 | 1.026614 |
| EMT-GSR | 3-4 | 0.414070 | 0.828566 |
| EMT-GSR | 5+ | 0.423717 | 0.897568 |
| ETT-SR | 0 | 0.777803 | 1.440001 |
| ETT-SR | 1-2 | 0.510284 | 1.041982 |
| ETT-SR | 3-4 | 0.426195 | 0.854664 |
| ETT-SR | 5+ | 0.432655 | 0.917026 |

## Runtime and trainable capacity

Maximum pre-clip gradient norm: 2.783947. NaN/Inf: false. First-stage result/checkpoint/manifest SHA256 inventory: unchanged.

| Model | Total params | Trainable params | Mean peak training allocated MiB | Full latency B=1,N=8 ms | Full latency B=128,N=8 ms |
|---|---|---|---|---|---|
| EMT-ZR | 406960 | 35992 | 151.681 | 1.622517 | 1.736235 |
| EMT-SR | 477360 | 106392 | 161.925 | 3.970542 | 13.680082 |
| EMT-GSR | 493873 | 122905 | 162.180 | 4.024784 | 13.752069 |
| ETT-SR | 722736 | 106392 | 162.769 | 2.621143 | 4.683298 |

Training uses cached eval contexts to avoid repeatedly encoding frozen target/neighbor histories. Training memory includes resident source caches; it is not model-only memory. Precomputation timing is stored per fold/backbone/seed. Training wall times include parallel GPU contention. Latency measures the full uncached model, with 20 warmups and 100 synchronized eval/no_grad forwards, including the single shared temporal encoder for target and valid neighbors. Zero-neighbor and eight-neighbor cases, batch 1 and 128, are saved for every run.

## Reproduction and verification

Use existing ped_intent environment. Run build_eth_ucy_social_sequences.py, pytest, then run_social_residual.py (four-model ETH/seed42 smoke gate before 60 runs). Raw data and checkpoints remain ignored. All output is in results/social_residual. Original stage-one files remain unchanged.

Both social and gate decisions are internal validation findings. No held-out scene metrics are computed.

Final verification: 45 pytest tests passed; all Python sources compiled. Stage-one inventory unchanged (214 files). Complete Chinese handoff: `brain_report.md`. One precomputation CUDA OOM was recovered without changing the training protocol; all 60 runs and inference benchmarks are complete.
