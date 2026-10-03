# 密度自适应社会交互残差诊断

本报告只使用冻结的 source train/validation split。`heldout_test_accessed=false`；`historical_test_already_accessed=true`。新模型为 Mamba 基础模型 + Social Cross-Attention + density-conditioned calibration + 原 residual decoder。校准量仅由 observation-visible `neighbor_mask.sum()/8` 计算，`s=1+0.5*tanh(f_d(n/8))`，不复用旧 scalar gate。

固定设置：五 fold × 三 seeds，共 15 组；baseline 直接复用上一阶段 EMT-fixed 和 EMT-SR checkpoint。Mamba 与 base decoder 冻结，shared Social Cross-Attention/Residual Decoder 使用与 EMT-SR 相同的初始化。AdamW 1e-3，weight decay 1e-4，batch 128，SmoothL1 beta=1，clip=5；每 fold 复用 EMT-SR 固定 epoch，last epoch 保存，无 scheduler、early stopping 或 best selection。

主要比较为每个 fold/seed 先对该次验证中的 scene 等权，再对 fold 和 seed 等权。Pooled 结果也报告。scene gain 为每个真实 scene 上 base ADE − social ADE，之后平均可见的 fold/seed 场景均值。邻居组表中的行数累加了所有 fold/seed，同一验证窗口会重复出现；它们不是独立样本数。邻居组与正式阈值在训练前写入 frozen config。

## 主要指标

| Model | Scene-equal ADE | Scene-equal FDE | Scene-equal ADE gain | Scene-equal FDE gain |
|---|---:|---:|---:|---:|
| Mamba社会交互残差模型 | 0.536650 | 1.088542 | 0.013314 | 0.026771 |
| Mamba密度自适应社会交互残差模型 | 0.531184 | 1.076313 | 0.018780 | 0.039001 |

Density-aware − baseline：ΔADE=-0.005466 (-1.019%)；ΔFDE=-0.012230 (-1.123%)。paired scene-equal wins：ADE 10/15，FDE 11/15。负差值有利于新模型。

| Fold | Seed | Baseline ADE/FDE (scene-equal) | Density-aware ADE/FDE (scene-equal) | ΔADE | ΔFDE |
|---|---:|---:|---:|---:|---:|
| eth | 42 | 0.399652 / 0.825673 | 0.385407 / 0.794220 | -0.014246 | -0.031453 |
| eth | 123 | 0.400350 / 0.829761 | 0.406924 / 0.850374 | 0.006575 | 0.020613 |
| eth | 2024 | 0.391138 / 0.809499 | 0.386699 / 0.798068 | -0.004439 | -0.011431 |
| hotel | 42 | 0.608444 / 1.248616 | 0.585206 / 1.201519 | -0.023239 | -0.047097 |
| hotel | 123 | 0.578118 / 1.214808 | 0.587013 / 1.227485 | 0.008895 | 0.012677 |
| hotel | 2024 | 0.573622 / 1.178453 | 0.568935 / 1.153841 | -0.004687 | -0.024612 |
| univ | 42 | 0.490818 / 0.984224 | 0.489143 / 0.983128 | -0.001674 | -0.001096 |
| univ | 123 | 0.491519 / 0.980646 | 0.490943 / 0.977322 | -0.000576 | -0.003325 |
| univ | 2024 | 0.496938 / 0.976619 | 0.495951 / 0.975353 | -0.000987 | -0.001266 |
| zara1 | 42 | 0.619706 / 1.240459 | 0.566611 / 1.133541 | -0.053095 | -0.106917 |
| zara1 | 123 | 0.567147 / 1.151439 | 0.564364 / 1.143084 | -0.002783 | -0.008356 |
| zara1 | 2024 | 0.598891 / 1.218584 | 0.608674 / 1.237623 | 0.009783 | 0.019040 |
| zara2 | 42 | 0.616295 / 1.241512 | 0.606450 / 1.223247 | -0.009845 | -0.018265 |
| zara2 | 123 | 0.626952 / 1.254011 | 0.627061 / 1.245779 | 0.000109 | -0.008232 |
| zara2 | 2024 | 0.590166 / 1.173826 | 0.598379 / 1.200105 | 0.008213 | 0.026278 |

## 按 fold 汇总

| Fold | Baseline ADE | Density ADE | ΔADE | Baseline FDE | Density FDE | ΔFDE |
|---|---:|---:|---:|---:|---:|---:|
| ETH | 0.397047 | 0.393010 | -0.004037 | 0.821645 | 0.814221 | -0.007424 |
| HOTEL | 0.586728 | 0.580385 | -0.006343 | 1.213959 | 1.194282 | -0.019678 |
| UNIV | 0.493091 | 0.492012 | -0.001079 | 0.980496 | 0.978601 | -0.001896 |
| ZARA1 | 0.595248 | 0.579883 | -0.015365 | 1.203494 | 1.171416 | -0.032078 |
| ZARA2 | 0.611138 | 0.610630 | -0.000508 | 1.223117 | 1.223044 | -0.000073 |

Pooled sample-weighted results are in `comparison.json` alongside these primary scene-equal metrics.

## Scene 间增益稳定性

| 指标 | Baseline | Density-aware | Relative improvement / change |
|---|---:|---:|---:|
| Scene gain population SD | 0.009066 | 0.004171 | 53.99% |
| Scene gain range | 0.023254 | 0.012417 | 46.60% |
| Worst scene gain | -0.001199 | 0.012596 | +0.013795 |
| Neighbor-count gain population SD | 0.018770 | 0.016572 | 11.71% |

| Scene | Baseline social gain | Density-aware social gain | Change |
|---|---:|---:|---:|
| ETH | -0.001199 | 0.017093 | +0.018292 |
| HOTEL | 0.019812 | 0.025013 | +0.005201 |
| UNIV | 0.022055 | 0.021289 | -0.000766 |
| ZARA1 | 0.006570 | 0.012596 | +0.006026 |
| ZARA2 | 0.019330 | 0.017908 | -0.001421 |

## 按有效邻居数量分组

相同组中分别给出 base、冻结 baseline 社会模型和 density-aware 模型的 ADE/FDE。主统计按 scene 等权，详细每 fold/seed 组计数见 `comparison.json`。表中计数是验证记录行数，同一窗口在多个 fold/seed 中会重复出现。

| Neighbor count | Validation rows across runs (repeated) | Base ADE/FDE | Baseline social ADE/FDE | Density-aware ADE/FDE | Baseline gain | Density gain |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 120 | 0.697773/1.278845 | 0.686008/1.257663 | 0.671200/1.235594 | 0.011765 | 0.026573 |
| 1-2 | 780 | 0.717073/1.564191 | 0.723723/1.490691 | 0.694546/1.418932 | -0.006650 | 0.022527 |
| 3-4 | 2700 | 0.660015/1.354033 | 0.616974/1.239060 | 0.611159/1.230201 | 0.043042 | 0.048857 |
| 5-6 | 2520 | 0.534484/1.097081 | 0.542475/1.140516 | 0.527935/1.103894 | -0.007990 | 0.006549 |
| 7-8 | 42264 | 0.512371/1.021366 | 0.511407/1.045049 | 0.510226/1.039679 | 0.000964 | 0.002145 |

## Density response

| n | Mean scale across 15 trained models | SD across models |
|---:|---:|---:|
| 0 | 0.60084278 | 0.16655240 |
| 1 | 0.60721498 | 0.16822216 |
| 2 | 0.61557096 | 0.16909124 |
| 3 | 0.62668796 | 0.16874869 |
| 4 | 0.64161042 | 0.16665570 |
| 5 | 0.66168194 | 0.16213538 |
| 6 | 0.68852888 | 0.15440533 |
| 7 | 0.72393744 | 0.14275971 |
| 8 | 0.76950971 | 0.12721625 |

15 模型平均 scale 从 n=0 的 0.6008 增至 n=8 的 0.7695；九个 n 组的平均值均低于 1，整体是社会交互特征衰减，且稀疏样本衰减更强、密集样本衰减较弱。这不是稀疏增强或密集抑制。不同 fold/seed LUT 离散度较大，逐模型曲线和尺度见 comparison.json。
Observation scale: mean=0.713003, std=0.095422, min/max=0.507527/1.020501; all finite/in bounds=True/True.
Varₙ(E[s|n])=0.00295584; Pearson/Spearman count vs scale=0.289137/0.408125; scene-equal scale vs residual norm Pearson/Spearman=-0.130760/-0.125322.
Neighbor count vs residual norm: baseline Pearson/Spearman=-0.130213/-0.147160; density-aware=-0.078197/-0.092281. Response direction: higher neighbor counts increase the social scale. Baseline scale is N/A because it has no density module.

Scale depends only on n; LUTs are direct evaluations of the trained scalar network at n/8. No scene, distance, motion, attention, error, or future value enters that network. Correlations are descriptive and not mechanism proof.

## Training integrity

15/15 fixed runs completed. Maximum pre-clip gradient norm: 0.561264. NaN/Inf: False. EMT backbone frozen and unchanged in all runs; initialization matches frozen EMT-SR shared modules; max zero-init prediction difference <1e-7. All checkpoints are the prescribed last epoch.

## 阶段结论

- DENSITY CALIBRATION: **GO**
- DENSITY RESPONSE: **MEANINGFUL**
- CROSS-SCENE STABILITY: **IMPROVED**

预定义 GO/STRONG GO/STOP 条件逐项判断保存在 `comparison.json`。该阶段只完成 source 诊断；未访问 formal held-out test，未改 density 公式或开展 SDD。
