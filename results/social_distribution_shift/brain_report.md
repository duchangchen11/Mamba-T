SOURCE VALIDATION SOCIAL DISTRIBUTION AUDIT ONLY

heldout_test_accessed=false；historical_test_already_accessed=true。未重新训练 trajectory 模型，未读取正式 test 或 prediction。所有 forward 使用本阶段前冻结的 source validation checkpoint，eval/no_grad；所有参数 hash 保持一致。

scene 指样本来源，fold 指上一阶段四 source scenes 的留一 fold 名称。首先按 scene + ped_id + 8 observation frame IDs 去重；每个窗口的模型响应在可用 fold/seed 上平均。主要表格/相关性使用窗口级统计，另保存全部重复预测和每 fold/seed 的原始结果，不能把重复窗口当独立观测。

pooled 对所有唯一窗口等权；scene_equal 在每个有效 scene 内对窗口等权，再对 scene 等权。距离和相对速度在无邻居时为 missing，统计中排除，不以 0m 当真实距离。closing 为正向接近邻居的均值，max clamped >=0；signed mean/max 另存。速度单位 meter / sampling step。

Attention 统计为四头权重平均后，在有效邻居内归一化；entropy 在 n<=1 时定义 0。权重通过原 SDPA 的 identity-value probe 提取，使用相同 packed projections 和 padding mask，同时重建 context 与原输出核对。prediction 始终使用原 need_weights=False 路径；padding 不参与统计。高 neighbor_count 本身会影响集中度；另报告 n>=2 和固定 count 内相关性，不能仅凭相关性推出机制。

std 为 population SD；scene 内分位数采用 numpy linear quantile；全体 scene 等权分位数采用 weighted mid-CDF interpolation。SMD 比较 scene A−B，排名平均 10 个 scene pair 的 |SMD|。Wasserstein 有单位，不能跨不同量纲直接比较或取总体大小排序。

标签阈值已在审计前固定：mean_abs_SMD>=0.5 为 SUPPORTED，>=0.2 为 WEAK，否则 NOT SUPPORTED。家族使用最强变量，EMT/ETT 的 mean_abs_SMD 等权平均。仅为描述性分档，不是显著性检验、因果证明或 predictor 阈值。

以下逐项覆盖要求的 35 项。结果 commit 无法引用自身 SHA；最终 SHA 与远端核验在交付回复给出，本分支 HEAD 可复核。

1. branch

diag/social_distribution_shift

2. latest SHA

代码 commit=a2467f438e237791c344b95830b11f0ee715008b；结果 commit 为本分支 HEAD，具体 SHA 在交付回复给出。

3. push status

代码和结果通过已认证 GitHub API 发布；远端完整 tree 与本地对比后才交付。

4. heldout_test_accessed

false；代码 guard 拒绝 final_eth_ucy_benchmark，含 predictions 和 symlink。

5. historical_test_already_accessed

true；旧五阶段结果完全冻结。

6. pytest

128 passed；完整日志 data_audit/pytest.txt。

7. sample 数量

| scene | 唯一窗口 | scene/ped 数 | 每模型重复预测记录 |
| --- | --- | --- | --- |
| eth | 66 | 9 | 792 |
| hotel | 154 | 19 | 1848 |
| univ | 2836 | 100 | 34032 |
| zara1 | 304 | 22 | 3648 |
| zara2 | 672 | 19 | 8064 |

每模型共 48384 条预测，去重后 4032 个窗口；模型间样本完全配对。

8. neighbor count 五 scene 统计

| 模型 | 变量 | eth | hotel | univ | zara1 | zara2 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | neighbor_count | 4.590909 | 4.344156 | 8.000000 | 5.029605 | 7.260417 |
| ett_sr | neighbor_count | 4.590909 | 4.344156 | 8.000000 | 5.029605 | 7.260417 |

| scene | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth | 8 | 1 | 7 | 11 | 4 | 9 | 3 | 8 | 15 |
| hotel | 0 | 10 | 2 | 35 | 61 | 8 | 13 | 12 | 13 |
| univ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2836 |
| zara1 | 2 | 19 | 26 | 51 | 39 | 47 | 13 | 20 | 87 |
| zara2 | 0 | 0 | 0 | 13 | 11 | 62 | 55 | 92 | 439 |


9. distance 五 scene 统计

| 模型 | 变量 | eth | hotel | univ | zara1 | zara2 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | nearest_neighbor_distance | 1.263385 | 1.248832 | 0.783167 | 1.431258 | 0.935651 |
| emt_sr | mean_neighbor_distance | 4.088691 | 4.046756 | 2.122583 | 3.573365 | 3.180815 |
| emt_sr | median_neighbor_distance | 3.829866 | 4.359965 | 2.201879 | 3.443903 | 3.032997 |
| emt_sr | max_neighbor_distance | 7.275473 | 6.414411 | 3.181189 | 5.845605 | 5.893300 |
| ett_sr | nearest_neighbor_distance | 1.263385 | 1.248832 | 0.783167 | 1.431258 | 0.935651 |
| ett_sr | mean_neighbor_distance | 4.088691 | 4.046756 | 2.122583 | 3.573365 | 3.180815 |
| ett_sr | median_neighbor_distance | 3.829866 | 4.359965 | 2.201879 | 3.443903 | 3.032997 |
| ett_sr | max_neighbor_distance | 7.275473 | 6.414411 | 3.181189 | 5.845605 | 5.893300 |

rank1…rank8 每窗口原始距离已保存；per_scene CSV/JSON 包含各 rank 分位数。

10. relative speed 五 scene 统计

| 模型 | 变量 | eth | hotel | univ | zara1 | zara2 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | mean_relative_speed | 0.664197 | 0.346178 | 0.290488 | 0.471585 | 0.291432 |
| emt_sr | max_relative_speed | 1.206700 | 0.669485 | 0.535750 | 0.815776 | 0.612035 |
| ett_sr | mean_relative_speed | 0.664197 | 0.346178 | 0.290488 | 0.471585 | 0.291432 |
| ett_sr | max_relative_speed | 1.206700 | 0.669485 | 0.535750 | 0.815776 | 0.612035 |


11. closing speed 五 scene 统计

| 模型 | 变量 | eth | hotel | univ | zara1 | zara2 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | mean_closing_speed | 0.209951 | 0.209020 | 0.169291 | 0.211478 | 0.155249 |
| emt_sr | max_closing_speed | 0.315772 | 0.287628 | 0.297485 | 0.361686 | 0.283496 |
| emt_sr | mean_signed_closing_speed | -0.260691 | -0.054193 | -0.002067 | -0.086190 | -0.026769 |
| ett_sr | mean_closing_speed | 0.209951 | 0.209020 | 0.169291 | 0.211478 | 0.155249 |
| ett_sr | max_closing_speed | 0.315772 | 0.287628 | 0.297485 | 0.361686 | 0.283496 |
| ett_sr | mean_signed_closing_speed | -0.260691 | -0.054193 | -0.002067 | -0.086190 | -0.026769 |

mean_closing_speed 是 mean_positive_closing_speed 的别名；无接近邻居为 0。signed mean/max 用于保存远离的负值。

12. emt_sr attention_entropy

| scene | mean | std | median | p10 | p25 | p75 | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth | 0.705053 | 0.290941 | 0.807069 | 0.000000 | 0.722780 | 0.867649 | 0.914127 | 0.940010 |
| hotel | 0.645938 | 0.198241 | 0.680193 | 0.519950 | 0.602127 | 0.735724 | 0.833631 | 0.880259 |
| univ | 0.814091 | 0.075606 | 0.823456 | 0.716788 | 0.774358 | 0.868566 | 0.899425 | 0.913431 |
| zara1 | 0.753578 | 0.217123 | 0.809886 | 0.656722 | 0.754255 | 0.856468 | 0.885244 | 0.917004 |
| zara2 | 0.824863 | 0.075284 | 0.839229 | 0.730842 | 0.792986 | 0.871083 | 0.905709 | 0.920774 |


13. ett_sr attention_entropy

| scene | mean | std | median | p10 | p25 | p75 | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth | 0.473392 | 0.217889 | 0.497657 | 0.000000 | 0.447330 | 0.583039 | 0.703486 | 0.789796 |
| hotel | 0.556089 | 0.170170 | 0.578516 | 0.460638 | 0.521171 | 0.647693 | 0.705489 | 0.725410 |
| univ | 0.439966 | 0.048098 | 0.439292 | 0.376566 | 0.406182 | 0.475604 | 0.501784 | 0.517293 |
| zara1 | 0.504639 | 0.171408 | 0.509719 | 0.388827 | 0.459959 | 0.593642 | 0.671366 | 0.730377 |
| zara2 | 0.416092 | 0.076155 | 0.402432 | 0.328566 | 0.358072 | 0.463238 | 0.518630 | 0.546855 |


14. emt_sr residual_norm

| scene | mean | std | median | p10 | p25 | p75 | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth | 0.450178 | 0.233930 | 0.423115 | 0.184844 | 0.270546 | 0.572353 | 0.784306 | 0.937186 |
| hotel | 0.202922 | 0.122899 | 0.171337 | 0.069387 | 0.094711 | 0.284311 | 0.369938 | 0.409652 |
| univ | 0.201281 | 0.110213 | 0.180087 | 0.079699 | 0.116681 | 0.256196 | 0.347330 | 0.409113 |
| zara1 | 0.228546 | 0.083744 | 0.210909 | 0.146405 | 0.179160 | 0.258871 | 0.323338 | 0.371239 |
| zara2 | 0.128298 | 0.106266 | 0.067314 | 0.054341 | 0.059620 | 0.189103 | 0.263733 | 0.317694 |


15. ett_sr residual_norm

| scene | mean | std | median | p10 | p25 | p75 | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth | 0.358850 | 0.192095 | 0.320454 | 0.100464 | 0.236209 | 0.459089 | 0.650458 | 0.726172 |
| hotel | 0.119507 | 0.067327 | 0.102617 | 0.045946 | 0.062502 | 0.167214 | 0.214743 | 0.240373 |
| univ | 0.102003 | 0.050379 | 0.090849 | 0.060288 | 0.068026 | 0.118223 | 0.152411 | 0.187112 |
| zara1 | 0.158474 | 0.064358 | 0.147795 | 0.103066 | 0.118472 | 0.176511 | 0.227847 | 0.249943 |
| zara2 | 0.100678 | 0.059093 | 0.072338 | 0.054025 | 0.059806 | 0.134012 | 0.170653 | 0.197500 |


16. correction ratio

| 模型 | 变量 | eth | hotel | univ | zara1 | zara2 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | residual_final_norm | 0.970768 | 0.410157 | 0.420745 | 0.470058 | 0.252885 |
| emt_sr | correction_ratio | 0.471053 | 0.591411 | 0.315749 | 0.108564 | 0.844778 |
| ett_sr | residual_final_norm | 0.758547 | 0.224995 | 0.190570 | 0.296736 | 0.185372 |
| ett_sr | correction_ratio | 0.458577 | 0.434928 | 0.179767 | 0.069803 | 0.768215 |

R=mean_t ||delta_t||，R_final=||delta_12||，ratio=R/(mean_t||base_t||+1e-8)；base 相对最后 observation 位置，近静止 base 会放大 ratio。完整分位数见 per_scene。

17. 10 组 pairwise SMD table

| 模型 | scene pair | 变量 | mean A−B | median A−B | abs(SMD) | Wasserstein |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | eth ↔ hotel | neighbor_count | 0.246753 | 1.000000 | 0.107438 | 1.168831 |
| emt_sr | eth ↔ hotel | nearest_neighbor_distance | 0.014552 | 0.282234 | 0.016045 | 0.507967 |
| emt_sr | eth ↔ hotel | mean_neighbor_distance | 0.041935 | -0.524980 | 0.020250 | 0.611111 |
| emt_sr | eth ↔ hotel | median_neighbor_distance | -0.530098 | -1.364403 | 0.234618 | 0.921507 |
| emt_sr | eth ↔ hotel | max_neighbor_distance | 0.861062 | 0.871087 | 0.223526 | 1.634383 |
| emt_sr | eth ↔ hotel | mean_relative_speed | 0.318019 | 0.415033 | 1.439990 | 0.318189 |
| emt_sr | eth ↔ hotel | max_relative_speed | 0.537215 | 0.371676 | 1.207905 | 0.537215 |
| emt_sr | eth ↔ hotel | mean_closing_speed | 0.000931 | 0.023092 | 0.003609 | 0.025380 |
| emt_sr | eth ↔ hotel | max_closing_speed | 0.028144 | 0.087319 | 0.075858 | 0.046078 |
| emt_sr | eth ↔ hotel | attention_entropy | 0.059115 | 0.126876 | 0.237465 | 0.132728 |
| emt_sr | eth ↔ hotel | attention_max | -0.191962 | -0.178340 | 1.119797 | 0.191962 |
| emt_sr | eth ↔ hotel | attention_top2_sum | -0.194646 | -0.197868 | 0.947445 | 0.200438 |
| emt_sr | eth ↔ hotel | residual_norm | 0.247256 | 0.251778 | 1.323270 | 0.247256 |
| emt_sr | eth ↔ hotel | residual_final_norm | 0.560610 | 0.505243 | 1.333472 | 0.560610 |
| emt_sr | eth ↔ hotel | correction_ratio | -0.120358 | -0.021968 | 0.151959 | 0.205506 |
| emt_sr | eth ↔ univ | neighbor_count | -3.409091 | -3.000000 | 1.776756 | 3.409091 |
| emt_sr | eth ↔ univ | nearest_neighbor_distance | 0.480218 | 0.417295 | 1.011027 | 0.483831 |
| emt_sr | eth ↔ univ | mean_neighbor_distance | 1.966108 | 1.442356 | 1.124633 | 1.966108 |
| emt_sr | eth ↔ univ | median_neighbor_distance | 1.627988 | 0.689759 | 0.877685 | 1.629056 |
| emt_sr | eth ↔ univ | max_neighbor_distance | 4.094284 | 4.187221 | 1.204375 | 4.106232 |
| emt_sr | eth ↔ univ | mean_relative_speed | 0.373709 | 0.445768 | 1.903406 | 0.373709 |
| emt_sr | eth ↔ univ | max_relative_speed | 0.670950 | 0.546559 | 1.639626 | 0.670950 |
| emt_sr | eth ↔ univ | mean_closing_speed | 0.040661 | -0.039942 | 0.202062 | 0.101026 |
| emt_sr | eth ↔ univ | max_closing_speed | 0.018288 | -0.082636 | 0.058955 | 0.145136 |
| emt_sr | eth ↔ univ | attention_entropy | -0.109038 | -0.016387 | 0.512977 | 0.115304 |
| emt_sr | eth ↔ univ | attention_max | 0.061196 | 0.083347 | 0.427373 | 0.123207 |
| emt_sr | eth ↔ univ | attention_top2_sum | 0.074788 | 0.104800 | 0.373493 | 0.183783 |
| emt_sr | eth ↔ univ | residual_norm | 0.248896 | 0.243028 | 1.361187 | 0.248896 |
| emt_sr | eth ↔ univ | residual_final_norm | 0.550023 | 0.478078 | 1.312194 | 0.550023 |
| emt_sr | eth ↔ univ | correction_ratio | 0.155304 | -0.024040 | 0.241937 | 0.175787 |
| emt_sr | eth ↔ zara1 | neighbor_count | -0.438696 | 0.000000 | 0.172222 | 0.438696 |
| emt_sr | eth ↔ zara1 | nearest_neighbor_distance | -0.167873 | 0.212294 | 0.142471 | 0.529580 |
| emt_sr | eth ↔ zara1 | mean_neighbor_distance | 0.515325 | 0.017922 | 0.263214 | 0.796071 |
| emt_sr | eth ↔ zara1 | median_neighbor_distance | 0.385963 | -0.475981 | 0.182475 | 0.778894 |
| emt_sr | eth ↔ zara1 | max_neighbor_distance | 1.429868 | 1.507642 | 0.391843 | 2.136493 |
| emt_sr | eth ↔ zara1 | mean_relative_speed | 0.192612 | 0.248889 | 0.856616 | 0.192612 |
| emt_sr | eth ↔ zara1 | max_relative_speed | 0.390924 | 0.175692 | 0.906081 | 0.390924 |
| emt_sr | eth ↔ zara1 | mean_closing_speed | -0.001526 | -0.065882 | 0.006437 | 0.046292 |
| emt_sr | eth ↔ zara1 | max_closing_speed | -0.045914 | -0.048477 | 0.124942 | 0.076163 |
| emt_sr | eth ↔ zara1 | attention_entropy | -0.048525 | -0.002817 | 0.189033 | 0.061385 |
| emt_sr | eth ↔ zara1 | attention_max | -0.078382 | -0.025069 | 0.412702 | 0.079389 |
| emt_sr | eth ↔ zara1 | attention_top2_sum | -0.086398 | -0.056355 | 0.377911 | 0.087352 |
| emt_sr | eth ↔ zara1 | residual_norm | 0.221631 | 0.212206 | 1.261469 | 0.221656 |
| emt_sr | eth ↔ zara1 | residual_final_norm | 0.500710 | 0.421835 | 1.233351 | 0.501056 |
| emt_sr | eth ↔ zara1 | correction_ratio | 0.362489 | 0.055426 | 0.631240 | 0.362489 |
| emt_sr | eth ↔ zara2 | neighbor_count | -2.669508 | -3.000000 | 1.268598 | 2.669508 |
| emt_sr | eth ↔ zara2 | nearest_neighbor_distance | 0.327734 | 0.302901 | 0.572713 | 0.374304 |
| emt_sr | eth ↔ zara2 | mean_neighbor_distance | 0.907876 | 0.567528 | 0.489265 | 1.073557 |
| emt_sr | eth ↔ zara2 | median_neighbor_distance | 0.796869 | 0.005202 | 0.395655 | 0.917119 |
| emt_sr | eth ↔ zara2 | max_neighbor_distance | 1.382173 | 1.376715 | 0.376659 | 2.196270 |
| emt_sr | eth ↔ zara2 | mean_relative_speed | 0.372765 | 0.449532 | 1.895376 | 0.372765 |
| emt_sr | eth ↔ zara2 | max_relative_speed | 0.594666 | 0.468414 | 1.476204 | 0.594716 |
| emt_sr | eth ↔ zara2 | mean_closing_speed | 0.054702 | -0.018527 | 0.258772 | 0.074694 |
| emt_sr | eth ↔ zara2 | max_closing_speed | 0.032276 | -0.088989 | 0.098712 | 0.103518 |
| emt_sr | eth ↔ zara2 | attention_entropy | -0.119810 | -0.032160 | 0.563807 | 0.123726 |
| emt_sr | eth ↔ zara2 | attention_max | 0.071520 | 0.114677 | 0.474309 | 0.129824 |
| emt_sr | eth ↔ zara2 | attention_top2_sum | 0.077565 | 0.129075 | 0.378023 | 0.185189 |
| emt_sr | eth ↔ zara2 | residual_norm | 0.321880 | 0.355801 | 1.771680 | 0.321880 |
| emt_sr | eth ↔ zara2 | residual_final_norm | 0.717882 | 0.720863 | 1.738273 | 0.717882 |
| emt_sr | eth ↔ zara2 | correction_ratio | -0.373725 | -1.039885 | 0.515581 | 0.571891 |
| emt_sr | hotel ↔ univ | neighbor_count | -3.655844 | -4.000000 | 2.896200 | 3.655844 |
| emt_sr | hotel ↔ univ | nearest_neighbor_distance | 0.465666 | 0.135061 | 0.526034 | 0.465666 |
| emt_sr | hotel ↔ univ | mean_neighbor_distance | 1.924172 | 1.967336 | 1.323849 | 1.927166 |
| emt_sr | hotel ↔ univ | median_neighbor_distance | 2.158086 | 2.054161 | 1.298063 | 2.160554 |
| emt_sr | hotel ↔ univ | max_neighbor_distance | 3.233221 | 3.316134 | 1.418355 | 3.251939 |
| emt_sr | hotel ↔ univ | mean_relative_speed | 0.055690 | 0.030735 | 0.313289 | 0.056198 |
| emt_sr | hotel ↔ univ | max_relative_speed | 0.133735 | 0.174882 | 0.459805 | 0.145155 |
| emt_sr | hotel ↔ univ | mean_closing_speed | 0.039729 | -0.063035 | 0.192390 | 0.114477 |
| emt_sr | hotel ↔ univ | max_closing_speed | -0.009856 | -0.169956 | 0.033914 | 0.142369 |
| emt_sr | hotel ↔ univ | attention_entropy | -0.168153 | -0.143263 | 1.120824 | 0.168235 |
| emt_sr | hotel ↔ univ | attention_max | 0.253158 | 0.261687 | 2.141289 | 0.253158 |
| emt_sr | hotel ↔ univ | attention_top2_sum | 0.269434 | 0.302667 | 2.962145 | 0.269434 |
| emt_sr | hotel ↔ univ | residual_norm | 0.001641 | -0.008750 | 0.014056 | 0.017924 |
| emt_sr | hotel ↔ univ | residual_final_norm | -0.010587 | -0.027165 | 0.042773 | 0.032484 |
| emt_sr | hotel ↔ univ | correction_ratio | 0.275662 | -0.002072 | 0.443056 | 0.276179 |
| emt_sr | hotel ↔ zara1 | neighbor_count | -0.685449 | -1.000000 | 0.326756 | 0.894609 |
| emt_sr | hotel ↔ zara1 | nearest_neighbor_distance | -0.182426 | -0.069940 | 0.130757 | 0.258481 |
| emt_sr | hotel ↔ zara1 | mean_neighbor_distance | 0.473390 | 0.542903 | 0.278497 | 0.510767 |
| emt_sr | hotel ↔ zara1 | median_neighbor_distance | 0.916061 | 0.888422 | 0.470091 | 0.928769 |
| emt_sr | hotel ↔ zara1 | max_neighbor_distance | 0.568805 | 0.636555 | 0.215675 | 0.620644 |
| emt_sr | hotel ↔ zara1 | mean_relative_speed | -0.125407 | -0.166144 | 0.600534 | 0.132200 |
| emt_sr | hotel ↔ zara1 | max_relative_speed | -0.146291 | -0.195985 | 0.455189 | 0.155330 |
| emt_sr | hotel ↔ zara1 | mean_closing_speed | -0.002457 | -0.088974 | 0.010172 | 0.054583 |
| emt_sr | hotel ↔ zara1 | max_closing_speed | -0.074058 | -0.135796 | 0.210923 | 0.084550 |
| emt_sr | hotel ↔ zara1 | attention_entropy | -0.107640 | -0.129693 | 0.517757 | 0.111651 |
| emt_sr | hotel ↔ zara1 | attention_max | 0.113581 | 0.153271 | 0.660789 | 0.115567 |
| emt_sr | hotel ↔ zara1 | attention_top2_sum | 0.108248 | 0.141512 | 0.757059 | 0.118575 |
| emt_sr | hotel ↔ zara1 | residual_norm | -0.025624 | -0.039573 | 0.243668 | 0.047803 |
| emt_sr | hotel ↔ zara1 | residual_final_norm | -0.059900 | -0.083409 | 0.266849 | 0.088409 |
| emt_sr | hotel ↔ zara1 | correction_ratio | 0.482847 | 0.077394 | 0.874568 | 0.482847 |
| emt_sr | hotel ↔ zara2 | neighbor_count | -2.916261 | -4.000000 | 1.906430 | 2.916261 |
| emt_sr | hotel ↔ zara2 | nearest_neighbor_distance | 0.313181 | 0.020667 | 0.332812 | 0.341830 |
| emt_sr | hotel ↔ zara2 | mean_neighbor_distance | 0.865940 | 1.092509 | 0.547723 | 0.890838 |
| emt_sr | hotel ↔ zara2 | median_neighbor_distance | 1.326967 | 1.369605 | 0.721783 | 1.330020 |
| emt_sr | hotel ↔ zara2 | max_neighbor_distance | 0.521111 | 0.505628 | 0.195495 | 0.687787 |
| emt_sr | hotel ↔ zara2 | mean_relative_speed | 0.054746 | 0.034499 | 0.307340 | 0.055120 |
| emt_sr | hotel ↔ zara2 | max_relative_speed | 0.057451 | 0.096737 | 0.203861 | 0.148659 |
| emt_sr | hotel ↔ zara2 | mean_closing_speed | 0.053771 | -0.041620 | 0.248457 | 0.085963 |
| emt_sr | hotel ↔ zara2 | max_closing_speed | 0.004133 | -0.176308 | 0.013398 | 0.102899 |
| emt_sr | hotel ↔ zara2 | attention_entropy | -0.178925 | -0.159036 | 1.193270 | 0.178925 |
| emt_sr | hotel ↔ zara2 | attention_max | 0.263483 | 0.293018 | 2.069389 | 0.263483 |
| emt_sr | hotel ↔ zara2 | attention_top2_sum | 0.272211 | 0.326943 | 2.684879 | 0.272211 |
| emt_sr | hotel ↔ zara2 | residual_norm | 0.074624 | 0.104022 | 0.649558 | 0.075086 |
| emt_sr | hotel ↔ zara2 | residual_final_norm | 0.157272 | 0.215619 | 0.663831 | 0.157885 |
| emt_sr | hotel ↔ zara2 | correction_ratio | -0.253367 | -1.017917 | 0.358145 | 0.445150 |
| emt_sr | univ ↔ zara1 | neighbor_count | 2.970395 | 3.000000 | 1.772895 | 2.970395 |
| emt_sr | univ ↔ zara1 | nearest_neighbor_distance | -0.648091 | -0.205001 | 0.557895 | 0.648091 |
| emt_sr | univ ↔ zara1 | mean_neighbor_distance | -1.450782 | -1.424433 | 1.126972 | 1.450782 |
| emt_sr | univ ↔ zara1 | median_neighbor_distance | -1.242025 | -1.165739 | 0.850428 | 1.242025 |
| emt_sr | univ ↔ zara1 | max_neighbor_distance | -2.664416 | -2.679579 | 1.390273 | 2.665321 |
| emt_sr | univ ↔ zara1 | mean_relative_speed | -0.181097 | -0.196879 | 0.991177 | 0.182490 |
| emt_sr | univ ↔ zara1 | max_relative_speed | -0.280026 | -0.370867 | 1.036864 | 0.289037 |
| emt_sr | univ ↔ zara1 | mean_closing_speed | -0.042187 | -0.025939 | 0.234837 | 0.070842 |
| emt_sr | univ ↔ zara1 | max_closing_speed | -0.064202 | 0.034159 | 0.224395 | 0.138948 |
| emt_sr | univ ↔ zara1 | attention_entropy | 0.060513 | 0.013571 | 0.372225 | 0.062318 |
| emt_sr | univ ↔ zara1 | attention_max | -0.139577 | -0.108416 | 0.971032 | 0.142511 |
| emt_sr | univ ↔ zara1 | attention_top2_sum | -0.161186 | -0.161155 | 1.190428 | 0.166485 |
| emt_sr | univ ↔ zara1 | residual_norm | -0.027265 | -0.030822 | 0.278561 | 0.034976 |
| emt_sr | univ ↔ zara1 | residual_final_norm | -0.049313 | -0.056244 | 0.222010 | 0.066872 |
| emt_sr | univ ↔ zara1 | correction_ratio | 0.207185 | 0.079465 | 0.692387 | 0.207189 |
| emt_sr | univ ↔ zara2 | neighbor_count | 0.739583 | 0.000000 | 0.855941 | 0.739583 |
| emt_sr | univ ↔ zara2 | nearest_neighbor_distance | -0.152484 | -0.114395 | 0.283863 | 0.177185 |
| emt_sr | univ ↔ zara2 | mean_neighbor_distance | -1.058232 | -0.874827 | 0.939998 | 1.058393 |
| emt_sr | univ ↔ zara2 | median_neighbor_distance | -0.831119 | -0.684557 | 0.634551 | 0.840406 |
| emt_sr | univ ↔ zara2 | max_neighbor_distance | -2.712111 | -2.810506 | 1.387135 | 2.712593 |
| emt_sr | univ ↔ zara2 | mean_relative_speed | -0.000944 | 0.003763 | 0.006439 | 0.016205 |
| emt_sr | univ ↔ zara2 | max_relative_speed | -0.076285 | -0.078145 | 0.344380 | 0.081646 |
| emt_sr | univ ↔ zara2 | mean_closing_speed | 0.014042 | 0.021415 | 0.097498 | 0.030381 |
| emt_sr | univ ↔ zara2 | max_closing_speed | 0.013989 | -0.006353 | 0.060350 | 0.048632 |
| emt_sr | univ ↔ zara2 | attention_entropy | -0.010772 | -0.015772 | 0.142782 | 0.011857 |
| emt_sr | univ ↔ zara2 | attention_max | 0.010325 | 0.031331 | 0.120612 | 0.024868 |
| emt_sr | univ ↔ zara2 | attention_top2_sum | 0.002777 | 0.024275 | 0.030730 | 0.022003 |
| emt_sr | univ ↔ zara2 | residual_norm | 0.072983 | 0.112773 | 0.674162 | 0.073022 |
| emt_sr | univ ↔ zara2 | residual_final_norm | 0.167859 | 0.242784 | 0.715243 | 0.167859 |
| emt_sr | univ ↔ zara2 | correction_ratio | -0.529029 | -1.015845 | 0.990610 | 0.549629 |
| emt_sr | zara1 ↔ zara2 | neighbor_count | -2.230811 | -3.000000 | 1.183371 | 2.230811 |
| emt_sr | zara1 ↔ zara2 | nearest_neighbor_distance | 0.495607 | 0.090607 | 0.411389 | 0.495746 |
| emt_sr | zara1 ↔ zara2 | mean_neighbor_distance | 0.392550 | 0.549606 | 0.274561 | 0.426146 |
| emt_sr | zara1 ↔ zara2 | median_neighbor_distance | 0.410906 | 0.481183 | 0.247838 | 0.416350 |
| emt_sr | zara1 ↔ zara2 | max_neighbor_distance | -0.047694 | -0.130927 | 0.020187 | 0.344635 |
| emt_sr | zara1 ↔ zara2 | mean_relative_speed | 0.180153 | 0.200643 | 0.984078 | 0.180155 |
| emt_sr | zara1 ↔ zara2 | max_relative_speed | 0.203741 | 0.292722 | 0.782688 | 0.226533 |
| emt_sr | zara1 ↔ zara2 | mean_closing_speed | 0.056228 | 0.047354 | 0.294454 | 0.056518 |
| emt_sr | zara1 ↔ zara2 | max_closing_speed | 0.078190 | -0.040512 | 0.257021 | 0.091535 |
| emt_sr | zara1 ↔ zara2 | attention_entropy | -0.071285 | -0.029343 | 0.438688 | 0.071532 |
| emt_sr | zara1 ↔ zara2 | attention_max | 0.149902 | 0.139746 | 0.990682 | 0.152822 |
| emt_sr | zara1 ↔ zara2 | attention_top2_sum | 0.163963 | 0.185430 | 1.149682 | 0.169352 |
| emt_sr | zara1 ↔ zara2 | residual_norm | 0.100248 | 0.143595 | 1.047851 | 0.100815 |
| emt_sr | zara1 ↔ zara2 | residual_final_norm | 0.217172 | 0.299028 | 1.033014 | 0.217172 |
| emt_sr | zara1 ↔ zara2 | correction_ratio | -0.736214 | -1.095311 | 1.634410 | 0.736219 |
| ett_sr | eth ↔ hotel | neighbor_count | 0.246753 | 1.000000 | 0.107438 | 1.168831 |
| ett_sr | eth ↔ hotel | nearest_neighbor_distance | 0.014552 | 0.282234 | 0.016045 | 0.507967 |
| ett_sr | eth ↔ hotel | mean_neighbor_distance | 0.041935 | -0.524980 | 0.020250 | 0.611111 |
| ett_sr | eth ↔ hotel | median_neighbor_distance | -0.530098 | -1.364403 | 0.234618 | 0.921507 |
| ett_sr | eth ↔ hotel | max_neighbor_distance | 0.861062 | 0.871087 | 0.223526 | 1.634383 |
| ett_sr | eth ↔ hotel | mean_relative_speed | 0.318019 | 0.415033 | 1.439990 | 0.318189 |
| ett_sr | eth ↔ hotel | max_relative_speed | 0.537215 | 0.371676 | 1.207905 | 0.537215 |
| ett_sr | eth ↔ hotel | mean_closing_speed | 0.000931 | 0.023092 | 0.003609 | 0.025380 |
| ett_sr | eth ↔ hotel | max_closing_speed | 0.028144 | 0.087319 | 0.075858 | 0.046078 |
| ett_sr | eth ↔ hotel | attention_entropy | -0.082698 | -0.080859 | 0.423026 | 0.091026 |
| ett_sr | eth ↔ hotel | attention_max | -0.071062 | 0.018571 | 0.411085 | 0.095236 |
| ett_sr | eth ↔ hotel | attention_top2_sum | -0.103782 | -0.037063 | 0.483690 | 0.113029 |
| ett_sr | eth ↔ hotel | residual_norm | 0.239343 | 0.217837 | 1.662884 | 0.239343 |
| ett_sr | eth ↔ hotel | residual_final_norm | 0.533552 | 0.477528 | 1.644909 | 0.533552 |
| ett_sr | eth ↔ hotel | correction_ratio | 0.023648 | 0.033363 | 0.026040 | 0.240601 |
| ett_sr | eth ↔ univ | neighbor_count | -3.409091 | -3.000000 | 1.776756 | 3.409091 |
| ett_sr | eth ↔ univ | nearest_neighbor_distance | 0.480218 | 0.417295 | 1.011027 | 0.483831 |
| ett_sr | eth ↔ univ | mean_neighbor_distance | 1.966108 | 1.442356 | 1.124633 | 1.966108 |
| ett_sr | eth ↔ univ | median_neighbor_distance | 1.627988 | 0.689759 | 0.877685 | 1.629056 |
| ett_sr | eth ↔ univ | max_neighbor_distance | 4.094284 | 4.187221 | 1.204375 | 4.106232 |
| ett_sr | eth ↔ univ | mean_relative_speed | 0.373709 | 0.445768 | 1.903406 | 0.373709 |
| ett_sr | eth ↔ univ | max_relative_speed | 0.670950 | 0.546559 | 1.639626 | 0.670950 |
| ett_sr | eth ↔ univ | mean_closing_speed | 0.040661 | -0.039942 | 0.202062 | 0.101026 |
| ett_sr | eth ↔ univ | max_closing_speed | 0.018288 | -0.082636 | 0.058955 | 0.145136 |
| ett_sr | eth ↔ univ | attention_entropy | 0.033425 | 0.058365 | 0.211846 | 0.134334 |
| ett_sr | eth ↔ univ | attention_max | -0.009842 | 0.062398 | 0.063598 | 0.102162 |
| ett_sr | eth ↔ univ | attention_top2_sum | -0.044779 | 0.024990 | 0.211388 | 0.130209 |
| ett_sr | eth ↔ univ | residual_norm | 0.256847 | 0.229605 | 1.829067 | 0.256847 |
| ett_sr | eth ↔ univ | residual_final_norm | 0.567977 | 0.501936 | 1.778420 | 0.567977 |
| ett_sr | eth ↔ univ | correction_ratio | 0.278810 | 0.036881 | 0.344889 | 0.278810 |
| ett_sr | eth ↔ zara1 | neighbor_count | -0.438696 | 0.000000 | 0.172222 | 0.438696 |
| ett_sr | eth ↔ zara1 | nearest_neighbor_distance | -0.167873 | 0.212294 | 0.142471 | 0.529580 |
| ett_sr | eth ↔ zara1 | mean_neighbor_distance | 0.515325 | 0.017922 | 0.263214 | 0.796071 |
| ett_sr | eth ↔ zara1 | median_neighbor_distance | 0.385963 | -0.475981 | 0.182475 | 0.778894 |
| ett_sr | eth ↔ zara1 | max_neighbor_distance | 1.429868 | 1.507642 | 0.391843 | 2.136493 |
| ett_sr | eth ↔ zara1 | mean_relative_speed | 0.192612 | 0.248889 | 0.856616 | 0.192612 |
| ett_sr | eth ↔ zara1 | max_relative_speed | 0.390924 | 0.175692 | 0.906081 | 0.390924 |
| ett_sr | eth ↔ zara1 | mean_closing_speed | -0.001526 | -0.065882 | 0.006437 | 0.046292 |
| ett_sr | eth ↔ zara1 | max_closing_speed | -0.045914 | -0.048477 | 0.124942 | 0.076163 |
| ett_sr | eth ↔ zara1 | attention_entropy | -0.031248 | -0.012063 | 0.159403 | 0.043871 |
| ett_sr | eth ↔ zara1 | attention_max | -0.074276 | -0.000092 | 0.415341 | 0.078482 |
| ett_sr | eth ↔ zara1 | attention_top2_sum | -0.100841 | -0.037692 | 0.450618 | 0.100851 |
| ett_sr | eth ↔ zara1 | residual_norm | 0.200376 | 0.172660 | 1.398767 | 0.200805 |
| ett_sr | eth ↔ zara1 | residual_final_norm | 0.461811 | 0.404074 | 1.414756 | 0.462243 |
| ett_sr | eth ↔ zara1 | correction_ratio | 0.388774 | 0.058456 | 0.494118 | 0.388774 |
| ett_sr | eth ↔ zara2 | neighbor_count | -2.669508 | -3.000000 | 1.268598 | 2.669508 |
| ett_sr | eth ↔ zara2 | nearest_neighbor_distance | 0.327734 | 0.302901 | 0.572713 | 0.374304 |
| ett_sr | eth ↔ zara2 | mean_neighbor_distance | 0.907876 | 0.567528 | 0.489265 | 1.073557 |
| ett_sr | eth ↔ zara2 | median_neighbor_distance | 0.796869 | 0.005202 | 0.395655 | 0.917119 |
| ett_sr | eth ↔ zara2 | max_neighbor_distance | 1.382173 | 1.376715 | 0.376659 | 2.196270 |
| ett_sr | eth ↔ zara2 | mean_relative_speed | 0.372765 | 0.449532 | 1.895376 | 0.372765 |
| ett_sr | eth ↔ zara2 | max_relative_speed | 0.594666 | 0.468414 | 1.476204 | 0.594716 |
| ett_sr | eth ↔ zara2 | mean_closing_speed | 0.054702 | -0.018527 | 0.258772 | 0.074694 |
| ett_sr | eth ↔ zara2 | max_closing_speed | 0.032276 | -0.088989 | 0.098712 | 0.103518 |
| ett_sr | eth ↔ zara2 | attention_entropy | 0.057300 | 0.095224 | 0.351078 | 0.142930 |
| ett_sr | eth ↔ zara2 | attention_max | -0.075676 | -0.020923 | 0.485120 | 0.093780 |
| ett_sr | eth ↔ zara2 | attention_top2_sum | -0.105003 | -0.036581 | 0.495639 | 0.128163 |
| ett_sr | eth ↔ zara2 | residual_norm | 0.258172 | 0.248116 | 1.816666 | 0.258172 |
| ett_sr | eth ↔ zara2 | residual_final_norm | 0.573175 | 0.537146 | 1.778726 | 0.573175 |
| ett_sr | eth ↔ zara2 | correction_ratio | -0.309638 | -0.879940 | 0.344658 | 0.616006 |
| ett_sr | hotel ↔ univ | neighbor_count | -3.655844 | -4.000000 | 2.896200 | 3.655844 |
| ett_sr | hotel ↔ univ | nearest_neighbor_distance | 0.465666 | 0.135061 | 0.526034 | 0.465666 |
| ett_sr | hotel ↔ univ | mean_neighbor_distance | 1.924172 | 1.967336 | 1.323849 | 1.927166 |
| ett_sr | hotel ↔ univ | median_neighbor_distance | 2.158086 | 2.054161 | 1.298063 | 2.160554 |
| ett_sr | hotel ↔ univ | max_neighbor_distance | 3.233221 | 3.316134 | 1.418355 | 3.251939 |
| ett_sr | hotel ↔ univ | mean_relative_speed | 0.055690 | 0.030735 | 0.313289 | 0.056198 |
| ett_sr | hotel ↔ univ | max_relative_speed | 0.133735 | 0.174882 | 0.459805 | 0.145155 |
| ett_sr | hotel ↔ univ | mean_closing_speed | 0.039729 | -0.063035 | 0.192390 | 0.114477 |
| ett_sr | hotel ↔ univ | max_closing_speed | -0.009856 | -0.169956 | 0.033914 | 0.142369 |
| ett_sr | hotel ↔ univ | attention_entropy | 0.116123 | 0.139224 | 0.928666 | 0.160925 |
| ett_sr | hotel ↔ univ | attention_max | 0.061220 | 0.043827 | 0.656538 | 0.061220 |
| ett_sr | hotel ↔ univ | attention_top2_sum | 0.059003 | 0.062053 | 0.889731 | 0.059003 |
| ett_sr | hotel ↔ univ | residual_norm | 0.017503 | 0.011768 | 0.294374 | 0.026145 |
| ett_sr | hotel ↔ univ | residual_final_norm | 0.034425 | 0.024408 | 0.283010 | 0.051149 |
| ett_sr | hotel ↔ univ | correction_ratio | 0.255162 | 0.003518 | 0.518599 | 0.255218 |
| ett_sr | hotel ↔ zara1 | neighbor_count | -0.685449 | -1.000000 | 0.326756 | 0.894609 |
| ett_sr | hotel ↔ zara1 | nearest_neighbor_distance | -0.182426 | -0.069940 | 0.130757 | 0.258481 |
| ett_sr | hotel ↔ zara1 | mean_neighbor_distance | 0.473390 | 0.542903 | 0.278497 | 0.510767 |
| ett_sr | hotel ↔ zara1 | median_neighbor_distance | 0.916061 | 0.888422 | 0.470091 | 0.928769 |
| ett_sr | hotel ↔ zara1 | max_neighbor_distance | 0.568805 | 0.636555 | 0.215675 | 0.620644 |
| ett_sr | hotel ↔ zara1 | mean_relative_speed | -0.125407 | -0.166144 | 0.600534 | 0.132200 |
| ett_sr | hotel ↔ zara1 | max_relative_speed | -0.146291 | -0.195985 | 0.455189 | 0.155330 |
| ett_sr | hotel ↔ zara1 | mean_closing_speed | -0.002457 | -0.088974 | 0.010172 | 0.054583 |
| ett_sr | hotel ↔ zara1 | max_closing_speed | -0.074058 | -0.135796 | 0.210923 | 0.084550 |
| ett_sr | hotel ↔ zara1 | attention_entropy | 0.051450 | 0.068796 | 0.301246 | 0.052539 |
| ett_sr | hotel ↔ zara1 | attention_max | -0.003215 | -0.018663 | 0.024856 | 0.022883 |
| ett_sr | hotel ↔ zara1 | attention_top2_sum | 0.002941 | -0.000629 | 0.030013 | 0.025070 |
| ett_sr | hotel ↔ zara1 | residual_norm | -0.038967 | -0.045178 | 0.591674 | 0.038970 |
| ett_sr | hotel ↔ zara1 | residual_final_norm | -0.071742 | -0.073454 | 0.515738 | 0.071958 |
| ett_sr | hotel ↔ zara1 | correction_ratio | 0.365125 | 0.025094 | 0.801312 | 0.365125 |
| ett_sr | hotel ↔ zara2 | neighbor_count | -2.916261 | -4.000000 | 1.906430 | 2.916261 |
| ett_sr | hotel ↔ zara2 | nearest_neighbor_distance | 0.313181 | 0.020667 | 0.332812 | 0.341830 |
| ett_sr | hotel ↔ zara2 | mean_neighbor_distance | 0.865940 | 1.092509 | 0.547723 | 0.890838 |
| ett_sr | hotel ↔ zara2 | median_neighbor_distance | 1.326967 | 1.369605 | 0.721783 | 1.330020 |
| ett_sr | hotel ↔ zara2 | max_neighbor_distance | 0.521111 | 0.505628 | 0.195495 | 0.687787 |
| ett_sr | hotel ↔ zara2 | mean_relative_speed | 0.054746 | 0.034499 | 0.307340 | 0.055120 |
| ett_sr | hotel ↔ zara2 | max_relative_speed | 0.057451 | 0.096737 | 0.203861 | 0.148659 |
| ett_sr | hotel ↔ zara2 | mean_closing_speed | 0.053771 | -0.041620 | 0.248457 | 0.085963 |
| ett_sr | hotel ↔ zara2 | max_closing_speed | 0.004133 | -0.176308 | 0.013398 | 0.102899 |
| ett_sr | hotel ↔ zara2 | attention_entropy | 0.139997 | 0.176083 | 1.061966 | 0.178594 |
| ett_sr | hotel ↔ zara2 | attention_max | -0.004615 | -0.039495 | 0.048422 | 0.046537 |
| ett_sr | hotel ↔ zara2 | attention_top2_sum | -0.001221 | 0.000482 | 0.018391 | 0.015212 |
| ett_sr | hotel ↔ zara2 | residual_norm | 0.018829 | 0.030279 | 0.297247 | 0.023569 |
| ett_sr | hotel ↔ zara2 | residual_final_norm | 0.039622 | 0.059618 | 0.307203 | 0.051098 |
| ett_sr | hotel ↔ zara2 | correction_ratio | -0.333287 | -0.913302 | 0.529844 | 0.403430 |
| ett_sr | univ ↔ zara1 | neighbor_count | 2.970395 | 3.000000 | 1.772895 | 2.970395 |
| ett_sr | univ ↔ zara1 | nearest_neighbor_distance | -0.648091 | -0.205001 | 0.557895 | 0.648091 |
| ett_sr | univ ↔ zara1 | mean_neighbor_distance | -1.450782 | -1.424433 | 1.126972 | 1.450782 |
| ett_sr | univ ↔ zara1 | median_neighbor_distance | -1.242025 | -1.165739 | 0.850428 | 1.242025 |
| ett_sr | univ ↔ zara1 | max_neighbor_distance | -2.664416 | -2.679579 | 1.390273 | 2.665321 |
| ett_sr | univ ↔ zara1 | mean_relative_speed | -0.181097 | -0.196879 | 0.991177 | 0.182490 |
| ett_sr | univ ↔ zara1 | max_relative_speed | -0.280026 | -0.370867 | 1.036864 | 0.289037 |
| ett_sr | univ ↔ zara1 | mean_closing_speed | -0.042187 | -0.025939 | 0.234837 | 0.070842 |
| ett_sr | univ ↔ zara1 | max_closing_speed | -0.064202 | 0.034159 | 0.224395 | 0.138948 |
| ett_sr | univ ↔ zara1 | attention_entropy | -0.064673 | -0.070427 | 0.513745 | 0.112694 |
| ett_sr | univ ↔ zara1 | attention_max | -0.064435 | -0.062490 | 0.620192 | 0.070754 |
| ett_sr | univ ↔ zara1 | attention_top2_sum | -0.056062 | -0.062682 | 0.610259 | 0.067989 |
| ett_sr | univ ↔ zara1 | residual_norm | -0.056471 | -0.056945 | 0.977124 | 0.056525 |
| ett_sr | univ ↔ zara1 | residual_final_norm | -0.106166 | -0.097862 | 0.835796 | 0.106254 |
| ett_sr | univ ↔ zara1 | correction_ratio | 0.109964 | 0.021576 | 0.576301 | 0.109979 |
| ett_sr | univ ↔ zara2 | neighbor_count | 0.739583 | 0.000000 | 0.855941 | 0.739583 |
| ett_sr | univ ↔ zara2 | nearest_neighbor_distance | -0.152484 | -0.114395 | 0.283863 | 0.177185 |
| ett_sr | univ ↔ zara2 | mean_neighbor_distance | -1.058232 | -0.874827 | 0.939998 | 1.058393 |
| ett_sr | univ ↔ zara2 | median_neighbor_distance | -0.831119 | -0.684557 | 0.634551 | 0.840406 |
| ett_sr | univ ↔ zara2 | max_neighbor_distance | -2.712111 | -2.810506 | 1.387135 | 2.712593 |
| ett_sr | univ ↔ zara2 | mean_relative_speed | -0.000944 | 0.003763 | 0.006439 | 0.016205 |
| ett_sr | univ ↔ zara2 | max_relative_speed | -0.076285 | -0.078145 | 0.344380 | 0.081646 |
| ett_sr | univ ↔ zara2 | mean_closing_speed | 0.014042 | 0.021415 | 0.097498 | 0.030381 |
| ett_sr | univ ↔ zara2 | max_closing_speed | 0.013989 | -0.006353 | 0.060350 | 0.048632 |
| ett_sr | univ ↔ zara2 | attention_entropy | 0.023875 | 0.036860 | 0.374852 | 0.034104 |
| ett_sr | univ ↔ zara2 | attention_max | -0.065835 | -0.083322 | 1.173446 | 0.065835 |
| ett_sr | univ ↔ zara2 | attention_top2_sum | -0.060224 | -0.061571 | 1.057622 | 0.060224 |
| ett_sr | univ ↔ zara2 | residual_norm | 0.001325 | 0.018511 | 0.024136 | 0.013641 |
| ett_sr | univ ↔ zara2 | residual_final_norm | 0.005198 | 0.035210 | 0.044867 | 0.023130 |
| ett_sr | univ ↔ zara2 | correction_ratio | -0.588449 | -0.916820 | 1.242050 | 0.588449 |
| ett_sr | zara1 ↔ zara2 | neighbor_count | -2.230811 | -3.000000 | 1.183371 | 2.230811 |
| ett_sr | zara1 ↔ zara2 | nearest_neighbor_distance | 0.495607 | 0.090607 | 0.411389 | 0.495746 |
| ett_sr | zara1 ↔ zara2 | mean_neighbor_distance | 0.392550 | 0.549606 | 0.274561 | 0.426146 |
| ett_sr | zara1 ↔ zara2 | median_neighbor_distance | 0.410906 | 0.481183 | 0.247838 | 0.416350 |
| ett_sr | zara1 ↔ zara2 | max_neighbor_distance | -0.047694 | -0.130927 | 0.020187 | 0.344635 |
| ett_sr | zara1 ↔ zara2 | mean_relative_speed | 0.180153 | 0.200643 | 0.984078 | 0.180155 |
| ett_sr | zara1 ↔ zara2 | max_relative_speed | 0.203741 | 0.292722 | 0.782688 | 0.226533 |
| ett_sr | zara1 ↔ zara2 | mean_closing_speed | 0.056228 | 0.047354 | 0.294454 | 0.056518 |
| ett_sr | zara1 ↔ zara2 | max_closing_speed | 0.078190 | -0.040512 | 0.257021 | 0.091535 |
| ett_sr | zara1 ↔ zara2 | attention_entropy | 0.088547 | 0.107287 | 0.667638 | 0.129782 |
| ett_sr | zara1 ↔ zara2 | attention_max | -0.001400 | -0.020831 | 0.013238 | 0.048555 |
| ett_sr | zara1 ↔ zara2 | attention_top2_sum | -0.004162 | 0.001111 | 0.045282 | 0.037589 |
| ett_sr | zara1 ↔ zara2 | residual_norm | 0.057796 | 0.075457 | 0.935491 | 0.057796 |
| ett_sr | zara1 ↔ zara2 | residual_final_norm | 0.111364 | 0.133072 | 0.830640 | 0.111364 |
| ett_sr | zara1 ↔ zara2 | correction_ratio | -0.698412 | -0.938396 | 1.602263 | 0.698412 |


18. shift ranking

| 模型 | feature | mean abs(SMD) | max abs(SMD) | mean Wasserstein | 有效 pair 数 |
| --- | --- | --- | --- | --- | --- |
| emt_sr | neighbor_count | 1.226661 | 2.896200 | 2.109363 | 10 |
| emt_sr | attention_top2_sum | 1.085180 | 2.962145 | 0.167482 | 10 |
| emt_sr | attention_max | 0.938797 | 2.141289 | 0.147679 | 10 |
| emt_sr | mean_relative_speed | 0.929824 | 1.903406 | 0.187964 | 10 |
| emt_sr | residual_norm | 0.862546 | 1.771680 | 0.138931 | 10 |
| emt_sr | residual_final_norm | 0.856101 | 1.738273 | 0.306025 | 10 |
| emt_sr | max_relative_speed | 0.851260 | 1.639626 | 0.324016 | 10 |
| emt_sr | max_neighbor_distance | 0.682352 | 1.418355 | 2.035630 | 10 |
| emt_sr | correction_ratio | 0.653389 | 1.634410 | 0.401289 | 10 |
| emt_sr | mean_neighbor_distance | 0.638896 | 1.323849 | 1.071094 | 10 |
| emt_sr | median_neighbor_distance | 0.591319 | 1.298063 | 1.116470 | 10 |
| emt_sr | attention_entropy | 0.528883 | 1.193270 | 0.103766 | 10 |
| emt_sr | nearest_neighbor_distance | 0.398501 | 1.011027 | 0.428268 | 10 |
| emt_sr | mean_closing_speed | 0.154869 | 0.294454 | 0.066016 | 10 |
| emt_sr | max_closing_speed | 0.115847 | 0.257021 | 0.097983 | 10 |
| ett_sr | neighbor_count | 1.226661 | 2.896200 | 2.109363 | 10 |
| ett_sr | residual_norm | 0.982743 | 1.829067 | 0.117181 | 10 |
| ett_sr | residual_final_norm | 0.943407 | 1.778726 | 0.255190 | 10 |
| ett_sr | mean_relative_speed | 0.929824 | 1.903406 | 0.187964 | 10 |
| ett_sr | max_relative_speed | 0.851260 | 1.639626 | 0.324016 | 10 |
| ett_sr | max_neighbor_distance | 0.682352 | 1.418355 | 2.035630 | 10 |
| ett_sr | correction_ratio | 0.648007 | 1.602263 | 0.394480 | 10 |
| ett_sr | mean_neighbor_distance | 0.638896 | 1.323849 | 1.071094 | 10 |
| ett_sr | median_neighbor_distance | 0.591319 | 1.298063 | 1.116470 | 10 |
| ett_sr | attention_entropy | 0.499347 | 1.061966 | 0.108080 | 10 |
| ett_sr | attention_top2_sum | 0.429263 | 1.057622 | 0.073734 | 10 |
| ett_sr | nearest_neighbor_distance | 0.398501 | 1.011027 | 0.428268 | 10 |
| ett_sr | attention_max | 0.391184 | 1.173446 | 0.068544 | 10 |
| ett_sr | mean_closing_speed | 0.154869 | 0.294454 | 0.066016 | 10 |
| ett_sr | max_closing_speed | 0.115847 | 0.257021 | 0.097983 | 10 |


19. neighbor-count 分层

| 权重 | 模型 | 组 | unique n | prediction n | 非空 scene | base ADE | SR ADE | ADE gain | base FDE | SR FDE | FDE gain | residual | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene_equal | emt_sr | 0 | 10 | 120 | 2 | 0.697773 | 0.686008 | 0.011765 | 1.278845 | 1.257663 | 0.021182 | 0.203105 | 0.000000 |
| scene_equal | emt_sr | 1-2 | 65 | 780 | 3 | 0.717073 | 0.723723 | -0.006650 | 1.564191 | 1.490691 | 0.073500 | 0.378467 | 0.466212 |
| scene_equal | emt_sr | 3-4 | 225 | 2700 | 4 | 0.660015 | 0.616974 | 0.043042 | 1.354033 | 1.239060 | 0.114973 | 0.289866 | 0.800409 |
| scene_equal | emt_sr | 5-6 | 210 | 2520 | 4 | 0.534484 | 0.542475 | -0.007990 | 1.097081 | 1.140516 | -0.043435 | 0.257761 | 0.796634 |
| scene_equal | emt_sr | 7-8 | 3522 | 42264 | 5 | 0.512371 | 0.511407 | 0.000964 | 1.021366 | 1.045049 | -0.023683 | 0.244376 | 0.773087 |
| scene_equal | ett_sr | 0 | 10 | 120 | 2 | 0.690821 | 0.673323 | 0.017498 | 1.252737 | 1.272418 | -0.019681 | 0.317916 | 0.000000 |
| scene_equal | ett_sr | 1-2 | 65 | 780 | 3 | 0.740220 | 0.692667 | 0.047553 | 1.578893 | 1.414800 | 0.164094 | 0.239145 | 0.422709 |
| scene_equal | ett_sr | 3-4 | 225 | 2700 | 4 | 0.666672 | 0.602824 | 0.063848 | 1.360234 | 1.211815 | 0.148419 | 0.197772 | 0.597810 |
| scene_equal | ett_sr | 5-6 | 210 | 2520 | 4 | 0.534974 | 0.543261 | -0.008287 | 1.095707 | 1.123112 | -0.027404 | 0.174278 | 0.508890 |
| scene_equal | ett_sr | 7-8 | 3522 | 42264 | 5 | 0.500618 | 0.493912 | 0.006705 | 1.008985 | 1.014934 | -0.005950 | 0.177521 | 0.452915 |
| pooled | emt_sr | 0 | 10 | 120 | 2 | 0.948451 | 0.911725 | 0.036725 | 1.684609 | 1.594886 | 0.089723 | 0.194317 | 0.000000 |
| pooled | emt_sr | 1-2 | 65 | 780 | 3 | 0.509683 | 0.518998 | -0.009315 | 1.074651 | 1.034881 | 0.039770 | 0.320130 | 0.448090 |
| pooled | emt_sr | 3-4 | 225 | 2700 | 4 | 0.462797 | 0.442004 | 0.020792 | 0.929728 | 0.881779 | 0.047949 | 0.223048 | 0.767497 |
| pooled | emt_sr | 5-6 | 210 | 2520 | 4 | 0.362650 | 0.363829 | -0.001180 | 0.753825 | 0.776237 | -0.022411 | 0.193549 | 0.813019 |
| pooled | emt_sr | 7-8 | 3522 | 42264 | 5 | 0.480152 | 0.458484 | 0.021668 | 1.016707 | 0.976903 | 0.039804 | 0.191342 | 0.813634 |
| pooled | ett_sr | 0 | 10 | 120 | 2 | 0.936369 | 0.851823 | 0.084546 | 1.647737 | 1.549648 | 0.098089 | 0.330827 | 0.000000 |
| pooled | ett_sr | 1-2 | 65 | 780 | 3 | 0.527005 | 0.490490 | 0.036515 | 1.098460 | 0.989490 | 0.108970 | 0.200967 | 0.412773 |
| pooled | ett_sr | 3-4 | 225 | 2700 | 4 | 0.465263 | 0.431706 | 0.033558 | 0.933983 | 0.866367 | 0.067616 | 0.141704 | 0.606933 |
| pooled | ett_sr | 5-6 | 210 | 2520 | 4 | 0.359359 | 0.344702 | 0.014657 | 0.747971 | 0.727367 | 0.020604 | 0.140202 | 0.493009 |
| pooled | ett_sr | 7-8 | 3522 | 42264 | 5 | 0.487064 | 0.468413 | 0.018651 | 1.027926 | 0.995897 | 0.032029 | 0.104913 | 0.434619 |


20. nearest-distance 分层

| 权重 | 模型 | 组 | unique n | prediction n | 非空 scene | base ADE | SR ADE | ADE gain | base FDE | SR FDE | FDE gain | residual | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene_equal | emt_sr | no neighbor | 10 | 120 | 2 | 0.697773 | 0.686008 | 0.011765 | 1.278845 | 1.257663 | 0.021182 | 0.203105 | 0.000000 |
| scene_equal | emt_sr | <2m | 3859 | 46308 | 5 | 0.545858 | 0.534629 | 0.011230 | 1.122408 | 1.100290 | 0.022118 | 0.247033 | 0.785578 |
| scene_equal | emt_sr | 2-4m | 128 | 1536 | 5 | 0.647899 | 0.625910 | 0.021988 | 1.264558 | 1.236198 | 0.028359 | 0.284777 | 0.701689 |
| scene_equal | emt_sr | 4-8m | 28 | 336 | 4 | 0.729144 | 0.651384 | 0.077760 | 1.292931 | 1.082761 | 0.210170 | 0.327977 | 0.565891 |
| scene_equal | emt_sr | >8m | 7 | 84 | 1 | 0.332827 | 0.340221 | -0.007394 | 0.752307 | 0.619154 | 0.133153 | 0.333919 | 0.264967 |
| scene_equal | ett_sr | no neighbor | 10 | 120 | 2 | 0.690821 | 0.673323 | 0.017498 | 1.252737 | 1.272418 | -0.019681 | 0.317916 | 0.000000 |
| scene_equal | ett_sr | <2m | 3859 | 46308 | 5 | 0.549750 | 0.527300 | 0.022450 | 1.129189 | 1.080987 | 0.048202 | 0.163775 | 0.505135 |
| scene_equal | ett_sr | 2-4m | 128 | 1536 | 5 | 0.634256 | 0.629614 | 0.004642 | 1.261326 | 1.274733 | -0.013407 | 0.211916 | 0.434051 |
| scene_equal | ett_sr | 4-8m | 28 | 336 | 4 | 0.755026 | 0.707719 | 0.047307 | 1.309434 | 1.186872 | 0.122562 | 0.169635 | 0.385733 |
| scene_equal | ett_sr | >8m | 7 | 84 | 1 | 0.348030 | 0.275036 | 0.072994 | 0.783000 | 0.556158 | 0.226842 | 0.250176 | 0.256624 |
| pooled | emt_sr | no neighbor | 10 | 120 | 2 | 0.948451 | 0.911725 | 0.036725 | 1.684609 | 1.594886 | 0.089723 | 0.194317 | 0.000000 |
| pooled | emt_sr | <2m | 3859 | 46308 | 5 | 0.469267 | 0.449564 | 0.019702 | 0.992663 | 0.956911 | 0.035753 | 0.191769 | 0.810763 |
| pooled | emt_sr | 2-4m | 128 | 1536 | 5 | 0.573582 | 0.552168 | 0.021414 | 1.163850 | 1.128105 | 0.035746 | 0.275989 | 0.711853 |
| pooled | emt_sr | 4-8m | 28 | 336 | 4 | 0.637957 | 0.587269 | 0.050688 | 1.187851 | 0.995778 | 0.192072 | 0.280158 | 0.587850 |
| pooled | emt_sr | >8m | 7 | 84 | 1 | 0.332827 | 0.340221 | -0.007394 | 0.752307 | 0.619154 | 0.133153 | 0.333919 | 0.264967 |
| pooled | ett_sr | no neighbor | 10 | 120 | 2 | 0.936369 | 0.851823 | 0.084546 | 1.647737 | 1.549648 | 0.098089 | 0.330827 | 0.000000 |
| pooled | ett_sr | <2m | 3859 | 46308 | 5 | 0.475546 | 0.456465 | 0.019081 | 1.002598 | 0.969238 | 0.033359 | 0.107477 | 0.448244 |
| pooled | ett_sr | 2-4m | 128 | 1536 | 5 | 0.574761 | 0.551844 | 0.022917 | 1.173288 | 1.136040 | 0.037248 | 0.174808 | 0.427855 |
| pooled | ett_sr | 4-8m | 28 | 336 | 4 | 0.668073 | 0.610437 | 0.057635 | 1.224568 | 1.069619 | 0.154950 | 0.179073 | 0.404141 |
| pooled | ett_sr | >8m | 7 | 84 | 1 | 0.348030 | 0.275036 | 0.072994 | 0.783000 | 0.556158 | 0.226842 | 0.250176 | 0.256624 |

边界：no neighbor、<2、[2,4)、[4,8]、>8；单位米。未调整 radius。

21. closing-speed 分层

| 权重 | 模型 | 组 | unique n | prediction n | 非空 scene | base ADE | SR ADE | ADE gain | base FDE | SR FDE | FDE gain | residual | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene_equal | emt_sr | Q1 | 1008 | 12096 | 5 | 0.540652 | 0.524003 | 0.016650 | 1.110875 | 1.058300 | 0.052574 | 0.226345 | 0.734967 |
| scene_equal | emt_sr | Q2 | 1008 | 12096 | 5 | 0.512860 | 0.501418 | 0.011442 | 1.039877 | 1.004272 | 0.035605 | 0.229448 | 0.758131 |
| scene_equal | emt_sr | Q3 | 1008 | 12096 | 5 | 0.571583 | 0.610389 | -0.038806 | 1.168535 | 1.291350 | -0.122815 | 0.247537 | 0.771819 |
| scene_equal | emt_sr | Q4 | 1008 | 12096 | 5 | 0.590875 | 0.564171 | 0.026704 | 1.169796 | 1.133656 | 0.036140 | 0.269247 | 0.754972 |
| scene_equal | ett_sr | Q1 | 1008 | 12096 | 5 | 0.546243 | 0.512591 | 0.033652 | 1.116851 | 1.037255 | 0.079595 | 0.165914 | 0.484952 |
| scene_equal | ett_sr | Q2 | 1008 | 12096 | 5 | 0.501265 | 0.482065 | 0.019200 | 1.026110 | 0.985061 | 0.041049 | 0.150686 | 0.460228 |
| scene_equal | ett_sr | Q3 | 1008 | 12096 | 5 | 0.552014 | 0.573106 | -0.021092 | 1.137246 | 1.200235 | -0.062988 | 0.168588 | 0.477309 |
| scene_equal | ett_sr | Q4 | 1008 | 12096 | 5 | 0.597920 | 0.574182 | 0.023738 | 1.188047 | 1.154549 | 0.033498 | 0.180624 | 0.479340 |
| pooled | emt_sr | Q1 | 1008 | 12096 | 5 | 0.400492 | 0.381037 | 0.019454 | 0.834749 | 0.796435 | 0.038313 | 0.171896 | 0.802026 |
| pooled | emt_sr | Q2 | 1008 | 12096 | 5 | 0.421647 | 0.403517 | 0.018130 | 0.892998 | 0.857684 | 0.035314 | 0.174765 | 0.826295 |
| pooled | emt_sr | Q3 | 1008 | 12096 | 5 | 0.485702 | 0.466905 | 0.018796 | 1.031342 | 0.996223 | 0.035119 | 0.193773 | 0.807177 |
| pooled | emt_sr | Q4 | 1008 | 12096 | 5 | 0.590965 | 0.567477 | 0.023488 | 1.243920 | 1.204102 | 0.039818 | 0.240805 | 0.776967 |
| pooled | ett_sr | Q1 | 1008 | 12096 | 5 | 0.408527 | 0.386788 | 0.021739 | 0.848730 | 0.805832 | 0.042898 | 0.107457 | 0.457968 |
| pooled | ett_sr | Q2 | 1008 | 12096 | 5 | 0.427343 | 0.410265 | 0.017078 | 0.900116 | 0.871048 | 0.029067 | 0.096697 | 0.441246 |
| pooled | ett_sr | Q3 | 1008 | 12096 | 5 | 0.491541 | 0.473859 | 0.017683 | 1.043330 | 1.012412 | 0.030918 | 0.104209 | 0.434983 |
| pooled | ett_sr | Q4 | 1008 | 12096 | 5 | 0.596407 | 0.574000 | 0.022406 | 1.250932 | 1.214520 | 0.036411 | 0.135290 | 0.449187 |

分位点=[0.11583204297394213, 0.24823306232121983, 0.4323665380495827] meter / sampling step，来自全部唯一 source validation observation 窗口。仅诊断分组，重复分位点可产生空组，不随机打散相同值。

22. N=8 saturation

| 模型 | scene | N8比例 | 组 | unique n | ADE gain | FDE gain | residual | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emt_sr | eth | 0.227273 | N=8 | 15 | -0.148549 | -0.458609 | 0.496031 | 0.722154 |
| emt_sr | eth | 0.227273 | N<8 | 51 | 0.042139 | 0.155883 | 0.436691 | 0.700023 |
| emt_sr | hotel | 0.084416 | N=8 | 13 | 0.011924 | 0.004585 | 0.221490 | 0.680413 |
| emt_sr | hotel | 0.084416 | N<8 | 141 | 0.020539 | 0.038910 | 0.201210 | 0.642759 |
| emt_sr | univ | 1.000000 | N=8 | 2836 | 0.022055 | 0.042255 | 0.201281 | 0.814091 |
| emt_sr | univ | 1.000000 | N<8 | 0 | NA | NA | NA | NA |
| emt_sr | zara1 | 0.286184 | N=8 | 87 | 0.027294 | 0.051675 | 0.192884 | 0.818262 |
| emt_sr | zara1 | 0.286184 | N<8 | 217 | -0.001739 | -0.008723 | 0.242844 | 0.727644 |
| emt_sr | zara2 | 0.653274 | N=8 | 439 | 0.024722 | 0.041563 | 0.125804 | 0.822428 |
| emt_sr | zara2 | 0.653274 | N<8 | 233 | 0.009169 | 0.010525 | 0.132997 | 0.829451 |
| ett_sr | eth | 0.227273 | N=8 | 15 | -0.085657 | -0.270677 | 0.427473 | 0.465660 |
| ett_sr | eth | 0.227273 | N<8 | 51 | 0.062533 | 0.175599 | 0.338667 | 0.475665 |
| ett_sr | hotel | 0.084416 | N=8 | 13 | 0.000331 | -0.019302 | 0.147408 | 0.493680 |
| ett_sr | hotel | 0.084416 | N<8 | 141 | 0.025420 | 0.046157 | 0.116934 | 0.561843 |
| ett_sr | univ | 1.000000 | N=8 | 2836 | 0.018314 | 0.031968 | 0.102003 | 0.439966 |
| ett_sr | univ | 1.000000 | N<8 | 0 | NA | NA | NA | NA |
| ett_sr | zara1 | 0.286184 | N=8 | 87 | 0.029097 | 0.046262 | 0.135819 | 0.475280 |
| ett_sr | zara1 | 0.286184 | N<8 | 217 | 0.014231 | 0.024870 | 0.167557 | 0.516410 |
| ett_sr | zara2 | 0.653274 | N=8 | 439 | 0.023313 | 0.043118 | 0.100457 | 0.390606 |
| ett_sr | zara2 | 0.653274 | N<8 | 233 | 0.026825 | 0.043970 | 0.101096 | 0.464111 |

当前 N=8 上限造成邻居计数右截断，不能解释为所有 scene 的真实总邻居数最多为八。

23. EMT vs ETT paired response difference

| scene | Δ residual | Δ entropy | Δ ADE gain | Δ FDE gain | Δ ratio |
| --- | --- | --- | --- | --- | --- |
| eth | 0.091327 | 0.231661 | -0.030052 | -0.057947 | 0.012476 |
| hotel | 0.083415 | 0.089848 | -0.003490 | -0.004619 | 0.156483 |
| univ | 0.099278 | 0.374124 | 0.003741 | 0.010287 | 0.135982 |
| zara1 | 0.070072 | 0.248938 | -0.011915 | -0.022430 | 0.038761 |
| zara2 | 0.027620 | 0.408771 | -0.005201 | -0.012612 | 0.076562 |

方向 EMT−ETT；逐窗口、可用 fold/seed 完全配对。仅描述响应差异，不能证明 Mamba 导致某一交互机制。

24. feature vs ADE_gain correlations

| 权重 | 模型 | feature | n | scenes | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| scene_equal | emt_sr | neighbor_count | 4032 | 5 | -0.049160 | 0.069650 |
| scene_equal | emt_sr | mean_neighbor_distance | 4022 | 5 | 0.205038 | 0.031285 |
| scene_equal | emt_sr | nearest_neighbor_distance | 4022 | 5 | 0.017457 | -0.077556 |
| scene_equal | emt_sr | mean_relative_speed | 4022 | 5 | 0.040942 | -0.071525 |
| scene_equal | emt_sr | mean_closing_speed | 4032 | 5 | 0.006769 | 0.021899 |
| scene_equal | emt_sr | max_closing_speed | 4032 | 5 | -0.020870 | 0.034614 |
| scene_equal | emt_sr | attention_entropy | 4032 | 5 | -0.006472 | -0.032570 |
| scene_equal | emt_sr | attention_max | 4032 | 5 | -0.016761 | -0.101448 |
| scene_equal | emt_sr | residual_norm | 4032 | 5 | 0.187949 | -0.035554 |
| scene_equal | emt_sr | correction_ratio | 4032 | 5 | -0.066574 | -0.041895 |
| scene_equal | ett_sr | neighbor_count | 4032 | 5 | -0.177841 | -0.054065 |
| scene_equal | ett_sr | mean_neighbor_distance | 4022 | 5 | 0.257113 | 0.070763 |
| scene_equal | ett_sr | nearest_neighbor_distance | 4022 | 5 | 0.015327 | -0.112463 |
| scene_equal | ett_sr | mean_relative_speed | 4022 | 5 | 0.133461 | -0.047012 |
| scene_equal | ett_sr | mean_closing_speed | 4032 | 5 | -0.078520 | -0.069312 |
| scene_equal | ett_sr | max_closing_speed | 4032 | 5 | -0.089096 | -0.057246 |
| scene_equal | ett_sr | attention_entropy | 4032 | 5 | -0.009787 | -0.051572 |
| scene_equal | ett_sr | attention_max | 4032 | 5 | -0.025996 | -0.016247 |
| scene_equal | ett_sr | residual_norm | 4032 | 5 | 0.307480 | 0.101431 |
| scene_equal | ett_sr | correction_ratio | 4032 | 5 | -0.209441 | -0.118116 |
| pooled | emt_sr | neighbor_count | 4032 | 5 | 0.038574 | 0.063948 |
| pooled | emt_sr | mean_neighbor_distance | 4022 | 5 | 0.021658 | -0.013493 |
| pooled | emt_sr | nearest_neighbor_distance | 4022 | 5 | -0.023119 | -0.066549 |
| pooled | emt_sr | mean_relative_speed | 4022 | 5 | -0.020496 | 0.005873 |
| pooled | emt_sr | mean_closing_speed | 4032 | 5 | -0.005125 | 0.003209 |
| pooled | emt_sr | max_closing_speed | 4032 | 5 | -0.005299 | 0.013570 |
| pooled | emt_sr | attention_entropy | 4032 | 5 | 0.001145 | -0.009803 |
| pooled | emt_sr | attention_max | 4032 | 5 | -0.028170 | -0.035342 |
| pooled | emt_sr | residual_norm | 4032 | 5 | 0.180890 | 0.112203 |
| pooled | emt_sr | correction_ratio | 4032 | 5 | -0.063868 | -0.007648 |
| pooled | ett_sr | neighbor_count | 4032 | 5 | -0.085461 | -0.012358 |
| pooled | ett_sr | mean_neighbor_distance | 4022 | 5 | 0.138759 | 0.083248 |
| pooled | ett_sr | nearest_neighbor_distance | 4022 | 5 | 0.081515 | 0.053353 |
| pooled | ett_sr | mean_relative_speed | 4022 | 5 | 0.083572 | 0.040628 |
| pooled | ett_sr | mean_closing_speed | 4032 | 5 | 0.008187 | 0.016131 |
| pooled | ett_sr | max_closing_speed | 4032 | 5 | 0.012170 | 0.027913 |
| pooled | ett_sr | attention_entropy | 4032 | 5 | 0.013773 | -0.008064 |
| pooled | ett_sr | attention_max | 4032 | 5 | 0.014088 | 0.011949 |
| pooled | ett_sr | residual_norm | 4032 | 5 | 0.286660 | 0.145798 |
| pooled | ett_sr | correction_ratio | 4032 | 5 | -0.078764 | -0.041611 |

Pearson/Spearman 无 p-value，correlation != causality。

25. feature vs residual norm / attention response correlations

| 权重 | 模型 | feature | n | scenes | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| scene_equal | emt_sr | neighbor_count | 4032 | 5 | -0.180955 | -0.226573 |
| scene_equal | emt_sr | mean_neighbor_distance | 4022 | 5 | 0.278181 | 0.143191 |
| scene_equal | emt_sr | nearest_neighbor_distance | 4022 | 5 | 0.135258 | 0.300461 |
| scene_equal | emt_sr | mean_relative_speed | 4022 | 5 | 0.620549 | 0.638193 |
| scene_equal | emt_sr | mean_closing_speed | 4032 | 5 | 0.135044 | 0.146571 |
| scene_equal | emt_sr | max_closing_speed | 4032 | 5 | 0.128508 | 0.108810 |
| scene_equal | emt_sr | attention_entropy | 4032 | 5 | -0.027464 | -0.040469 |
| scene_equal | emt_sr | attention_max | 4032 | 5 | 0.182077 | 0.298135 |
| scene_equal | ett_sr | neighbor_count | 4032 | 5 | -0.215451 | -0.222439 |
| scene_equal | ett_sr | mean_neighbor_distance | 4022 | 5 | 0.431124 | 0.324544 |
| scene_equal | ett_sr | nearest_neighbor_distance | 4022 | 5 | 0.159279 | 0.338408 |
| scene_equal | ett_sr | mean_relative_speed | 4022 | 5 | 0.579655 | 0.664343 |
| scene_equal | ett_sr | mean_closing_speed | 4032 | 5 | 0.066102 | 0.131557 |
| scene_equal | ett_sr | max_closing_speed | 4032 | 5 | 0.058337 | 0.107130 |
| scene_equal | ett_sr | attention_entropy | 4032 | 5 | -0.027510 | 0.068397 |
| scene_equal | ett_sr | attention_max | 4032 | 5 | -0.102532 | -0.069178 |
| pooled | emt_sr | neighbor_count | 4032 | 5 | -0.114256 | -0.046936 |
| pooled | emt_sr | mean_neighbor_distance | 4022 | 5 | 0.192419 | 0.137665 |
| pooled | emt_sr | nearest_neighbor_distance | 4022 | 5 | 0.198851 | 0.187782 |
| pooled | emt_sr | mean_relative_speed | 4022 | 5 | 0.573677 | 0.615465 |
| pooled | emt_sr | mean_closing_speed | 4032 | 5 | 0.286063 | 0.332656 |
| pooled | emt_sr | max_closing_speed | 4032 | 5 | 0.244531 | 0.264643 |
| pooled | emt_sr | attention_entropy | 4032 | 5 | -0.238130 | -0.318766 |
| pooled | emt_sr | attention_max | 4032 | 5 | 0.244955 | 0.388098 |
| pooled | ett_sr | neighbor_count | 4032 | 5 | -0.265249 | -0.176048 |
| pooled | ett_sr | mean_neighbor_distance | 4022 | 5 | 0.335649 | 0.278180 |
| pooled | ett_sr | nearest_neighbor_distance | 4022 | 5 | 0.266118 | 0.254184 |
| pooled | ett_sr | mean_relative_speed | 4022 | 5 | 0.562470 | 0.661374 |
| pooled | ett_sr | mean_closing_speed | 4032 | 5 | 0.242601 | 0.354443 |
| pooled | ett_sr | max_closing_speed | 4032 | 5 | 0.209571 | 0.297576 |
| pooled | ett_sr | attention_entropy | 4032 | 5 | 0.062256 | 0.164278 |
| pooled | ett_sr | attention_max | 4032 | 5 | -0.025665 | -0.097086 |

| 权重 | 模型 | feature | n | scenes | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| scene_equal | emt_sr | neighbor_count vs attention_entropy | 4032 | 5 | 0.487736 | 0.178527 |
| scene_equal | emt_sr | neighbor_count vs attention_max | 4032 | 5 | -0.528279 | -0.628122 |
| scene_equal | emt_sr | max_closing_speed vs attention_max | 4032 | 5 | -0.081247 | -0.149417 |
| scene_equal | emt_sr | nearest_neighbor_distance vs attention_max | 4022 | 5 | 0.365733 | 0.278730 |
| scene_equal | emt_sr | mean_neighbor_distance vs attention_max | 4022 | 5 | 0.339421 | 0.380101 |
| scene_equal | ett_sr | neighbor_count vs attention_entropy | 4032 | 5 | -0.088915 | -0.479544 |
| scene_equal | ett_sr | neighbor_count vs attention_max | 4032 | 5 | -0.139552 | -0.420180 |
| scene_equal | ett_sr | max_closing_speed vs attention_max | 4032 | 5 | -0.058258 | -0.277891 |
| scene_equal | ett_sr | nearest_neighbor_distance vs attention_max | 4022 | 5 | 0.379432 | 0.192147 |
| scene_equal | ett_sr | mean_neighbor_distance vs attention_max | 4022 | 5 | 0.077622 | 0.035045 |
| pooled | emt_sr | neighbor_count vs attention_entropy | 4032 | 5 | 0.385513 | 0.089186 |
| pooled | emt_sr | neighbor_count vs attention_max | 4032 | 5 | -0.611715 | -0.376265 |
| pooled | emt_sr | max_closing_speed vs attention_max | 4032 | 5 | 0.036033 | 0.132082 |
| pooled | emt_sr | nearest_neighbor_distance vs attention_max | 4022 | 5 | 0.330782 | 0.123113 |
| pooled | emt_sr | mean_neighbor_distance vs attention_max | 4022 | 5 | 0.480373 | 0.408804 |
| pooled | ett_sr | neighbor_count vs attention_entropy | 4032 | 5 | -0.257958 | -0.319256 |
| pooled | ett_sr | neighbor_count vs attention_max | 4032 | 5 | -0.360112 | -0.335653 |
| pooled | ett_sr | max_closing_speed vs attention_max | 4032 | 5 | -0.060033 | -0.108675 |
| pooled | ett_sr | nearest_neighbor_distance vs attention_max | 4022 | 5 | 0.284219 | 0.089018 |
| pooled | ett_sr | mean_neighbor_distance vs attention_max | 4022 | 5 | 0.182742 | 0.136170 |

count>=2 及固定 neighbor_count 内的 attention 相关性完整保存于 emt_sr/correlation_audit.json、ett_sr/correlation_audit.json，避免 n<=1 entropy=0 的定义驱动解释。

26. scene classifier

| 模型 | 权重 | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| logistic | pooled | 0.661954 | 0.360315 |
| logistic | scene_equal | 0.404367 | 0.374766 |
| majority | pooled | 0.703373 | 0.165172 |
| majority | scene_equal | 0.200000 | 0.066667 |
| random_frequency | pooled | 0.506200 | 0.177237 |
| random_frequency | scene_equal | 0.177519 | 0.112111 |

{'executed': True, 'feature_names': ['neighbor_count', 'mean_neighbor_distance', 'nearest_neighbor_distance', 'mean_relative_speed', 'mean_closing_speed'], 'unique_window_samples': 4032, 'groups': 'scene,ped_id; repeated windows/folds/seeds deduplicated; all pedestrian windows remain together', 'cv': '5-fold StratifiedGroupKFold; standardization fitted inside each training fold; no tuning', 'folds': [{'fold': 0, 'train_samples': 3385, 'validation_samples': 647, 'train_groups': 135, 'validation_groups': 34, 'pedestrian_overlap': 0, 'train_scene_count': 5, 'validation_scene_count': 5, 'logistic_iterations': [46]}, {'fold': 1, 'train_samples': 3172, 'validation_samples': 860, 'train_groups': 137, 'validation_groups': 32, 'pedestrian_overlap': 0, 'train_scene_count': 5, 'validation_scene_count': 5, 'logistic_iterations': [44]}, {'fold': 2, 'train_samples': 3180, 'validation_samples': 852, 'train_groups': 134, 'validation_groups': 35, 'pedestrian_overlap': 0, 'train_scene_count': 5, 'validation_scene_count': 4, 'logistic_iterations': [38]}, {'fold': 3, 'train_samples': 3275, 'validation_samples': 757, 'train_groups': 136, 'validation_groups': 33, 'pedestrian_overlap': 0, 'train_scene_count': 5, 'validation_scene_count': 5, 'logistic_iterations': [52]}, {'fold': 4, 'train_samples': 3116, 'validation_samples': 916, 'train_groups': 134, 'validation_groups': 35, 'pedestrian_overlap': 0, 'train_scene_count': 5, 'validation_scene_count': 5, 'logistic_iterations': [43]}], 'no_significance_tests': True}
仅五个观察统计作为输入，无绝对坐标、ped/frame ID、prediction error、未来轨迹；scene/ped 只用于 CV 分组。ETH 仅 9 名目标 pedestrian，保留每 fold 样本/组明细，不作显著性宣称。

27. SOCIAL DENSITY SHIFT

SUPPORTED；{'leading_feature': 'neighbor_count', 'score_mean_abs_SMD': 1.2266607308472621}

28. SOCIAL DISTANCE SHIFT

SUPPORTED；{'leading_feature': 'max_neighbor_distance', 'score_mean_abs_SMD': 0.6823523484179881}

29. RELATIVE-MOTION SHIFT

SUPPORTED；{'leading_feature': 'mean_relative_speed', 'score_mean_abs_SMD': 0.9298244526308382}

30. ATTENTION RESPONSE SHIFT

SUPPORTED；{'leading_feature': 'attention_top2_sum', 'score_mean_abs_SMD': 0.757221438307551}

31. RESIDUAL RESPONSE SHIFT

SUPPORTED；{'leading_feature': 'residual_norm', 'score_mean_abs_SMD': 0.9226445358567972}

32. PRIMARY SHIFT FACTOR

neighbor_count；由 source validation 10 scene pairs 的 mean_abs_SMD 排名选出。

33. Candidate A/B/C

Candidate A: density normalization；依据 SOCIAL DENSITY SHIFT/neighbor_count，mean |SMD|=1.226661；仅建议，未实现。
Candidate B: motion-aware interaction representation；依据 RELATIVE-MOTION SHIFT/mean_relative_speed，mean |SMD|=0.929824；仅建议，未实现。
Candidate C: confidence-normalized residual correction；依据 RESIDUAL RESPONSE SHIFT/residual_norm，mean |SMD|=0.922645；仅建议，未实现。

34. summary.md

results/social_distribution_shift/summary.md

35. brain_report.md

results/social_distribution_shift/brain_report.md；comparison.json、per_scene、per_fold、pairwise_shift 保存完整精度和全部分布摘要。

主要 source-validation shift factor 为 neighbor_count，EMT/ETT 等权 mean |SMD|=1.226661。它是分布依赖性最大变量，不是因果解释或模型优劣排名。
N=8 饱和比例：eth=0.227273, hotel=0.084416, univ=1.000000, zara1=0.286184, zara2=0.653274. 计数右截断使该审计无法观察超过八名邻居的密度差异。
EMT scene-equal ADE gain 相关性（按 |Spearman| 排序）：attention_max: Pearson=-0.016761, Spearman=-0.101448, nearest_neighbor_distance: Pearson=0.017457, Spearman=-0.077556, mean_relative_speed: Pearson=0.040942, Spearman=-0.071525, neighbor_count: Pearson=-0.049160, Spearman=0.069650, correction_ratio: Pearson=-0.066574, Spearman=-0.041895. 残差大小、比值和 gain 共享预测变量，相关性不能解释为观测因素的因果作用。
Attention 响应需结合 n>=2 和固定邻居 count 内相关性；n<=1 entropy=0、top2 在 n<=2 时饱和是定义引入的效应。不同 scene 的 prediction response 还混合 checkpoint、pedestrian 和观测轨迹组成，source 审计不能证明正式 held-out 失败的机制。


补充解释（所有数字来自上述 source-validation 统计）：

计数的 scene dependency 最大，但不代表它最能解释 gain。最近距离的 mean |SMD|=0.398501；mean/max closing 的值分别为 0.154869/0.115847。RELATIVE-MOTION SHIFT 的 SUPPORTED 由相对速度而非径向 closing 驱动；SOCIAL DISTANCE SHIFT 由距离集合变量驱动，不能推广为每个距离变量都强。

EMT scene 等权：观察变量中 ADE gain 的最大 |Pearson| 来自 mean_neighbor_distance (0.205038)，最大 |Spearman| 来自 nearest_neighbor_distance (-0.077556)；相对速度与 residual norm 的 Pearson/Spearman 为 0.620549/0.638193。gain 的单调关联总体较弱，残差大小与相对速度的关联较强。两种相关性排序和符号均保存在 association_rankings，不把残差与 gain 的共享预测量当成观察因素的因果作用。

ETT scene 等权：观察变量中 ADE gain 的最大 |Pearson| 来自 mean_neighbor_distance (0.257113)，最大 |Spearman| 来自 nearest_neighbor_distance (-0.112463)；相对速度与 residual norm 的 Pearson/Spearman 为 0.579655/0.664343。gain 的单调关联总体较弱，残差大小与相对速度的关联较强。两种相关性排序和符号均保存在 association_rankings，不把残差与 gain 的共享预测量当成观察因素的因果作用。

EMT 的 count–entropy Pearson 从全体 0.487736 降为 n>=2 的 0.032970 (Spearman=0.003447)，全体相关性受到 n<=1 entropy=0 的定义影响。n>=2 的 count–attention_max 仍有关联 (-0.779271/-0.749458)；固定 N=8 后，mean distance–attention_max 为 0.642386/0.624061，max closing–attention_max 为 0.322871/0.334225。因此 attention concentration 与密度相关，但本审计不支持“仅由数量驱动、与交互状态无关”。

ETT 的 n>=2 count–entropy 为 -0.800570/-0.770540；固定 N=8 的 mean distance–attention_max 为 -0.121852/-0.119281，max closing–attention_max 为 -0.098293/-0.048915。这显示 EMT/ETT 响应不同；backbone 和各自训练的 SR 模块共同参与响应，不能归因于单一 Mamba 机制。

ETH source-validation 的 N=8 只有 15 个唯一窗口，EMT ADE gain=-0.148549，N<8 为 0.042139；ETT 对应 -0.085657/0.062533。这不是 ETH formal test。其他 scene 的 N=8 ADE gain 为正，UNIV 没有 N<8 样本可作内部比较；不能断言 N=8 普遍有害，不能据此修改 N 或 radius。

同一窗口配对后，EMT 的 mean residual norm 和 entropy 的 scene 均值在五个 scene 都高于 ETT。EMT−ETT 的 ADE gain 均值只有 UNIV 为正，其他四个 scene 为负；更大的 correction 不意味着更大的 gain。本阶段只报告响应现象，不给出机制因果解释或模型优劣排名。

按 pedestrian 分组的 scene classifier，logistic scene 等权 Accuracy=0.404367，Macro F1=0.374766，高于 majority 的 0.200000/0.066667；但 pooled Accuracy=0.661954 低于 majority 0.703373。五个观察统计含有一定 scene 识别信息，不能称为非常容易识别 scene，也不宣称显著性或证明 domain shift。ETH 只有 9 名目标 pedestrian，窗口相关，CV 第 2 fold 的 validation 仅含 4 类；分组避免目标 pedestrian 跨 fold，但不是对新 recording/domain 的验证。

本阶段 scene 等权 ADE/FDE gain：EMT=0.013314/0.026771，ETT=0.022697/0.044235。先对同一窗口的四 source folds×三 seeds 平均，再对五 scene 等权；这与上一阶段按 fold 等权、fold 内按样本数汇总的统计对象不同。UNIV 占唯一窗口的 70.3373%，主结论使用 scene 等权；新的等权均值不覆盖或重算旧阶段结论。

已停止；候选方向由大脑 AI 后续选择。
SOCIAL DENSITY SHIFT: SUPPORTED
SOCIAL DISTANCE SHIFT: SUPPORTED
RELATIVE-MOTION SHIFT: SUPPORTED
ATTENTION RESPONSE SHIFT: SUPPORTED
RESIDUAL RESPONSE SHIFT: SUPPORTED
PRIMARY SHIFT FACTOR: neighbor_count
