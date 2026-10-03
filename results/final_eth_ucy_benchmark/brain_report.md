# Final ETH/UCY Benchmark：大脑 AI 交付报告

正式 held-out 未复现 Clean Validation 的 social 收益：EMT-SR 相比 EMT，ADE 增加 3.0380%、FDE 增加 3.3809%，两指标各为 7/15 wins。退化主要集中在 HOTEL，ZARA2 也退化。结果原样保留，不调参、不重测。

60/60 正式训练与 60/60 正式 held-out test 均完成。协议在 test 打开前冻结；所有结果原样报告，不根据 test 调整模型。

## 1. Branch

`feat/final_eth_ucy_benchmark`，基于 `feat/clean_social_validation` / `458dc02950d3949a653be729f35bfc3cc7d090c6`。

## 2. Latest commit

最新为包含本报告的 `exp: evaluate final Mamba social residual benchmark` 提交；最终 SHA 见交付消息和 GitHub 分支提交记录。协议记录的代码 commit：`2ba6f509aab415ff2704b62348e4278a171fe4dd`。

## 3. Push status

交付前核对本地与 GitHub 分支 HEAD、完整文件树一致，最终推送状态见交付消息。Checkpoint 和数据保留在本地；逐样本预测以 NPZ 保存在本阶段 results/predictions/。

## 4. Protocol frozen SHA

`e9240c2f042282e370f007ff9310c39a21fcf747e3b93a0f96317960f9f28914`。协议文件：`results/final_eth_ucy_benchmark/protocol_frozen.json`；配置/代码/数据 SHA：`data_audit/protocol_hashes.json`。

## 5. Final epoch determination

ETT/EMT 从第一阶段、ETT-SR/EMT-SR 从 Clean Social Validation 读取各 fold/model 三 seed 的 best_epoch，中位数四舍五入取整数。来源及哈希在 `data_audit/epoch_determination.json`，数字不人工选择；test 打开后保持不变。Final train 无 validation，因此冻结为无 scheduler、固定原始 lr=1e-3；其余 optimizer/loss/batch/clip/结构不变。

## 6. 每 fold/model final epoch

`configs/final_training_epochs.json`：

| Fold | ETT | EMT | ETT-SR | EMT-SR |
|---|---|---|---|---|
| ETH | 38 | 10 | 10 | 14 |
| HOTEL | 11 | 14 | 11 | 15 |
| UNIV | 21 | 12 | 13 | 4 |
| ZARA1 | 30 | 12 | 34 | 19 |
| ZARA2 | 27 | 14 | 30 | 16 |

## 7. 60/60 training

全部完成，均使用四个 source scene 的所有 eligible target windows，target-only 从随机初始化重训。SR 加载同 fold/seed 的 final full-source backbone，并冻结全部 temporal encoder/base decoder。最后一个冻结 epoch 的 checkpoint 为最终模型。`FINAL_TRAINING_COMPLETE.json` 记录 60 组 checkpoint/log 哈希。

## 8. Test 打开顺序

全部 60 个模型完成、checkpoint/log/协议/epoch/finite 核验通过后，才生成 completion 文件并提交 pre-test training 证据；随后显式 `--enable-heldout-evaluation`。全部 access timestamp 晚于 completion timestamp，60 个 fold/model/seed 各访问一次。`test_metrics_accessed_before_freeze=false`。Smoke 只导出 source 样本，未预测 ETH 正式 test。

## 9. Train/test sample counts

五个正式 fold 的全量窗口数：

| Held-out | Train scenes | Train samples | Test samples |
|---|---|---|---|
| ETH | HOTEL, UNIV, ZARA1, ZARA2 | 33797 | 364 |
| HOTEL | ETH, UNIV, ZARA1, ZARA2 | 32964 | 1197 |
| UNIV | ETH, HOTEL, ZARA1, ZARA2 | 9827 | 24334 |
| ZARA1 | ETH, HOTEL, UNIV, ZARA2 | 31805 | 2356 |
| ZARA2 | ETH, HOTEL, UNIV, ZARA1 | 28251 | 5910 |

## 10. Social neighbor 统计

Train/test 均为同 recording、全部 8 个 observation frame 可见、排除自己、最后观测帧距离最近 8 个，不使用 future eligibility、不使用 radius、不再保留 train/val 过滤。以下按 scene 记录；train fold 为相应四个 scene 的合并。

| Scene | Samples | Mean | Median | p90 | Zero % | 0 % | 1–2 % | 3–4 % | 5+ % |
|---|---|---|---|---|---|---|---|---|---|
| ETH | 364 | 3.818681 | 3.0 | 8.0 | 10.7143 | 10.7143 | 26.9231 | 28.2967 | 34.0659 |
| HOTEL | 1197 | 5.198830 | 6.0 | 8.0 | 4.0100 | 4.0100 | 10.4428 | 29.0727 | 56.4745 |
| UNIV | 24334 | 8.000000 | 8.0 | 8.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 |
| ZARA1 | 2356 | 4.978778 | 5.0 | 8.0 | 0.6367 | 0.6367 | 14.8132 | 30.3905 | 54.1596 |
| ZARA2 | 5910 | 7.037056 | 8.0 | 8.0 | 0.2876 | 0.2876 | 2.6904 | 6.9205 | 90.1015 |

| Fold | Split | Samples | Mean | Median | p90 | Zero % |
|---|---|---|---|---|---|---|
| ETH | train | 33797 | 7.521792 | 8.0 | 8.0 | 0.2367 |
| ETH | test | 364 | 3.818681 | 3.0 | 8.0 | 10.7143 |
| HOTEL | train | 32964 | 7.565253 | 8.0 | 8.0 | 0.2154 |
| HOTEL | test | 1197 | 5.198830 | 6.0 | 8.0 | 4.0100 |
| UNIV | train | 9827 | 6.200468 | 7.0 | 8.0 | 1.2109 |
| UNIV | test | 24334 | 8.000000 | 8.0 | 8.0 | 0.0000 |
| ZARA1 | train | 31805 | 7.667788 | 8.0 | 8.0 | 0.3270 |
| ZARA1 | test | 2356 | 4.978778 | 5.0 | 8.0 | 0.6367 |
| ZARA2 | train | 28251 | 7.575484 | 8.0 | 8.0 | 0.3610 |
| ZARA2 | test | 5910 | 7.037056 | 8.0 | 8.0 | 0.2876 |

## 11. ETT：15 组正式 test ADE/FDE

单位 meter；确定性 K=1，非 minADE20/minFDE20。

| Held-out | Seed | ADE | FDE |
|---|---|---|---|
| ETH | 42 | 1.033304813 | 2.024985863 |
| ETH | 123 | 1.058418996 | 2.078874765 |
| ETH | 2024 | 1.051346738 | 2.060846526 |
| HOTEL | 42 | 0.571277325 | 1.123478177 |
| HOTEL | 123 | 0.534662171 | 1.085092955 |
| HOTEL | 2024 | 0.541956672 | 1.067339502 |
| UNIV | 42 | 0.606446332 | 1.279858086 |
| UNIV | 123 | 0.608575738 | 1.287511958 |
| UNIV | 2024 | 0.608916065 | 1.276753553 |
| ZARA1 | 42 | 0.463742460 | 0.987329649 |
| ZARA1 | 123 | 0.555642875 | 1.119156669 |
| ZARA1 | 2024 | 0.436074176 | 0.934677730 |
| ZARA2 | 42 | 0.344214797 | 0.747535956 |
| ZARA2 | 123 | 0.357775489 | 0.762970359 |
| ZARA2 | 2024 | 0.362136371 | 0.772749851 |

## 12. EMT：15 组正式 test ADE/FDE

单位 meter；确定性 K=1，非 minADE20/minFDE20。

| Held-out | Seed | ADE | FDE |
|---|---|---|---|
| ETH | 42 | 1.070914761 | 2.135178844 |
| ETH | 123 | 1.074814634 | 2.140886953 |
| ETH | 2024 | 1.093168360 | 2.171526213 |
| HOTEL | 42 | 0.543374018 | 1.098043200 |
| HOTEL | 123 | 0.501185687 | 1.006874874 |
| HOTEL | 2024 | 0.544617115 | 1.128416103 |
| UNIV | 42 | 0.591110692 | 1.257557938 |
| UNIV | 123 | 0.590204919 | 1.259561228 |
| UNIV | 2024 | 0.577435764 | 1.238399882 |
| ZARA1 | 42 | 0.483911981 | 1.042394037 |
| ZARA1 | 123 | 0.407163296 | 0.878610338 |
| ZARA1 | 2024 | 0.471725399 | 1.012849213 |
| ZARA2 | 42 | 0.350604087 | 0.746250577 |
| ZARA2 | 123 | 0.346943383 | 0.745157190 |
| ZARA2 | 2024 | 0.356563177 | 0.755059122 |

## 13. ETT-SR：15 组正式 test ADE/FDE

单位 meter；确定性 K=1，非 minADE20/minFDE20。

| Held-out | Seed | ADE | FDE |
|---|---|---|---|
| ETH | 42 | 1.051729004 | 2.046859400 |
| ETH | 123 | 1.144509845 | 2.231112606 |
| ETH | 2024 | 1.071404685 | 2.060905257 |
| HOTEL | 42 | 0.517912261 | 1.067689864 |
| HOTEL | 123 | 0.476764531 | 0.972712517 |
| HOTEL | 2024 | 0.503430960 | 0.990917549 |
| UNIV | 42 | 0.598275001 | 1.268824077 |
| UNIV | 123 | 0.600472162 | 1.272659031 |
| UNIV | 2024 | 0.609621030 | 1.294428036 |
| ZARA1 | 42 | 0.419080388 | 0.900016089 |
| ZARA1 | 123 | 0.433739072 | 0.917614189 |
| ZARA1 | 2024 | 0.424687882 | 0.911597247 |
| ZARA2 | 42 | 0.381626672 | 0.837393151 |
| ZARA2 | 123 | 0.363381637 | 0.780496595 |
| ZARA2 | 2024 | 0.345792655 | 0.735919191 |

## 14. EMT-SR：15 组正式 test ADE/FDE

单位 meter；确定性 K=1，非 minADE20/minFDE20。

| Held-out | Seed | ADE | FDE |
|---|---|---|---|
| ETH | 42 | 1.075579717 | 2.100334914 |
| ETH | 123 | 1.045574861 | 2.046279003 |
| ETH | 2024 | 1.072674728 | 2.113637235 |
| HOTEL | 42 | 0.569875959 | 1.187343442 |
| HOTEL | 123 | 0.656125824 | 1.355529249 |
| HOTEL | 2024 | 0.698644131 | 1.455639252 |
| UNIV | 42 | 0.585130530 | 1.255046852 |
| UNIV | 123 | 0.582641906 | 1.252160708 |
| UNIV | 2024 | 0.584913637 | 1.255102671 |
| ZARA1 | 42 | 0.444013476 | 0.957603409 |
| ZARA1 | 123 | 0.423832680 | 0.922405318 |
| ZARA1 | 2024 | 0.432290138 | 0.932547749 |
| ZARA2 | 42 | 0.378482462 | 0.830309186 |
| ZARA2 | 123 | 0.377622422 | 0.819889282 |
| ZARA2 | 2024 | 0.349867861 | 0.762356474 |

## 15. 每 fold 3-seed mean ± sample SD

SD 使用 n−1 分母；主表：

| Held-out | ETT ADE / FDE (m) | EMT ADE / FDE (m) | ETT-SR ADE / FDE (m) | EMT-SR ADE / FDE (m) |
|---|---|---|---|---|
| ETH | 1.047690 ± 0.012950 / 2.054902 ± 0.027432 | 1.079633 ± 0.011883 / 2.149197 ± 0.019547 | 1.089215 ± 0.048887 / 2.112959 ± 0.102565 | 1.064610 ± 0.016549 / 2.086750 ± 0.035675 |
| HOTEL | 0.549299 ± 0.019380 / 1.091970 ± 0.028694 | 0.529726 ± 0.024724 / 1.077778 ± 0.063254 | 0.499369 ± 0.020872 / 1.010440 ± 0.050409 | 0.641549 ± 0.065610 / 1.332837 ± 0.135580 |
| UNIV | 0.607979 ± 0.001339 / 1.281375 ± 0.005537 | 0.586250 ± 0.007647 / 1.251840 ± 0.011682 | 0.602789 ± 0.006017 / 1.278637 ± 0.013809 | 0.584229 ± 0.001378 / 1.254103 ± 0.001683 |
| ZARA1 | 0.485153 ± 0.062594 / 1.013721 ± 0.095029 | 0.454267 ± 0.041245 / 0.977951 ± 0.087291 | 0.425836 ± 0.007396 / 0.909743 ± 0.008944 | 0.433379 ± 0.010134 / 0.937519 ± 0.018118 |
| ZARA2 | 0.354709 ± 0.009346 / 0.761085 ± 0.012712 | 0.351370 ± 0.004855 / 0.748822 ± 0.005429 | 0.363600 ± 0.017918 / 0.784603 ± 0.050861 | 0.368658 ± 0.016278 / 0.804185 ± 0.036597 |
| Average | 0.608966 / 1.240611 | 0.600249 / 1.241118 | 0.596162 / 1.219276 | 0.618485 / 1.283079 |

## 16. 五 fold 等权平均

先 fold 内平均三 seed，然后五 fold 等权。以下没有 pooled-window weighting：

| Model | ADE | FDE |
|---|---|---|
| ETT | 0.608966068 | 1.240610773 |
| EMT | 0.600249152 | 1.241117714 |
| ETT-SR | 0.596161852 | 1.219276320 |
| EMT-SR | 0.618484689 | 1.283078983 |

## 17. EMT vs ETT

差值=treated−control，relative 分母为 control；负值表示 treated 更好。仅描述，不新增 GO/STOP。

| Metric | Δ (m) | Relative % | Paired wins /15 |
|---|---|---|---|
| ADE | -0.008716916 | -1.431429 | 8 |
| FDE | +0.000506941 | +0.040862 | 9 |

两指标同时改善：8/15。

## 18. EMT-SR vs EMT

差值=treated−control，relative 分母为 control；负值表示 treated 更好。仅描述，不新增 GO/STOP。

| Metric | Δ (m) | Relative % | Paired wins /15 |
|---|---|---|---|
| ADE | +0.018235537 | +3.037995 | 7 |
| FDE | +0.041961269 | +3.380926 | 7 |

两指标同时改善：6/15。

## 19. EMT-SR vs ETT-SR

差值=treated−control，relative 分母为 control；负值表示 treated 更好。仅描述，不新增 GO/STOP。

| Metric | Δ (m) | Relative % | Paired wins /15 |
|---|---|---|---|
| ADE | +0.022322836 | +3.744425 | 6 |
| FDE | +0.063802663 | +5.232830 | 5 |

两指标同时改善：5/15。

## 20. Parameters

SR trainable 仅 relation embedding、cross-attention、residual decoder；backbone frozen。

| Model | Total | Trainable | Frozen |
|---|---|---|---|
| ETT | 616344 | 616344 | 0 |
| EMT | 370968 | 370968 | 0 |
| ETT-SR | 722736 | 106392 | 616344 |
| EMT-SR | 477360 | 106392 | 370968 |

## 21. Latency

独占 GPU，完整 uncached 模型，eval/no_grad，20 warmups +100 synchronized CUDA forwards；B=1/128、SR N=8。不包含 attention 导出/数据传输。不据此作未经支持的 Mamba 更快声明。

| Model | Total params | Trainable | B=1 ms | B=128 ms | Mean train peak MiB | Max train peak MiB | Mean B=128 inference peak MiB |
|---|---|---|---|---|---|---|---|
| ETT | 616344 | 616344 | 0.795524 | 0.836066 | 58.249 | 58.249 | 16.164 |
| EMT | 370968 | 370968 | 1.448930 | 1.562408 | 67.354 | 67.354 | 21.062 |
| ETT-SR | 722736 | 106392 | 2.679340 | 4.677900 | 162.916 | 193.698 | 49.315 |
| EMT-SR | 477360 | 106392 | 4.034381 | 13.653183 | 162.253 | 193.535 | 101.611 |

## 22. GPU memory

表中 train 峰值为单 process PyTorch allocated，SR 包括 frozen context cache；四 worker 共用 GPU，训练 wall time 含竞争，不能视为独立速度。各 run 与 inference 峰值完整保存。

## 23. NaN/Inf

**false**。训练 loss、gradient、checkpoint tensor、预测、attention 与 artifact metrics 全部 finite。

## 24. Max gradient norm

裁剪前全局最大：**8.348210335**；clip 阈值保持 5.0。所有 SR zero-init max_abs_diff <1e-7，backbone state before/after SHA 相同。历史冻结文件 932 个保持不变。

## 25. Pytest

**90 passed**，完整输出见 `data_audit/pytest.txt`。覆盖 source/test 隔离、full-source 数量、observation-only neighbor、N=8、frozen backbone、zero-init、protocol 修改拒绝、60-run guard、attention 导出预测不变、artifact 重算指标。

## 26. Test access log

`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/data_audit/test_access_log.json`，记录每次首次 evaluation 的 UTC timestamp、branch、commit、protocol SHA、checkpoint SHA、fold/model/seed 和完成状态。

## 27. Prediction artifacts

`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/predictions/heldout_<fold>/<model>/seed_<seed>.npz`，共 60 个。包括 scene/ped/frame、obs_abs、future_gt、prediction_abs/rel、last_obs_pos、逐样本 ADE/FDE。SR 额外包含 neighbor_abs/mask/relation/IDs/frames、attention_weights、social_residual、base_prediction。Attention 诊断另行计算，不替换原模型预测路径。保存预测可精确重算表中指标。

## 28. Visualization candidates

`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/visualization_candidates.json`：每 fold/seed 按 EMT ADE−EMT-SR ADE 保存 top/median/near-zero/failure 每类至多 10 个候选，仅数据索引，未画图。

## 29. Summary

`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/summary.md`。完整机器可读比较：`comparison.json`；完整性：`data_audit/result_integrity.json`。

## 30. Brain report 与停止

`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/brain_report.md`。正式 EMT-SR vs EMT 同时包含 capacity 和 social 效果；validation EMT-ZR 的消融单独标注来源，不与正式 held-out 数值混表。没有执行 bootstrap/t-test/Wilcoxon、SDD、K=20、radius tuning、新模块、SOTA 对比或绘图。全部结果原样汇报，等待大脑 AI 审查；本阶段到此停止。
