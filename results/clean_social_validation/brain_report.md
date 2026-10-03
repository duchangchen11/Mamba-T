# Clean Social Validation：大脑 AI 交付报告

**CLEAN SOCIAL: GO**。30 个新 run 已完成；只使用 source-scene validation。指标单位为米，均为 K=1。五 fold 各先平均三个 seed，再等权平均五 fold。

## 1. Branch

`feat/clean_social_validation`，从已冻结的 `feat/social_residual` / `f9605c691800d6a884fd2e9179273ebedfddeafb` 创建。

## 2. Latest commit

代码提交：`ec87ad256d611fc67dc3646f15f7f0030421065e`，`fix: enforce split-clean social neighbors`。最新实验提交为包含本报告的 `exp: validate clean social residual on ETH-UCY`；其最终 SHA 在交付消息及 [分支提交记录](https://github.com/duchangchen11/Mamba-T/commits/feat/clean_social_validation) 中给出，避免把提交自身的 SHA 写入自身内容。

## 3. Push status

交付前核对 GitHub 分支与本地 HEAD 的 SHA 和完整文件树一致；最终 push 状态及最新 SHA 见交付消息。数据和 checkpoint 保留在本地，不提交到 GitHub；本轮代码、配置、测试、审计及实验结果提交到新分支。

## 4. Clean neighbor 规则

复用五份第一阶段 manifest，target pedestrian split、样本顺序和 target 数值不变。Train 候选先按当前 scene 的 train target 集合过滤，再按最后观测帧距离选最近 8 个；必须同 recording、全部 8 个观测帧存在且不是 target。Pedestrian 以 recording-qualified ID 判定。Val 使用全部 observation-visible 邻居，允许 train/val/other，不查询 neighbor future、future length 或 future visibility。N=8，无 radius。新数据保存于 `data/processed/eth_ucy_social_clean/<fold>/<scene>/<split>.npz`；split 标签和 ID 只用于审计，不进入模型。旧数据和模型源码保留原样。

## 5. Train cross-split violations

**成立：train cross-split neighbor violations = 0。** 五 fold 的有效 train neighbor 100% 属于该 scene 的 train target 集合，与 val target 集合交集为空。原始 ID 重新核对与保存的 split 标签一致；发现违规会直接报错。`data_audit/neighbor_split_audit.json` 记录全部统计。

| Fold | 有效 train 邻居 | Train 比例 | Val 数量 | Other 数量 | 违规数量 |
|---|---|---|---|---|---|
| ETH | 219659 | 100% | 0 | 0 | 0 |
| HOTEL | 215979 | 100% | 0 | 0 | 0 |
| UNIV | 48122 | 100% | 0 | 0 | 0 |
| ZARA1 | 210930 | 100% | 0 | 0 | 0 |
| ZARA2 | 185678 | 100% | 0 | 0 | 0 |

## 6. 每 fold clean neighbor 统计

每个 fold 汇总其四个 source scene。比例为该 split 内样本比例；validation 的全部模型输入逐数组与旧版本完全一致。

| Fold | Split | Samples | Mean | Median | p90 | Zero 比例 | Old mean | 0 | 1–2 | 3–4 | 5+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ETH | train | 29831 | 7.363447 | 8.000000 | 8.000000 | 0.5397% | 7.524019 | 0.5397% | 3.0673% | 5.0887% | 91.3043% |
| ETH | val | 3966 | 7.505043 | 8.000000 | 8.000000 | 0.0504% | 7.505043 | 0.0504% | 1.4372% | 5.2950% | 93.2173% |
| HOTEL | train | 29086 | 7.425531 | 8.000000 | 8.000000 | 0.6257% | 7.563158 | 0.6257% | 2.5717% | 4.7446% | 92.0580% |
| HOTEL | val | 3878 | 7.580970 | 8.000000 | 8.000000 | 0.2579% | 7.580969 | 0.2579% | 1.3667% | 3.3265% | 95.0490% |
| UNIV | train | 8631 | 5.575484 | 6.000000 | 8.000000 | 3.3368% | 6.204611 | 3.3368% | 11.9337% | 18.0396% | 66.6898% |
| UNIV | val | 1196 | 6.170569 | 7.000000 | 8.000000 | 0.8361% | 6.170568 | 0.8361% | 5.4348% | 18.8127% | 74.9164% |
| ZARA1 | train | 28077 | 7.512555 | 8.000000 | 8.000000 | 0.9153% | 7.669445 | 0.9153% | 2.0729% | 3.1556% | 93.8562% |
| ZARA1 | val | 3728 | 7.655311 | 8.000000 | 8.000000 | 0.2146% | 7.655311 | 0.2146% | 0.5365% | 3.6212% | 95.6277% |
| ZARA2 | train | 24891 | 7.459644 | 8.000000 | 8.000000 | 1.0606% | 7.586115 | 1.0606% | 3.3948% | 3.5635% | 91.9810% |
| ZARA2 | val | 3360 | 7.496726 | 8.000000 | 8.000000 | 0.2976% | 7.496726 | 0.2976% | 1.9345% | 5.9821% | 91.7857% |

Train 平均邻居数在所有 fold 均下降；UNIV 下降最多，6.204611 → 5.575484（约 10.14%）。其他 fold 下降约 1.67%–2.13%。

Val 的有效邻居来源比例（分母为有效邻居数，不是样本数）：

| Fold | Train-side | Val-side | Other/non-target-eligible |
|---|---|---|---|
| ETH | 88.7351% | 10.1226% | 1.1423% |
| HOTEL | 88.6017% | 9.8745% | 1.5239% |
| UNIV | 87.2764% | 7.8726% | 4.8509% |
| ZARA1 | 87.8692% | 10.2526% | 1.8781% |
| ZARA2 | 87.1412% | 10.8341% | 2.0247% |

## 7. Distance 统计

仅统计被选中的有效邻居与 target 在最后观测帧的距离，单位 meter。每个 window 的邻居观测计一次，不做 pedestrian 去重。本轮不据此设置 radius 或调整选择规则。完整计数和 rank 分母见 `data_audit/distance_statistics.json`。

| Fold | Split | 有效邻居数 | Mean | Median | p25 | p75 | p90 | Max |
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

## 8. Pytest

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -m pytest -q`：**59 passed, 8 warnings in 29.53s**；全部 src/scripts 的 py_compile 通过。新增 14 项测试覆盖 train membership、过滤后排序、同 recording、padding、val 与旧版本一致、future 访问隔离、污染文件拒绝、base regression、backbone 无梯度及 zero initialization。原始输出：`data_audit/pytest.txt`；验证：`data_audit/verification.json`。

## 9. Clean EMT-SR：15 组 ADE/FDE

5 folds × seeds 42/123/2024，共 15 个新 run。使用对应第一阶段最佳 frozen backbone，选择最小 source-validation ADE 的 checkpoint。

| Fold | Seed | ADE (m) | FDE (m) |
|---|---|---|---|
| ETH | 42 | 0.439763 | 0.938716 |
| ETH | 123 | 0.435447 | 0.927684 |
| ETH | 2024 | 0.438540 | 0.932252 |
| HOTEL | 42 | 0.451360 | 0.962969 |
| HOTEL | 123 | 0.452768 | 0.959166 |
| HOTEL | 2024 | 0.456088 | 0.966086 |
| UNIV | 42 | 0.321701 | 0.661187 |
| UNIV | 123 | 0.315885 | 0.643238 |
| UNIV | 2024 | 0.328231 | 0.672572 |
| ZARA1 | 42 | 0.449297 | 0.956604 |
| ZARA1 | 123 | 0.453720 | 0.967997 |
| ZARA1 | 2024 | 0.450857 | 0.959019 |
| ZARA2 | 42 | 0.493618 | 1.039260 |
| ZARA2 | 123 | 0.495706 | 1.040296 |
| ZARA2 | 2024 | 0.493732 | 1.046892 |

## 10. Clean ETT-SR：15 组 ADE/FDE

5 folds × seeds 42/123/2024，共 15 个新 run。使用对应第一阶段最佳 frozen backbone，选择最小 source-validation ADE 的 checkpoint。

| Fold | Seed | ADE (m) | FDE (m) |
|---|---|---|---|
| ETH | 42 | 0.449696 | 0.956784 |
| ETH | 123 | 0.448608 | 0.955920 |
| ETH | 2024 | 0.443267 | 0.941065 |
| HOTEL | 42 | 0.453560 | 0.962556 |
| HOTEL | 123 | 0.457323 | 0.973211 |
| HOTEL | 2024 | 0.453467 | 0.959984 |
| UNIV | 42 | 0.322895 | 0.657340 |
| UNIV | 123 | 0.327633 | 0.662985 |
| UNIV | 2024 | 0.316411 | 0.639963 |
| ZARA1 | 42 | 0.456356 | 0.969042 |
| ZARA1 | 123 | 0.456679 | 0.970203 |
| ZARA1 | 2024 | 0.463738 | 0.988361 |
| ZARA2 | 42 | 0.503216 | 1.064375 |
| ZARA2 | 123 | 0.498625 | 1.053820 |
| ZARA2 | 2024 | 0.509733 | 1.080600 |

## 11. Frozen EMT-ZR reference：15 组 ADE/FDE

直接复用旧阶段 15 个 control run，未重新训练。对应 metrics 与 checkpoint SHA256 全部核验，完整 15 组哈希保存于 `data_audit/emt_zr_references.json`。

| Fold | Seed | ADE (m) | FDE (m) |
|---|---|---|---|
| ETH | 42 | 0.445040 | 0.957707 |
| ETH | 123 | 0.445291 | 0.958745 |
| ETH | 2024 | 0.446200 | 0.960200 |
| HOTEL | 42 | 0.456481 | 0.983586 |
| HOTEL | 123 | 0.458979 | 0.984695 |
| HOTEL | 2024 | 0.459309 | 0.985053 |
| UNIV | 42 | 0.320634 | 0.668953 |
| UNIV | 123 | 0.327342 | 0.680384 |
| UNIV | 2024 | 0.326429 | 0.680644 |
| ZARA1 | 42 | 0.459855 | 0.990959 |
| ZARA1 | 123 | 0.461424 | 0.993034 |
| ZARA1 | 2024 | 0.462053 | 0.993511 |
| ZARA2 | 42 | 0.503518 | 1.073864 |
| ZARA2 | 123 | 0.502985 | 1.070762 |
| ZARA2 | 2024 | 0.504953 | 1.076295 |

## 12. Clean EMT-SR vs Frozen EMT-ZR

以下统一定义差值为 Clean EMT-SR − control，relative % 的分母为 control，负值表示 Clean EMT-SR 更好。配对单位为同 fold × seed。

| 指标 | Δ (m) | Relative change | Clean wins /15 |
|---|---|---|---|
| ADE | -0.006918627 | -1.577077% | 13 |
| FDE | -0.025630280 | -2.734695% | 15 |

两指标同时改善：13/15。

| 模型 | 五 fold 等权 ADE | 五 fold 等权 FDE |
|---|---|---|
| Frozen EMT-ZR | 0.438699 | 0.937226 |
| Old EMT-SR | 0.427041 | 0.900124 |
| Clean EMT-SR | 0.431781 | 0.911596 |
| Clean ETT-SR | 0.437414 | 0.922414 |

ADE 和 FDE 都达到预设 GO 门槛，且分别有 13/15、15/15 配对改善。ADE 未达 2%、FDE 未达 3%，因此未达到 STRONG GO。

## 13. Clean EMT-SR vs Clean ETT-SR

以下统一定义差值为 Clean EMT-SR − control，relative % 的分母为 control，负值表示 Clean EMT-SR 更好。配对单位为同 fold × seed。本项仅为 descriptive comparison，不用于 GO 判定。

| 指标 | Δ (m) | Relative change | Clean wins /15 |
|---|---|---|---|
| ADE | -0.005633033 | -1.287804% | 13 |
| FDE | -0.010817971 | -1.172789% | 11 |

两指标同时改善：11/15。

## 14. Old EMT-SR vs Clean EMT-SR

以下统一定义差值为 Clean EMT-SR − control，relative % 的分母为 control，负值表示 Clean EMT-SR 更好。配对单位为同 fold × seed。本项 control 为 Old EMT-SR，仅用于记录清洗后的退化。

| 指标 | Δ (m) | Relative change | Clean wins /15 |
|---|---|---|---|
| ADE | +0.004739890 | +1.109938% | 1 |
| FDE | +0.011471768 | +1.274465% | 2 |

两指标同时改善：0/15。

| 指标 | Old 相对 ZR 改善 | Clean 相对 ZR 改善 | 增益减少（百分点） | 保留旧增益 | 旧增益收缩比例 |
|---|---|---|---|---|---|
| ADE | 2.657518% | 1.577077% | 1.080441 | 59.3440% | 40.6560% |
| FDE | 3.958708% | 2.734695% | 1.224013 | 69.0805% | 30.9195% |

清洗后误差变大，不能把旧的 2.66%/3.96% 增益当作严格 split-clean 结果。清洗后仍保留约 59.34% ADE、69.08% FDE 增益，符合本阶段 GO；以上是固定协议下三个 seed 的观测结果。

## 15. Neighbor count 分层

分组使用 validation 的有效邻居数，各模型及 seed 的组成员完全相同。每 fold 先平均三个 seed，再对有样本的 fold 等权平均。Samples 是每个 seed 的 validation window 数，三个 seed 重复同一组样本，不能把样本数乘三当作独立数据。每 fold / 每 seed 的原始结果见 `comparison.json`。

| Group | Samples /seed | ZR ADE | ZR FDE | Clean ADE | Clean FDE | ADE change | FDE change |
|---|---|---|---|---|---|---|---|
| 0 | 40 | 0.815972 | 1.468171 | 0.800030 | 1.453463 | -1.9538% | -1.0018% |
| 1-2 | 260 | 0.508356 | 1.087837 | 0.519798 | 1.057457 | +2.2508% | -2.7928% |
| 3-4 | 900 | 0.452469 | 0.926344 | 0.433503 | 0.869296 | -4.1916% | -6.1584% |
| 5+ | 14928 | 0.432473 | 0.928101 | 0.426544 | 0.905873 | -1.3708% | -2.3950% |

| Fold | Group | Samples /seed | ZR ADE/FDE | Clean ADE/FDE |
|---|---|---|---|---|
| ETH | 0 | 2 | 0.239413 / 0.556594 | 0.317752 / 0.701010 |
| ETH | 1-2 | 57 | 0.383148 / 0.807521 | 0.436900 / 0.891337 |
| ETH | 3-4 | 210 | 0.387972 / 0.794109 | 0.383796 / 0.771177 |
| ETH | 5+ | 3697 | 0.449852 / 0.970795 | 0.441072 / 0.942835 |
| HOTEL | 0 | 10 | 0.906507 / 1.631709 | 0.863724 / 1.608968 |
| HOTEL | 1-2 | 53 | 0.464797 / 1.007028 | 0.480431 / 0.980173 |
| HOTEL | 3-4 | 129 | 0.513088 / 1.098382 | 0.510165 / 1.076675 |
| HOTEL | 5+ | 3686 | 0.455027 / 0.978377 | 0.449917 / 0.956749 |
| UNIV | 0 | 10 | 0.907709 / 1.600199 | 0.842028 / 1.449582 |
| UNIV | 1-2 | 65 | 0.420637 / 0.896150 | 0.424449 / 0.852213 |
| UNIV | 3-4 | 225 | 0.384127 / 0.781656 | 0.364816 / 0.724647 |
| UNIV | 5+ | 896 | 0.296446 / 0.624064 | 0.297931 / 0.619674 |
| ZARA1 | 0 | 8 | 1.098391 / 1.911717 | 1.050864 / 1.804174 |
| ZARA1 | 1-2 | 20 | 0.790553 / 1.735471 | 0.751449 / 1.543798 |
| ZARA1 | 3-4 | 135 | 0.501862 / 1.007913 | 0.466713 / 0.911238 |
| ZARA1 | 5+ | 3565 | 0.456289 / 0.985687 | 0.447678 / 0.957939 |
| ZARA2 | 0 | 10 | 0.927842 / 1.640638 | 0.925783 / 1.703578 |
| ZARA2 | 1-2 | 65 | 0.482643 / 0.993017 | 0.505759 / 1.019762 |
| ZARA2 | 3-4 | 201 | 0.475294 / 0.949659 | 0.442025 / 0.862743 |
| ZARA2 | 5+ | 3084 | 0.504749 / 1.081582 | 0.496123 / 1.052169 |

3–4 组的相对改善最大；5+ 组也改善，且占 14,928/16,128 个窗口，构成多数数据。1–2 组 ADE 退化、FDE 改善；不能声称所有密度均改善。0 组仅 40 个窗口，各 fold 为 2–10 个；其残差仍可利用 target context，因此该组收益不能单独证明邻居信息的作用。

## 16. 最大 gradient norm

30 个新 run 的最大裁剪前 gradient norm：**4.470528603**，clip 阈值维持 5.0。每个 run 的数值在 metrics/comparison 中保存。

## 17. NaN/Inf

**false**。30 个训练/验证过程及保存的 residual checkpoint 均为有限数值；所有 validation history 数值再次核对。初始预测与 frozen base 的最大绝对差在全部新 run 中均为 **0.0**（要求 <1e-7）。

## 18. Backbone hash 检查

**全部通过。** 30 个 run 的第一阶段 base checkpoint SHA256 和 state SHA256 与冻结清单一致；trainer 在训练后再次检查 backbone state hash 相等、全部 backbone 参数无梯度，且始终 eval()/requires_grad=False。新 checkpoint 仅保存新模块。全部 671 个冻结文件（旧两阶段结果、checkpoint、旧 social 数据、manifest、模型源码等）SHA256 未变化。30 个新 run 的残差和 trainable 初始化 hash 也与旧相应 run 完全一致。证据见 `data_audit/frozen_inventory.json`、`data_audit/result_integrity.json` 与各 run 的 initialization/metrics。

## 19. Held-out test

**heldout_test_accessed = false**。所有结果均是对应 heldout fold 的其余四个 source scene validation；目录名 `heldout_eth` 等表示被排除的 scene，不代表测试该 scene。未计算任何正式 ETH/HOTEL/UNIV/ZARA1/ZARA2 held-out test 指标。没有运行 GSR 或其他新模型、没有修改结构/超参数、没有根据距离调整邻居。

## 20. Summary 路径与判定

`/home/lrj/Mamba-T/results/clean_social_validation/summary.md`；机器可读对照为同目录 `comparison.json`。本报告为 `brain_report.md`。

**CLEAN SOCIAL: GO。未达到 STRONG GO。** 在严格 train pedestrian isolation 下，ADE 改善 1.577077%、FDE 改善 2.734695%，配对 wins 为 13/15、15/15。旧增益确有收缩，结论限定于预设 source-validation 协议。本阶段完成后停止，不进入正式 held-out test。
