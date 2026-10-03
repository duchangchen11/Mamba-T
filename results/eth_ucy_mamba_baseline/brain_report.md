# 第一阶段完成汇报（大脑 AI）

**最终：STOP。** 仅为 source-scene 内部 validation 结果，没有评估 held-out 场景。

## 1. GitHub repository

https://github.com/duchangchen11/Mamba-T

## 2. Local repository path

`/home/lrj/Mamba-T`

## 3. Branch

`feat/eth_ucy_mamba_baseline`

## 4. Latest commit

本报告纳入 `exp: validate Mamba trajectory backbone on ETH-UCY` 提交；最终发布 SHA 见交付回复，或运行 `git log -1 --oneline`。实验代码基础 commit：`c594c1645e5807049a1e0e069b07e496e38129ab`。

## 5. Push status

通过已连接的 GitHub 工具发布实验分支，最终回复给出发布后远端与本地一致性核验结果。本机普通 HTTPS git push 缺少凭据，故使用 GitHub API 上传 tree/commit 并更新 ref。

## 6. Python / Torch / CUDA / GPU

Python 3.10.21；Torch 2.5.1+cu124；CUDA runtime 12.4；NVIDIA GeForce RTX 3080。Python 路径：`/home/lrj/anaconda3/envs/ped_intent/bin/python`。

## 7. mamba-ssm version

2.2.6.post3

## 8. ETH/UCY 数据来源

[Social-STGCNN 固定版本原始轨迹](https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/datasets/raw/all_data)，commit `333d3a57b4d2705e129b21aefefa09c79b2b9ae1`。2026-10-03 获取。六个文件：biwi_eth、biwi_hotel、students001、students003、crowds_zara01、crowds_zara02；不使用 crowds_zara03 或 uni_examples。UNIV 的两个录像使用 recording-qualified ped_id。

## 9. Provenance 路径

原始：`/home/lrj/Mamba-T/data/raw/eth_ucy/DATA_PROVENANCE.md`；提交副本：`results/eth_ucy_mamba_baseline/data_audit/DATA_PROVENANCE.md`；机器可读 SHA256、来源 URL、采样定义：`data_audit/data_provenance.json`。

## 10. 五 scene pedestrian 数量

| Scene | 原始 pedestrians | 可构成连续 20 步的 pedestrians |
|---|---|---|
| ETH | 360 | 44 |
| HOTEL | 389 | 122 |
| UNIV | 849 | 722 |
| ZARA1 | 148 | 142 |
| ZARA2 | 204 | 189 |

## 11. 五 scene window 数量

| Scene | Windows |
|---|---|
| ETH | 364 |
| HOTEL | 1197 |
| UNIV | 24334 |
| ZARA1 | 2356 |
| ZARA2 | 5910 |

总计 34,161；窗口 stride=1，无插值。

## 12. 每 fold train / validation 数量

| Fold | Train pedestrians | Val pedestrians | Train windows | Val windows |
|---|---|---|---|---|
| ETH | 1015 | 160 | 29831 | 3966 |
| HOTEL | 947 | 150 | 29086 | 3878 |
| UNIV | 428 | 69 | 8631 | 1196 |
| ZARA1 | 930 | 147 | 28077 | 3728 |
| ZARA2 | 880 | 150 | 24891 | 3360 |

固定 SHA1 阈值划分，约 85%/15%，不要求精确数量配额；三个 seed 共享相同 manifest。

## 13. Leakage test 结果

全部通过：track 不交叉；held-out 不在 train/val；实际文件读取拦截；input 不利用未来；最后 observation 相对位置为零；canonical 唯一且排序；连续帧、shape 和有限值检查。

## 14. ETT 参数量

616,344（全部可训练）

## 15. EMT 参数量

370,968（全部可训练）

## 16. 15 组 ETT validation ADE / FDE

| Fold | Seed | ADE (m) | FDE (m) |
|---|---|---|---|
| ETH | 42 | 0.449600 | 0.958002 |
| ETH | 123 | 0.451385 | 0.962596 |
| ETH | 2024 | 0.448674 | 0.957712 |
| HOTEL | 42 | 0.465835 | 0.988899 |
| HOTEL | 123 | 0.462927 | 0.982849 |
| HOTEL | 2024 | 0.466359 | 0.992991 |
| UNIV | 42 | 0.334469 | 0.687584 |
| UNIV | 123 | 0.335133 | 0.685318 |
| UNIV | 2024 | 0.330035 | 0.674688 |
| ZARA1 | 42 | 0.466944 | 0.994244 |
| ZARA1 | 123 | 0.470773 | 1.005388 |
| ZARA1 | 2024 | 0.465719 | 0.992787 |
| ZARA2 | 42 | 0.512778 | 1.086908 |
| ZARA2 | 123 | 0.516562 | 1.090714 |
| ZARA2 | 2024 | 0.511275 | 1.082635 |

## 17. 15 组 EMT validation ADE / FDE

| Fold | Seed | ADE (m) | FDE (m) |
|---|---|---|---|
| ETH | 42 | 0.454713 | 0.964892 |
| ETH | 123 | 0.453825 | 0.967809 |
| ETH | 2024 | 0.458093 | 0.980044 |
| HOTEL | 42 | 0.466940 | 0.992438 |
| HOTEL | 123 | 0.466371 | 0.990488 |
| HOTEL | 2024 | 0.467493 | 0.997949 |
| UNIV | 42 | 0.334149 | 0.677880 |
| UNIV | 123 | 0.333929 | 0.690215 |
| UNIV | 2024 | 0.331085 | 0.682616 |
| ZARA1 | 42 | 0.465207 | 0.992948 |
| ZARA1 | 123 | 0.464167 | 0.995000 |
| ZARA1 | 2024 | 0.466981 | 0.998411 |
| ZARA2 | 42 | 0.512331 | 1.083310 |
| ZARA2 | 123 | 0.512635 | 1.078843 |
| ZARA2 | 2024 | 0.506680 | 1.078264 |

## 18. 每组 EMT − ETT

| Fold | Seed | ΔADE (m) | ΔFDE (m) |
|---|---|---|---|
| ETH | 42 | +0.005113 | +0.006889 |
| ETH | 123 | +0.002440 | +0.005213 |
| ETH | 2024 | +0.009419 | +0.022332 |
| HOTEL | 42 | +0.001105 | +0.003539 |
| HOTEL | 123 | +0.003444 | +0.007639 |
| HOTEL | 2024 | +0.001135 | +0.004958 |
| UNIV | 42 | -0.000320 | -0.009704 |
| UNIV | 123 | -0.001204 | +0.004897 |
| UNIV | 2024 | +0.001049 | +0.007928 |
| ZARA1 | 42 | -0.001737 | -0.001296 |
| ZARA1 | 123 | -0.006606 | -0.010388 |
| ZARA1 | 2024 | +0.001262 | +0.005624 |
| ZARA2 | 42 | -0.000447 | -0.003598 |
| ZARA2 | 123 | -0.003927 | -0.011871 |
| ZARA2 | 2024 | -0.004595 | -0.004371 |

## 19. 五 fold 等权平均

| Model | ADE (m) | FDE (m) |
|---|---|---|
| ETT | 0.445898 | 0.942888 |
| EMT | 0.446307 | 0.944740 |

每 fold 的三个 seed mean ± sample SD 见 `summary.md`。

## 20. ADE 相对变化

+0.091658%（EMT / ETT − 1）；正值代表恶化。

## 21. FDE 相对变化

+0.196482%（EMT / ETT − 1）。

## 22. EMT 胜出数量 / 15

ADE：7/15；FDE：6/15；两项均胜出：6/15。

## 23. 最大 pre-clip gradient

1.636078；裁剪阈值 5.0；逐 run 值见 metrics_validation.json。

## 24. NaN / Inf 情况

所有正式 run 及 smoke 均无 NaN/Inf；loss、pre-clip gradient、ADE/FDE 均检查有限性。

## 25. Latency

| Model | Batch | Mean ms | Min ms | Max ms |
|---|---|---|---|---|
| ETT | 1 | 0.810348 | 0.768456 | 1.289434 |
| ETT | 128 | 0.819207 | 0.813612 | 0.835678 |
| EMT | 1 | 1.428217 | 1.421616 | 1.435879 |
| EMT | 128 | 1.545193 | 1.535357 | 1.555784 |

全部训练结束后独立测量。GPU resident 零输入 [B,8,4]；eval/no_grad；20 次 warmup，100 次 forward，CUDA synchronize；报告整 batch latency，不含数据传输。15 个 checkpoint 的逐 run 测量见 summary.md。

## 26. GPU memory

| Model | Mean MiB | Min MiB | Max MiB |
|---|---|---|---|
| ETT | 58.399 | 58.249 | 58.624 |
| EMT | 67.354 | 67.354 | 67.354 |

每进程训练/validation 期间 PyTorch peak allocated memory。六进程并行时训练耗时包含资源竞争，不用于独立速度结论。

## 27. Pytest / 编译检查

23 passed；4 个 Torch Transformer nested-tensor 提示。全部要求文件 py_compile 通过。证据：`data_audit/pytest.txt`、`data_audit/verification.json`。

## 28. test_accessed=false 是否成立

成立。训练加载器只接受 train/val 且跳过 held-out NPZ。所有场景的数据构建/完整性审计可以读取文件，但没有任何 held-out 模型评估或指标计算，也未根据 held-out 选择 checkpoint。

## 29. summary.md 路径

`/home/lrj/Mamba-T/results/eth_ucy_mamba_baseline/summary.md`

## 30. 最终 GO / STOP

**STOP**。条件 A 未满足：ADE/FDE 均没有改善；条件 B 未满足：ADE 7/15、FDE 6/15；条件 C 满足：没有 fold 平均 ADE/FDE 恶化超过 10%。未开始超参数搜索或 Social Transformer 第二阶段。
