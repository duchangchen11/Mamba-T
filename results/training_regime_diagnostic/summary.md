SOURCE VALIDATION DIAGNOSTIC ONLY

本阶段 heldout_test_accessed=false；historical_test_already_accessed=true。所有新指标来自原 source validation；正式均值仅引用用户请求，没有加载或重算正式 test。

原 pedestrian manifest、clean neighbor 数据、模型结构和 final_training_epochs.json 保持冻结。先新训练 fixed backbone，再冻结其最后 epoch checkpoint 训练 SR。固定 AdamW lr=0.001，无 scheduler/early stop；validation 隔离 RNG 且验证模型状态不变；只保存最后 epoch。

五 fold 等权，fold 内三个 seed 等权；标准差为三个 seed 的样本标准差。分层先在每 fold 按样本数汇总三个 seed，再对非空 fold 等权。sample_count 是三次 seed 预测的数量，同一 validation 样本被计数三次，不表示独立样本。

diagnostic best 按 minimum ADE 取 epoch，FDE 是同一 epoch 的 FDE；独立 minimum FDE 另列。best 仅事后分析，未用于 checkpoint 或正式 test。

历史对比同时改变了 backbone 和 SR 的训练链、epoch、LR schedule、停止规则和 checkpoint 选择；不能解释为 scheduler 单因素因果消融。量纲：ADE/FDE/residual norm 为米，correction ratio 无量纲，epsilon=1e-8。

| 模型 | ADE | FDE |
| --- | --- | --- |
| ETT-fixed | 0.465043 | 0.973106 |
| EMT-fixed | 0.458223 | 0.962098 |
| ETT-SR-fixed | 0.445248 | 0.937964 |
| EMT-SR-fixed | 0.439169 | 0.927060 |

| 比较 candidate − reference | ΔADE | ADE % | ADE wins/15 | ΔFDE | FDE % | FDE wins/15 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr_vs_emt | -0.019054 | -4.158173 | 15 | -0.035038 | -3.641856 | 14 |
| ett_sr_vs_ett | -0.019795 | -4.256623 | 15 | -0.035142 | -3.611292 | 15 |
| emt_sr_vs_ett_sr | -0.006078 | -1.365168 | 13 | -0.010905 | -1.162575 | 12 |
| fixed_ett_vs_old_clean_base_ett | 0.019145 | 4.293552 | 1 | 0.030218 | 3.204846 | 1 |
| fixed_emt_vs_old_clean_base_emt | 0.011916 | 2.669994 | 2 | 0.017357 | 1.837276 | 3 |
| fixed_ett_sr_vs_old_clean_ett_sr | 0.007834 | 1.790943 | 0 | 0.015550 | 1.685837 | 2 |
| fixed_emt_sr_vs_old_clean_emt_sr | 0.007388 | 1.711166 | 0 | 0.015464 | 1.696347 | 0 |

模型 final − best ADE gap（15 run 等权）：ETT-fixed=0.012714, EMT-fixed=0.007539, ETT-SR-fixed=0.003195, EMT-SR-fixed=0.005755.

分层、每 run checkpoint/state SHA、曲线和全部数值见 comparison.json / brain_report.md。

诊断标签为描述性证据判断，不代表因果证明或统计显著性。Source validation 中 fixed EMT-SR 对自身 base 的变化和 final−best gap 为训练制度证据；旧 Clean 对比是整个训练链改变。若 source 社交收益保留而用户提供的正式 test 社交收益为负，训练制度单独不能解释正式失败，支持跨场景泛化风险。Mamba 专属兼容风险需要 ETT 收益保留、EMT 收益丢失以及更大 EMT 修正共同支持。标签不代表统计显著性，15 个运行共享 fold 数据；不将残差大小与错误率关联解释为因果。

Q1：EMT-SR-fixed 仍优于新 EMT-fixed：ADE -4.158173%（15/15），FDE -3.641856%（14/15）。但相对 Old Clean EMT-SR，ADE +1.711166%、FDE +1.696347%，且 final−best ADE gap=0.005755 m、FDE gap=0.011808 m。训练制度存在部分损失证据，未在 source validation 复现社交收益消失，不能单独解释正式失败。新 EMT base 相对旧 Clean base 也退化 ADE +2.669994%、FDE +1.837276%，因此不能把 SR 的变化全部归因于适配器。
Q2：EMT-SR 的绝对均值低于 ETT-SR（ADE -1.365168%、FDE -1.162575%，wins 13/15、12/15）。ETT-SR 的 final−best ADE gap 更小（0.003195 vs 0.005755 m），表明对最后 epoch 规则的损失较小；准确率和这一稳定性指标给出不同排序，不据此宣布新主模型。
Q3：EMT residual norm 均值 0.189084 m，大于 ETT 的 0.117398 m；mean correction/base ratio 为 0.432249 vs 0.330544。这是潜在过度修正的幅度线索。EMT 四个 ratio 组的 ADE improvement rate 为 0.534844、0.547003、0.564907、0.571700，未呈现随幅度增大更容易修坏的单调趋势；ETT 为 0.557273、0.592630、0.603589、0.594329，只在最高组有小幅回落。分组 base 难度和轨迹尺度不同，不能作因果判定。
Q4：邻居数量/距离影响分层收益。EMT 的 1–2 邻居组 ADE gain=-0.017504 m，FDE gain=+0.038833 m；3–4 与 5+ 邻居组 ADE gain 分别 +0.020439、+0.019033 m。>4m 距离组两种 SR 均有约 +0.034 m ADE gain；这不是支持修改 radius 的消融。0 邻居仅 120 个重复 seed 预测（40 个 fold 内窗口实例），且 target-conditioned residual 在无邻居时仍可输出，不能归因于 social attention。
历史正式 EMT-SR 均值比 EMT 差，而本阶段 source validation 社交收益保留，支持跨场景泛化风险。诊断同时有训练链和 full-source 数据条件差异，不能证明单一机制；本阶段没有加载、重算或再次选择正式 test。

TRAINING REGIME MISMATCH: PARTIAL
MAMBA-SOCIAL COMPATIBILITY RISK: UNCLEAR
CROSS-SCENE GENERALIZATION RISK: SUPPORTED
