SOURCE VALIDATION SOCIAL DISTRIBUTION AUDIT ONLY

heldout_test_accessed=false；historical_test_already_accessed=true。未重新训练 trajectory 模型，未读取正式 test 或 prediction。所有 forward 使用本阶段前冻结的 source validation checkpoint，eval/no_grad；所有参数 hash 保持一致。

scene 指样本来源，fold 指上一阶段四 source scenes 的留一 fold 名称。首先按 scene + ped_id + 8 observation frame IDs 去重；每个窗口的模型响应在可用 fold/seed 上平均。主要表格/相关性使用窗口级统计，另保存全部重复预测和每 fold/seed 的原始结果，不能把重复窗口当独立观测。

pooled 对所有唯一窗口等权；scene_equal 在每个有效 scene 内对窗口等权，再对 scene 等权。距离和相对速度在无邻居时为 missing，统计中排除，不以 0m 当真实距离。closing 为正向接近邻居的均值，max clamped >=0；signed mean/max 另存。速度单位 meter / sampling step。

Attention 统计为四头权重平均后，在有效邻居内归一化；entropy 在 n<=1 时定义 0。权重通过原 SDPA 的 identity-value probe 提取，使用相同 packed projections 和 padding mask，同时重建 context 与原输出核对。prediction 始终使用原 need_weights=False 路径；padding 不参与统计。高 neighbor_count 本身会影响集中度；另报告 n>=2 和固定 count 内相关性，不能仅凭相关性推出机制。

std 为 population SD；scene 内分位数采用 numpy linear quantile；全体 scene 等权分位数采用 weighted mid-CDF interpolation。SMD 比较 scene A−B，排名平均 10 个 scene pair 的 |SMD|。Wasserstein 有单位，不能跨不同量纲直接比较或取总体大小排序。

标签阈值已在审计前固定：mean_abs_SMD>=0.5 为 SUPPORTED，>=0.2 为 WEAK，否则 NOT SUPPORTED。家族使用最强变量，EMT/ETT 的 mean_abs_SMD 等权平均。仅为描述性分档，不是显著性检验、因果证明或 predictor 阈值。

完成 30/30 个 SR source validation forwards；每模型 48384 条 fold/seed 预测，去重后 4032 个窗口、169 名 scene/ped 目标。

| 模型 | 变量 | eth | hotel | univ | zara1 | zara2 |
| --- | --- | --- | --- | --- | --- | --- |
| emt_sr | neighbor_count | 4.590909 | 4.344156 | 8.000000 | 5.029605 | 7.260417 |
| emt_sr | nearest_neighbor_distance | 1.263385 | 1.248832 | 0.783167 | 1.431258 | 0.935651 |
| emt_sr | mean_neighbor_distance | 4.088691 | 4.046756 | 2.122583 | 3.573365 | 3.180815 |
| emt_sr | mean_relative_speed | 0.664197 | 0.346178 | 0.290488 | 0.471585 | 0.291432 |
| emt_sr | mean_closing_speed | 0.209951 | 0.209020 | 0.169291 | 0.211478 | 0.155249 |
| emt_sr | attention_entropy | 0.705053 | 0.645938 | 0.814091 | 0.753578 | 0.824863 |
| emt_sr | residual_norm | 0.450178 | 0.202922 | 0.201281 | 0.228546 | 0.128298 |
| emt_sr | correction_ratio | 0.471053 | 0.591411 | 0.315749 | 0.108564 | 0.844778 |
| ett_sr | neighbor_count | 4.590909 | 4.344156 | 8.000000 | 5.029605 | 7.260417 |
| ett_sr | nearest_neighbor_distance | 1.263385 | 1.248832 | 0.783167 | 1.431258 | 0.935651 |
| ett_sr | mean_neighbor_distance | 4.088691 | 4.046756 | 2.122583 | 3.573365 | 3.180815 |
| ett_sr | mean_relative_speed | 0.664197 | 0.346178 | 0.290488 | 0.471585 | 0.291432 |
| ett_sr | mean_closing_speed | 0.209951 | 0.209020 | 0.169291 | 0.211478 | 0.155249 |
| ett_sr | attention_entropy | 0.473392 | 0.556089 | 0.439966 | 0.504639 | 0.416092 |
| ett_sr | residual_norm | 0.358850 | 0.119507 | 0.102003 | 0.158474 | 0.100678 |
| ett_sr | correction_ratio | 0.458577 | 0.434928 | 0.179767 | 0.069803 | 0.768215 |

| 排名 | 变量 | mean abs(SMD) | max abs(SMD) |
| --- | --- | --- | --- |
| 1 | neighbor_count | 1.226661 | 2.896200 |
| 2 | mean_relative_speed | 0.929824 | 1.903406 |
| 3 | residual_norm | 0.922645 | 1.829067 |
| 4 | residual_final_norm | 0.899754 | 1.778726 |
| 5 | max_relative_speed | 0.851260 | 1.639626 |
| 6 | attention_top2_sum | 0.757221 | 2.962145 |
| 7 | max_neighbor_distance | 0.682352 | 1.418355 |
| 8 | attention_max | 0.664991 | 2.141289 |
| 9 | correction_ratio | 0.650698 | 1.634410 |
| 10 | mean_neighbor_distance | 0.638896 | 1.323849 |
| 11 | median_neighbor_distance | 0.591319 | 1.298063 |
| 12 | attention_entropy | 0.514115 | 1.193270 |
| 13 | nearest_neighbor_distance | 0.398501 | 1.011027 |
| 14 | mean_closing_speed | 0.154869 | 0.294454 |
| 15 | max_closing_speed | 0.115847 | 0.257021 |

关键注意事项：N=8 只表示当前最多保留的八个观测可见邻居已满，实际邻居总数被右截断。attention/residual 随 scene 改变还混合了 frozen checkpoint 与观测组成差异；主分析不能将其当成 scene 的因果效应。分类器仅使用五个 observation 统计，按 pedestrian 分组划分，全部预处理在训练 fold 内拟合。

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

SOCIAL DENSITY SHIFT: SUPPORTED
SOCIAL DISTANCE SHIFT: SUPPORTED
RELATIVE-MOTION SHIFT: SUPPORTED
ATTENTION RESPONSE SHIFT: SUPPORTED
RESIDUAL RESPONSE SHIFT: SUPPORTED
PRIMARY SHIFT FACTOR: neighbor_count

Candidate A: density normalization（仅建议，未实现；依据 neighbor_count）
Candidate B: motion-aware interaction representation（仅建议，未实现；依据 mean_relative_speed）
Candidate C: confidence-normalized residual correction（仅建议，未实现；依据 residual_norm）
