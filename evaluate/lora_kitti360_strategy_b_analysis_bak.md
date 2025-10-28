# KITTI-360 建筑场景 LoRA 微调评估对比与建议（VGGT）

本报告基于 `evaluate/` 目录下两次评估产物，对比分析：
- 训练一（strategy_b）：先用 `training/config/lora_kitti360_strategy_b_15epoch.yaml` 训练
  20 epoch，随后放松 LoRA 参数，改用 `training/config/lora_kitti360_strategy_b.yaml`
  继续训练与评估。
- 训练二（strategy_b_soft）：全程使用温和 LoRA（`training/config/lora_kitti360_strategy_b_soft.yaml`），
  更低的学习率与 α 等。

两次训练都仅在预测头与后 8 层 Transformer 上注入 LoRA。评估与可视化通过
`scripts/evaluate_kitti360_buildings.py` 与 `scripts/visualize_kitti360_metrics.py` 完成，
结果见 `evaluate/` 下相应子目录。

参考证据（可点击查看）：
- 训练一综合摘要：`evaluate/lora_kitti360_strategy_b/plots/evaluation_summary.txt:1`
- 主要图表（训练一）：
  - 深度主指标：`evaluate/lora_kitti360_strategy_b/plots/01_main_depth_metrics.png`
  - 建筑 vs 非建筑：`evaluate/lora_kitti360_strategy_b/plots/02_building_vs_nonbuilding.png`
  - 几何质量：`evaluate/lora_kitti360_strategy_b/plots/03_geometry_quality.png`
  - 位姿误差：`evaluate/lora_kitti360_strategy_b/plots/04_pose_errors.png`
  - 雷达图对比：`evaluate/lora_kitti360_strategy_b/plots/05_radar_comparison.png`
  - 稳定性：`evaluate/lora_kitti360_strategy_b/plots/06_training_stability.png`
  - 综合评分：`evaluate/lora_kitti360_strategy_b/plots/07_comprehensive_score.png`
- 主要图表（训练二）位于：`evaluate/lora_kitti360_strategy_b_soft/plots/`（其中缺少
  05、07 两张图与整体 CSV 汇总，见文末改进建议）。


## 核心结论（TL;DR）
- 训练一（strategy_b）在“建筑区域深度质量”上明显优于训练二（soft）。
  - Building AbsRel 最优（ckpt_34）：从基线 1.721 降至 0.611（↓64.5%）。
  - Building δ<1.25 最优（ckpt_36）：从 0.403 提升至 0.442（+9.7%）。
  - Building RMSE 最优（ckpt_34）：16.50 → 11.02（↓33.2%）。
- 训练二（soft）也带来改进，但幅度较小，且建筑 δ<1.25 未超过基线：
  - Building AbsRel（ckpt_45）≈ 0.819（较基线↓约52.4%，不及训练一）。
  - Building δ<1.25（ckpt_40）≈ 0.386（低于基线 0.403）。
- 非建筑区域与全局深度指标，两次训练均优于基线，但训练一仍占优（AbsRel 更低）。
- 几何质量方面（法线/平面）：两次训练均提升“建筑区域”法线与平面一致性；
  训练二在“建筑法线角度均值”上略优，但训练一综合表现更稳。
- 位姿误差差异很小，对整体结论影响不大。
- 代价侧：两次 LoRA 都明显增加了推理显存（≈2×）并降低了 FPS（较基线下降 50%~80%）。


## 数据与可比性说明（重要）
- 训练一评估统计量中样本数（full/building）多处为约 4000/3300；
  训练二为约 12240/8500。两次评估使用的帧量级不同，绝对数值不完全可比。
- 本报告因此采用“相对基线的改变量”进行对比为主，并在同一 run 内选用最优
  checkpoint 对应的指标进行横向比较。

参考：
- 基线统计：`evaluate/lora_kitti360_strategy_b/baseline_metrics.json:1`
- 训练一最优样例：
  - ckpt_34（建筑 AbsRel 最优）：`evaluate/lora_kitti360_strategy_b/checkpoint_34_metrics.json:1`
  - ckpt_36（建筑 δ<1.25 最优）：`evaluate/lora_kitti360_strategy_b/checkpoint_36_metrics.json:1`
- 训练二最优样例：
  - ckpt_45（建筑 AbsRel 最优，不计 ckpt_0 异常点）：
    `evaluate/lora_kitti360_strategy_b_soft/checkpoint_45_metrics.json:1`
  - ckpt_40（建筑 δ<1.25 最优）：
    `evaluate/lora_kitti360_strategy_b_soft/checkpoint_40_metrics.json:1`


## 关键指标对比与分析
以下百分比均以训练一目录下的“基线”数值为参照（同一评估脚本生成）。

- 深度（建筑区域）
  - AbsRel（越低越好）
    - 基线：1.721
    - 训练一：0.611（ckpt_34，↓64.5%）
    - 训练二：0.819（ckpt_45，↓52.4%）
    - 结论：训练一明显更优，soft 参数下欠拟合更明显。
  - δ<1.25（越高越好）
    - 基线：0.403
    - 训练一：0.442（ckpt_36，+9.7%）
    - 训练二：0.386（ckpt_40，-4.2%）
    - 结论：训练一显著改善建筑区域“可接受误差内”的比例，训练二未超基线。
  - RMSE（越低越好）
    - 基线：16.50 → 训练一：11.02（↓33.2%） → 训练二：≈11.90（↓27%）

- 深度（全局与非建筑）
  - 全局 AbsRel：基线 0.394 → 训练一 0.259（↓34%） → 训练二 0.270（↓30%）。
  - 非建筑 AbsRel：基线 0.347 → 训练一 0.240（↓31%） → 训练二 0.261（↓25%）。
  - 结论：两者均提升泛化，但训练一更佳，且无明显牺牲非建筑区域。

- 几何质量（法线与平面）
  - 建筑法线角度均值（越低越好）：
    - 基线：83.76° → 训练一：82.67°（-1.3°） → 训练二：81.27°（-2.5°）。
    - 结论：两者均提升建筑表面法线一致性，训练二略优，但差距较小。
  - 全局法线角度均值（越低越好）：
    - 基线：92.80° → 训练一：93.60° → 训练二：93.63°（全局略有变差）。
    - 解释：LoRA 仅在后 8 层与预测头，偏向建筑域的提升，
      对非建筑的法线可能未受益甚至略受损；
      但平面一致性（建筑）有显著提升：
      - 平面内点比：基线 0.141 → 训练一 0.200（+42%） → 训练二 0.207~0.216（+47%）。

- 位姿（ATE/RPE）
  - ATE_Trans（越低越好）：基线 3603 → 训练一 ≈3556（小幅改善）→ 训练二 ≈3620（微弱变差）。
  - ATE_Rot：数值差异很小（±1° 内），不影响总体判断。

- 光度一致性（建筑）
  - Photometric Error：基线 0.075 → 训练一 ≈0.131 → 训练二 ≈0.135（数值更大）。
  - 解释：该误差的定义可能对外观差异更敏感，而 LoRA 专注结构/几何，
    因此该项未同步改善；可辅以更直接的几何/深度度量解读结果。

- 速度与显存
  - FPS：基线 ≈2.33 → 训练一 ≈1.12 → 训练二 ≈0.45（存在评测环境差异，但总体下降）。
  - 显存：基线 ≈10.7GB → 两次 LoRA ≈20.3GB（≈2×）。
  - 结论：LoRA 注入带来推理开销，应按部署预算评估收益/代价比。


## 训练行为诊断与假设
- 训练一优于训练二的主要原因：
  - 先“较强约束（15epoch）→ 再放松参数”的两阶段策略，让后 8 层 Transformer
    获得更强的域适配能力，同时避免早期大步长直接破坏已学到的通用能力。
  - 在建筑区域指标（AbsRel、RMSE、δ<1.25）上持续提升并在 34~36 号 ckpt 附近达到
    平台期；配套的平面一致性指标也有显著增益。
- 训练二（soft）全程温和设置可能导致欠拟合：
  - 建筑 δ<1.25 未超过基线，说明“低误差阈值内准确率”未被有效推高。
  - 尽管建筑法线更优，但深度误差主指标仍逊于训练一，
    反映出几何特征层面收益未充分传导至像素级深度精度。
- 评估样本量不同（4k vs 12k）提醒我们：
  - 训练二的统计可能更稳健，但为公平性，建议统一评估子集与随机种子。


## 建议与下一步计划（可执行）
- 选型与导出
  - 面向“建筑场景深度精度”优先：推荐使用训练一 `ckpt_34`（或 `ckpt_36` 若更看重 δ<1.25）。
  - 若部署对平面一致性极其敏感且对 δ<1.25 要求一般，可评估 `ckpt_41`
    （训练一全局 AbsRel 最低点），但其建筑 δ<1.25 略低于 `ckpt_36`。
- 继续微调（在训练一基础上“再精修”）
  - 分阶段策略：
    1) 以训练一 `ckpt_34` 为起点，较小学习率“温和收敛”3~5 epoch；
    2) 逐步减小 LoRA α 或冻结部分 LoRA 分支（比如仅保留 Q/K，冻结 V/Proj），
       保住已得益处同时稳住法线全局退化。
  - 采样与损失：
    - 提高“建筑像素/块”的采样权重；
    - 增加平面一致性/几何一致性损失的权重（但设上限，避免压制 δ<1.25 提升）。
  - 结构与注入位点：
    - 试验仅在后 8 层的自注意力 Q/K 分支上保留 LoRA（减小参数与显存）；
    - 或在后 10~12 层进行更“深”的注入，对高层语义/结构更敏感的建筑几何可能更受益。
- 推理代价优化
  - 使用低秩、更小 α 的 LoRA，或采用按通道稀疏/分组 LoRA，减小 VRAM；
  - 部署时将 LoRA 合并权重固化（若代码路径支持）以减少运行时开销。


## 对评估脚本与可视化的改进建议
为更全面、可比和可复现，建议在评估脚本中新增/统一以下输出：
- 统一的“评估子集”标识与帧数统计
  - 在 summary 中明确列出：数据子集名称、序列 ID、帧数（full/building/non-building）。
  - 将该信息写入 `evaluation_summary.txt` 与 `metrics_overall.csv`（若存在）。
- 面向两次 run 生成一致的汇总产物
  - 为 `strategy_b_soft` 也输出：`metrics_overall.csv`、`evaluation_summary.txt`、
    `05_radar_comparison.png`、`07_comprehensive_score.png`。
- 自动选优与排行榜
  - 计算并保存每个 ckpt 的“综合评分”（例如对 AbsRel↓、δ<1.25↑、Planarity↑ 做归一化
    加权），输出 top-k 列表与曲线；
  - 在 `evaluation_summary.txt` 中附上 “Top-5 ckpt 及关键指标表”。
- 细粒度可视化
  - 分场景/分类别（建筑立面、玻璃、树木等邻近结构）误差分解柱状图；
  - 建筑区域误差分布直方图/箱线图（AbsRel、RMSE、δ 指标）；
  - 法线角误差热力图与平面内点比的空间分布（叠加在图像/深度上）。
- 质量-代价权衡图
  - 在一张图中联合展示（AbsRel、δ<1.25、Planarity）与（FPS、VRAM）的 Pareto 前沿；
  - 便于选取部署 ckpt。
- 置信度/不确定性关联
  - 若模型输出置信度，绘制“置信度-误差”曲线，辅助阈值选取与后处理策略。


## 结论
- 训练一（strategy_b）在“建筑领域深度估计”上总体优势更明显：
  - 建筑 AbsRel、RMSE、δ<1.25 全面领先；
  - 非建筑与全局也同步受益，显示出更好的域适配与泛化平衡；
  - 代价是推理显存与 FPS 的开销，与训练二相近。
- 训练二（soft）提供了平滑而稳健的几何改进（法线/平面）但深度主指标
  尤其是建筑 δ<1.25 未能超越基线，显示出欠拟合迹象。
- 建议优先采用训练一 `ckpt_34`（或 `ckpt_36` 取决于 δ<1.25 的重要性）作为当前
  建筑场景应用的候选权重；并基于该点进行小步精修与代价优化。

附：本报告分析所依据的关键文件（起始行）：
- `evaluate/lora_kitti360_strategy_b/plots/evaluation_summary.txt:1`
- `evaluate/lora_kitti360_strategy_b/baseline_metrics.json:1`
- `evaluate/lora_kitti360_strategy_b/checkpoint_34_metrics.json:1`
- `evaluate/lora_kitti360_strategy_b/checkpoint_36_metrics.json:1`
- `evaluate/lora_kitti360_strategy_b_soft/checkpoint_40_metrics.json:1`
- `evaluate/lora_kitti360_strategy_b_soft/checkpoint_45_metrics.json:1`
