# LoRA Fine-tuning Guide for VGGT on ETH3D Building Facades

本指南详细说明如何使用LoRA在ETH3D建筑外立面数据集上微调VGGT模型。

## 📋 目录

- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练策略](#训练策略)
- [快速开始](#快速开始)
- [训练监控](#训练监控)
- [评估与分析](#评估与分析)
- [常见问题](#常见问题)

---

## 🔧 环境准备

### 1. 安装依赖

```bash
# 基础依赖（已安装）
pip install -r requirements.txt

# 训练依赖
pip install -r requirements_training.txt

# 确认PEFT库已安装
pip install peft>=0.7.0
```

### 2. 硬件要求

- **GPU**: NVIDIA RTX 5090 (32GB VRAM) ✓
- **CUDA**: 支持bfloat16（Compute Capability >= 8.0）✓
- **系统**: Linux (推荐)

**关于BF16**：
- ✅ VGGT原项目完全支持bfloat16（见`training/config/default.yaml:127`）
- ✅ RTX 5090支持原生BF16硬件加速
- ✅ 所有配置文件已启用`amp_dtype: bfloat16`
- 相比FP32，BF16可节约约50%显存，提升1.5-2倍训练速度

---

## 📦 数据准备

### 1. ETH3D数据集

你已经下载的数据：
```
/home/zerun/data/dataset/ETH3D/Stereo/High-res_multi-view/
├── multi_view_training_dslr_undistorted/     # 主图像数据 ✓
├── multi_view_training_dslr_occlusion/       # 遮挡掩膜 ✓
└── multi_view_training_dslr_scan_eval/       # Ground Truth ✓
```

### 2. 预训练模型

下载VGGT预训练模型：

```bash
# 选项1: 非商业版本
huggingface-cli download facebook/VGGT-1B --local-dir ./pretrained/vggt-1b

# 选项2: 商业版本
huggingface-cli download facebook/VGGT-1B-Commercial --local-dir ./pretrained/vggt-1b-commercial
```

### 3. 更新配置文件

修改配置文件中的checkpoint路径：

```bash
# 编辑配置文件
nano training/config/lora_eth3d_strategy_a.yaml

# 找到这一行并更新：
resume_checkpoint_path: /path/to/your/pretrained/checkpoint.pt
```

例如：
```yaml
resume_checkpoint_path: ./pretrained/vggt-1b/model.pt
```

---

## 🎯 训练策略

我们提供三种渐进式训练策略：

### 策略A: Depth Head Only（推荐起点）

**特点：**
- ✅ 最快训练速度
- ✅ 最少显存占用（~10GB）
- ✅ 适合验证pipeline
- ⚠️ 性能提升有限

**配置：** `lora_eth3d_strategy_a.yaml`

**LoRA参数：**
- Rank: 16
- Target: Depth Head only
- Trainable params: ~8M (0.8%)

### 策略B: Aggregator后期层 + Depth Head（推荐最优）

**特点：**
- ✅ 平衡性能与效率
- ✅ 针对性学习建筑特征
- ✅ 显存占用适中（~18GB）
- ✅ **最推荐用于生产**

**配置：** `lora_eth3d_strategy_b.yaml`

**LoRA参数：**
- Rank: 16
- Target: Aggregator layers 12-23 + Depth Head
- Trainable params: ~45M (4.5%)

### 策略C: Full LoRA（最大容量）

**特点：**
- ✅ 最强学习能力
- ⚠️ 最慢训练速度
- ⚠️ 显存占用最高（~28GB）
- ⚠️ 可能过拟合

**配置：** `lora_eth3d_strategy_c.yaml`

**LoRA参数：**
- Rank: 16
- Target: All Aggregator + Depth Head + Camera Head
- Trainable params: ~90M (9%)

---

## 🚀 快速开始

### 方法1: 使用Shell脚本（推荐）

```bash
# 给脚本添加执行权限
chmod +x scripts/train_lora_eth3d.sh

# 训练策略A（测试pipeline）
bash scripts/train_lora_eth3d.sh strategy_a

# 训练策略B（生产使用）
bash scripts/train_lora_eth3d.sh strategy_b

# 训练策略C（最大性能）
bash scripts/train_lora_eth3d.sh strategy_c
```

### 方法2: 直接使用torchrun（推荐）

```bash
# 策略A
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_a

# 策略B
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_b

# 策略C
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_c
```

**重要**：即使是单GPU训练，也必须使用`torchrun`启动器，因为训练代码需要分布式环境变量（`LOCAL_RANK`、`RANK`）。

### 训练参数说明

配置文件中的关键参数：

```yaml
# 训练基础参数
max_epochs: 20                 # 训练轮数
max_img_per_gpu: 4             # 每GPU图像数（调整以适应显存）
accum_steps: 1                 # 梯度累积步数

# LoRA参数
lora:
  rank: 16                     # LoRA秩（越大容量越大，显存越多）
  alpha: 32                    # 缩放因子（alpha/rank=2.0）
  dropout: 0.1                 # Dropout率

# 优化器
optimizer:
  lr: 1e-4                     # 学习率（LoRA通常需要更高LR）
  weight_decay: 0.01           # 权重衰减

# 损失权重
loss:
  camera:
    weight: 2.0                # 相机损失权重
  depth:
    weight: 5.0                # 深度损失权重（主要优化目标）
```

---

## 📊 训练监控

### 1. TensorBoard

启动TensorBoard监控：

```bash
# 策略A
tensorboard --logdir logs/lora_eth3d_strategy_a_r16/tensorboard

# 策略B
tensorboard --logdir logs/lora_eth3d_strategy_b_r16/tensorboard

# 策略C
tensorboard --logdir logs/lora_eth3d_strategy_c_r16/tensorboard
```

访问 `http://localhost:6006`

### 2. 监控指标

**关键损失：**
- `loss_objective`: 总损失（应该稳定下降）
- `loss_conf_depth`: 深度置信损失（主要优化目标）
- `loss_reg_depth`: 深度回归损失
- `loss_grad_depth`: 深度梯度损失（平滑性）

**学习率调度：**
- 前10% warmup: 1e-6 → 1e-4
- 后90% cosine decay: 1e-4 → 1e-6

### 3. 检查点

检查点自动保存在：
```
logs/{exp_name}/ckpts/
├── checkpoint.pth              # 最新检查点
├── checkpoint_2.pth            # 第2个epoch
├── checkpoint_4.pth            # 第4个epoch
└── ...
```

---

## 🎯 评估与分析

### 1. 运行评估

```bash
# 评估策略A
python scripts/evaluate_eth3d.py \
    --checkpoint logs/lora_eth3d_strategy_a_r16/ckpts/checkpoint.pth \
    --config lora_eth3d_strategy_a \
    --save-vis

# 评估策略B
python scripts/evaluate_eth3d.py \
    --checkpoint logs/lora_eth3d_strategy_b_r16/ckpts/checkpoint.pth \
    --config lora_eth3d_strategy_b \
    --save-vis
```

### 2. 评估指标

输出示例：
```
Depth Metrics:
  MAE:       0.1234 ± 0.0456 (median: 0.1123)
  RMSE:      0.2345 ± 0.0789
  Abs Rel:   0.0567 ± 0.0123
  Sq Rel:    0.0234 ± 0.0089

Threshold Accuracy:
  δ < 1.25:  0.9234
  δ < 1.25²: 0.9789
  δ < 1.25³: 0.9912
```

**指标说明：**
- **MAE**: 平均绝对误差（越小越好）
- **RMSE**: 均方根误差（越小越好）
- **δ < 1.25**: 阈值准确度（越大越好，>0.9为优秀）

### 3. 可视化结果

可视化保存在 `eval_results/visualizations/`：
- 输入图像
- 预测深度图
- Ground truth深度图
- 误差热力图

---

## 🔬 实验流程建议

### 第1阶段：Pipeline验证（1-2天）

```bash
# 1. 训练策略A测试2个epoch
python training/launch.py --config lora_eth3d_strategy_a \
    max_epochs=2

# 2. 检查是否正常运行
# - 查看TensorBoard损失曲线
# - 确认没有OOM错误
# - 验证checkpoint保存成功
```

### 第2阶段：超参数调优（3-5天）

**LoRA Rank扫描：**
```bash
# 尝试不同rank值
# 编辑配置文件修改lora.rank: [8, 16, 32, 64]

# Rank 8
sed -i 's/rank: 16/rank: 8/g' training/config/lora_eth3d_strategy_b.yaml
python training/launch.py --config lora_eth3d_strategy_b

# Rank 32
sed -i 's/rank: 8/rank: 32/g' training/config/lora_eth3d_strategy_b.yaml
python training/launch.py --config lora_eth3d_strategy_b
```

**学习率扫描：**
```bash
# 尝试不同学习率: [5e-5, 1e-4, 2e-4, 5e-4]
# 修改配置文件中的optimizer.lr值
```

### 第3阶段：完整训练（1周）

```bash
# 使用最优超参数训练策略B
python training/launch.py --config lora_eth3d_strategy_b \
    max_epochs=20

# 评估
python scripts/evaluate_eth3d.py \
    --checkpoint logs/lora_eth3d_strategy_b_r16/ckpts/checkpoint_20.pth \
    --save-vis
```

### 第4阶段：对比实验

对比三种策略的性能：
1. 训练速度
2. 深度估计精度
3. 窗户区域专项评估
4. 泛化能力测试

---

## ❓ 常见问题

### Q1: OOM错误（显存不足）

**解决方案：**
```yaml
# 减小batch size
max_img_per_gpu: 2  # 从4降到2

# 启用梯度累积
accum_steps: 2      # 从1增加到2

# 减小图像分辨率
img_size: 420       # 从518降到420

# 使用更小的LoRA rank
lora:
  rank: 8           # 从16降到8
```

### Q2: 训练不收敛

**检查清单：**
1. 学习率是否过高/过低？
   - 推荐范围：5e-5 to 2e-4
2. 是否加载了预训练权重？
   - 检查`resume_checkpoint_path`
3. 损失权重是否合理？
   - depth.weight建议2-5之间

### Q3: 验证集性能不提升

**可能原因：**
1. **过拟合**：训练集loss下降但验证集不降
   - 增加dropout: `lora.dropout: 0.2`
   - 减小rank: `lora.rank: 8`
   - 添加数据增强

2. **学习率过高**：验证集loss震荡
   - 降低学习率：`lr: 5e-5`

3. **Epoch不足**：还未收敛
   - 增加训练轮数：`max_epochs: 30`

### Q4: PEFT导入错误

```bash
# 安装PEFT
pip install peft>=0.7.0

# 如果还有问题，尝试从源码安装
pip install git+https://github.com/huggingface/peft.git
```

### Q5: 报错 "int() argument must be a string... not 'NoneType'"

**问题**：直接运行`python training/launch.py`会报错，因为缺少分布式环境变量。

**解决方案**：
```bash
# ❌ 错误：直接用 python 运行
python training/launch.py --config lora_eth3d_strategy_b

# ✅ 正确：使用 torchrun（即使单GPU）
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_b

# 或使用提供的脚本
bash scripts/train_lora_eth3d.sh strategy_b
```

### Q6: 如何恢复中断的训练？

训练会自动从最新checkpoint恢复：
```bash
# 只需重新运行相同命令
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_b

# 或指定特定checkpoint
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_b \
    checkpoint.resume_checkpoint_path=logs/lora_eth3d_strategy_b_r16/ckpts/checkpoint_10.pth
```

### Q7: 如何调整训练/验证集划分？

编辑配置文件：
```yaml
data:
  train:
    dataset:
      train_val_split: 0.85  # 85%训练，15%验证
```

### Q8: 只想训练特定场景怎么办？

修改ETH3D数据集初始化：
```python
# 在 training/data/datasets/eth3d.py 中
# 修改 BUILDING_SCENES 列表
BUILDING_SCENES = ['facade', 'terrace']  # 只使用这两个场景
```

或在配置中指定：
```yaml
data:
  train:
    dataset:
      scenes: ['facade', 'terrace', 'electro']  # 自定义场景列表
      use_building_scenes_only: False
```

---

## 📈 预期效果

### 训练时间估计（RTX 5090）

| 策略 | Epoch耗时 | 20 Epochs总时长 | 峰值显存 |
|------|-----------|----------------|---------|
| A    | ~15分钟   | ~5小时         | ~12GB   |
| B    | ~30分钟   | ~10小时        | ~20GB   |
| C    | ~45分钟   | ~15小时        | ~28GB   |

### 性能提升预期

相比预训练模型在ETH3D建筑场景上：
- **策略A**: 深度MAE降低 10-15%
- **策略B**: 深度MAE降低 20-30% ⭐推荐
- **策略C**: 深度MAE降低 25-35%（可能过拟合）

窗户区域专项改善：
- 窗框深度估计更准确
- 玻璃反射区域置信度建模更好
- 平面墙面更平滑

---

## 🎓 进阶技巧

### 1. 窗户区域专项优化

如果你有窗户区域的分割掩膜，可以增加窗户区域的损失权重：

```python
# 在loss.py的regression_loss函数中
# 根据window_mask加权
if window_mask is not None:
    loss_conf = loss_conf * (1 + window_mask * window_weight)
```

### 2. 混合精度训练

已默认启用bfloat16：
```yaml
amp:
  enabled: True
  amp_dtype: bfloat16  # RTX 5090支持
```

### 3. 保存LoRA Adapter

仅保存LoRA权重（而非整个模型）：
```python
from training.lora_utils import save_lora_checkpoint

save_lora_checkpoint(
    model,
    save_path='./lora_adapters/facade_adapter'
)
```

### 4. 合并LoRA权重用于推理

```python
from training.lora_utils import merge_lora_weights

merged_model = merge_lora_weights(peft_model)
# merged_model现在是普通nn.Module，可以正常保存和推理
```

---

## 📝 许可证

- **代码**: MIT License
- **预训练模型**:
  - `facebook/VGGT-1B`: 非商业使用
  - `facebook/VGGT-1B-Commercial`: 商业使用（不包括军事）
- **ETH3D数据集**: 仅用于研究

---

## 🤝 贡献与反馈

如有问题或建议，请：
1. 检查本指南的[常见问题](#常见问题)部分
2. 查看TensorBoard日志和训练输出
3. 在项目Issue中提问

祝训练顺利！ 🚀
