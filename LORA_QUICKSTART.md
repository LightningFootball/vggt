# LoRA微调快速上手 ⚡

**5分钟开始训练！**

## 1️⃣ 安装依赖

```bash
pip install peft>=0.7.0
```

## 2️⃣ 下载预训练模型

```bash
# 方式1: 使用huggingface-cli
huggingface-cli download facebook/VGGT-1B --local-dir ./pretrained/vggt-1b

# 方式2: 使用wget（如果有直接链接）
# 或者手动从Hugging Face下载
```

## 3️⃣ 更新配置

编辑 `training/config/lora_eth3d_strategy_a.yaml`:

```yaml
checkpoint:
  resume_checkpoint_path: ./pretrained/vggt-1b/model.pt  # 更新这里！
```

## 4️⃣ 开始训练

```bash
# 策略A（最快，测试用）
bash scripts/train_lora_eth3d.sh strategy_a

# 策略B（推荐，生产用）
bash scripts/train_lora_eth3d.sh strategy_b

# 策略C（最强，但慢）
bash scripts/train_lora_eth3d.sh strategy_c

# 或直接使用 torchrun（推荐，更灵活）
torchrun --nproc_per_node=1 training/launch.py --config lora_eth3d_strategy_b
```

**注意**：即使单GPU训练也需要使用`torchrun`来设置分布式环境变量。

## 5️⃣ 监控训练

```bash
# 另开一个终端
tensorboard --logdir logs/lora_eth3d_strategy_b_r16/tensorboard

# 访问 http://localhost:6006
```

## 6️⃣ 评估结果

```bash
python scripts/evaluate_eth3d.py \
    --checkpoint logs/lora_eth3d_strategy_b_r16/ckpts/checkpoint.pth \
    --config lora_eth3d_strategy_b \
    --save-vis
```

---

## 📊 三种策略对比

| 策略 | 速度 | 效果 | 显存 | 适用场景 |
|------|------|------|------|---------|
| **A** | ⚡⚡⚡ | ⭐⭐ | 12GB | 快速测试 |
| **B** | ⚡⚡ | ⭐⭐⭐⭐ | 20GB | 🎯生产推荐 |
| **C** | ⚡ | ⭐⭐⭐⭐⭐ | 28GB | 最大性能 |

---

## 🔧 遇到问题？

### OOM（显存不足）

编辑配置文件：
```yaml
max_img_per_gpu: 2  # 减小batch size
img_size: 420       # 减小图像尺寸
```

### 训练太慢

```yaml
max_epochs: 10      # 减少epoch数
```

### 想调整学习率

```yaml
optimizer:
  lr: 5e-5          # 试试更小的学习率
```

---

## 📚 完整文档

详细说明请查看: [LORA_TRAINING_GUIDE.md](LORA_TRAINING_GUIDE.md)

---

## ✅ 检查清单

- [ ] 安装了PEFT库
- [ ] 下载了预训练模型
- [ ] 更新了配置文件中的checkpoint路径
- [ ] ETH3D数据在正确位置（`/home/zerun/data/dataset/ETH3D/`）
- [ ] GPU显存足够（建议>= 16GB）

全部勾选后，就可以开始训练了！🚀
