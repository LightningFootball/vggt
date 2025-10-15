# DataLoader调优指南 - 速度与内存平衡

## 你的系统配置
- CPU: 32核心
- RAM: 46GB
- GPU: 32GB (RTX 5090)
- 数据集: KITTI-360 (LiDAR密集型)

## 内存消耗分析

### DataLoader内存公式
```
总CPU内存 ≈ num_workers × prefetch_factor × batch_size × images_per_batch × memory_per_image

其中:
- batch_size = max_img_per_gpu / sequence_length = 32 / 8 = 4
- images_per_batch = 8 (KITTI-360序列长度)
- memory_per_image ≈ 120MB (8个LiDAR文件 × 15MB)
```

### 不同配置的内存占用

| 配置 | Workers | Prefetch | 并发图像数 | CPU内存 | 训练速度 | 稳定性 |
|------|---------|----------|-----------|---------|---------|--------|
| **原始 (爆内存)** | 16 | 8 | 512 | ~60GB ❌ | 100% | ❌ OOM |
| **安全 (当前)** | 4 | 2 | 32 | ~4GB | ~60% | ✅✅✅ |
| **平衡** | 8 | 3 | 96 | ~12GB | ~85% | ✅✅ |
| **激进** | 12 | 4 | 192 | ~23GB | ~95% | ✅ |
| **极限** | 16 | 4 | 256 | ~31GB | ~100% | ⚠️ (需监控) |

## 推荐配置方案

### 方案1：平衡配置 (推荐) ⭐
**适用**: 日常训练，稳定性优先

```yaml
data:
  train:
    num_workers: 8
    prefetch_factor: 3
    persistent_workers: True
```

**优点**:
- CPU内存 ~12GB (安全余量)
- 训练速度恢复到85%
- 稳定性高

**缺点**:
- 速度略低于原始配置

---

### 方案2：激进配置
**适用**: 追求速度，愿意承担小风险

```yaml
data:
  train:
    num_workers: 12
    prefetch_factor: 4
    persistent_workers: True
```

**优点**:
- 训练速度恢复到95%
- CPU内存 ~23GB (仍有余量)

**缺点**:
- 内存余量较小
- 建议关闭其他程序

---

### 方案3：极限配置 ⚠️
**适用**: 追求最大速度，密切监控

```yaml
data:
  train:
    num_workers: 16
    prefetch_factor: 4  # 降低prefetch (不是8!)
    persistent_workers: True
```

**优点**:
- 训练速度接近100%

**缺点**:
- CPU内存 ~31GB (接近上限)
- **必须实时监控**，防止内存尖峰
- 不能同时运行其他程序

---

### 方案4：GPU加速替代 🚀
**思路**: 用更大的GPU batch替代CPU prefetch

```yaml
max_img_per_gpu: 48    # 从32增加到48 (6 batches而不是4)
accum_steps: 6         # 匹配batch数量

data:
  train:
    num_workers: 6     # 适度减少workers
    prefetch_factor: 2  # 降低prefetch
    persistent_workers: True
```

**原理**:
- 增加GPU batch size → GPU利用率更高
- 减少CPU workers → 降低CPU内存压力
- GPU显存: 19GB → ~25GB (仍在32GB内)

**优点**:
- 训练速度可能**更快** (GPU并行效率高)
- CPU内存压力更小 (~8GB)

**缺点**:
- GPU显存占用增加
- 如果出现NaN可能更难恢复

## 如何选择？

### 场景1: 稳定训练，不想操心
→ **方案1 (平衡配置)**

### 场景2: 追求速度，可以监控
→ **方案2 (激进配置)** 或 **方案4 (GPU加速)**

### 场景3: 极限性能，愿意调试
→ **方案3 (极限配置)** + 实时监控

## 监控命令

### 终端1: 训练
```bash
./train_single_gpu.sh lora_kitti360_strategy_b
```

### 终端2: 实时监控
```bash
# 监控CPU和内存
watch -n 1 'echo "=== CPU Memory ===" && free -h && echo "" && echo "=== GPU Memory ===" && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader && echo "" && echo "=== Python Processes ===" && ps aux | grep "python.*launch.py" | grep -v grep | awk "{print \$2, \$4, \$11}" | head -5'
```

### 内存告警脚本
创建 `monitor_memory.sh`:

```bash
#!/bin/bash
THRESHOLD=40  # GB (当内存超过40GB时告警)

while true; do
    USED=$(free -g | awk 'NR==2 {print $3}')
    if [ $USED -gt $THRESHOLD ]; then
        echo "⚠️  WARNING: Memory usage ${USED}GB exceeds ${THRESHOLD}GB!"
        # 可选: 发送邮件或自动降低workers
    fi
    sleep 5
done
```

## 动态调优策略

如果选择激进/极限配置，可以这样监控调整：

1. **启动训练，观察3-5分钟**
   - 如果内存稳定在30GB以下 → ✅ 可以继续
   - 如果内存超过35GB → ⚠️ 降低workers或prefetch

2. **中途调整**（不重启训练）
   - 虽然DataLoader配置运行时不可改，但可以：
   - 关闭其他程序释放内存
   - 监控swap使用（如果swap增长 → 危险）

3. **下次训练前调整**
   - 根据日志中的GPU利用率调整
   - GPU利用率 < 80% → 可以增加prefetch
   - GPU利用率 > 95% → CPU workers已足够

## 最佳实践

### ✅ DO
- 先用保守配置测试1个epoch
- 逐步增加workers/prefetch，观察内存曲线
- 同时监控GPU利用率和CPU内存
- 记录不同配置的训练吞吐（samples/sec）

### ❌ DON'T
- 不要直接用极限配置（可能训练到一半崩溃）
- 不要在训练时运行其他内存密集型程序
- 不要忽略swap使用量（swap增长 = 即将OOM）
- 不要盲目增加batch_size（可能影响收敛）

## 快速决策树

```
开始
  │
  ├─ 追求最大稳定性？
  │   └─ YES → 方案1 (workers=8, prefetch=3)
  │
  ├─ 追求速度，可监控？
  │   ├─ 偏好CPU并行 → 方案2 (workers=12, prefetch=4)
  │   └─ 偏好GPU并行 → 方案4 (更大batch_size)
  │
  └─ 极限性能实验？
      └─ 方案3 (workers=16, prefetch=4) + 实时监控
```

## 总结

对于你的系统（32核 + 46GB RAM + 32GB GPU），**推荐使用方案1（平衡配置）**:

```yaml
num_workers: 8
prefetch_factor: 3
persistent_workers: True
```

这将恢复约**85%的训练速度**，同时保持内存安全（~12GB CPU，远低于46GB上限）。

如果实际测试中内存占用更低，可以逐步调整为方案2或方案4以获得更高速度。
