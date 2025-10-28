# Docker 环境恢复指南

本文档记录了在 Docker 重启后恢复 VGGT 训练环境的完整步骤。

## 背景说明

- 宿主机原始 home 目录挂载到：`/home/zerun/workspace/zerun/`
- Docker 内需要创建符号链接，将路径映射回标准位置
- 原始路径格式：`/home/zerun/workspace/zerun/XXX`
- 目标路径格式：`/home/zerun/XXX`

## 快速恢复脚本

将以下内容保存为 `/home/zerun/restore_env.sh`，每次 Docker 重启后执行一次：

```bash
#!/bin/bash
# VGGT Docker 环境快速恢复脚本

echo "=========================================="
echo "VGGT Docker 环境恢复"
echo "=========================================="

# 1. 创建 miniconda3 符号链接
echo "[1/5] 创建 miniconda3 符号链接..."
if [ ! -L /home/zerun/miniconda3 ]; then
    sudo ln -s /home/zerun/workspace/zerun/miniconda3 /home/zerun/miniconda3
    echo "  ✓ miniconda3 链接已创建"
else
    echo "  ✓ miniconda3 链接已存在"
fi

# 2. 创建数据集符号链接
echo "[2/5] 创建数据集符号链接..."
if [ ! -L /home/zerun/data ]; then
    sudo ln -s /home/zerun/workspace/zerun/data /home/zerun/data
    echo "  ✓ data 链接已创建"
else
    echo "  ✓ data 链接已存在"
fi

# 3. 创建项目符号链接（通常由挂载自动创建，检查即可）
echo "[3/5] 检查项目符号链接..."
if [ ! -L /home/zerun/workspace/vggt ]; then
    sudo ln -s /home/zerun/workspace/zerun/workspace/vggt /home/zerun/workspace/vggt
    echo "  ✓ vggt 项目链接已创建"
else
    echo "  ✓ vggt 项目链接已存在"
fi

# 4. 安装 OpenGL 依赖（Docker 环境必需）
echo "[4/5] 安装 OpenGL 依赖..."
if ! dpkg -l | grep -q libgl1-mesa-glx; then
    sudo apt-get update -qq
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    echo "  ✓ OpenGL 依赖已安装"
else
    echo "  ✓ OpenGL 依赖已存在"
fi

# 5. 更新 conda 环境中的 vggt 包路径
echo "[5/5] 更新 vggt 包路径..."
FINDER_FILE="/home/zerun/miniconda3/envs/vggt-train/lib/python3.12/site-packages/__editable___vggt_0_0_1_finder.py"
if [ -f "$FINDER_FILE" ]; then
    # 检查是否需要更新
    if grep -q "/home/zerun/workspace/zerun/workspace/vggt" "$FINDER_FILE"; then
        sed -i 's|/home/zerun/workspace/zerun/workspace/vggt|/home/zerun/workspace/vggt|g' "$FINDER_FILE"
        echo "  ✓ vggt 包路径已更新"
    else
        echo "  ✓ vggt 包路径已正确"
    fi
else
    echo "  ⚠ 警告: 找不到 vggt finder 文件，可能需要重新安装"
fi

# 6. 初始化 conda（仅首次需要）
echo "[6/5] 初始化 conda..."
if ! grep -q "conda initialize" ~/.bashrc; then
    /home/zerun/miniconda3/bin/conda init bash
    echo "  ✓ conda 已初始化到 ~/.bashrc"
    echo "  ⚠ 请运行: source ~/.bashrc"
else
    echo "  ✓ conda 已初始化"
fi

echo "=========================================="
echo "环境恢复完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 重新加载 shell: source ~/.bashrc"
echo "2. 激活环境: conda activate vggt-train"
echo "3. 开始训练: cd /home/zerun/workspace/vggt && ./train_single_gpu.sh lora_kitti360_strategy_b"
echo ""
```

## 手动恢复步骤

如果自动脚本不可用，可以按以下步骤手动执行：

### 1. 创建 miniconda3 符号链接

```bash
sudo ln -s /home/zerun/workspace/zerun/miniconda3 /home/zerun/miniconda3
```

验证：
```bash
ls -la /home/zerun/miniconda3
```

### 2. 创建数据集符号链接

```bash
sudo ln -s /home/zerun/workspace/zerun/data /home/zerun/data
```

验证：
```bash
ls /home/zerun/data/dataset/KITTI-360/
```

### 3. 创建项目符号链接（如果不存在）

```bash
sudo ln -s /home/zerun/workspace/zerun/workspace/vggt /home/zerun/workspace/vggt
```

验证：
```bash
ls /home/zerun/workspace/vggt/
```

### 4. 安装 OpenGL 依赖

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

这是 cv2 (OpenCV) 在 Docker 环境中所需的依赖。

### 5. 更新 vggt 包的安装路径

编辑文件：
```bash
nano /home/zerun/miniconda3/envs/vggt-train/lib/python3.12/site-packages/__editable___vggt_0_0_1_finder.py
```

将所有 `/home/zerun/workspace/zerun/workspace/vggt` 替换为 `/home/zerun/workspace/vggt`

或者使用 sed 命令：
```bash
sed -i 's|/home/zerun/workspace/zerun/workspace/vggt|/home/zerun/workspace/vggt|g' \
  /home/zerun/miniconda3/envs/vggt-train/lib/python3.12/site-packages/__editable___vggt_0_0_1_finder.py
```

### 6. 初始化 conda

```bash
/home/zerun/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 验证环境

```bash
# 1. 激活 conda 环境
conda activate vggt-train

# 2. 验证 Python 版本
python --version  # 应该显示 Python 3.12.11

# 3. 验证 vggt 包导入
python -c "import vggt; from vggt.models.aggregator import Aggregator; print('✓ vggt import successful')"

# 4. 验证数据集路径
ls /home/zerun/data/dataset/KITTI-360/calibration/

# 5. 验证 GPU
nvidia-smi
```

## 启动训练

```bash
cd /home/zerun/workspace/vggt
./train_single_gpu.sh lora_kitti360_strategy_b
```

## 路径映射总结

| 原始路径（挂载点） | 符号链接 | 说明 |
|------------------|---------|------|
| `/home/zerun/workspace/zerun/miniconda3` | `/home/zerun/miniconda3` | Conda 环境 |
| `/home/zerun/workspace/zerun/data` | `/home/zerun/data` | 数据集目录 |
| `/home/zerun/workspace/zerun/workspace/vggt` | `/home/zerun/workspace/vggt` | 项目目录 |

## 常见问题

### Q1: `conda: command not found`
**解决方案：**
```bash
source ~/.bashrc
# 或者
/home/zerun/miniconda3/bin/conda init bash && source ~/.bashrc
```

### Q2: `ImportError: libGL.so.1: cannot open shared object file`
**解决方案：**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Q3: `FileNotFoundError: /home/zerun/data/dataset/KITTI-360/...`
**解决方案：**
检查数据集符号链接是否正确：
```bash
ls -la /home/zerun/data
# 应该显示: lrwxrwxrwx ... /home/zerun/data -> /home/zerun/workspace/zerun/data
```

### Q4: 训练时提示找不到 vggt 模块
**解决方案：**
重新更新 vggt 包路径（参见步骤 5）

## 注意事项

1. **每次 Docker 重启都需要重新创建符号链接**（因为 Docker 容器文件系统会重置）
2. **conda init 只需要执行一次**（会写入 `~/.bashrc`）
3. **OpenGL 依赖安装一次即可**（除非使用全新镜像）
4. **训练会自动从最新 checkpoint 恢复**（保存在项目的 `logs/` 目录）

## 制作持久化脚本

为了更方便，建议将恢复脚本添加到 `~/.bashrc`：

```bash
echo "" >> ~/.bashrc
echo "# Auto-restore VGGT environment on Docker restart" >> ~/.bashrc
echo "if [ ! -L /home/zerun/miniconda3 ]; then" >> ~/.bashrc
echo "    echo '检测到 Docker 重启，正在恢复环境...'" >> ~/.bashrc
echo "    bash /home/zerun/restore_env.sh" >> ~/.bashrc
echo "fi" >> ~/.bashrc
```

这样每次打开新终端时会自动检查并恢复环境。
