#!/bin/bash
# VGGT Docker 环境快速恢复脚本
# 新挂载方式: /media/.../home/zerun -> /home/zerun

echo "=========================================="
echo "VGGT Docker 环境恢复"
echo "=========================================="

# 1. 检查 miniconda3 是否存在
echo "[1/3] 检查 miniconda3..."
if [ -d /home/zerun/miniconda3 ]; then
    echo "  ✓ miniconda3 目录已存在"
else
    echo "  ✗ 错误: miniconda3 目录不存在，请检查挂载"
    exit 1
fi

# 2. 检查数据集目录是否存在
echo "[2/3] 检查数据集目录..."
if [ -d /home/zerun/data ]; then
    echo "  ✓ data 目录已存在"
else
    echo "  ⚠ 警告: data 目录不存在"
fi

# 3. 安装 OpenGL 依赖（Docker 环境必需）
echo "[3/3] 安装 OpenGL 依赖..."
if ! dpkg -l | grep -q libgl1-mesa-glx; then
    sudo apt-get update -qq
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    echo "  ✓ OpenGL 依赖已安装"
else
    echo "  ✓ OpenGL 依赖已存在"
fi

# 4. 初始化 conda（仅首次需要）
echo "[4/3] 初始化 conda..."
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
