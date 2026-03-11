#!/bin/bash
# ============================================================
# S3Gaussian 路侧数据集环境安装脚本
# CUDA 11.8 + PyTorch 2.2.1
# ============================================================
set -e

echo "=========================================="
echo " S3Gaussian Roadside Setup"
echo " CUDA 11.8 + PyTorch 2.2.1"
echo "=========================================="

# 1. 创建 conda 环境
echo "[1/5] Creating conda environment..."
conda create -n s3gs python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate s3gs

# 2. 安装 PyTorch 2.2.1 + CUDA 11.8
echo "[2/5] Installing PyTorch 2.2.1 + CUDA 11.8..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装基础依赖
echo "[3/5] Installing base dependencies..."
pip install plyfile opencv-python open3d imageio imageio-ffmpeg
pip install tqdm numpy scipy Pillow

# 4. 编译 CUDA 子模块
echo "[4/5] Building CUDA submodules..."
cd "$(dirname "$0")/../../"
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn

# 5. 验证安装
echo "[5/5] Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import diff_gaussian_rasterization
print('diff_gaussian_rasterization: OK')

from simple_knn._C import distCUDA2
print('simple_knn: OK')

print('All checks passed!')
"

echo "=========================================="
echo " Setup complete!"
echo " Activate with: conda activate s3gs"
echo "=========================================="
