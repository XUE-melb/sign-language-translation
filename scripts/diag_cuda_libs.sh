#!/bin/bash
# 查清这台机器上到底有哪些 CUDA/cuDNN 运行库，以及各自的版本。
echo "===== 1. 系统 CUDA ====="
ls -d /usr/local/cuda* 2>/dev/null
echo "  nvcc: $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*')"
echo "  驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"

echo
echo "===== 2. 系统库里的 cublasLt / cudnn ====="
for pat in libcublasLt libcudnn; do
  echo "  --- $pat ---"
  ldconfig -p 2>/dev/null | grep "$pat" | head -4 || echo "    (ldconfig 无)"
  find /usr/local/cuda*/lib64 /usr/lib/x86_64-linux-gnu -maxdepth 1 -name "${pat}.so.*" 2>/dev/null | head -4
done

echo
echo "===== 3. 训练环境 torch 自带的 nvidia 运行库 ====="
TSITE=/root/miniconda3/lib/python3.12/site-packages
ls -d $TSITE/nvidia/*/lib 2>/dev/null | head -12
echo "  --- 具体版本 ---"
find $TSITE/nvidia -name "libcublasLt.so.*" -o -name "libcudnn.so.*" 2>/dev/null | head -6
/root/miniconda3/bin/pip list 2>/dev/null | grep -iE "nvidia-cudnn|nvidia-cublas"

echo
echo "===== 4. rtmpose venv 里 pip 装的 nvidia 库 ====="
RSITE=/root/autodl-tmp/slt/.venv-rtmpose/lib/python3.12/site-packages
find $RSITE/nvidia -name "libcublasLt.so.*" -o -name "libcudnn.so.*" 2>/dev/null | head -6

echo
echo "===== 5. 当前 onnxruntime-gpu 版本 ====="
/root/autodl-tmp/slt/.venv-rtmpose/bin/pip list 2>/dev/null | grep -i onnxruntime
