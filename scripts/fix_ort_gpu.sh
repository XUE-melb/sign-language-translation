#!/bin/bash
# onnxruntime 没走 GPU 的诊断与修复。
V=/root/autodl-tmp/slt/.venv-rtmpose
unset http_proxy https_proxy all_proxy

echo "===== 1. 当前 provider 与冲突包 ====="
"$V/bin/python" -c "
import onnxruntime as ort
print('  onnxruntime', ort.__version__)
print('  file:', ort.__file__)
print('  available:', ort.get_available_providers())
"
echo "--- 同时装了哪些 onnxruntime 包（冲突源）---"
"$V/bin/pip" list 2>/dev/null | grep -i onnxruntime

echo
echo "===== 2. CUDA / cuDNN 运行库是否存在 ====="
echo "  nvidia-smi 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
ldconfig -p 2>/dev/null | grep -cE "libcudnn|libcublas" | xargs -I{} echo "  系统 ldconfig 中 cudnn/cublas 条目数: {}"
ls /root/miniconda3/lib/libcudnn*.so* 2>/dev/null | head -3 || echo "  conda lib 下无 libcudnn"
"$V/bin/pip" list 2>/dev/null | grep -iE "nvidia-cudnn|nvidia-cublas" || echo "  venv 内无 nvidia-cudnn/cublas 包"

echo
echo "===== 3. 修复：卸掉 CPU 版，只留 GPU 版 ====="
"$V/bin/pip" uninstall -y -q onnxruntime 2>&1 | tail -2
"$V/bin/pip" install -q --force-reinstall --no-deps onnxruntime-gpu 2>&1 | tail -2
# onnxruntime-gpu 需要 cudnn9 / cublas 的 pip 运行库
"$V/bin/pip" install -q "nvidia-cudnn-cu12" "nvidia-cublas-cu12" "nvidia-cufft-cu12" "nvidia-curand-cu12" 2>&1 | tail -3

echo
echo "===== 4. 修复后再看 ====="
"$V/bin/python" - <<'PY'
import os, glob, sys
# 把 pip 装的 nvidia 运行库目录加进搜索路径
import site
sp = site.getsitepackages()[0]
libs = [p for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib")) if os.path.isdir(p)]
os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
print("  注入 LD_LIBRARY_PATH 条目数:", len(libs))
import onnxruntime as ort
print("  available:", ort.get_available_providers())
PY
echo
echo "（注意：LD_LIBRARY_PATH 必须在 python 启动前设好才生效，上面只是探测。）"
echo "真正可用的启动方式见 scripts/run_rtmpose.sh"
