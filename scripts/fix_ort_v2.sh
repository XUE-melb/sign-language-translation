#!/bin/bash
# 正确地配 onnxruntime-gpu：这次把 LD_LIBRARY_PATH 设对再判断。
#
# 本机事实：
#   系统 CUDA 12.1，libcublasLt.so.12 在 /usr/local/cuda/targets/x86_64-linux/lib
#   系统 cuDNN 8.9.0，libcudnn.so.8 在 /usr/lib/x86_64-linux-gnu
#   -> onnxruntime-gpu 1.18.1 要 CUDA12+cuDNN8，和系统完全匹配
#   -> onnxruntime-gpu 1.19.2 要 CUDA12+cuDNN9，需用 venv 里 pip 装的 cudnn9
V=/root/autodl-tmp/slt/.venv-rtmpose
RSITE=$V/lib/python3.12/site-packages
SYS_CUDA=/usr/local/cuda/targets/x86_64-linux/lib
SYS_LIB=/usr/lib/x86_64-linux-gnu
unset http_proxy https_proxy all_proxy

probe() {   # $1 = LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$1" "$V/bin/python" - <<'PY' 2>&1 | tail -3
import glob, os, sys
import onnxruntime as ort
cks = glob.glob("/root/.cache/rtmlib/hub/checkpoints/*.onnx")
m = min(cks, key=os.path.getsize)
try:
    s = ort.InferenceSession(m, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    p = s.get_providers()
    print("RESULT", "OK" if "CUDAExecutionProvider" in p else "CPU_ONLY", p)
except Exception as e:
    print("RESULT ERR", type(e).__name__, str(e)[:100])
PY
}

echo "############ 方案 A: onnxruntime-gpu 1.18.1 + 系统 cuDNN 8 ############"
"$V/bin/pip" install -q "onnxruntime-gpu==1.18.1" 2>&1 | tail -1
"$V/bin/pip" uninstall -y -q nvidia-cudnn-cu12 2>&1 | tail -1   # 移掉冲突的 cudnn9
echo "LD_LIBRARY_PATH = $SYS_CUDA:$SYS_LIB"
probe "$SYS_CUDA:$SYS_LIB"

echo
echo "############ 方案 B: onnxruntime-gpu 1.19.2 + pip cuDNN 9 ############"
"$V/bin/pip" install -q "onnxruntime-gpu==1.19.2" "nvidia-cudnn-cu12==9.*" 2>&1 | tail -1
NV=""
for d in "$RSITE"/nvidia/*/lib; do [ -d "$d" ] && NV="$NV:$d"; done
NV="${NV#:}"
echo "LD_LIBRARY_PATH = $NV:$SYS_CUDA"
probe "$NV:$SYS_CUDA"

echo
echo "############ 当前状态 ############"
"$V/bin/pip" list 2>/dev/null | grep -iE "onnxruntime|nvidia-cudnn|nvidia-cublas"
