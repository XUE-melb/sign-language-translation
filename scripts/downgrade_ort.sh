#!/bin/bash
# onnxruntime-gpu 1.30 要求 CUDA 13，本机是 CUDA 12.1 -> 降级适配（CLAUDE.md 硬约束）。
# 依次尝试几个已知面向 CUDA 12 + cuDNN 9 的版本，第一个能真正建起
# CUDA session 的就停下。判据不是 get_available_providers()，
# 而是实际建 session 后 get_providers() 里有没有 CUDA。
V=/root/autodl-tmp/slt/.venv-rtmpose
unset http_proxy https_proxy all_proxy

verify() {
"$V/bin/python" - <<'PY'
import sys, numpy as np, onnxruntime as ort
try:
    # 构造一个最小模型来真实建 session
    from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions
    import onnx  # noqa
except Exception:
    pass
import glob, os
cks = glob.glob("/root/.cache/rtmlib/hub/checkpoints/*.onnx")
if not cks:
    print("NO_MODEL"); sys.exit(1)
m = min(cks, key=os.path.getsize)
try:
    s = ort.InferenceSession(m, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    p = s.get_providers()
    print("PROVIDERS:", p)
    sys.exit(0 if "CUDAExecutionProvider" in p else 2)
except Exception as e:
    print("ERR:", type(e).__name__, str(e)[:120]); sys.exit(3)
PY
}

for VER in 1.20.1 1.19.2 1.18.1; do
    echo "================ 尝试 onnxruntime-gpu==$VER ================"
    "$V/bin/pip" install -q "onnxruntime-gpu==$VER" 2>&1 | tail -2
    OUT=$(verify 2>&1); RC=$?
    echo "$OUT" | tail -2
    if [ $RC -eq 0 ]; then
        echo ">>> 成功：onnxruntime-gpu==$VER 能真正建起 CUDA session"
        "$V/bin/pip" list 2>/dev/null | grep -iE "onnxruntime|nvidia-cudnn|nvidia-cublas"
        exit 0
    fi
    echo ">>> $VER 不行（rc=$RC），继续"
done

echo "!!! 所有候选版本都失败，需要换思路（见下方说明）"
exit 1
