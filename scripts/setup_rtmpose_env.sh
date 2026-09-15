#!/bin/bash
# RTMPose 隔离环境：走 rtmlib + onnxruntime，绕开 mmcv/mmpose/mmdet 的版本地狱。
# 和 .venv-pose（mediapipe）、训练环境（torch）三者互不干扰。
set -e
VENV=/root/autodl-tmp/slt/.venv-rtmpose

if [ ! -d "$VENV" ]; then
    /root/miniconda3/bin/python -m venv "$VENV"
    echo "venv 已创建: $VENV"
fi

unset http_proxy https_proxy all_proxy      # pip 走国内源，不要代理
"$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1

echo "--- 安装 rtmlib + onnxruntime-gpu + opencv ---"
"$VENV/bin/pip" install -q rtmlib "onnxruntime-gpu" "opencv-python-headless<5" "numpy<2" 2>&1 | tail -5

echo
echo "===== 版本 ====="
"$VENV/bin/python" - <<'PY'
import numpy, cv2, onnxruntime as ort
print("  numpy       ", numpy.__version__)
print("  cv2         ", cv2.__version__)
print("  onnxruntime ", ort.__version__)
print("  可用 provider:", ort.get_available_providers())
import rtmlib
print("  rtmlib      ", getattr(rtmlib, "__version__", "(无版本号)"))
from rtmlib import Wholebody
print("  Wholebody 可导入 OK")
PY
