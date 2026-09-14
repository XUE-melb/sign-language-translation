#!/bin/bash
set -e
unset http_proxy https_proxy all_proxy
VENV=/root/autodl-tmp/slt/.venv-pose

if [ ! -d "$VENV" ]; then
  /root/miniconda3/bin/python -m venv "$VENV"
  echo "venv created: $VENV"
fi

"$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -2

# 先钉死 numpy<2，再让 mediapipe 在这个约束下解析
echo "--- installing numpy<2 ---"
"$VENV/bin/pip" install -q "numpy<2" 2>&1 | tail -3

echo "--- resolving mediapipe under numpy<2 ---"
for V in 0.10.21 0.10.20 0.10.18 0.10.14; do
  echo "try mediapipe==$V"
  if "$VENV/bin/pip" install -q "mediapipe==$V" "numpy<2" 2>&1 | tail -3; then
    echo "INSTALLED mediapipe==$V"
    break
  fi
done

echo "===== 结果 ====="
"$VENV/bin/python" -c "
import numpy, mediapipe as mp
print('numpy', numpy.__version__)
print('mediapipe', mp.__version__)
has_legacy = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'holistic')
print('legacy mp.solutions.holistic:', has_legacy)
try:
    from mediapipe.tasks.python.vision import HolisticLandmarker
    print('tasks HolisticLandmarker: True')
except Exception as e:
    print('tasks HolisticLandmarker: False', type(e).__name__)
import cv2; print('cv2', cv2.__version__)
"
