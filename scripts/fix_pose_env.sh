#!/bin/bash
unset http_proxy https_proxy all_proxy
VENV=/root/autodl-tmp/slt/.venv-pose
PIP="$VENV/bin/pip"

pkill -f "pip install" 2>/dev/null; sleep 2
echo "--- 清掉回溯留下的缓存 ---"
"$PIP" cache purge 2>&1 | tail -1

echo "--- 装 mediapipe (--no-deps，绕开 jax) ---"
"$PIP" install --no-deps mediapipe==0.10.14 2>&1 | tail -3

echo "--- 补 holistic 真正需要的运行时依赖 ---"
"$PIP" install "numpy<2" "protobuf>=4.25.3,<5" absl-py attrs flatbuffers \
    "opencv-contrib-python<5" sounddevice 2>&1 | tail -4

echo "===== 验证 ====="
"$VENV/bin/python" - <<'PY'
import numpy, cv2
import mediapipe as mp
print("numpy     ", numpy.__version__)
print("cv2       ", cv2.__version__)
print("mediapipe ", mp.__version__)
h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
print("Holistic 实例化: OK")
fm = mp.solutions.face_mesh
idx = set()
for g in (fm.FACEMESH_LEFT_EYEBROW, fm.FACEMESH_RIGHT_EYEBROW,
          fm.FACEMESH_LEFT_EYE, fm.FACEMESH_RIGHT_EYE, fm.FACEMESH_LIPS):
    for a, b in g:
        idx.add(a); idx.add(b)
print("compact 面部子集点数:", len(idx))
print("特征维度 D =", 33*4 + 21*3 + 21*3 + len(idx)*3)
h.close()
PY
