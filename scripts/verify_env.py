import numpy, cv2
import mediapipe as mp

print("numpy    ", numpy.__version__)
print("cv2      ", cv2.__version__)
print("mediapipe", mp.__version__)

h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
print("Holistic 实例化: OK")

fm = mp.solutions.face_mesh
idx = set()
for g in (fm.FACEMESH_LEFT_EYEBROW, fm.FACEMESH_RIGHT_EYEBROW,
          fm.FACEMESH_LEFT_EYE, fm.FACEMESH_RIGHT_EYE, fm.FACEMESH_LIPS):
    for a, b in g:
        idx.add(a)
        idx.add(b)
print("compact 面部点数:", len(idx))
print("特征维度 D =", 33 * 4 + 21 * 3 + 21 * 3 + len(idx) * 3)
h.close()
