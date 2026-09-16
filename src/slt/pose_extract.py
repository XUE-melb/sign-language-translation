"""关键点提取（C 层，D-027）：与训练时 scripts/extract_rtmpose.py 逐项一致的 rtmlib 封装。

一致的点（改任何一项都会造成 demo 输入分布与训练不同）：
  - rtmlib Wholebody，mode="lightweight"，backend="onnxruntime"，to_openpose=False（D-020）
  - 每帧只取平均置信度最高的那个人；整帧没检到人 → 133 点全 0、置信度 0
  - 坐标按 [w, h] 归一化到 0~1；原生帧率，不抽帧（抽帧在 dataloader / infer 里按 max_frames 做）
输出 {"keypoints": (T,133,2) float32, "scores": (T,133) float32}，与 CE-CSL/pose_rtm 的 pkl 同格式。

两种用法：
  extract_video(path)                 一段视频 → (K, S)
  PoseExtractor().frame(img, w, h)    逐帧（摄像头 / WebSocket 端侧），返回 (133,2), (133,)
"""
import time

import numpy as np

N_KP = 133


class PoseExtractor:
    def __init__(self, mode="lightweight", device=None):
        import onnxruntime as ort
        from rtmlib import Wholebody
        if device is None:
            device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        self.device = device
        self.model = Wholebody(to_openpose=False, mode=mode, backend="onnxruntime", device=device)
        # onnxruntime 报 CUDA 可用但实际跑 CPU 的坑（D-020）：看 session 真正用的 provider
        try:
            self.providers = self.model.det_model.session.get_providers()
        except Exception:
            self.providers = ["?"]

    def frame(self, img_bgr, w=None, h=None):
        """一帧 BGR 图像 -> (133,2) 归一化坐标, (133,) 置信度。"""
        if w is None or h is None:
            h, w = img_bgr.shape[:2]
        kp, sc = self.model(img_bgr)
        kp, sc = np.asarray(kp), np.asarray(sc)
        if kp.size == 0:
            return np.zeros((N_KP, 2), np.float32), np.zeros((N_KP,), np.float32)
        best = int(sc.mean(axis=1).argmax())
        k = kp[best].astype(np.float32) / np.array([w, h], np.float32)[None, :]
        return k, sc[best].astype(np.float32)

    def video(self, path, max_seconds=None):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError("无法打开视频: {}".format(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        limit = int(max_seconds * fps) if max_seconds else None
        kps, scs, t0 = [], [], time.time()
        while True:
            ok, frame = cap.read()
            if not ok or (limit and len(kps) >= limit):
                break
            k, s = self.frame(frame, w, h)
            kps.append(k); scs.append(s)
        cap.release()
        if not kps:
            return np.zeros((0, N_KP, 2), np.float32), np.zeros((0, N_KP), np.float32), \
                {"w": w, "h": h, "fps": fps, "frames": 0, "seconds": time.time() - t0}
        return np.stack(kps), np.stack(scs), \
            {"w": w, "h": h, "fps": fps, "frames": len(kps), "seconds": time.time() - t0}


_EXTRACTOR = None


def extract_video(path, max_seconds=None):
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = PoseExtractor()
    return _EXTRACTOR.video(path, max_seconds)
