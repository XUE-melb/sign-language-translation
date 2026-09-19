#!/usr/bin/env python
"""单个视频 → RTMPose 关键点 pkl，在 .venv-rtmpose（GPU onnxruntime）里跑，供 demo 服务端子进程调用（D-030）。

为什么要子进程：demo 服务跑在训练环境（miniconda，torch 2.3），那里只能装 CPU 版 onnxruntime（1.2 s/帧）；
GPU 版 onnxruntime 1.19.2 + pip cuDNN 只在 .venv-rtmpose 里能起（D-020），两个环境合不到一起。
用法：bash scripts/run_rtmpose.sh scripts/extract_one.py <video> <out.pkl> [max_seconds]
输出 pkl 与 CE-CSL/pose_rtm 同格式：{"keypoints": (T,133,2) float32 归一化, "scores": (T,133), "meta": {...}}
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from slt.pose_extract import PoseExtractor  # noqa: E402  只依赖 numpy / cv2 / onnxruntime / rtmlib


def main():
    src, dst = sys.argv[1], sys.argv[2]
    max_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else None
    px = PoseExtractor()
    K, S, meta = px.video(src, max_seconds)
    meta["device"] = px.device
    meta["providers"] = list(px.providers)
    with open(dst, "wb") as f:
        pickle.dump({"keypoints": K, "scores": S, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("ok frames={} device={} {:.1f}s".format(meta["frames"], px.device, meta["seconds"]))


if __name__ == "__main__":
    main()
