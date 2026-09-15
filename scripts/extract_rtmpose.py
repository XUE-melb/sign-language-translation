#!/usr/bin/env python
"""用 RTMPose (rtmlib) 提 COCO-WholeBody 133 关键点，格式对齐 Uni-Sign。

为什么要再提一遍（已有 MediaPipe 版本）：
  阶段 1 要用 Uni-Sign 的冻结预训练 pose 编码器，它在 RTMPose 提取的
  133 点上预训练（1,985 小时）。冻结模型对分布偏移没有适应能力，
  必须用同一个提取器。详见 DECISIONS.md D-020。

输出格式严格对齐 Uni-Sign 的 demo/pose_extraction.py：
  {"keypoints": (T, 133, 2) float32,  # 已按 [w, h] 归一化到 0~1
   "scores":    (T, 133)    float32}
存 .pkl，目录结构沿用 video/ 的约定。

69 点的筛选与分组中心化由下游 dataloader 做，这里**不做**任何裁剪，
保留全部 133 点 —— 和 Uni-Sign 一致，也给将来换分组留余地。

用法：
    python extract_rtmpose.py --split dev --workers 2
"""
import argparse
import csv
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

N_KP = 133

# Uni-Sign 的 69 点分组（仅用于质检统计，不改变落盘内容）
BODY_IDX = [0] + list(range(3, 11))          # 9
LH_IDX = list(range(91, 112))                # 21
RH_IDX = list(range(112, 133))               # 21
FACE_IDX = list(range(23, 40, 2)) + list(range(83, 91)) + [53]   # 18


_MODEL = None
_CFG = None


def _init_worker(cfg):
    global _MODEL, _CFG
    import onnxruntime as ort
    from rtmlib import Wholebody
    _CFG = cfg
    dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    _MODEL = Wholebody(to_openpose=False, mode=cfg["mode"],
                       backend="onnxruntime", device=dev)


def process_one(task):
    import cv2
    src, dst, split, translator, number = task
    t0 = time.time()
    try:
        if os.path.exists(dst) and not _CFG["overwrite"]:
            with open(dst, "rb") as f:
                d = pickle.load(f)
            sc = d["scores"]
            return _stat_row(split, translator, number, d["keypoints"], sc,
                             0.0, "cached")

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError("无法打开视频: " + src)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        kps, scs = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kp, sc = _MODEL(frame)          # (n_person, 133, 2), (n_person, 133)
            kp, sc = np.asarray(kp), np.asarray(sc)
            if kp.size == 0:                # 整帧没检到人：填 0，置信度 0
                kps.append(np.zeros((N_KP, 2), np.float32))
                scs.append(np.zeros((N_KP,), np.float32))
                continue
            # 只取置信度最高的那个人（手语数据是单人录制）
            best = int(sc.mean(axis=1).argmax())
            kps.append(kp[best].astype(np.float32))
            scs.append(sc[best].astype(np.float32))
        cap.release()

        if not kps:
            return _stat_row(split, translator, number, np.zeros((0, N_KP, 2)),
                             np.zeros((0, N_KP)), time.time() - t0, "empty")

        K = np.stack(kps)                                    # (T, 133, 2) 像素
        S = np.stack(scs)                                    # (T, 133)
        K = K / np.array([w, h], np.float32)[None, None, :]  # 归一化，同 Uni-Sign

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"keypoints": K, "scores": S}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dst)
        return _stat_row(split, translator, number, K, S, time.time() - t0, "ok")
    except Exception as e:
        return dict(split=split, translator=translator, number=number,
                    frames=0, body_conf=0, lh_conf=0, rh_conf=0, face_conf=0,
                    lh_det=0, rh_det=0, seconds=round(time.time() - t0, 2),
                    status="error:{}:{}".format(type(e).__name__, e))


def _stat_row(split, translator, number, K, S, secs, status):
    """质检统计用 Uni-Sign 的分组，口径与 MediaPipe 版的 manifest 可比。"""
    if len(S) == 0:
        m = dict(body=0, lh=0, rh=0, face=0, lh_det=0, rh_det=0)
    else:
        m = dict(
            body=float(S[:, BODY_IDX].mean()),
            lh=float(S[:, LH_IDX].mean()),
            rh=float(S[:, RH_IDX].mean()),
            face=float(S[:, FACE_IDX].mean()),
            # "检出"口径：该部位平均置信度 > 0.3（Uni-Sign dataloader 的阈值）
            lh_det=float((S[:, LH_IDX].mean(axis=1) > 0.3).mean()),
            rh_det=float((S[:, RH_IDX].mean(axis=1) > 0.3).mean()),
        )
    return dict(split=split, translator=translator, number=number,
                frames=len(S),
                body_conf=round(m["body"], 4), lh_conf=round(m["lh"], 4),
                rh_conf=round(m["rh"], 4), face_conf=round(m["face"], 4),
                lh_det=round(m["lh_det"], 4), rh_det=round(m["rh_det"], 4),
                seconds=round(secs, 2), status=status)


def collect_tasks(root, split, csv_path, out_root, limit=0):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    tasks, missing = [], []
    for r in rows:
        number, translator = r["Number"].strip(), r["Translator"].strip()
        src = os.path.join(root, "video", split, translator, number + ".mp4")
        if not os.path.exists(src):
            missing.append(number)
            continue
        tasks.append((src,
                      os.path.join(out_root, split, translator, number + ".pkl"),
                      split, translator, number))
    return (tasks[:limit] if limit else tasks), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/slt/CE-CSL")
    ap.add_argument("--csv-dir", default="/root/autodl-tmp/slt/TFNet/data/CE-CSL")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--mode", default="lightweight",
                    choices=["performance", "balanced", "lightweight"],
                    help="lightweight = Uni-Sign pose_extraction.py 的默认值。"
                         "论文写 RTMPose-x 但代码默认 lightweight，以代码为准："
                         "只有 lightweight 的置信度在 [0,1] 尺度上，"
                         "其 dataloader 的 conf>0.3 阈值才讲得通（见 D-020）")
    ap.add_argument("--workers", type=int, default=2,
                    help="GPU 推理，进程别开太多，2-3 通常就跑满了")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = os.path.join(args.root, "pose_rtm")
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, args.split + ".csv")
    tasks, missing = collect_tasks(args.root, args.split, csv_path, out_root, args.limit)
    print("[{}] 待处理 {} 个视频，缺视频 {} 个，mode={}".format(
        args.split, len(tasks), len(missing), args.mode), flush=True)
    if not tasks:
        return

    cfg = {"mode": args.mode, "overwrite": args.overwrite}
    man = os.path.join(out_root, "manifest_{}.csv".format(args.split))
    fields = ["split", "translator", "number", "frames", "body_conf", "lh_conf",
              "rh_conf", "face_conf", "lh_det", "rh_det", "seconds", "status"]
    t0 = time.time()
    done = ok = 0
    with open(man, "w", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        w.writeheader()
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for rec in pool.imap_unordered(process_one, tasks, chunksize=1):
                w.writerow(rec)
                done += 1
                ok += rec["status"] in ("ok", "cached")
                if rec["status"] not in ("ok", "cached"):
                    print("  !! {} {}".format(rec["number"], rec["status"]), flush=True)
                if done % 20 == 0 or done == len(tasks):
                    el = time.time() - t0
                    mf.flush()
                    print("  {}/{} ok={} 用时 {:.1f}min ETA {:.1f}min".format(
                        done, len(tasks), ok, el / 60,
                        el / done * (len(tasks) - done) / 60), flush=True)
    print("\n完成 {}/{}，manifest -> {}".format(ok, len(tasks), man), flush=True)


if __name__ == "__main__":
    main()
