#!/usr/bin/env python
"""CE-CSL 关键点提取：视频 -> 每视频一个 .npy。

输出目录结构沿用 video/ 的约定：pose/{split}/{translator}/{number}.npy
每个 .npy 形状 (T, D) float16，D 的分段布局写在 pose/layout.json。

用法：
    python extract_pose.py --split dev --workers 12
    python extract_pose.py --split dev --verify-channel-order   # 验通道顺序
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

# ---------------------------------------------------------------- 布局定义

POSE_N, POSE_C = 33, 4          # x, y, z, visibility
HAND_N, HAND_C = 21, 3          # x, y, z
FACE_C = 3                      # x, y, z


def build_face_indices(face_set):
    """从 mediapipe 的 FACEMESH 常量推导面部子集索引，不硬编码。"""
    if face_set == "none":
        return []
    import mediapipe as mp
    fm = mp.solutions.face_mesh
    groups = {
        "compact": [
            fm.FACEMESH_LEFT_EYEBROW, fm.FACEMESH_RIGHT_EYEBROW,
            fm.FACEMESH_LEFT_EYE, fm.FACEMESH_RIGHT_EYE,
            fm.FACEMESH_LIPS,
        ],
        "contours": [
            fm.FACEMESH_LEFT_EYEBROW, fm.FACEMESH_RIGHT_EYEBROW,
            fm.FACEMESH_LEFT_EYE, fm.FACEMESH_RIGHT_EYE,
            fm.FACEMESH_LIPS, fm.FACEMESH_NOSE, fm.FACEMESH_FACE_OVAL,
        ],
    }
    idx = set()
    for conn in groups[face_set]:
        for a, b in conn:
            idx.add(a)
            idx.add(b)
    return sorted(idx)


def layout_spec(face_idx):
    """返回 (总维度, 分段描述)。分段描述给 dataloader 切片用。

    末尾的 mask 段是 4 个 0/1 标志 (pose, left_hand, right_hand, face)。
    必须有：未检出时坐标填 0，而 0 本身是合法的归一化坐标（图像左上角），
    没有 mask 就无法区分"手不在画面里"和"手在左上角"。手垂下休息本身
    携带语义（句子边界、非签名段），不能和真实坐标混在一起。
    """
    segs, off = [], 0
    for name, n, c in (("pose", POSE_N, POSE_C),
                       ("left_hand", HAND_N, HAND_C),
                       ("right_hand", HAND_N, HAND_C),
                       ("face", len(face_idx), FACE_C),
                       ("mask", 4, 1)):
        if n == 0:
            continue
        segs.append({"name": name, "start": off, "end": off + n * c,
                     "n_points": n, "n_channels": c})
        off += n * c
    return off, segs


# ---------------------------------------------------------------- 每帧打包

def pack_frame(res, face_idx):
    """把一帧 holistic 结果打成扁平向量；未检出的部位填 0。"""
    pose = np.zeros((POSE_N, POSE_C), np.float32)
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark[:POSE_N]):
            pose[i] = (lm.x, lm.y, lm.z, lm.visibility)

    hands = []
    for lms in (res.left_hand_landmarks, res.right_hand_landmarks):
        h = np.zeros((HAND_N, HAND_C), np.float32)
        if lms:
            for i, lm in enumerate(lms.landmark[:HAND_N]):
                h[i] = (lm.x, lm.y, lm.z)
        hands.append(h)

    parts = [pose.ravel(), hands[0].ravel(), hands[1].ravel()]
    if face_idx:
        face = np.zeros((len(face_idx), FACE_C), np.float32)
        if res.face_landmarks:
            lms = res.face_landmarks.landmark
            for i, fi in enumerate(face_idx):
                if fi < len(lms):
                    face[i] = (lms[fi].x, lms[fi].y, lms[fi].z)
        parts.append(face.ravel())

    det = (res.pose_landmarks is not None,
           res.left_hand_landmarks is not None,
           res.right_hand_landmarks is not None,
           res.face_landmarks is not None)
    # mask 段：把"这个部位这一帧到底检到没有"显式写进特征，
    # 否则零填充和真实的 0 坐标无法区分。
    parts.append(np.asarray(det, dtype=np.float32))
    return np.concatenate(parts), det


# ---------------------------------------------------------------- worker

_H = None
_FACE_IDX = None
_CFG = None


def _init_worker(cfg, face_idx):
    global _H, _FACE_IDX, _CFG
    import mediapipe as mp
    _CFG, _FACE_IDX = cfg, face_idx
    _H = mp.solutions.holistic.Holistic(
        static_image_mode=False,          # 视频流：跨帧跟踪，更快也更稳
        model_complexity=cfg["model_complexity"],
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _read_frames(path, resize_width=0):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("无法打开视频: " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if resize_width and bgr.shape[1] > resize_width:
            h = int(bgr.shape[0] * resize_width / bgr.shape[1])
            bgr = cv2.resize(bgr, (resize_width, h), interpolation=cv2.INTER_AREA)
        yield bgr, fps
    cap.release()


def process_one(task):
    """返回一行 manifest 记录。"""
    import cv2
    src, dst, split, translator, number = task
    t0 = time.time()
    try:
        if os.path.exists(dst) and not _CFG["overwrite"]:
            # 已提取过：检出率直接从 mask 段反算，不重跑视频。
            # 这样边下边提、分多次运行时，最终 manifest 仍然是完整可统计的。
            arr = np.load(dst, mmap_mode="r")
            ms, me = _CFG["mask_span"]
            m = np.asarray(arr[:, ms:me], dtype=np.float32).mean(axis=0)
            return dict(split=split, translator=translator, number=number,
                        frames=arr.shape[0], dim=arr.shape[1], fps=0.0,
                        pose_rate=round(float(m[0]), 4), lh_rate=round(float(m[1]), 4),
                        rh_rate=round(float(m[2]), 4), face_rate=round(float(m[3]), 4),
                        seconds=0.0, status="cached")

        feats, dets, fps = [], [], 0.0
        for bgr, fps in _read_frames(src, _CFG["resize_width"]):
            # cv2.VideoCapture 返回 BGR，mediapipe 要 RGB —— 这里只转一次
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = _H.process(rgb)
            v, d = pack_frame(res, _FACE_IDX)
            feats.append(v)
            dets.append(d)

        if not feats:
            return dict(split=split, translator=translator, number=number,
                        frames=0, dim=0, fps=fps, pose_rate=0, lh_rate=0,
                        rh_rate=0, face_rate=0, seconds=time.time() - t0,
                        status="empty")

        arr = np.stack(feats).astype(np.float16)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, dst)

        d = np.array(dets, dtype=np.float32).mean(axis=0)
        return dict(split=split, translator=translator, number=number,
                    frames=arr.shape[0], dim=arr.shape[1], fps=round(fps, 2),
                    pose_rate=round(float(d[0]), 4), lh_rate=round(float(d[1]), 4),
                    rh_rate=round(float(d[2]), 4), face_rate=round(float(d[3]), 4),
                    seconds=round(time.time() - t0, 2), status="ok")
    except Exception as e:
        return dict(split=split, translator=translator, number=number,
                    frames=0, dim=0, fps=0.0, pose_rate=0, lh_rate=0, rh_rate=0,
                    face_rate=0, seconds=round(time.time() - t0, 2),
                    status="error:{}:{}".format(type(e).__name__, e))


# ---------------------------------------------------------------- 通道顺序验证

def verify_channel_order(video, face_set, model_complexity, max_frames=60):
    """实测 RGB vs BGR 两种喂法的检出率，验证转换是否必要。

    CLAUDE.md 提到原脚本 imageio 已是 RGB 却又做了一次 BGR2RGB。
    这里不靠推理，直接比检出率。
    """
    import cv2
    import mediapipe as mp
    face_idx = build_face_indices(face_set)
    out = {}
    for mode in ("RGB", "BGR"):
        h = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=model_complexity,
            refine_face_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        dets, n = [], 0
        cap = cv2.VideoCapture(video)
        while n < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if mode == "RGB" else bgr
            _, d = pack_frame(h.process(img), face_idx)
            dets.append(d)
            n += 1
        cap.release()
        h.close()
        m = np.array(dets, dtype=np.float32).mean(axis=0)
        out[mode] = dict(pose=float(m[0]), left_hand=float(m[1]),
                         right_hand=float(m[2]), face=float(m[3]), frames=n)
    return out


# ---------------------------------------------------------------- 主流程

def collect_tasks(root, split, csv_path, pose_root, limit=0):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    tasks, missing = [], []
    for r in rows:
        number = r["Number"].strip()
        translator = r["Translator"].strip()
        src = os.path.join(root, "video", split, translator, number + ".mp4")
        if not os.path.exists(src):
            missing.append(number)
            continue
        dst = os.path.join(pose_root, split, translator, number + ".npy")
        tasks.append((src, dst, split, translator, number))
    if limit:
        tasks = tasks[:limit]
    return tasks, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/slt/CE-CSL")
    ap.add_argument("--csv-dir", default="/root/autodl-tmp/slt/TFNet/data/CE-CSL")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--face-set", default="compact",
                    choices=["none", "compact", "contours"])
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--resize-width", type=int, default=0,
                    help="0 表示不缩放；1080p 下可设 960 提速")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-channel-order", action="store_true")
    args = ap.parse_args()

    pose_root = os.path.join(args.root, "pose")
    csv_path = os.path.join(args.csv_dir, args.split + ".csv")

    if args.verify_channel_order:
        tasks, _ = collect_tasks(args.root, args.split, csv_path, pose_root, limit=1)
        if not tasks:
            sys.exit("没有可用视频")
        print("验证视频: " + tasks[0][0])
        res = verify_channel_order(tasks[0][0], args.face_set, args.model_complexity)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        rgb, bgr = res["RGB"], res["BGR"]
        print("\n结论: RGB 手部检出 {:.3f}/{:.3f}, BGR {:.3f}/{:.3f}".format(
            rgb["left_hand"], rgb["right_hand"], bgr["left_hand"], bgr["right_hand"]))
        return

    face_idx = build_face_indices(args.face_set)
    dim, segs = layout_spec(face_idx)
    os.makedirs(pose_root, exist_ok=True)
    layout = {"dim": dim, "dtype": "float16", "face_set": args.face_set,
              "face_indices": face_idx, "segments": segs,
              "model_complexity": args.model_complexity,
              "resize_width": args.resize_width}
    with open(os.path.join(pose_root, "layout.json"), "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    tasks, missing = collect_tasks(args.root, args.split, csv_path, pose_root, args.limit)
    print("[{}] 待处理 {} 个视频，CSV 中缺失视频 {} 个，特征维度 D={} (face {} 点)".format(
        args.split, len(tasks), len(missing), dim, len(face_idx)), flush=True)
    if missing[:5]:
        print("  缺失示例:", missing[:5], flush=True)
    if not tasks:
        return

    mask_seg = [g for g in segs if g["name"] == "mask"][0]
    cfg = {"model_complexity": args.model_complexity,
           "resize_width": args.resize_width, "overwrite": args.overwrite,
           "mask_span": (mask_seg["start"], mask_seg["end"])}

    man_path = os.path.join(pose_root, "manifest_{}.csv".format(args.split))
    fields = ["split", "translator", "number", "frames", "dim", "fps",
              "pose_rate", "lh_rate", "rh_rate", "face_rate", "seconds", "status"]
    t0 = time.time()
    done = ok = 0
    with open(man_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        w.writeheader()
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(cfg, face_idx)) as pool:
            for rec in pool.imap_unordered(process_one, tasks, chunksize=1):
                w.writerow(rec)
                done += 1
                if rec["status"] in ("ok", "cached"):
                    ok += 1
                else:
                    print("  !! {} {}".format(rec["number"], rec["status"]), flush=True)
                if done % 25 == 0 or done == len(tasks):
                    el = time.time() - t0
                    eta = el / done * (len(tasks) - done)
                    mf.flush()
                    print("  {}/{} ok={} 用时 {:.1f}min ETA {:.1f}min".format(
                        done, len(tasks), ok, el / 60, eta / 60), flush=True)

    print("\n完成: {}/{}，manifest -> {}".format(ok, len(tasks), man_path), flush=True)


if __name__ == "__main__":
    main()
