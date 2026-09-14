#!/usr/bin/env python
"""诊断手部检出率偏低的成因。

pose 检出率是 1.0，说明骨架一直在。那就从已提取的 .npy 里直接算：
  - 人在画面里占多大（肩宽、躯干高，归一化坐标）
  - 手腕的 visibility（骨架认为手腕可见吗）
再和 manifest 里的手部检出率做相关。

不重跑视频，纯用已有数据。
    python diag_hands.py --split dev
"""
import argparse
import csv
import json
import os

import numpy as np

# MediaPipe Pose 33 点索引
L_SH, R_SH = 11, 12
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/slt/CE-CSL")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    pose_root = os.path.join(args.root, "pose")
    layout = json.load(open(os.path.join(pose_root, "layout.json"), encoding="utf-8"))
    seg = {s["name"]: (s["start"], s["end"]) for s in layout["segments"]}
    rows = [r for r in csv.DictReader(
        open(os.path.join(pose_root, "manifest_{}.csv".format(args.split)),
             encoding="utf-8")) if r["status"] == "ok"]

    recs = []
    for r in rows:
        p = os.path.join(pose_root, args.split, r["translator"], r["number"] + ".npy")
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float32)
        ps, pe = seg["pose"]
        pose = a[:, ps:pe].reshape(len(a), 33, 4)      # x, y, z, visibility
        x, y, vis = pose[:, :, 0], pose[:, :, 1], pose[:, :, 3]

        shoulder_w = np.abs(x[:, L_SH] - x[:, R_SH]).mean()
        torso_h = np.abs(y[:, L_SH] - y[:, L_HIP]).mean()
        lw_vis, rw_vis = vis[:, L_WR].mean(), vis[:, R_WR].mean()
        # 手腕在画面内的比例（归一化坐标越界说明被裁掉）
        lw_in = float(((x[:, L_WR] > 0) & (x[:, L_WR] < 1) &
                       (y[:, L_WR] > 0) & (y[:, L_WR] < 1)).mean())
        rw_in = float(((x[:, R_WR] > 0) & (x[:, R_WR] < 1) &
                       (y[:, R_WR] > 0) & (y[:, R_WR] < 1)).mean())

        recs.append(dict(
            t=r["translator"], n=r["number"],
            lh=float(r["lh_rate"]), rh=float(r["rh_rate"]),
            shoulder_w=float(shoulder_w), torso_h=float(torso_h),
            lw_vis=float(lw_vis), rw_vis=float(rw_vis),
            lw_in=lw_in, rw_in=rw_in, frames=int(r["frames"])))

    print("样本 {} 条".format(len(recs)))
    hand = [(r["lh"] + r["rh"]) / 2 for r in recs]

    print("=" * 74)
    print("与「双手平均检出率」的相关系数:")
    for k, label in (("shoulder_w", "肩宽(归一化)"), ("torso_h", "躯干高"),
                     ("lw_vis", "左腕visibility"), ("rw_vis", "右腕visibility"),
                     ("frames", "帧数")):
        print("  {:<16} r = {:+.3f}".format(label, corr([r[k] for r in recs], hand)))

    print("=" * 74)
    print("按 translator: 看人在画面里的尺度差异")
    print("  {:<4}{:>5}{:>11}{:>10}{:>9}{:>9}{:>9}{:>9}".format(
        "T", "n", "肩宽", "躯干高", "左手", "右手", "左腕vis", "右腕vis"))
    by = {}
    for r in recs:
        by.setdefault(r["t"], []).append(r)
    for t in sorted(by):
        g = by[t]
        print("  {:<4}{:>5}{:>11.4f}{:>10.4f}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}".format(
            t, len(g),
            np.mean([r["shoulder_w"] for r in g]),
            np.mean([r["torso_h"] for r in g]),
            np.mean([r["lh"] for r in g]), np.mean([r["rh"] for r in g]),
            np.mean([r["lw_vis"] for r in g]), np.mean([r["rw_vis"] for r in g])))

    print("=" * 74)
    print("手腕越界(出画)比例 —— 若手腕跑出画面，手部必然检不到")
    print("  {:<4}{:>5}{:>12}{:>12}".format("T", "n", "左腕在画内", "右腕在画内"))
    for t in sorted(by):
        g = by[t]
        print("  {:<4}{:>5}{:>12.3f}{:>12.3f}".format(
            t, len(g), np.mean([r["lw_in"] for r in g]),
            np.mean([r["rw_in"] for r in g])))

    print("=" * 74)
    print("按肩宽分档看手部检出率（检验「人太小所以手检不到」）:")
    sw = np.array([r["shoulder_w"] for r in recs])
    hand_a = np.array(hand)
    edges = np.percentile(sw, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        m = (sw >= lo) & (sw <= hi if i == 4 else sw < hi)
        if m.sum():
            print("  肩宽 [{:.3f},{:.3f})  n={:<4} 手部检出 {:.3f}".format(
                lo, hi, int(m.sum()), hand_a[m].mean()))


if __name__ == "__main__":
    main()
