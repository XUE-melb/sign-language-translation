#!/usr/bin/env python
"""关键点提取质检：读 manifest + layout，报告帧数/检出率/存储量分布。

重点不是"跑完没有"，而是找出检出率低的样本——手语里手部丢失等于信号丢失。
    python qc_pose.py --split dev
"""
import argparse
import csv
import json
import os

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def describe(name, a, fmt="{:.3f}"):
    a = np.asarray(a, dtype=np.float64)
    if not len(a):
        print("  {:<12} (空)".format(name))
        return
    row = "  {:<12} min " + fmt + "  p05 " + fmt + "  中位 " + fmt + \
          "  p95 " + fmt + "  max " + fmt + "  均值 " + fmt
    print(row.format(name, a.min(), pct(a, 5), pct(a, 50), pct(a, 95), a.max(), a.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/slt/CE-CSL")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--csv-dir", default="/root/autodl-tmp/slt/TFNet/data/CE-CSL")
    ap.add_argument("--hand-threshold", type=float, default=0.5,
                    help="双手平均检出率低于此值的样本列为可疑")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    pose_root = os.path.join(args.root, "pose")
    man_path = os.path.join(pose_root, "manifest_{}.csv".format(args.split))
    rows = list(csv.DictReader(open(man_path, encoding="utf-8")))

    layout = json.load(open(os.path.join(pose_root, "layout.json"), encoding="utf-8"))
    print("=" * 72)
    print("布局: D={} dtype={} face_set={} ({} 点)".format(
        layout["dim"], layout["dtype"], layout["face_set"], len(layout["face_indices"])))
    for s in layout["segments"]:
        print("  {:<11} [{:>3}:{:>3}]  {} 点 x {} 通道".format(
            s["name"], s["start"], s["end"], s["n_points"], s["n_channels"]))

    ok = [r for r in rows if r["status"] in ("ok", "skipped")]
    bad = [r for r in rows if r["status"] not in ("ok", "skipped")]
    print("=" * 72)
    print("[{}] manifest {} 条，成功 {}，失败 {}".format(
        args.split, len(rows), len(ok), len(bad)))
    for r in bad[:args.show]:
        print("  失败: {} {}".format(r["number"], r["status"]))

    # 与标注 CSV 对账
    ann = list(csv.DictReader(open(os.path.join(args.csv_dir, args.split + ".csv"),
                                   encoding="utf-8")))
    ann_ids = {r["Number"].strip() for r in ann}
    got_ids = {r["number"] for r in ok}
    print("标注 {} 条 | 提取成功 {} 条 | 标注有而提取缺 {} 条".format(
        len(ann_ids), len(got_ids), len(ann_ids - got_ids)))
    for n in sorted(ann_ids - got_ids)[:args.show]:
        print("  缺:", n)

    real = [r for r in ok if float(r["pose_rate"]) >= 0]   # 跳过 skipped 的占位行
    if not real:
        print("\n(本次全部为 skipped，无新统计)")
        return

    frames = [int(r["frames"]) for r in real]
    print("=" * 72)
    print("帧数分布 (CLAUDE.md 记载平均约 158 帧):")
    describe("frames", frames, "{:.0f}")

    print("\n检出率分布:")
    for key, label in (("pose_rate", "pose"), ("lh_rate", "左手"),
                       ("rh_rate", "右手"), ("face_rate", "face")):
        describe(label, [float(r[key]) for r in real])

    secs = [float(r["seconds"]) for r in real]
    describe("\n单视频耗时", secs, "{:.2f}")

    # 存储量
    total = 0
    for r in real:
        p = os.path.join(pose_root, args.split, r["translator"], r["number"] + ".npy")
        if os.path.exists(p):
            total += os.path.getsize(p)
    per = total / max(len(real), 1)
    print("\n存储: 本 split {:.1f} MB，单视频均值 {:.0f} KB".format(
        total / 1e6, per / 1e3))
    print("      按 5988 条全量外推约 {:.2f} GB".format(per * 5988 / 1e9))

    # 可疑样本：双手检出率过低
    susp = sorted(real, key=lambda r: (float(r["lh_rate"]) + float(r["rh_rate"])) / 2)
    low = [r for r in susp
           if (float(r["lh_rate"]) + float(r["rh_rate"])) / 2 < args.hand_threshold]
    print("=" * 72)
    print("双手平均检出率 < {:.2f} 的样本: {} 条 ({:.1f}%)".format(
        args.hand_threshold, len(low), 100 * len(low) / len(real)))
    for r in susp[:args.show]:
        print("  {} {}  左手 {:.2f} 右手 {:.2f} pose {:.2f} 帧 {}".format(
            r["translator"], r["number"], float(r["lh_rate"]), float(r["rh_rate"]),
            float(r["pose_rate"]), r["frames"]))

    # 按 translator 聚合，看是否某个手语员系统性偏差
    print("=" * 72)
    print("按 translator 聚合 (看是否某人系统性检出偏低):")
    by = {}
    for r in real:
        by.setdefault(r["translator"], []).append(r)
    print("  {:<4}{:>6}{:>10}{:>10}{:>10}".format("T", "n", "左手", "右手", "帧数中位"))
    for t in sorted(by):
        g = by[t]
        print("  {:<4}{:>6}{:>10.3f}{:>10.3f}{:>10.0f}".format(
            t, len(g),
            np.mean([float(r["lh_rate"]) for r in g]),
            np.mean([float(r["rh_rate"]) for r in g]),
            np.median([int(r["frames"]) for r in g])))

    # 抽一条实际 .npy 验证形状与数值范围
    print("=" * 72)
    sample = real[0]
    p = os.path.join(pose_root, args.split, sample["translator"], sample["number"] + ".npy")
    a = np.load(p)
    print("抽样 {}: shape={} dtype={}".format(sample["number"], a.shape, a.dtype))
    for s in layout["segments"]:
        seg = a[:, s["start"]:s["end"]].astype(np.float32)
        nz = seg[np.any(seg != 0, axis=1)]
        print("  {:<11} 非零帧 {:>4}/{:<4} 取值 [{:.3f}, {:.3f}]".format(
            s["name"], len(nz), len(seg),
            float(nz.min()) if len(nz) else 0.0,
            float(nz.max()) if len(nz) else 0.0))


if __name__ == "__main__":
    main()
