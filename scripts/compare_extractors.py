"""在同一批视频上对比 RTMPose 与 MediaPipe 的输出，并验证 pkl 内容。"""
import csv
import os
import pickle

import numpy as np

RTM = "/root/autodl-tmp/slt/CE-CSL/pose_rtm"
MP = "/root/autodl-tmp/slt/CE-CSL/pose"
SPLIT = "dev"

rtm_rows = [r for r in csv.DictReader(
    open(os.path.join(RTM, "manifest_%s.csv" % SPLIT), encoding="utf-8"))
    if r["status"] in ("ok", "cached")]
mp_rows = {r["number"]: r for r in csv.DictReader(
    open(os.path.join(MP, "manifest_%s.csv" % SPLIT), encoding="utf-8"))}

print("=" * 72)
print("RTMPose 提取 %d 条" % len(rtm_rows))

# --- 1. 验证 pkl 内容 ---
r0 = rtm_rows[0]
p = os.path.join(RTM, SPLIT, r0["translator"], r0["number"] + ".pkl")
with open(p, "rb") as f:
    d = pickle.load(f)
K, S = d["keypoints"], d["scores"]
print()
print("抽样 %s:" % r0["number"])
print("  keypoints %s %s   scores %s %s" % (K.shape, K.dtype, S.shape, S.dtype))
print("  坐标范围 x [%.3f, %.3f]  y [%.3f, %.3f]  (应在 0~1 附近)" % (
    K[..., 0].min(), K[..., 0].max(), K[..., 1].min(), K[..., 1].max()))
print("  置信度范围 [%.3f, %.3f]" % (S.min(), S.max()))
print("  文件大小 %.0f KB" % (os.path.getsize(p) / 1e3))

# --- 2. 帧数是否一致 ---
print()
print("=" * 72)
print("帧数一致性（RTMPose vs MediaPipe，同一视频应完全相同）")
bad = 0
for r in rtm_rows:
    m = mp_rows.get(r["number"])
    if m and int(r["frames"]) != int(m["frames"]):
        bad += 1
        if bad <= 3:
            print("  不一致 %s: rtm %s vs mp %s" % (r["number"], r["frames"], m["frames"]))
print("  %d/%d 条帧数完全一致" % (len(rtm_rows) - bad, len(rtm_rows)))

# --- 3. 手部"检出率"对比 ---
print()
print("=" * 72)
print("手部检出率对比（同一批 %d 条视频）" % len(rtm_rows))
print("  RTMPose 口径: 该部位平均置信度 > 0.3 的帧占比")
print("  MediaPipe 口径: holistic 是否返回该手")
print()
print("  {:<14}{:>12}{:>12}".format("", "左手", "右手"))
rl = np.mean([float(r["lh_det"]) for r in rtm_rows])
rr = np.mean([float(r["rh_det"]) for r in rtm_rows])
ml = np.mean([float(mp_rows[r["number"]]["lh_rate"]) for r in rtm_rows if r["number"] in mp_rows])
mr = np.mean([float(mp_rows[r["number"]]["rh_rate"]) for r in rtm_rows if r["number"] in mp_rows])
print("  {:<14}{:>12.3f}{:>12.3f}".format("RTMPose", rl, rr))
print("  {:<14}{:>12.3f}{:>12.3f}".format("MediaPipe", ml, mr))
print("  {:<14}{:>+12.3f}{:>+12.3f}".format("差值", rl - ml, rr - mr))

print()
print("  RTMPose 各部位平均置信度:")
for k, label in (("body_conf", "身体"), ("lh_conf", "左手"),
                 ("rh_conf", "右手"), ("face_conf", "面部")):
    print("    {:<8}{:.3f}".format(label, np.mean([float(r[k]) for r in rtm_rows])))

# 置信度分布：RTMW 的 SimCC 输出不是归一化概率，可能 >1，
# 需要知道实际尺度才能解释 Uni-Sign dataloader 里的 0.3 阈值
print()
print("=" * 72)
print("置信度尺度（关系到 Uni-Sign dataloader 的 0.3 阈值怎么解释）")
allS = []
for r in rtm_rows[:10]:
    with open(os.path.join(RTM, SPLIT, r["translator"], r["number"] + ".pkl"), "rb") as f:
        allS.append(pickle.load(f)["scores"])
A = np.concatenate(allS, axis=0)
print("  全体分位数: p01 %.3f  p25 %.3f  中位 %.3f  p75 %.3f  p99 %.3f  max %.3f" % (
    *np.percentile(A, [1, 25, 50, 75, 99]), A.max()))
print("  > 0.3 的比例: %.3f" % (A > 0.3).mean())
for name, idx in (("身体", [0] + list(range(3, 11))),
                  ("左手", list(range(91, 112))),
                  ("右手", list(range(112, 133))),
                  ("面部", list(range(23, 40, 2)) + list(range(83, 91)) + [53])):
    sub = A[:, idx]
    print("    {:<6} 中位 {:.3f}  > 0.3 比例 {:.3f}".format(
        name, float(np.median(sub)), float((sub > 0.3).mean())))

secs = [float(r["seconds"]) for r in rtm_rows if float(r["seconds"]) > 0]
if secs:
    print()
    print("  单视频耗时: 中位 %.1fs  均值 %.1fs  max %.1fs" % (
        np.median(secs), np.mean(secs), max(secs)))
    print("  -> 5988 条 / 2 worker 约 %.1f 小时" % (np.mean(secs) * 5988 / 2 / 3600))
