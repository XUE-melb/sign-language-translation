"""RTMPose 提取质检：与标注 CSV 对账 + 置信度分布 + 按手语者聚合。"""
import csv
import os
import pickle

import numpy as np

ROOT = "/root/autodl-tmp/slt/CE-CSL/pose_rtm"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
BODY = [0] + list(range(3, 11))
LH = list(range(91, 112))
RH = list(range(112, 133))
FACE = list(range(23, 40))[::2] + list(range(83, 91)) + [53]

print("=" * 76)
print("RTMPose 提取质检（mode=lightweight，对齐 Uni-Sign）")
allok = True
for split in ("train", "dev", "test"):
    man_p = os.path.join(ROOT, "manifest_%s.csv" % split)
    if not os.path.exists(man_p):
        print("  [%s] manifest 缺失" % split)
        allok = False
        continue
    rows = list(csv.DictReader(open(man_p, encoding="utf-8")))
    ok = [r for r in rows if r["status"] in ("ok", "cached")]
    bad = [r for r in rows if r["status"] not in ("ok", "cached")]
    ann = {r["Number"].strip() for r in csv.DictReader(
        open(os.path.join(CSVD, split + ".csv"), encoding="utf-8"))}
    got = {r["number"] for r in ok}
    n_pkl = sum(len(fs) for _, _, fs in os.walk(os.path.join(ROOT, split)))

    print("-" * 76)
    print("[%s] 标注 %d | manifest 成功 %d 失败 %d | 磁盘 pkl %d | 缺 %d"
          % (split, len(ann), len(ok), len(bad), n_pkl, len(ann - got)))
    for r in bad[:5]:
        print("    失败:", r["number"], r["status"])
    for n in sorted(ann - got)[:5]:
        print("    缺:", n)
    allok &= (len(bad) == 0 and len(ann - got) == 0)

    frames = [int(r["frames"]) for r in ok]
    print("  帧数 min %d 中位 %d max %d 均值 %.0f" % (
        min(frames), int(np.median(frames)), max(frames), np.mean(frames)))
    for k, label in (("body_conf", "身体"), ("lh_conf", "左手"),
                     ("rh_conf", "右手"), ("face_conf", "面部")):
        v = np.array([float(r[k]) for r in ok])
        print("  {:<6} 置信度 均值 {:.3f}  p05 {:.3f}  中位 {:.3f}".format(
            label, v.mean(), np.percentile(v, 5), np.median(v)))

    by = {}
    for r in ok:
        by.setdefault(r["translator"], []).append(r)
    print("  按手语者的手部置信度均值:")
    line = "   "
    for t in sorted(by):
        g = by[t]
        m = np.mean([float(r["lh_conf"]) + float(r["rh_conf"]) for r in g]) / 2
        line += " {}={:.2f}".format(t, m)
    print(line)

# 抽样验证 pkl 内容
print("=" * 76)
p = None
for dp, _, fs in os.walk(os.path.join(ROOT, "dev")):
    for f in fs:
        if f.endswith(".pkl"):
            p = os.path.join(dp, f)
            break
    if p:
        break
with open(p, "rb") as f:
    d = pickle.load(f)
K, S = d["keypoints"], d["scores"]
print("抽样 %s" % os.path.basename(p))
print("  keypoints %s %s | scores %s %s" % (K.shape, K.dtype, S.shape, S.dtype))
print("  坐标 x [%.3f, %.3f]  y [%.3f, %.3f]" % (
    K[..., 0].min(), K[..., 0].max(), K[..., 1].min(), K[..., 1].max()))
print("  置信度 [%.3f, %.3f] 中位 %.3f  | > 0.3 比例 %.3f" % (
    S.min(), S.max(), np.median(S), (S > 0.3).mean()))
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(ROOT) for f in fs if f.endswith(".pkl"))
print("  总体积 %.2f GB" % (sz / 1e9))
print("=" * 76)
print("质检结论:", "全部通过" if allok else "有问题，需检查")
