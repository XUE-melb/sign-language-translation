"""验证 Uni-Sign 预处理复刻正确。

这块是最容易藏 bug 的地方——冻结编码器对预处理差异没有适应能力，
写错了不会报错，只会让阶段 1 的分数莫名偏低。所以逐条验。
"""
import numpy as np
import torch

from slt.data import CharVocab
from slt.data_rtm import (BODY_IDX, CONF_THR, FACE_IDX, GROUPS, LH_IDX,
                          N_KP, RH_IDX, RTMPoseSLTDataset, build_groups,
                          crop_scale)

ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"

ok = True
print("=" * 70)
print("检查 1：索引数量与 Uni-Sign 论文一致")
for name, idx, _ in GROUPS:
    print("  {:<11}{:>4} 点".format(name, len(idx)))
print("  合计 {} 点（论文: 9+21+21+18 = 69）".format(N_KP))
ok &= (len(BODY_IDX), len(LH_IDX), len(RH_IDX), len(FACE_IDX)) == (9, 21, 21, 18)
ok &= N_KP == 69
print("  索引无重复:", len(set(BODY_IDX + LH_IDX + RH_IDX + FACE_IDX)) == 69)
print("  face 最后一个索引 =", FACE_IDX[-1], "(应为 53，中心化锚点)")
ok &= FACE_IDX[-1] == 53

print("=" * 70)
print("检查 2：crop_scale 的输出范围与清零行为（构造数据）")
T, N = 5, 10
m = np.zeros((T, N, 3), np.float32)
m[..., 0] = np.random.RandomState(0).uniform(100, 500, (T, N))
m[..., 1] = np.random.RandomState(1).uniform(100, 500, (T, N))
m[..., 2] = 0.9
m[:, 3, 2] = 0.1                       # 第 3 个点置信度低于阈值
r = crop_scale(m, CONF_THR)
print("  输出范围 [{:.3f}, {:.3f}]（应在 [-1,1] 内）".format(r.min(), r.max()))
print("  低置信度点是否整行清零:", np.all(r[:, 3, :] == 0))
print("    -> 该点 x/y/conf 三个通道都为 0:", r[:, 3, :].tolist()[0])
ok &= r.min() >= -1 - 1e-6 and r.max() <= 1 + 1e-6
ok &= np.all(r[:, 3, :] == 0)

print("=" * 70)
print("检查 3：bbox 是在整段序列上算的，不是逐帧")
m2 = m.copy()
m2[0, :, 0] += 2000                    # 只改第 0 帧，若逐帧归一化则其他帧不受影响
r2 = crop_scale(m2, CONF_THR)
changed = not np.allclose(r[1:], r2[1:], atol=1e-5)
print("  改第 0 帧后，其余帧的结果是否也变了:", changed, "（应为 True）")
ok &= changed

print("=" * 70)
print("检查 4：真实数据上的分组中心化")
vocab = CharVocab.build(["测试"])
ds = RTMPoseSLTDataset(ROOT, CSVD, "dev", vocab, frame_stride=2, max_frames=256)
print("  dev 样本 {} 条，缺 {} 条，dim={}".format(len(ds), len(ds.missing), ds.dim))
assert len(ds) > 0, "没有 RTMPose 数据，先跑提取"

import pickle
with open(ds.items[0]["path"], "rb") as f:
    d = pickle.load(f)
feat, spans = build_groups(d["keypoints"], d["scores"])
print("  build_groups 输出 {} （应为 (T, 69, 3)）".format(feat.shape))
print("  分段:", {k: v for k, v in spans.items()})
ok &= feat.shape[1:] == (69, 3)

# 中心化验证：归一化前，手腕相对自身应为原点
K, S = d["keypoints"], d["scores"]
for name, idx, anchor in GROUPS:
    if anchor is None:
        continue
    kp = K[:, idx, :].astype(np.float32)
    a = anchor if anchor >= 0 else len(idx) + anchor
    centered = kp - kp[:, a:a + 1, :]
    mx = float(np.abs(centered[:, a, :]).max())
    print("  {:<11} 锚点(第{}点)中心化后最大绝对值 {:.8f}".format(name, a, mx))
    ok &= mx == 0.0

print("=" * 70)
print("检查 5：低置信度点在最终输出里确实为 0")
lo = (d["scores"] <= CONF_THR)
print("  原始低置信度点占比 {:.4f}".format(lo.mean()))
sel = BODY_IDX + LH_IDX + RH_IDX + FACE_IDX
lo_sel = lo[:, sel]
if lo_sel.any():
    vals = feat[lo_sel]
    print("  这些点在输出中的最大绝对值 {:.8f}（应为 0）".format(float(np.abs(vals).max())))
    ok &= float(np.abs(vals).max()) == 0.0
else:
    print("  本样本无低置信度点，跳过")

print("=" * 70)
print("检查 6：Dataset 输出形状与展平顺序")
s = ds[0]
print("  feat {}  tokens {}".format(tuple(s["feat"].shape), tuple(s["tokens"].shape)))
ok &= s["feat"].shape[1] == 207
T = s["feat"].shape[0]
back = s["feat"].numpy().reshape(T, 69, 3)
print("  展平可逆:", back.shape == (T, 69, 3))
print("  取值范围 [{:.3f}, {:.3f}]".format(float(back.min()), float(back.max())))
ok &= back.min() >= -1 - 1e-6 and back.max() <= 1 + 1e-6

print("=" * 70)
print("总判定:", "全部通过" if ok else "有失败项")
