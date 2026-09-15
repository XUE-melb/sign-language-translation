# -*- coding: utf-8 -*-
"""汇总五个 lr 的 dev 结果，并和 seed 噪声比较，判断差异是否显著。"""
import json
import os

import torch

RUNS = "/root/autodl-tmp/slt/runs"
LRS = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]

print("=" * 64)
print("lr 曲线（seed 1234，dev 最佳 BLEU-4）")
print("  {:<8}{:>12}{:>10}{:>12}".format("lr", "dev BLEU-4", "峰值epoch", "末轮loss"))
best_lr, best_v = None, -1.0
rows = []
for lr in LRS:
    d = os.path.join(RUNS, "stage0_lr%s_s1234" % lr)
    try:
        ck = torch.load(os.path.join(d, "best.pt"), map_location="cpu")
        h = json.load(open(os.path.join(d, "history.json"), encoding="utf-8"))
        v, ep = ck["dev_bleu4"], ck["epoch"]
        last = h["history"][-1]["loss"]
    except Exception as e:
        print("  {:<8}{:>12}".format(lr, "读取失败"))
        continue
    rows.append((lr, v, ep, last))
    print("  {:<8}{:>12.4f}{:>10}{:>12.3f}".format(lr, v, ep, last))
    if v > best_v:
        best_v, best_lr = v, lr

print()
print("  最优: lr=%s (dev BLEU-4 %.4f)" % (best_lr, best_v))
i = LRS.index(best_lr)
print("  位置: %s（%s）" % (
    "区间内部 ✓ 边界问题已解决" if 0 < i < len(LRS) - 1 else "仍在边界 ✗",
    "左右都有更差的点" if 0 < i < len(LRS) - 1 else "需继续扩展"))

# 与 seed 噪声比较
print("=" * 64)
print("seed 噪声（lr=3e-4 的三个 seed，dev 选点值）")
vals = []
for s in (1234, 2345, 3456):
    d = os.path.join(RUNS, "stage0_lr3e-4_s%d" % s)
    try:
        vals.append(torch.load(os.path.join(d, "best.pt"), map_location="cpu")["dev_bleu4"])
    except Exception:
        pass
if len(vals) >= 2:
    m = sum(vals) / len(vals)
    sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    print("  取值 %s" % ", ".join("%.4f" % v for v in vals))
    print("  均值 %.4f  总体标准差 %.4f  极差 %.4f" % (m, sd, max(vals) - min(vals)))

    top2 = sorted(rows, key=lambda r: -r[1])[:2]
    gap = top2[0][1] - top2[1][1]
    print()
    print("  前两名: lr=%s (%.4f) vs lr=%s (%.4f)，差距 %.4f" % (
        top2[0][0], top2[0][1], top2[1][0], top2[1][1], gap))
    print("  该差距 / seed 标准差 = %.2f" % (gap / sd if sd else float("inf")))
    print()
    if gap < sd:
        print("  判定: 差距小于单个 seed 的标准差 -> 两者在统计上不可区分。")
    else:
        print("  判定: 差距超过 seed 标准差 -> 值得进一步用多 seed 确认。")
