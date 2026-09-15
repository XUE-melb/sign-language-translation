"""看清 pose 侧各模块的分组与输入输出维度，决定阶段 1 怎么接线。"""
import collections
import re

import torch

P = "/root/autodl-tmp/slt/weights/unisign/csl_stage1_weight.pth"
sd = torch.load(P, map_location="cpu", weights_only=False)
for k in ("model", "state_dict"):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
        sd = sd[k]
        break
T = {k: v for k, v in sd.items() if hasattr(v, "shape")}

print("=" * 76)
print("gcn_modules 的分组与各组邻接矩阵（A 的维度 = 该组关键点数）")
for k, v in sorted(T.items()):
    if k.startswith("gcn_modules") and k.endswith(".gcn.A"):
        print("  {:<52} {}".format(k, tuple(v.shape)))

print()
print("=" * 76)
print("fusion_gcn_modules 的分组")
groups = collections.OrderedDict()
for k in T:
    if k.startswith("fusion_gcn_modules"):
        g = ".".join(k.split(".")[:2])
        groups.setdefault(g, 0)
        groups[g] += T[k].numel()
for g, n in groups.items():
    print("  {:<40}{:>12,}".format(g, n))
print("  (共 {} 组)".format(len(groups)))
for k, v in sorted(T.items()):
    if k.startswith("fusion_gcn_modules") and k.endswith(".gcn.A"):
        print("  A: {:<48} {}".format(k, tuple(v.shape)))

print()
print("=" * 76)
print("pose_proj / proj_linear / part_para —— 编码器到 mT5 的接口")
for k, v in sorted(T.items()):
    if k.startswith(("pose_proj", "proj_linear", "part_para")):
        print("  {:<52} {}".format(k, tuple(v.shape)))

print()
print("=" * 76)
print("各分组 GCN 的最后一层输出通道（决定编码器输出维度）")
pat = re.compile(r"^(gcn_modules|fusion_gcn_modules)\.([a-z_]+)\.layer(\d+)_(\d+)\.gcn\.conv\.weight$")
last = {}
for k, v in T.items():
    m = pat.match(k)
    if m:
        mod, grp, li, lj = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        key = (mod, grp)
        if key not in last or (li, lj) > last[key][0]:
            last[key] = ((li, lj), tuple(v.shape))
for (mod, grp), (li, shape) in sorted(last.items()):
    print("  {:<22}{:<12} 最后层 layer{}_{}  conv.weight {}  -> 输出通道 {}".format(
        mod, grp, li[0], li[1], shape, shape[0]))

print()
print("=" * 76)
print("mT5 的规格")
for k in ("mt5_model.shared.weight",):
    if k in T:
        v = T[k]
        print("  词表 {} x 隐层 {}  -> mT5-base".format(v.shape[0], v.shape[1]))
nblk = len({k.split(".")[3] for k in T if k.startswith("mt5_model.encoder.block.")})
print("  encoder block 数:", nblk)
