"""看清 Uni-Sign checkpoint 的结构，找出 pose 编码器是哪一部分。"""
import collections

import torch

P = "/root/autodl-tmp/slt/weights/unisign/csl_stage1_weight.pth"

ck = torch.load(P, map_location="cpu", weights_only=False)
print("顶层类型:", type(ck).__name__)
if isinstance(ck, dict):
    print("顶层键:", [k for k in ck.keys()][:15])

sd = ck
for key in ("model", "state_dict", "module"):
    if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
        print("-> 使用子键:", key)
        sd = sd[key]
        break

tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
print()
print("参数张量数:", len(tensors))
total = sum(v.numel() for v in tensors.values())
print("总参数量: {:.2f}M".format(total / 1e6))

# 按一级前缀聚合
groups = collections.OrderedDict()
for k, v in tensors.items():
    g = k.split(".")[0]
    groups.setdefault(g, [0, 0])
    groups[g][0] += v.numel()
    groups[g][1] += 1

print()
print("按一级前缀分组:")
print("  {:<28}{:>14}{:>8}{:>8}".format("前缀", "参数量", "占比", "张量数"))
for g, (n, c) in sorted(groups.items(), key=lambda x: -x[1][0]):
    print("  {:<28}{:>14,}{:>7.1f}%{:>8}".format(g, n, 100 * n / total, c))

# 找 pose 编码器相关的键
print()
print("含 gcn / pose / part 字样的键（前 25 个）:")
hits = [k for k in tensors if any(s in k.lower() for s in ("gcn", "pose", "part", "spatial", "temporal"))]
for k in hits[:25]:
    print("  {:<60} {}".format(k, tuple(tensors[k].shape)))
print("  ... 共 {} 个，合计 {:.2f}M 参数".format(
    len(hits), sum(tensors[k].numel() for k in hits) / 1e6))

print()
print("含 mt5 / decoder / shared 字样的键（前 8 个）:")
mt5 = [k for k in tensors if any(s in k.lower() for s in ("mt5", "t5", "decoder", "shared", "encoder.block"))]
for k in mt5[:8]:
    print("  {:<60} {}".format(k, tuple(tensors[k].shape)))
print("  ... 共 {} 个，合计 {:.2f}M 参数".format(
    len(mt5), sum(tensors[k].numel() for k in mt5) / 1e6))
