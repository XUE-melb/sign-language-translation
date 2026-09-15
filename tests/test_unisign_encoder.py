"""验证 Uni-Sign pose 编码器能独立实例化、权重能严格加载、前向输出正确。

要点：
  - 权重必须**严格**匹配（missing/unexpected 都为空），否则说明我搭的结构
    和他们的不一致，冻结权重就废了
  - 输出维度必须是 1024（4 组 × 256）
  - 喂真实 dev 数据跑通
"""
import time

import numpy as np
import torch

from slt.data import CharVocab
from slt.data_rtm import RTMPoseSLTDataset
from slt.models.unisign_encoder import (MODES, OUT_DIM, UniSignPoseEncoder,
                                        load_pretrained, split_groups)

CKPT = "/root/autodl-tmp/slt/weights/unisign/csl_stage1_weight.pth"
ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"

print("=" * 72)
print("检查 1：结构能否独立实例化（不加载 mT5）")
t0 = time.time()
enc = UniSignPoseEncoder()
n = sum(p.numel() for p in enc.parameters())
print("  实例化耗时 {:.2f}s，参数量 {:.2f}M".format(time.time() - t0, n / 1e6))
print("  各组邻接矩阵 A 的形状:")
for m in MODES:
    print("    {:<10} {}".format(m, tuple(torch.tensor(enc.graph[m].A).shape)))
print("  左右手是否共享权重:",
      enc.gcn_modules["left"] is enc.gcn_modules["right"],
      enc.proj_linear["left"] is enc.proj_linear["right"])

print("=" * 72)
print("检查 2：严格加载预训练权重")
enc, info = load_pretrained(CKPT, strict=False)
print("  载入张量 {} 个".format(info["loaded"]))
print("  missing    {} 个 {}".format(len(info["missing"]), info["missing"][:3]))
print("  unexpected {} 个 {}".format(len(info["unexpected"]), info["unexpected"][:3]))
print("  丢弃 mt5_model 张量 {} 个，pose_proj {} 个（留给阶段 2）".format(
    info["dropped_mt5"], info["dropped_pose_proj"]))
strict_ok = not info["missing"] and not info["unexpected"]
print("  严格匹配:", strict_ok)

print("=" * 72)
print("检查 3：喂真实 dev 数据前向")
vocab = CharVocab.build(["测试"])
ds = RTMPoseSLTDataset(ROOT, CSVD, "dev", vocab, frame_stride=2, max_frames=256)
s = ds[0]
x = s["feat"].unsqueeze(0)                    # (1, T, 207)
print("  输入 {}".format(tuple(x.shape)))
groups = split_groups(x)
for m in MODES:
    print("    {:<10} {}".format(m, tuple(groups[m].shape)))

enc.eval()
with torch.no_grad():
    y = enc(groups)
print("  输出 {}  （应为 (1, T, {})）".format(tuple(y.shape), OUT_DIM))
print("  取值 min {:.3f} 中位 {:.3f} max {:.3f}".format(
    float(y.min()), float(y.median()), float(y.max())))
shape_ok = y.shape == (1, x.shape[1], OUT_DIM)

print("=" * 72)
print("检查 4：冻结后不产生梯度")
for p in enc.parameters():
    p.requires_grad_(False)
y2 = enc(groups)
print("  输出 requires_grad:", y2.requires_grad, "（冻结后应为 False）")

print("=" * 72)
print("检查 5：GPU 上跑一个 batch")
if torch.cuda.is_available():
    enc_c = enc.cuda()
    # 各样本长度不同，按最短的截齐（真实训练里由 collate_fn 做 padding）
    fs = [ds[i]["feat"] for i in range(4)]
    L = min(f.shape[0] for f in fs)
    xb = torch.stack([f[:L] for f in fs]).cuda()
    gb = split_groups(xb)
    t0 = time.time()
    with torch.no_grad():
        yb = enc_c(gb)
    torch.cuda.synchronize()
    print("  batch 输入 {} -> 输出 {}，耗时 {:.3f}s".format(
        tuple(xb.shape), tuple(yb.shape), time.time() - t0))
else:
    print("  (无 GPU，跳过)")

print("=" * 72)
print("总判定:", "通过" if (strict_ok and shape_ok) else "有问题")
