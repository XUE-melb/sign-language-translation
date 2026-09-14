"""dataloader 正确性检查 —— 重点验归一化没有把零填充变成假坐标。"""
import numpy as np
import torch

from slt.data import (CharVocab, PoseSLTDataset, collate_fn, normalize_body,
                      L_SHOULDER, R_SHOULDER)

ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSV = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"

vocab = CharVocab.build(["你好世界", "今天天气很好"])
ds = PoseSLTDataset(ROOT, CSV, "dev", vocab, frame_stride=2,
                    max_frames=256, normalize="none")
print("dev 样本 {} 条，缺关键点 {} 条，D={}".format(
    len(ds), len(ds.missing), ds.dim))

raw = np.load(ds.items[0]["path"]).astype(np.float32)
segs = ds.segs
print("原始 shape", raw.shape)

norm = normalize_body(raw, segs)
ms, me, _, _ = segs["mask"]
mask = raw[:, ms:me]

print("=" * 66)
print("检查 1：未检出的部位归一化后必须仍是全 0")
ok = True
for name, mi in (("left_hand", 1), ("right_hand", 2), ("face", 3)):
    s, e, _, _ = segs[name]
    absent = mask[:, mi] < 0.5
    if absent.sum() == 0:
        print("  {:<11} 本视频无未检出帧，跳过".format(name))
        continue
    blk = norm[absent, s:e]
    mx = float(np.abs(blk).max())
    good = mx == 0.0
    ok &= good
    print("  {:<11} 未检出 {:>4} 帧，归一化后最大绝对值 {:.6f}  {}".format(
        name, int(absent.sum()), mx, "OK" if good else "<<< 失败：凭空造出了坐标"))

print("=" * 66)
print("检查 2：归一化后肩中点应在原点、肩宽应为 1")
ps, pe, pn, pc = segs["pose"]
pose_n = norm[:, ps:pe].reshape(-1, pn, pc)
present = mask[:, 0] > 0.5
center = (pose_n[present, L_SHOULDER, :2] + pose_n[present, R_SHOULDER, :2]) / 2
width = np.linalg.norm(
    pose_n[present, L_SHOULDER, :2] - pose_n[present, R_SHOULDER, :2], axis=-1)
print("  肩中点 |x| 最大 {:.6f}，|y| 最大 {:.6f}（应约为 0）".format(
    float(np.abs(center[:, 0]).max()), float(np.abs(center[:, 1]).max())))
print("  肩宽 min {:.6f} max {:.6f}（应约为 1）".format(
    float(width.min()), float(width.max())))
ok &= float(np.abs(center).max()) < 1e-5 and abs(float(width.mean()) - 1.0) < 1e-4

print("=" * 66)
print("检查 3：归一化确实改变了取值范围（说明它真的生效了）")
print("  原始 pose xy 范围 [{:.3f}, {:.3f}]".format(
    float(raw[:, ps:pe].reshape(-1, pn, pc)[:, :, :2].min()),
    float(raw[:, ps:pe].reshape(-1, pn, pc)[:, :, :2].max())))
print("  归一后 pose xy 范围 [{:.3f}, {:.3f}]".format(
    float(pose_n[:, :, :2].min()), float(pose_n[:, :, :2].max())))

print("=" * 66)
print("检查 4：batch 拼装与长度")
ds_n = PoseSLTDataset(ROOT, CSV, "dev", vocab, frame_stride=2,
                      max_frames=256, normalize="body")
batch = collate_fn([ds_n[i] for i in range(4)])
print("  feats", tuple(batch["feats"].shape), "feat_lens", batch["feat_lens"].tolist())
print("  tokens", tuple(batch["tokens"].shape), "token_lens", batch["token_lens"].tolist())
for i, L in enumerate(batch["feat_lens"].tolist()):
    tail = batch["feats"][i, L:]
    assert tail.numel() == 0 or float(tail.abs().max()) == 0.0, "padding 区非零"
print("  padding 区全 0  OK")
print("  样例文本:", batch["texts"][0])

print("=" * 66)
print("检查 5：max_frames 上限生效")
longest = max(np.load(it["path"], mmap_mode="r").shape[0] for it in ds.items[:200])
print("  前 200 条里最长原始帧数 {}，stride=2 后 {}".format(longest, longest // 2))
mx = max(ds_n[i]["feat"].shape[0] for i in range(0, len(ds_n), 37))
print("  抽样后实际最大帧数 {}（上限 256）".format(mx))
ok &= mx <= 256

print("=" * 66)
print("总判定:", "全部通过" if ok else "有失败项")
