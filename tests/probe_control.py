"""零样本探针的对照：模型到底有没有在看 pose 输入？

D-015 的方法：同一模型，喂真实关键点 vs 全零关键点 vs 打乱配对，比输出。
  - 若真实 ≈ 全零 → 解码器在靠先验瞎说（域偏移把 pose 信号淹没了）
  - 若真实 ≠ 全零 但都错 → pose 有影响，只是映射不到 CE-CSL 的句子
两种情况的处置不同，所以要分清。

另外直接量 pose_proj 输出在不同样本间的差异，排除"embedding 常数化"这种接线 bug。

用法：python probe_control.py [ckpt_path]
"""
import sys

import torch
from torch.utils.data import DataLoader

from slt.data import CharVocab, collate_fn
from slt.data_rtm import RTMPoseSLTDataset
from slt.metrics import corpus_bleu4, corpus_rouge
from slt.models.unisign_full import CKPT, UniSignFull

ckpt = sys.argv[1] if len(sys.argv) > 1 else CKPT
ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
dev = "cuda"

model = UniSignFull(ckpt_path=ckpt).to(dev).eval()
print("checkpoint:", ckpt.split("/")[-1])
vocab = CharVocab.build(["x"])
ds = RTMPoseSLTDataset(ROOT, CSVD, "dev", vocab, frame_stride=1, max_frames=256)
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)

# ---- 1. pose_proj 输出是否随样本变化（接线 sanity）----
b = next(iter(dl))
f, L = b["feats"].to(dev), b["feat_lens"].to(dev)
with torch.no_grad():
    emb, mask = model.build_inputs(f, L)
P = mask.size(1) - f.size(1)
print("prefix 长度 {} token | pose 段 {} 帧 | inputs_embeds {}".format(P, f.size(1), tuple(emb.shape)))
pose_part = emb[:, P:, :]
valid = mask[:, P:].bool()
means = torch.stack([pose_part[i][valid[i]].mean(0) for i in range(f.size(0))])
pair = torch.cdist(means, means)
print("各样本 pose 段均值向量的两两 L2 距离: 均值 {:.3f}  最小 {:.3f}  （接近 0 = embedding 常数化 = 接线有问题）".format(
    pair[pair > 0].mean().item(), pair[pair > 0].min().item()))
print("pose 段 embedding 幅度: |x| 均值 {:.3f}   prefix 段 |x| 均值 {:.3f}".format(
    pose_part[valid].abs().mean().item(), emb[:, :P, :].abs().mean().item()))

# ---- 2. 真实 / 全零 / 打乱 三种输入 ----
def run(mode):
    hyps, refs = [], []
    for b in dl:
        f, L = b["feats"].to(dev), b["feat_lens"].to(dev)
        if mode == "zero":
            f = torch.zeros_like(f)
        elif mode == "shuffle":
            f = f[torch.randperm(f.size(0), device=dev)]
        hyps += model.generate(f, L, num_beams=1, max_new_tokens=100)
        refs += b["texts"]
    return hyps, refs

out = {}
for mode in ("real", "zero", "shuffle"):
    torch.manual_seed(0)
    h, r = run(mode)
    out[mode] = h
    rg = corpus_rouge(h, r)
    print("  {:<8} BLEU-4 {:.2f}  ROUGE-L {:.2f}  不同句子 {}/{}".format(
        mode, corpus_bleu4(h, r), rg["rouge-l"], len(set(h)), len(h)))
same = sum(a == b for a, b in zip(out["real"], out["zero"]))
print("  真实 vs 全零 预测完全相同: {}/{}".format(same, len(out["real"])))
print("  全零输入的前 3 条:")
for s in out["zero"][:3]:
    print("    ", s[:60])
