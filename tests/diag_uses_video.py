"""模型到底有没有在用视觉输入？

判据：把关键点换成全零 / 换成别的视频的关键点，如果预测几乎不变，
说明解码器完全靠语言模型在猜，视觉通路等于没接上。
"""
import sys

import torch
from torch.utils.data import DataLoader

from slt.data import CharVocab, PoseSLTDataset, collate_fn
from slt.metrics import corpus_bleu4
from slt.models.stage0 import Seq2SeqLSTM

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/slt/runs/stage0_smoke/best.pt"
ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
dev = "cuda"

ck = torch.load(CKPT, map_location=dev)
a = ck["args"]
vocab = CharVocab.load("/root/autodl-tmp/slt/runs/stage0_smoke/vocab.json")
ds = PoseSLTDataset(ROOT, CSVD, a["eval_split"], vocab,
                    frame_stride=a["frame_stride"], max_frames=a["max_frames"],
                    normalize=a["normalize"])
dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = Seq2SeqLSTM(in_dim=ds.dim, vocab_size=len(vocab)).to(dev)
model.load_state_dict(ck["model"])
model.eval()
print("checkpoint epoch {} (BLEU {:.2f})".format(ck["epoch"], ck["bleu4"]))


@torch.no_grad()
def decode_all(mode):
    hyps, refs = [], []
    for b in dl:
        f = b["feats"].to(dev)
        L = b["feat_lens"].to(dev)
        if mode == "zero":
            f = torch.zeros_like(f)
        elif mode == "shuffle":
            f = f[torch.randperm(f.size(0), device=dev)]   # 与句子错配
        ids = model.greedy_decode(f, L, max_len=60)
        hyps += [vocab.decode(ids[i].tolist()) for i in range(ids.size(0))]
        refs += b["texts"]
    return hyps, refs


res = {}
for mode in ("real", "zero", "shuffle"):
    torch.manual_seed(0)
    h, r = decode_all(mode)
    res[mode] = h
    print("{:<8} BLEU-4 {:.2f}".format(mode, corpus_bleu4(h, r)))

same_zero = sum(x == y for x, y in zip(res["real"], res["zero"]))
same_shuf = sum(x == y for x, y in zip(res["real"], res["shuffle"]))
n = len(res["real"])
print("=" * 62)
print("真实输入 vs 全零输入：预测完全相同的样本 {}/{} ({:.1f}%)".format(
    same_zero, n, 100 * same_zero / n))
print("真实输入 vs 错配输入：预测完全相同的样本 {}/{} ({:.1f}%)".format(
    same_shuf, n, 100 * same_shuf / n))
print("不同预测句子的种类数：真实 {} / 全零 {}".format(
    len(set(res["real"])), len(set(res["zero"]))))
print("=" * 62)
for i in range(5):
    print("[{}]\n  真实 {}\n  全零 {}\n  错配 {}".format(
        i, res["real"][i], res["zero"][i], res["shuffle"][i]))
