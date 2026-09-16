"""零样本探针：Uni-Sign 完整管线原样装回，不训练，在 CE-CSL dev 上评。

回答两个问题：
  1. "借来的系统"在 CE-CSL 上的起点是多少（决定 E-002 的训练贡献了多少）
  2. beam=4 相对 greedy 值多少（决定 E-002 怎么和 greedy 的 E-001 比）

只跑 dev。test 留给最终行。
"""
import json
import sys
import time

import torch
from torch.utils.data import DataLoader

from slt.data import CharVocab, collate_fn
from slt.data_rtm import RTMPoseSLTDataset
from slt.metrics import evaluate, format_report
from slt.models.unisign_full import CKPT, UniSignFull

ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
CKPT_PATH = sys.argv[2] if len(sys.argv) > 2 else CKPT
ENC_CKPT = sys.argv[3] if len(sys.argv) > 3 else None
TAG = CKPT_PATH.split("/")[-1].replace(".pth", "")
if ENC_CKPT:
    TAG += "__enc_" + ENC_CKPT.split("/")[-1].replace(".pth", "")
assert SPLIT != "test", "探针只在 dev 上跑，test 留给最终行"

dev = "cuda"
t0 = time.time()
model = UniSignFull(ckpt_path=CKPT_PATH, encoder_ckpt=ENC_CKPT).to(dev).eval()
print("checkpoint:", TAG)
print("加载 {:.1f}s | 权重: {}".format(time.time() - t0, model.load_info))
n_par = sum(p.numel() for p in model.parameters())
print("总参数 {:.1f}M".format(n_par / 1e6))

vocab = CharVocab.build(["占位"])                # 只为 dataset 接口，不用其 encode
ds = RTMPoseSLTDataset(ROOT, CSVD, SPLIT, vocab, frame_stride=1, max_frames=256)
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
print("{} {} 条，frame_stride=1，max_frames=256".format(SPLIT, len(ds)))

results = {}
for name, beams in (("greedy", 1), ("beam4", 4)):
    hyps, refs, trs = [], [], []
    t0 = time.time()
    for b in dl:
        f = b["feats"].to(dev)
        L = b["feat_lens"].to(dev)
        hyps += model.generate(f, L, num_beams=beams, max_new_tokens=100)
        refs += b["texts"]
        trs += b["translators"]
    res = evaluate(hyps, refs, trs, with_floor=True)
    res["_meta"] = {"decode": "{}, max_new_tokens=100".format(name),
                    "seconds": round(time.time() - t0, 1)}
    results[name] = res
    print()
    print(format_report(res, "[Uni-Sign 零样本 · {} · {}]  耗时 {:.0f}s".format(
        SPLIT, name, time.time() - t0)))
    print("  输出多样性: {} 种 / {} 条".format(len(set(hyps)), len(hyps)))
    for i in range(3):
        print("    预测: {}\n    参考: {}".format(hyps[i], refs[i]))
    results[name]["_samples"] = list(zip(hyps[:10], refs[:10]))

out = "/root/autodl-tmp/slt/runs/probe_zeroshot_{}_{}.json".format(TAG, SPLIT)
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n已写入", out)
print("\nbeam4 - greedy: BLEU-4 {:+.2f}  chrF {:+.2f}  ROUGE-L {:+.2f}".format(
    results["beam4"]["bleu4"] - results["greedy"]["bleu4"],
    results["beam4"]["chrf"] - results["greedy"]["chrf"],
    results["beam4"]["rouge-l"] - results["greedy"]["rouge-l"]))
