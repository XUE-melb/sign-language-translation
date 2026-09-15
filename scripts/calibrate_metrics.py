"""给指标定标：一句"通顺但完全无关"的中文，本身能拿多少分？

如果随机配对就能拿到接近我们的分数，说明那个指标在这个任务上几乎没有分辨力，
报告时必须说明。这不是推测，直接测。
"""
import csv
import random
import sys

sys.path.insert(0, "/root/autodl-tmp/slt/src")
from slt.metrics import corpus_bleu4, corpus_rouge

P = "/root/autodl-tmp/slt/runs/stage0_lr3e-4_s1234/predictions_test.csv"
rows = list(csv.DictReader(open(P, encoding="utf-8")))
hyps = [r["hyp"] for r in rows]
refs = [r["ref"] for r in rows]

print("=" * 70)
print("对照组设计：预测句子原样不动，只打乱它和参考的配对关系。")
print("这样'中文的流畅度'完全保留，只抹掉'内容是否对应'。")
print("=" * 70)


def report(name, h, r):
    b = corpus_bleu4(h, r)
    g = corpus_rouge(h, r)
    print("  {:<22} BLEU-4 {:>6.2f}   R-1 {:>6.2f}   R-2 {:>6.2f}   R-L {:>6.2f}".format(
        name, b, g["rouge-1"], g["rouge-2"], g["rouge-l"]))
    return b, g


report("① 真实配对（模型）", hyps, refs)

random.seed(0)
bl, r1, r2, rl = [], [], [], []
for i in range(5):
    sh = hyps[:]
    random.shuffle(sh)
    b = corpus_bleu4(sh, refs)
    g = corpus_rouge(sh, refs)
    bl.append(b); r1.append(g["rouge-1"]); r2.append(g["rouge-2"]); rl.append(g["rouge-l"])
print("  {:<22} BLEU-4 {:>6.2f}   R-1 {:>6.2f}   R-2 {:>6.2f}   R-L {:>6.2f}".format(
    "② 打乱配对（5 次均值）", sum(bl)/5, sum(r1)/5, sum(r2)/5, sum(rl)/5))

# ③ 参考句互相打乱：真人写的句子之间的"基础重合度"上限参照
sh = refs[:]
random.seed(1)
random.shuffle(sh)
report("③ 参考句互相打乱", sh, refs)

# ④ 完美预测的上限
report("④ 完美预测（理论上限）", refs, refs)

print("=" * 70)
print("解读：② 是'通顺但无关'的地板分。模型分数若贴近 ②，说明")
print("      该指标下模型没有比'随口说一句中文'好多少。")
