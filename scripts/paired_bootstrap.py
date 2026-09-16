"""配对 bootstrap 显著性检验（机器翻译标准做法，Koehn 2004）。

回答的问题和 seed 标准差不同：
  seed 标准差 —— 换个随机种子，这个架构的数字晃多少
  配对 bootstrap —— 在**同一批** test 句子上重采样，系统 B 比 A 高的差距
                     换一批测试句子还在不在

两者都要报，互相不能替代。3 个 seed 不足以做统计检验，但 500 条 test 句子
足以做 bootstrap。

做法：把 500 条 test 有放回重采样 B 次，每次对两个系统算同一子集上的 corpus
分数，统计"B 不优于 A"的比例作为 p 值。两个系统用同一子集，所以是配对的。

用法：
  python paired_bootstrap.py runs/E000b_lr1e-4_s1234 runs/E001_lr1e-4_s1234
"""
import csv
import os
import sys

import numpy as np
import sacrebleu


def load(run_dir):
    p = os.path.join(run_dir, "predictions_test.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return ([r["hyp"] for r in rows], [r["ref"] for r in rows],
            [r["number"] for r in rows])


def bleu4(h, r):
    return sacrebleu.corpus_bleu(h, [r], tokenize="zh").score


def chrf(h, r):
    return sacrebleu.corpus_chrf(h, [r]).score


def paired(hA, hB, refs, metric, B=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(refs)
    full = metric(hB, refs) - metric(hA, refs)
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.randint(0, n, n)
        rs = [refs[j] for j in idx]
        deltas[i] = (metric([hB[j] for j in idx], rs)
                     - metric([hA[j] for j in idx], rs))
    p = float((deltas <= 0).mean())          # B 不优于 A 的比例
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return full, p, lo, hi


def main():
    a, b = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    hA, rA, nA = load(a)
    hB, rB, nB = load(b)
    assert nA == nB and rA == rB, "两个 run 的 test 句子顺序/内容不一致，无法配对"
    print("A = {}\nB = {}\nn = {} 句，bootstrap {} 次".format(
        os.path.basename(a), os.path.basename(b), len(rA), B))
    for name, m in (("BLEU-4", bleu4), ("chrF", chrf)):
        full, p, lo, hi = paired(hA, hB, rA, m, B)
        print("  {:<7} B-A = {:+.2f}   95% CI [{:+.2f}, {:+.2f}]   p(B<=A) = {:.4f}   {}".format(
            name, full, lo, hi, p,
            "显著" if p < 0.05 else "不显著"))


if __name__ == "__main__":
    main()
