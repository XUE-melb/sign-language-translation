#!/usr/bin/env python
"""查官方划分是不是 signer-independent：三个 split 的 translator 是否重叠。

只读 CSV，不依赖视频是否下载完。
"""
import collections
import csv
import os

CSV_DIR = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
SPLITS = ["train", "dev", "test"]

counts, sents = {}, {}
for s in SPLITS:
    rows = list(csv.DictReader(open(os.path.join(CSV_DIR, s + ".csv"), encoding="utf-8")))
    counts[s] = collections.Counter(r["Translator"].strip() for r in rows)
    sents[s] = [r["Chinese Sentences"].strip() for r in rows]

all_t = sorted(set().union(*[set(c) for c in counts.values()]))
print("=" * 64)
print("各 split 的 translator 分布")
print("  {:<4}{:>9}{:>7}{:>7}{:>8}".format("T", "train", "dev", "test", "合计"))
for t in all_t:
    row = [counts[s].get(t, 0) for s in SPLITS]
    print("  {:<4}{:>9}{:>7}{:>7}{:>8}".format(t, *row, sum(row)))
print("  {:<4}{:>9}{:>7}{:>7}{:>8}".format(
    "总", *[sum(counts[s].values()) for s in SPLITS],
    sum(sum(counts[s].values()) for s in SPLITS)))

sets = {s: set(counts[s]) for s in SPLITS}
print("=" * 64)
print("signer 重叠情况")
print("  train 有 {} 人: {}".format(len(sets["train"]), "".join(sorted(sets["train"]))))
print("  dev   有 {} 人: {}".format(len(sets["dev"]), "".join(sorted(sets["dev"]))))
print("  test  有 {} 人: {}".format(len(sets["test"]), "".join(sorted(sets["test"]))))
print()
print("  test 中出现过在 train 里的人: {} / {}".format(
    len(sets["test"] & sets["train"]), len(sets["test"])))
print("  test 独有(train 从未见过)的人: {}".format(
    sorted(sets["test"] - sets["train"]) or "无"))

indep = not (sets["test"] & sets["train"])
print()
print("  结论: 官方划分是 signer-{}".format("INDEPENDENT" if indep else "DEPENDENT"))

# 句子层面是否也重叠（更严重的泄漏）
print("=" * 64)
print("句子重叠（同一中文句子跨 split 出现）")
tr = set(sents["train"])
for s in ["dev", "test"]:
    ov = [x for x in sents[s] if x in tr]
    print("  {}: {}/{} 条的句子在 train 中出现过 ({:.1f}%)".format(
        s, len(ov), len(sents[s]), 100 * len(ov) / len(sents[s])))
    for x in ov[:5]:
        print("     重复例:", x)

print("=" * 64)
print("test 内部句子唯一性: {} 条中有 {} 个不同句子".format(
    len(sents["test"]), len(set(sents["test"]))))
