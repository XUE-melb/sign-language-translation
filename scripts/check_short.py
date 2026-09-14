"""查极短视频：帧数过少的样本，配的句子是不是也短？不匹配就是坏样本。"""
import csv
import os

CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
POSE = "/root/autodl-tmp/slt/CE-CSL/pose"

for split in ("train", "dev", "test"):
    man = list(csv.DictReader(
        open(os.path.join(POSE, "manifest_{}.csv".format(split)), encoding="utf-8")))
    ann = {r["Number"].strip(): r["Chinese Sentences"].strip()
           for r in csv.DictReader(
               open(os.path.join(CSVD, split + ".csv"), encoding="utf-8"))}

    rows = []
    for r in man:
        if r["status"] not in ("ok", "cached"):
            continue
        f = int(r["frames"])
        t = ann.get(r["number"], "")
        rows.append((f, len(t), r["number"], r["translator"], t))

    rows.sort()
    n = len(rows)
    print("=" * 74)
    print("[{}] {} 条".format(split, n))
    for thr in (40, 60, 90):
        c = sum(1 for f, *_ in rows if f < thr)
        print("  帧数 < {:<4} {:>4} 条 ({:.2f}%)".format(thr, c, 100 * c / n))

    # 每字对应多少帧：正常打一个字总要若干帧，比值过低说明视频配不上句子
    print("  最短的 8 条：")
    print("    {:<6}{:<14}{:>6}{:>6}{:>8}  {}".format(
        "T", "number", "帧", "字数", "帧/字", "句子"))
    for f, L, num, t, txt in rows[:8]:
        print("    {:<6}{:<14}{:>6}{:>6}{:>8.1f}  {}".format(
            t, num, f, L, f / max(L, 1), txt[:28]))

    ratio = sorted(f / max(L, 1) for f, L, *_ in rows)
    print("  帧/字 分布: min {:.1f}  p01 {:.1f}  中位 {:.1f}  max {:.1f}".format(
        ratio[0], ratio[max(0, n // 100)], ratio[n // 2], ratio[-1]))
