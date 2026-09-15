"""汇总 E-000b 与 E-001，直接输出可粘进 EXPERIMENTS.md 的表格。"""
import json
import os
import statistics as st
import sys

LR = sys.argv[1] if len(sys.argv) > 1 else "3e-4"
RUNS = "/root/autodl-tmp/slt/runs"
SEEDS = (1234, 2345, 3456)


def collect(tag):
    rows = []
    for s in SEEDS:
        p = os.path.join(RUNS, "%s_lr%s_s%d" % (tag, LR, s), "eval_test.json")
        if not os.path.exists(p):
            continue
        e = json.load(open(p, encoding="utf-8"))
        rows.append(dict(seed=s, bleu4=e["bleu4"], rougeL=e["rouge-l"],
                         r1=e["rouge-1"], r2=e["rouge-2"],
                         dev=e["_meta"]["dev_bleu4_at_select"],
                         epoch=e["_meta"]["epoch"], per=e.get("per_translator", {})))
    return rows


def agg(rows, key):
    v = [r[key] for r in rows]
    return (st.mean(v), st.pstdev(v)) if v else (float("nan"), float("nan"))


print("=" * 78)
print("阶段 0 / 阶段 1 汇总   lr=%s  epochs=100  batch=16  RTMPose 输入" % LR)
print("协议: signer-dependent 官方划分 | dev 选 checkpoint | test 只评一次")
print("BLEU: sacrebleu tokenize=zh 字级 | ROUGE: rouge-chinese 字级")

data = {}
for tag, label in (("E000b", "E-000b 阶段0"), ("E001", "E-001 阶段1")):
    rows = collect(tag)
    data[tag] = rows
    print("-" * 78)
    print("%s  （%d/3 个 seed 完成）" % (label, len(rows)))
    if not rows:
        print("  尚无结果")
        continue
    print("  {:<8}{:>12}{:>12}{:>14}{:>8}".format(
        "seed", "test BLEU4", "test R-L", "选点dev BLEU", "epoch"))
    for r in rows:
        print("  {:<8}{:>12.2f}{:>12.2f}{:>14.2f}{:>8}".format(
            r["seed"], r["bleu4"], r["rougeL"], r["dev"], r["epoch"]))
    mb, sb = agg(rows, "bleu4")
    ml, sl = agg(rows, "rougeL")
    print("  {:<8}{:>12}{:>12}".format(
        "均值±std", "%.2f±%.2f" % (mb, sb), "%.2f±%.2f" % (ml, sl)))

print("=" * 78)
a, b = data.get("E000b", []), data.get("E001", [])
if a and b:
    mb0, sb0 = agg(a, "bleu4")
    mb1, sb1 = agg(b, "bleu4")
    ml0, _ = agg(a, "rougeL")
    ml1, _ = agg(b, "rougeL")
    d = mb1 - mb0
    pooled = (sb0 ** 2 + sb1 ** 2) ** 0.5
    print("阶段 0 -> 阶段 1（只换视觉编码器）")
    print("  BLEU-4  %.2f -> %.2f   差值 %+.2f" % (mb0, mb1, d))
    print("  ROUGE-L %.2f -> %.2f   差值 %+.2f" % (ml0, ml1, ml1 - ml0))
    print("  两组标准差合成 %.2f；差值/合成std = %.2f" % (
        pooled, d / pooled if pooled else float("nan")))
    if abs(d) < pooled:
        print("  ⚠️ 差值小于合成标准差 —— 不能声称显著提升，只能报「未见显著差异」")
    else:
        print("  差值超过合成标准差，可以谈方向性，但 3 个 seed 仍不足以做统计检验")
    print()
    print("可粘进 EXPERIMENTS.md 的行：")
    print("| E-000b | 0 | 老架构 baseline（RTMPose 69 点输入） | %.2f ± %.2f | R-L %.2f | | 3 seed，lr=%s |"
          % (mb0, sb0, ml0, LR))
    print("| E-001 | 1 | 只换视觉编码器：冻结 Uni-Sign pose 编码器 | %.2f ± %.2f | R-L %.2f | | 解码器与 E-000b 逐张量相同 |"
          % (mb1, sb1, ml1))
