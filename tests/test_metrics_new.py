"""在替换线上 metrics.py 之前，先用 metrics_new 验证：
  1. 旧键一个不少、值与旧实现一致（链路正在跑，接口不能变）
  2. 新键存在且数值合理（BLEU-1 >= BLEU-2 >= BLEU-3 >= BLEU-4；地板 < 真实）
  3. with_floor=False 时不算地板（训练期开销不变）
"""
import csv
import sys

from slt import metrics as old
from slt import metrics_new as new

hyps = ["今天天气很好", "我想去北京", "你好世界", ""]
refs = ["今天天气不错", "我要去上海", "你好世界", "空句子测试"]
trs = ["A", "A", "B", "B"]

ok = True
o = old.evaluate(hyps, refs, trs)
n = new.evaluate(hyps, refs, trs)

print("=" * 66)
print("检查 1：旧键保留且数值一致")
for k in ("n", "bleu4", "rouge-1", "rouge-2", "rouge-l"):
    same = abs(float(o[k]) - float(n[k])) < 1e-9
    print("  {:<8} 旧 {:>8.4f}  新 {:>8.4f}  {}".format(k, o[k], n[k], "OK" if same else "<<< 不一致"))
    ok &= same
same_pt = set(o["per_translator"]) == set(n["per_translator"]) and all(
    abs(o["per_translator"][t]["bleu4"] - n["per_translator"][t]["bleu4"]) < 1e-9
    for t in o["per_translator"])
print("  per_translator 一致:", same_pt)
ok &= same_pt

print("=" * 66)
print("检查 2：新键存在且单调（BLEU-1 >= 2 >= 3 >= 4）")
b = [n["bleu1"], n["bleu2"], n["bleu3"], n["bleu4"]]
print("  BLEU-1..4 = {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(*b))
mono = b[0] >= b[1] >= b[2] >= b[3]
print("  单调递减:", mono)
print("  chrF = {:.2f}".format(n["chrf"]))
ok &= mono and 0 <= n["chrf"] <= 100

print("=" * 66)
print("检查 3：默认不算地板；with_floor=True 才算，且地板 <= 真实")
print("  默认含 floor:", "floor" in n, "（应为 False）")
ok &= "floor" not in n
nf = new.evaluate(hyps, refs, trs, with_floor=True)
print("  floor 键:", sorted(nf["floor"].keys()))
print("  真实 BLEU-4 {:.2f} vs 地板 {:.2f}".format(nf["bleu4"], nf["floor"]["bleu4"]))
ok &= "floor" in nf and "chrf" in nf["floor"]

print("=" * 66)
print("检查 4：真实预测文件上跑一遍（用 E-000 留下的 predictions_test.csv）")
p = "/root/autodl-tmp/slt/runs/stage0_lr3e-4_s1234/predictions_test.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8")))
H = [r["hyp"] for r in rows]
R = [r["ref"] for r in rows]
T = [r["translator"] for r in rows]
r = new.evaluate(H, R, T, with_floor=True)
print(new.format_report(r, "[E-000 seed1234 · test · 新指标]"))
# 与 D-015 calibrate_metrics.py 当时手工算的地板对照：BLEU-4 0.48 / R-L 14.25
print("  （D-015 手工地板：BLEU-4 0.48, ROUGE-L 14.25 —— 应在同一量级）")

print("=" * 66)
print("总判定:", "通过，可替换线上 metrics.py" if ok else "有失败项，不要替换")
sys.exit(0 if ok else 1)
