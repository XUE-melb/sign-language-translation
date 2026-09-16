#!/bin/bash
# 等 STAGE2 CHAIN DONE，然后自动：配对 bootstrap + 汇总 + 生成可粘贴的 EXPERIMENTS 段落。
# 产物：logs/post_stage2.log 与 runs/stage2_summary.md
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/slt/logs/post_stage2.log
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

log "等待 STAGE2 CHAIN DONE ..."
while ! grep -q "STAGE2 CHAIN DONE" logs/stage2_chain.log 2>/dev/null; do sleep 300; done
log "链路已完成，开始后处理"

# E-002 的预测文件名带解码方式后缀，bootstrap 脚本按 predictions_test.csv 读 -> 建软链
for S in 1234 2345 3456; do
    D=runs/E002_lora16_s$S
    [ -f "$D/predictions_test_greedy.csv" ] && ln -sf predictions_test_greedy.csv "$D/predictions_test.csv"
done

log "---- 配对 bootstrap: E-000b -> E-001b ----"
for S in 1234 2345 3456; do
    $PY scripts/paired_bootstrap.py runs/E000b_lr1e-4_s$S runs/E001b_lr1e-4_s$S 1000 2>&1 | grep -vE "arn" | tee -a "$LOG"
done
log "---- 配对 bootstrap: E-001b -> E-002 (greedy) ----"
for S in 1234 2345 3456; do
    $PY scripts/paired_bootstrap.py runs/E001b_lr1e-4_s$S runs/E002_lora16_s$S 1000 2>&1 | grep -vE "arn" | tee -a "$LOG"
done

log "---- 生成 EXPERIMENTS 段落 ----"
$PY - <<'PY' 2>&1 | tee runs/stage2_summary.md | tee -a "$LOG"
import json, os, statistics as st
R = "/root/autodl-tmp/slt/runs"
def load(pat, fn):
    out = []
    for s in (1234, 2345, 3456):
        p = os.path.join(R, pat % s, fn)
        if os.path.exists(p):
            out.append(json.load(open(p, encoding="utf-8")))
    return out
def ms(rs, k):
    v = [r[k] for r in rs]; return st.mean(v), st.pstdev(v)
def row(tag, stage, desc, rs, note):
    if not rs: return "| %s | %s | %s | 未完成 | | | |" % (tag, stage, desc)
    b, bs = ms(rs, "bleu4"); c, _ = ms(rs, "chrf"); r, rsd = ms(rs, "rouge-l")
    r1, _ = ms(rs, "rouge-1"); r2, _ = ms(rs, "rouge-2")
    fl = st.mean(x.get("floor", {}).get("bleu4", float("nan")) for x in rs)
    return ("| **%s** | %s | %s | **%.2f ± %.2f** | ROUGE-L **%.2f ± %.2f**（R-1 %.1f / R-2 %.1f）；chrF %.2f | | %s；地板 B4 %.2f |"
            % (tag, stage, desc, b, bs, r, rsd, r1, r2, c, note, fl))
e1b = load("E001b_lr1e-4_s%d", "eval_test.json")
e2g = load("E002_lora16_s%d", "eval_test_greedy.json")
e2b = load("E002_lora16_s%d", "eval_test_beam4.json")
print("### 可粘进 EXPERIMENTS.md 主表的行\n")
print(row("E-001b", 1, "只换视觉编码器：冻结 **CSL-Daily 版** Uni-Sign 编码器（与 E-002 一致）；解码器同 E-000b", e1b,
          "3 seed；lr=1e-4 epochs=100 batch=16；与 E-001 差异 = 编码器 checkpoint（stage-1 vs CSL-Daily）"))
print(row("E-002", 2, "只换解码器：LSTM → CSL-Daily 版 mT5（主干冻结 + LoRA r=16）+ 可训练 pose_proj；视觉端同 E-001b；**greedy**", e2g,
          "3 seed；epochs=30 batch=8 label_smoothing=0（D-024 偏离已注明）；零样本起点 dev 3.09"))
print(row("E-002 (beam4)", 2, "同上，beam=4（Uni-Sign 设置），非主口径", e2b, "仅参照"))
print()
for name, rs in (("E-001b", e1b), ("E-002 greedy", e2g), ("E-002 beam4", e2b)):
    print("### %s 各 seed（test）" % name)
    for r in rs:
        m = r["_meta"]
        print("  seed %s  BLEU-1/2/3/4 %.2f/%.2f/%.2f/%.2f  chrF %.2f  R-L %.2f  地板B4/RL %.2f/%.2f  epoch %s" % (
            m.get("seed"), r["bleu1"], r["bleu2"], r["bleu3"], r["bleu4"], r["chrf"], r["rouge-l"],
            r.get("floor", {}).get("bleu4", float("nan")), r.get("floor", {}).get("rouge-l", float("nan")), m.get("epoch")))
if e1b and e2g:
    a, _ = ms(e1b, "bleu4"); b, _ = ms(e2g, "bleu4")
    print("\nE-001b -> E-002 (greedy)  BLEU-4 %+.2f   |  零样本 3.09(dev) -> E-002 test %.2f（适配训练贡献，注意 dev/test 口径不同）" % (b - a, b))
# 训练曲线检查：30 轮时 dev 是否仍在涨
print("\n### E-002 dev 曲线末段（判断 30 epoch 够不够）")
for s in (1234, 2345, 3456):
    p = os.path.join(R, "E002_lora16_s%d" % s, "history.json")
    if os.path.exists(p):
        h = json.load(open(p, encoding="utf-8"))["history"]
        pk = max(h, key=lambda x: x["bleu4"])
        tail = h[-5:]
        print("  seed %d  峰值 ep%d %.2f | 末 5 轮 dev: %s" % (
            s, pk["epoch"], pk["bleu4"], " ".join("%.2f" % x["bleu4"] for x in tail)))
PY
log "================ POST DONE ================"
