#!/bin/bash
# 路线 B 无人值守链（D-024）：
#   E-001b  阶段 1，编码器换成 CSL-Daily checkpoint 的（与 E-002 一致），3 seed，协议同 E-001
#   E-002   阶段 2，冻结 CSL-Daily 编码器 + 其 mT5 + LoRA r=16 + 训 pose_proj，3 seed
#   每个 run 各评一次 test；E-002 同时给 greedy（与 E-001b 同口径）和 beam4
# 已有 best 的 run 跳过，可重入。E-001b seed 1234 可能已由手动启动在跑/跑完。

cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
CK=/root/autodl-tmp/slt/weights/unisign/csl_daily_pose_only_slt.pth
RUNS=/root/autodl-tmp/slt/runs
LOG=/root/autodl-tmp/slt/logs/stage2_chain.log
LR=1e-4

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

wait_manual() {   # 等手动启动的 E-001b s1234 结束（若在跑）
    while pgrep -f "E001b_lr1e-4_s1234" >/dev/null 2>&1; do
        log "等待手动启动的 E-001b s1234 ..."
        sleep 120
    done
}

log "================ 路线 B 链路启动 ================"
wait_manual

# ---------- E-001b ----------
for SEED in 1234 2345 3456; do
    D="$RUNS/E001b_lr${LR}_s${SEED}"
    if [ -f "$D/best.pt" ]; then log "跳过 E-001b s$SEED（已有 best.pt）"; continue; fi
    log ">>> E-001b s$SEED"
    $PY -m slt.train --stage 1 --input rtm --unisign-ckpt "$CK" \
        --train-split train --eval-split dev --epochs 100 --batch-size 16 \
        --lr $LR --seed $SEED --out "$D" >> "$LOG" 2>&1
    [ -f "$D/best.pt" ] || log "!! E-001b s$SEED 没有产出 best.pt，检查日志"
done
for SEED in 1234 2345 3456; do
    D="$RUNS/E001b_lr${LR}_s${SEED}"
    [ -f "$D/eval_test.json" ] && continue
    log ">>> test E-001b s$SEED"
    $PY -m slt.evaluate --ckpt "$D/best.pt" --split test 2>&1 | tee -a "$LOG" | grep -E "n=|chrF"
done

# ---------- E-002 ----------
for SEED in 1234 2345 3456; do
    D="$RUNS/E002_lora16_s${SEED}"
    if [ -f "$D/best/meta.json" ]; then log "跳过 E-002 s$SEED（已有 best）"; continue; fi
    log ">>> E-002 s$SEED（LoRA r=16，label_smoothing 0，batch 8，30 epoch）"
    $PY -m slt.train_stage2 --out "$D" --unisign-ckpt "$CK" --lora-r 16 \
        --lr $LR --epochs 30 --batch-size 8 --label-smoothing 0.0 --seed $SEED >> "$LOG" 2>&1
    [ -f "$D/best/meta.json" ] || log "!! E-002 s$SEED 没有产出 best，检查日志"
done
for SEED in 1234 2345 3456; do
    D="$RUNS/E002_lora16_s${SEED}"
    [ -f "$D/eval_test_greedy.json" ] && continue
    log ">>> test E-002 s$SEED（greedy + beam4）"
    $PY -m slt.train_stage2 --test-only "$D" --out "$D" 2>&1 | tee -a "$LOG" | grep -E "best:|n=|chrF|多样性"
done

# ---------- 汇总 ----------
log "---------------- 汇总 ----------------"
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, statistics as st
R = "/root/autodl-tmp/slt/runs"
def rows(pattern, fname):
    out = []
    for s in (1234, 2345, 3456):
        p = os.path.join(R, pattern % s, fname)
        if os.path.exists(p):
            e = json.load(open(p, encoding="utf-8"))
            out.append((s, e["bleu4"], e["chrf"], e["rouge-l"], e.get("floor", {}).get("bleu4", float("nan"))))
    return out
def show(name, rs):
    print("-" * 70); print(name, "(%d/3 seed)" % len(rs))
    for s, b, c, r, f in rs:
        print("  s{}  BLEU-4 {:.2f}  chrF {:.2f}  R-L {:.2f}  地板B4 {:.2f}".format(s, b, c, r, f))
    if len(rs) >= 2:
        b = [x[1] for x in rs]; c = [x[2] for x in rs]; r = [x[3] for x in rs]
        print("  均值±std  BLEU-4 {:.2f}±{:.2f}  chrF {:.2f}±{:.2f}  R-L {:.2f}±{:.2f}".format(
            st.mean(b), st.pstdev(b), st.mean(c), st.pstdev(c), st.mean(r), st.pstdev(r)))
    return rs
a = show("E-001b 阶段1（CSL-Daily 编码器，greedy）", rows("E001b_lr1e-4_s%d", "eval_test.json"))
b = show("E-002 阶段2 greedy（与 E-001b 同口径）", rows("E002_lora16_s%d", "eval_test_greedy.json"))
c = show("E-002 阶段2 beam4（Uni-Sign 设置）", rows("E002_lora16_s%d", "eval_test_beam4.json"))
if a and b:
    d = st.mean(x[1] for x in b) - st.mean(x[1] for x in a)
    ps = (st.pstdev([x[1] for x in a])**2 + st.pstdev([x[1] for x in b])**2) ** 0.5
    print("=" * 70)
    print("E-001b -> E-002 (greedy)  BLEU-4 差值 {:+.2f}  合成std {:.2f}  比值 {:.1f}".format(d, ps, d / ps if ps else float("nan")))
PY
log "================ STAGE2 CHAIN DONE ================"
