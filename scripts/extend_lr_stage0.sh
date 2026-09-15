#!/bin/bash
# 阶段 0 的 lr 搜索向下扩展（E-000 修订）
#
# 背景：首轮搜索 3e-4 / 1e-3 / 3e-3，胜出的 3e-4 是范围下边界，趋势指向更小的
# lr，baseline 可能被低估（违反 D-012"不要把 baseline 做弱"的原则）。
#
# ⚠️ 扩展的判断依据只能来自 dev。本脚本在决定是否重跑之前**不碰 test**。
# 只有当更低的 lr 在 dev 上胜出时，才重跑 seed 并重评 test 一次。
# 若 3e-4 仍然最优，则直接退出，E-000 保持原样，test 不再曝光。

cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
RUNS=/root/autodl-tmp/slt/runs
LOG=/root/autodl-tmp/slt/logs/stage0_extend.log
EPOCHS=100
ALL_LR="3e-5 1e-4 3e-4 1e-3 3e-3"

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

train_one() {   # $1=lr $2=seed $3=outdir
    log ">>> 训练 lr=$1 seed=$2"
    $PY -m slt.train --train-split train --eval-split dev \
        --epochs $EPOCHS --batch-size 16 --lr "$1" --seed "$2" \
        --out "$3" >> "$LOG" 2>&1
}

best_dev() {    # $1=rundir -> 最佳 dev BLEU-4
    $PY -c "
import torch
try:
    print('%.4f' % torch.load('$1/best.pt', map_location='cpu')['dev_bleu4'])
except Exception:
    print('-1')" 2>/dev/null
}

log "============ lr 搜索向下扩展开始 ============"

# ---------- 第 1 步：只在 dev 上补跑两个更低的 lr ----------
for LR in 1e-4 3e-5; do
    D="$RUNS/stage0_lr${LR}_s1234"
    [ -f "$D/best.pt" ] && { log "跳过已完成: lr=$LR"; continue; }
    train_one "$LR" 1234 "$D"
    log "<<< lr=$LR 最佳 dev BLEU-4 = $(best_dev "$D")"
done

# ---------- 第 2 步：五个 lr 一起比（仅 dev）----------
log "---------- 完整 lr 曲线（dev，seed 1234）----------"
BEST_LR=""; BEST_V="-1"
for LR in $ALL_LR; do
    V=$(best_dev "$RUNS/stage0_lr${LR}_s1234")
    log "  lr=${LR}  dev 最佳 BLEU-4 = $V"
    if $PY -c "import sys;sys.exit(0 if float('$V')>float('$BEST_V') else 1)"; then
        BEST_V="$V"; BEST_LR="$LR"
    fi
done
log "dev 上最优: lr=$BEST_LR (BLEU-4 $BEST_V)"

OLD_LR=$(cat "$RUNS/stage0_selected_lr.txt" 2>/dev/null)
if [ "$BEST_LR" = "$OLD_LR" ]; then
    log "最优 lr 未变（仍是 $OLD_LR）。不重跑、不重评 test，E-000 保持原样。"
    log "============ EXTEND DONE (无变化) ============"
    exit 0
fi

log "最优 lr 由 $OLD_LR 变为 $BEST_LR —— 需要重跑 seed 并重评 test"
echo "$BEST_LR" > "$RUNS/stage0_selected_lr.txt"

# ---------- 第 3 步：补跑另外两个 seed ----------
for SEED in 2345 3456; do
    D="$RUNS/stage0_lr${BEST_LR}_s${SEED}"
    [ -f "$D/best.pt" ] && { log "跳过已完成: seed=$SEED"; continue; }
    train_one "$BEST_LR" "$SEED" "$D"
done

# ---------- 第 4 步：test 评测（本轮唯一一次）----------
log "---------- test 评测（lr=$BEST_LR，三个 seed）----------"
for SEED in 1234 2345 3456; do
    $PY -m slt.evaluate --ckpt "$RUNS/stage0_lr${BEST_LR}_s${SEED}/best.pt" \
        --split test --tag "final" 2>&1 | tee -a "$LOG"
done

# ---------- 汇总 ----------
$PY - <<PY 2>&1 | tee -a "$LOG"
import json, statistics as st
lr = "$BEST_LR"
rows = []
for s in (1234, 2345, 3456):
    e = json.load(open("$RUNS/stage0_lr%s_s%d/eval_test_final.json" % (lr, s)))
    rows.append((s, e["bleu4"], e["rouge-l"], e["_meta"]["dev_bleu4_at_select"],
                 e["_meta"]["epoch"]))
print("E-000（修订）  阶段 0  lr=%s  epochs=$EPOCHS  batch=16" % lr)
print("  {:<8}{:>12}{:>12}{:>14}{:>8}".format("seed","test BLEU4","test RGE-L","选点dev BLEU","epoch"))
for s, b, r, dv, ep in rows:
    print("  {:<8}{:>12.2f}{:>12.2f}{:>14.2f}{:>8}".format(s, b, r, dv, ep))
bs=[r[1] for r in rows]; rs=[r[2] for r in rows]
print("  {:<8}{:>12}{:>12}".format("均值±std",
      "%.2f±%.2f"%(st.mean(bs), st.pstdev(bs)),
      "%.2f±%.2f"%(st.mean(rs), st.pstdev(rs))))
PY

log "============ EXTEND DONE (lr 已更新为 $BEST_LR) ============"
