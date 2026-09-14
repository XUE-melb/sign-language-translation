#!/bin/bash
# 阶段 0 正式实验编排（E-000）
#
# 协议（见 DECISIONS.md D-017）：
#   1. lr 搜索：3 个学习率各跑一趟，seed 固定 1234，用 **dev** BLEU-4 选最优
#   2. 多 seed：在选定的 lr 上补跑 seed 2345 / 3456
#      —— lr 搜索里胜出的那一趟本身就是 seed 1234，直接复用，不重跑
#   3. test 评测：每个 seed 用各自的 best checkpoint 在 test 上评**一次**
#
# ⚠️ test 只评一次。看了 test 再回头调参，test 就废了。
#
# 用法：tmux new-window -n stage0; bash scripts/run_stage0.sh

cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
RUNS=/root/autodl-tmp/slt/runs
LOG=/root/autodl-tmp/slt/logs/stage0.log
EPOCHS=100

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

train_one() {   # $1=lr  $2=seed  $3=outdir
    log ">>> 训练 lr=$1 seed=$2 -> $3"
    $PY -m slt.train --train-split train --eval-split dev \
        --epochs $EPOCHS --batch-size 16 --lr "$1" --seed "$2" \
        --out "$3" >> "$LOG" 2>&1
    local b
    b=$($PY -c "import json;print(json.load(open('$3/history.json'))['final']['bleu4'])" 2>/dev/null)
    log "<<< 完成 lr=$1 seed=$2  末轮 dev BLEU-4=$b"
}

best_dev() {    # 打印某个 run 的最佳 dev BLEU-4（来自 best.pt）
    $PY -c "
import torch,sys
print('%.4f' % torch.load('$1/best.pt', map_location='cpu')['dev_bleu4'])" 2>/dev/null
}

log "================ 阶段 0 正式实验开始 ================"
log "协议: train 训练 / dev 选 checkpoint / test 只评一次; epochs=$EPOCHS batch=16"

# ---------- 第 1 步：lr 搜索（seed 固定 1234）----------
log "---------- 第 1 步：lr 搜索 ----------"
for LR in 3e-4 1e-3 3e-3; do
    D="$RUNS/stage0_lr${LR}_s1234"
    [ -f "$D/best.pt" ] && { log "跳过已完成: $D"; continue; }
    train_one "$LR" 1234 "$D"
done

BEST_LR=""; BEST_V="-1"
for LR in 3e-4 1e-3 3e-3; do
    V=$(best_dev "$RUNS/stage0_lr${LR}_s1234")
    log "lr=$LR  最佳 dev BLEU-4 = $V"
    if $PY -c "import sys;sys.exit(0 if float('$V')>float('$BEST_V') else 1)"; then
        BEST_V="$V"; BEST_LR="$LR"
    fi
done
log "选定 lr=$BEST_LR（dev BLEU-4 $BEST_V）。四个阶段今后统一用这个 lr。"
echo "$BEST_LR" > "$RUNS/stage0_selected_lr.txt"

# ---------- 第 2 步：补跑另外两个 seed ----------
log "---------- 第 2 步：多 seed（seed 1234 复用 lr 搜索的结果）----------"
for SEED in 2345 3456; do
    D="$RUNS/stage0_lr${BEST_LR}_s${SEED}"
    [ -f "$D/best.pt" ] && { log "跳过已完成: $D"; continue; }
    train_one "$BEST_LR" "$SEED" "$D"
done

# ---------- 第 3 步：test 评测（每个 seed 一次）----------
log "---------- 第 3 步：test 评测 ----------"
for SEED in 1234 2345 3456; do
    D="$RUNS/stage0_lr${BEST_LR}_s${SEED}"
    log ">>> test 评测 seed=$SEED"
    $PY -m slt.evaluate --ckpt "$D/best.pt" --split test 2>&1 | tee -a "$LOG"
done

# ---------- 汇总 ----------
log "---------- 汇总 ----------"
$PY - <<PY 2>&1 | tee -a "$LOG"
import json, statistics as st
lr = open("$RUNS/stage0_selected_lr.txt").read().strip()
rows = []
for s in (1234, 2345, 3456):
    d = "$RUNS/stage0_lr%s_s%d" % (lr, s)
    e = json.load(open(d + "/eval_test.json"))
    rows.append((s, e["bleu4"], e["rouge-l"], e["_meta"]["dev_bleu4_at_select"],
                 e["_meta"]["epoch"]))
print("E-000  阶段 0  lr=%s  epochs=$EPOCHS  batch=16" % lr)
print("  {:<8}{:>10}{:>10}{:>12}{:>8}".format("seed","test BLEU4","test RGE-L","选点dev BLEU","epoch"))
for s, b, r, dv, ep in rows:
    print("  {:<8}{:>10.2f}{:>10.2f}{:>12.2f}{:>8}".format(s, b, r, dv, ep))
bs = [r[1] for r in rows]; rs = [r[2] for r in rows]
print("  {:<8}{:>10}{:>10}".format("均值±std",
      "%.2f±%.2f" % (st.mean(bs), st.pstdev(bs)),
      "%.2f±%.2f" % (st.mean(rs), st.pstdev(rs))))
print()
print("协议: signer-dependent，CE-CSL 官方划分（test 中训练未见过的手语者 0/12）")
print("BLEU: sacrebleu tokenize=zh 字级 | ROUGE: rouge-chinese 字级")
PY

log "================ STAGE0 DONE ================"
log "下一步：把汇总结果写入 docs/EXPERIMENTS.md 的 E-000 行"
