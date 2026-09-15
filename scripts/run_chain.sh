#!/bin/bash
# 无人值守链：等提取 -> 质检 -> lr 复查 -> E-000b -> E-001 -> 汇总
#
# 协议见 DECISIONS.md D-017（三 seed、四阶段共用 lr、test 只评一次）
# 与 D-021（阶段 0/1 共用同一个解码器）。
#
# lr 切换判据：**必须超过 seed 标准差**才换，不是"谁大选谁"。
# 这是 D-018 的教训 —— 曾差点为 0.06 的噪声差异多消耗一次 test 曝光。

cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
RUNS=/root/autodl-tmp/slt/runs
LOG=/root/autodl-tmp/slt/logs/chain.log
EPOCHS=100
SEED_STD=0.1096          # D-018 实测：lr=3e-4 三个 seed 的 dev 选点值标准差
LRS="1e-4 3e-4 1e-3"

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

best_dev() {   # $1=rundir
    $PY - "$1" <<'PY' 2>/dev/null
import sys, torch
try:
    print("%.4f" % torch.load(sys.argv[1] + "/best.pt", map_location="cpu")["dev_bleu4"])
except Exception:
    print("-1")
PY
}

train_one() {  # $1=stage $2=lr $3=seed $4=outdir
    [ -f "$4/best.pt" ] && { log "跳过已完成 $4"; return; }
    log ">>> 训练 stage=$1 lr=$2 seed=$3"
    $PY -m slt.train --stage "$1" --input rtm \
        --train-split train --eval-split dev \
        --epochs $EPOCHS --batch-size 16 --lr "$2" --seed "$3" \
        --out "$4" >> "$LOG" 2>&1
    log "<<< 完成，最佳 dev BLEU-4 = $(best_dev "$4")"
}

# ---------------- 0. 等提取 ----------------
log "================ 链路启动 ================"
while ! grep -q "RTMPOSE DONE" logs/rtmpose_extract.log 2>/dev/null; do
    n=$(find CE-CSL/pose_rtm -name "*.pkl" 2>/dev/null | wc -l)
    log "等待 RTMPose 提取... $n/5988"
    sleep 300
done
log "RTMPose 提取已完成"

# ---------------- 1. 质检 ----------------
log "---------------- 质检 ----------------"
$PY scripts/qc_rtmpose.py 2>&1 | tee -a "$LOG"

# ---------------- 2. lr 复查（只看 dev） ----------------
log "---------------- lr 复查（dev-only，seed 1234） ----------------"
for LR in $LRS; do
    train_one 0 "$LR" 1234 "$RUNS/s0b_lr${LR}_s1234"
done

BEST_LR=""; BEST_V="-1"
for LR in $LRS; do
    V=$(best_dev "$RUNS/s0b_lr${LR}_s1234")
    log "  lr=$LR  dev 最佳 BLEU-4 = $V"
    if $PY -c "import sys;sys.exit(0 if float('$V')>float('$BEST_V') else 1)"; then
        BEST_V="$V"; BEST_LR="$LR"
    fi
done

# 只有超过 seed 标准差才真正切换，否则沿用 D-017 定的 3e-4
V34=$(best_dev "$RUNS/s0b_lr3e-4_s1234")
GAP=$($PY -c "print('%.4f' % (float('$BEST_V') - float('$V34')))")
log "  最高 lr=$BEST_LR ($BEST_V)  vs  原定 3e-4 ($V34)  差距 $GAP  阈值 $SEED_STD"
if $PY -c "import sys;sys.exit(0 if float('$GAP') > float('$SEED_STD') else 1)"; then
    LR="$BEST_LR"
    log "  差距超过 seed 标准差 -> 切换到 lr=$LR（四阶段统一）"
else
    LR="3e-4"
    log "  差距未超过 seed 标准差 -> 判定等价，沿用 lr=3e-4"
fi
echo "$LR" > "$RUNS/chain_selected_lr.txt"

# ---------------- 3. E-000b（阶段 0，三 seed） ----------------
log "---------------- E-000b 阶段 0（lr=$LR，三 seed） ----------------"
for SEED in 1234 2345 3456; do
    D="$RUNS/E000b_lr${LR}_s${SEED}"
    # lr 复查里 seed 1234 已跑过同样配置，直接复用不重跑
    if [ "$SEED" = "1234" ] && [ -f "$RUNS/s0b_lr${LR}_s1234/best.pt" ]; then
        [ -d "$D" ] || cp -r "$RUNS/s0b_lr${LR}_s1234" "$D"
        log "seed 1234 复用 lr 复查的结果"
        continue
    fi
    train_one 0 "$LR" "$SEED" "$D"
done

# ---------------- 4. E-001（阶段 1，三 seed） ----------------
log "---------------- E-001 阶段 1（冻结 Uni-Sign，lr=$LR，三 seed） ----------------"
for SEED in 1234 2345 3456; do
    train_one 1 "$LR" "$SEED" "$RUNS/E001_lr${LR}_s${SEED}"
done

# ---------------- 5. test 评测（每个 run 各一次） ----------------
log "---------------- test 评测 ----------------"
for TAG in E000b E001; do
    for SEED in 1234 2345 3456; do
        D="$RUNS/${TAG}_lr${LR}_s${SEED}"
        [ -f "$D/eval_test.json" ] && { log "跳过已评 $D"; continue; }
        log ">>> test: $TAG seed=$SEED"
        $PY -m slt.evaluate --ckpt "$D/best.pt" --split test 2>&1 | tee -a "$LOG"
    done
done

# ---------------- 6. 汇总 ----------------
log "---------------- 汇总 ----------------"
$PY scripts/summarize_chain.py "$LR" 2>&1 | tee -a "$LOG"
log "================ CHAIN DONE ================"
