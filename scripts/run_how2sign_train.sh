#!/bin/bash
# How2Sign（ASL → 英文）第一轮训练链（D-029）。等数据准备链第二次打出 HOW2SIGN PREP DONE 后自动起跑。
#  E-101  冻结 Uni-Sign 编码器（CSL-Daily 版）+ 原版 mT5-base（LoRA r=16）+ pose_proj，30 轮   ← 与第一章 E-002 同构
#  E-102  同上但解冻编码器（lr 1e-4），30 轮                                              ← 与第一章 E-003 同构
#  各自 test（greedy + beam4）→ 汇总。日志 logs/how2sign_train.log，结束打 HOW2SIGN TRAIN DONE
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/slt/logs/how2sign_train.log
H2S=/root/autodl-tmp/How2Sign
COMMON="--lang en --hf-mt5 --root $H2S --csv-dir $H2S/csv --unisign-ckpt weights/unisign/csl_daily_pose_only_slt.pth --lora-r 16 --lr 1e-4 --epochs 30 --batch-size 8 --label-smoothing 0.0"
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

log "等待数据准备链完成（第二次 PREP DONE）..."
while [ "$(grep -c 'HOW2SIGN PREP DONE' logs/how2sign_prep.log 2>/dev/null)" -lt 2 ]; do sleep 300; done
log "数据就绪：train $(ls $H2S/pose_rtm/train/H2S | wc -l) / dev $(ls $H2S/pose_rtm/dev/H2S | wc -l) / test $(ls $H2S/pose_rtm/test/H2S | wc -l) 个 pkl"

train_run() {
    local D=$1; shift
    if [ -f "$D/best/meta.json" ] && [ -f "$D/history.json" ]; then log "跳过 $D（已有）"; return; fi
    log ">>> train $D  $*"
    $PY -m slt.train_stage2 --out "$D" $COMMON "$@" >> "$LOG" 2>&1
    [ -f "$D/best/meta.json" ] || log "!! $D 没有产出 best，检查日志"
}
test_run() {
    local D=$1; shift
    [ -f "$D/eval_test_greedy.json" ] && { log "跳过 test $D（已有）"; return; }
    [ -f "$D/best/meta.json" ] || { log "!! $D 无 best，跳过 test"; return; }
    log ">>> test $D"
    $PY -m slt.train_stage2 --test-only "$D" 2>&1 | tee -a "$LOG" | grep -E "best:|n=|chrF"
    ln -sf predictions_test_greedy.csv "$D/predictions_test.csv"
}

train_run runs/E101_h2s_frozen_s1234 --seed 1234
test_run  runs/E101_h2s_frozen_s1234
train_run runs/E102_h2s_unfrozen_s1234 --unfreeze-encoder --encoder-lr 1e-4 --seed 1234
test_run  runs/E102_h2s_unfrozen_s1234

log "---------------- 汇总 ----------------"
$PY - 2>&1 <<'PY' | tee -a "$LOG"
import json, os
R = "/root/autodl-tmp/slt/runs"
for name, d in (("E-101 冻结编码器 + 原版 mT5", "E101_h2s_frozen_s1234"), ("E-102 解冻编码器", "E102_h2s_unfrozen_s1234")):
    g = os.path.join(R, d, "eval_test_greedy.json"); b = os.path.join(R, d, "eval_test_beam4.json"); h = os.path.join(R, d, "history.json")
    if os.path.exists(g):
        G = json.load(open(g, encoding="utf-8")); B = json.load(open(b, encoding="utf-8")); H = json.load(open(h, encoding="utf-8"))["history"]
        pk = max(H, key=lambda x: x["bleu4"])
        print("  %-28s test greedy BLEU-4 %.2f (B1-3 %.1f/%.1f/%.1f chrF %.2f R-L %.2f) | beam4 %.2f | dev 峰值 ep%d %.2f | 地板 %.2f" % (
            name, G["bleu4"], G["bleu1"], G["bleu2"], G["bleu3"], G["chrf"], G["rouge-l"], B["bleu4"], pk["epoch"], pk["bleu4"], G.get("floor", {}).get("bleu4", float("nan"))))
    else:
        print("  %-28s 未完成" % name)
PY
log "================ HOW2SIGN TRAIN DONE ================"
