#!/bin/bash
# How2Sign 第二轮（D-029 §七）：E-102 解冻编码器已离开地板（dev 2.61 仍在涨），下一档容量：
#  E-103  解冻编码器（lr 1e-4）+ 全量微调 mT5（lr 1e-4，不用 LoRA）+ bf16，60 轮
# 先在 dev 上跑 1 轮冒烟看显存，过了再正式跑。等 HOW2SIGN TRAIN DONE 后起跑。日志 logs/how2sign_train2.log
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/slt/logs/how2sign_train2.log
H2S=/root/autodl-tmp/How2Sign
COMMON="--lang en --hf-mt5 --root $H2S --csv-dir $H2S/csv --unisign-ckpt weights/unisign/csl_daily_pose_only_slt.pth --lr 1e-4 --batch-size 8 --label-smoothing 0.0 --unfreeze-encoder --encoder-lr 1e-4 --train-mt5 --mt5-lr 1e-4 --bf16"
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

log "等待 HOW2SIGN TRAIN DONE ..."
while ! grep -q "HOW2SIGN TRAIN DONE" logs/how2sign_train.log 2>/dev/null; do sleep 120; done

log "显存冒烟：dev 当 train，1 轮，全量微调 + bf16"
rm -rf runs/smoke_e103
$PY -m slt.train_stage2 --out runs/smoke_e103 $COMMON --epochs 1 --train-split dev --eval-split dev --smoke --seed 1234 2>&1 | grep -E "^epoch|可训练|OutOfMemory|Error|Traceback" | tee -a "$LOG"
if [ ! -f runs/smoke_e103/best/meta.json ]; then
    log "!! 冒烟失败（多半是显存），退到 LoRA r=64 方案"
    COMMON="--lang en --hf-mt5 --root $H2S --csv-dir $H2S/csv --unisign-ckpt weights/unisign/csl_daily_pose_only_slt.pth --lr 1e-4 --batch-size 8 --label-smoothing 0.0 --unfreeze-encoder --encoder-lr 1e-4 --lora-r 64 --lora-alpha 128 --bf16"
    NAME=E103b_h2s_unfrozen_lora64_s1234
else
    NAME=E103_h2s_unfrozen_fullmt5_s1234
fi
rm -rf runs/smoke_e103
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"

D=runs/$NAME
if [ ! -f "$D/best/meta.json" ]; then
    log ">>> train $D（60 轮）"
    $PY -m slt.train_stage2 --out "$D" $COMMON --epochs 60 --seed 1234 >> "$LOG" 2>&1
fi
if [ -f "$D/best/meta.json" ] && [ ! -f "$D/eval_test_greedy.json" ]; then
    log ">>> test $D"
    $PY -m slt.train_stage2 --test-only "$D" 2>&1 | tee -a "$LOG" | grep -E "best:|n=|chrF"
fi
$PY - "$D" 2>&1 <<'PY' | tee -a "$LOG"
import json, os, sys
d = sys.argv[1]; R = "/root/autodl-tmp/slt/runs"
for name, dd in (("E-101 冻结+LoRA", "E101_h2s_frozen_s1234"), ("E-102 解冻+LoRA", "E102_h2s_unfrozen_s1234"), ("E-103 " + os.path.basename(d), d)):
    g = os.path.join(R, dd, "eval_test_greedy.json"); h = os.path.join(R, dd, "history.json")
    if os.path.exists(g):
        G = json.load(open(g, encoding="utf-8")); H = json.load(open(h, encoding="utf-8"))["history"]; pk = max(H, key=lambda x: x["bleu4"])
        print("  %-28s test greedy %.2f  B1 %.1f chrF %.2f R-L %.2f | dev 峰值 ep%d %.2f 末轮 %.2f | 地板 %.2f" % (
            name, G["bleu4"], G["bleu1"], G["chrf"], G["rouge-l"], pk["epoch"], pk["bleu4"], H[-1]["bleu4"], G.get("floor", {}).get("bleu4", float("nan"))))
    else:
        print("  %-28s 未完成" % name)
PY
log "================ HOW2SIGN TRAIN2 DONE ================"
