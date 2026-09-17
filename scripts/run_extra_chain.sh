#!/bin/bash
# 补充实验链（本人 09-17 批准"让 GPU 跑起来"，DECISIONS D-028 待写）。顺序：
#  1) E-005  去掉 LoRA 只训 pose_proj（E-002 其余配置不变）        → "为什么加 LoRA"
#  2) E-004b 留一手语者 E + 训练增广（--augment）                     → "陌生人对折，做了什么"
#  3) E-004c 留一手语者 E，阶段 3 配置（解冻编码器 lr 1e-4）           → "解冻会不会过拟合手语者"
#  4) E-004  留一手语者 A / D（E-002 配置）                          → 把 43% 变成均值±波动
#  5) 汇总。结束打 EXTRA CHAIN DONE。日志 logs/extra_chain.log
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/slt/logs/extra_chain.log
CK=weights/unisign/csl_daily_pose_only_slt.pth
COMMON="--unisign-ckpt $CK --lr 1e-4 --epochs 60 --batch-size 8 --label-smoothing 0.0"
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

train_run() {   # 用法: train_run 输出目录 [额外参数...]（含 --lora-r）
    local D=$1; shift
    if [ -f "$D/best/meta.json" ] && [ -f "$D/history.json" ]; then log "跳过 $D（已有）"; return; fi
    log ">>> train $D  $*"
    $PY -m slt.train_stage2 --out "$D" $COMMON "$@" >> "$LOG" 2>&1
    [ -f "$D/best/meta.json" ] || log "!! $D 没有产出 best，检查日志"
}
test_run() {    # 用法: test_run 输出目录 [额外参数...]
    local D=$1; shift
    [ -f "$D/eval_test_greedy.json" ] && { log "跳过 test $D（已有）"; return; }
    [ -f "$D/best/meta.json" ] || { log "!! $D 无 best，跳过 test"; return; }
    log ">>> test $D $*"
    $PY -m slt.train_stage2 --test-only "$D" "$@" 2>&1 | tee -a "$LOG" | grep -E "best:|n=|留一|其中原 test"
    ln -sf predictions_test_greedy.csv "$D/predictions_test.csv"
}

# 1) E-005 无 LoRA
train_run runs/E005_noLoRA_s1234 --lora-r 0 --seed 1234
test_run  runs/E005_noLoRA_s1234

# 2) E-004b 留 E + 增广
train_run runs/E004b_holdoutE_aug_s1234 --lora-r 16 --exclude-signer E --augment --seed 1234
test_run  runs/E004b_holdoutE_aug_s1234 --eval-signer E

# 3) E-004c 留 E，阶段 3 配置
train_run runs/E004c_holdoutE_enc_s1234 --lora-r 16 --exclude-signer E --unfreeze-encoder --encoder-lr 1e-4 --seed 1234
test_run  runs/E004c_holdoutE_enc_s1234 --eval-signer E

# 4) 留 A、留 D
for SG in A D; do
    train_run runs/E004_holdout${SG}_s1234 --lora-r 16 --exclude-signer $SG --seed 1234
    test_run  runs/E004_holdout${SG}_s1234 --eval-signer $SG
done

# 5) 汇总
log "---------------- 汇总 ----------------"
$PY - 2>&1 <<'PY' | tee -a "$LOG"
import json, os
R = "/root/autodl-tmp/slt/runs"
def J(p):
    p = os.path.join(R, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
def line(name, d):
    g = J(d + "/eval_test_greedy.json"); b = J(d + "/eval_test_beam4.json")
    if g: print("  %-28s test500 greedy %.2f  beam4 %.2f  R-L %.2f  ep %s" % (name, g["bleu4"], b["bleu4"] if b else float("nan"), g["rouge-l"], g["_meta"]["epoch"]))
    else: print("  %-28s 未完成" % name)
print("E-002 60ep 三 seed 均值 16.79 / beam4 17.74（对照）")
line("E-005 无 LoRA（只训 pose_proj）", "E005_noLoRA_s1234")
print("-" * 70); print("留一手语者（E-002 配置除非注明）| 参照 E-004 留 E：E-600 7.64，E-test54 7.53；见过 E 时 17.6")
for name, d, sg in (("E-004b 留 E + 增广", "E004b_holdoutE_aug_s1234", "E"),
                    ("E-004c 留 E，阶段 3 配置", "E004c_holdoutE_enc_s1234", "E"),
                    ("E-004 留 A", "E004_holdoutA_s1234", "A"),
                    ("E-004 留 D", "E004_holdoutD_s1234", "D")):
    e = J("%s/eval_signer%s_greedy.json" % (d, sg)); t = J(d + "/eval_test_greedy.json")
    if e: print("  %-26s %s 全部 %d 条 %.2f（test 子集 %d 条 %.2f）| 标准 test500 %.2f" % (
        name, sg, e["_meta"]["n_all"], e["all"]["bleu4"], e["_meta"]["n_test_only"], e["test_only"]["bleu4"], t["bleu4"] if t else float("nan")))
    else: print("  %-26s 未完成" % name)
# 见过 A / D 时的对照（E-002 60ep 三 seed 均值，分手语者）
import statistics as st
for sg in ("A", "D"):
    v = [J("E002_lora16_ep60_s%d/eval_test_greedy.json" % s)["per_translator"][sg]["bleu4"] for s in (1234, 2345, 3456)]
    print("  参照：E-002 见过 %s 的 test 子集 %.2f（3 seed 均值）" % (sg, st.mean(v)))
PY
log "================ EXTRA CHAIN DONE ================"
