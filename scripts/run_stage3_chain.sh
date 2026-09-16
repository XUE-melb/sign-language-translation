#!/bin/bash
# 阶段 3 + E-004 无人值守链（D-025 定 epochs=60；D-026）。顺序：
#  1) E-004 留一手语者 E：E-002 配置（60 轮，seed 1234），train/dev 去掉 E → test + 在 E 的 600 条上评
#  2) E-003 编码器 lr 探针：seed 1234，encoder-lr 1e-4 与 1e-5 各一趟（60 轮）
#  3) 按 dev 峰值选 lr，补 seed 2345 / 3456 → 三 seed test；落选探针也评 test 留档
#  4) 汇总。结束打 STAGE3 CHAIN DONE。日志 logs/stage3_chain.log
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/slt/logs/stage3_chain.log
CK=weights/unisign/csl_daily_pose_only_slt.pth
COMMON="--unisign-ckpt $CK --lora-r 16 --lr 1e-4 --epochs 60 --batch-size 8 --label-smoothing 0.0"
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

train_run() {   # 用法: train_run 输出目录 [额外参数...]
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
    $PY -m slt.train_stage2 --test-only "$D" --out "$D" "$@" 2>&1 | tee -a "$LOG" | grep -E "best:|n=|留一|其中原 test"
    ln -sf predictions_test_greedy.csv "$D/predictions_test.csv"
}

# ---------- 1) E-004 ----------
train_run runs/E004_holdoutE_s1234 --exclude-signer E --seed 1234
test_run  runs/E004_holdoutE_s1234 --eval-signer E

# ---------- 2) E-003 lr 探针 ----------
for LR in 1e-4 1e-5; do
    train_run runs/E003_enc${LR}_s1234 --unfreeze-encoder --encoder-lr $LR --seed 1234
done
BEST=$($PY - 2>>"$LOG" <<'PY'
import json, sys
best = None
for lr in ("1e-4", "1e-5"):
    try:
        v = json.load(open("runs/E003_enc%s_s1234/history.json" % lr, encoding="utf-8"))["best_dev_bleu4"]
    except Exception as e:
        v = -1.0
    print("  探针 encoder-lr %s: dev 峰值 %.2f" % (lr, v), file=sys.stderr)
    if best is None or v > best[1]:
        best = (lr, v)
print(best[0])
PY
)
log "E-003 编码器 lr 选定: $BEST（按 seed 1234 的 dev 峰值）"

# ---------- 3) 补 seed + test ----------
for S in 2345 3456; do
    train_run runs/E003_enc${BEST}_s$S --unfreeze-encoder --encoder-lr $BEST --seed $S
done
for S in 1234 2345 3456; do test_run runs/E003_enc${BEST}_s$S; done
for LR in 1e-4 1e-5; do [ "$LR" != "$BEST" ] && test_run runs/E003_enc${LR}_s1234; done

# ---------- 4) 汇总 ----------
log "---------------- 汇总 ----------------"
BEST=$BEST $PY - 2>&1 <<'PY' | tee -a "$LOG"
import json, os, statistics as st
R = "/root/autodl-tmp/slt/runs"; best = os.environ["BEST"]
def rows(pat, fn):
    out = []
    for s in (1234, 2345, 3456):
        p = os.path.join(R, pat % s, fn)
        if os.path.exists(p):
            e = json.load(open(p, encoding="utf-8"))
            out.append((s, e["bleu4"], e["chrf"], e["rouge-l"]))
    return out
def show(name, rs):
    print("-" * 70); print(name, "(%d/3 seed)" % len(rs))
    for s, b, c, r in rs:
        print("  s{}  BLEU-4 {:.2f}  chrF {:.2f}  R-L {:.2f}".format(s, b, c, r))
    if len(rs) >= 2:
        b = [x[1] for x in rs]; r = [x[3] for x in rs]
        print("  均值±std  BLEU-4 {:.2f}±{:.2f}  R-L {:.2f}±{:.2f}".format(st.mean(b), st.pstdev(b), st.mean(r), st.pstdev(r)))
    return rs
a = show("E-002 60ep greedy（正式行）", rows("E002_lora16_ep60_s%d", "eval_test_greedy.json"))
b = show("E-003 encoder-lr %s greedy" % best, rows("E003_enc%s_s%%d" % best, "eval_test_greedy.json"))
show("E-003 encoder-lr %s beam4" % best, rows("E003_enc%s_s%%d" % best, "eval_test_beam4.json"))
if a and b:
    d = st.mean(x[1] for x in b) - st.mean(x[1] for x in a)
    print("=" * 70); print("E-002 -> E-003 (greedy)  BLEU-4 差值 {:+.2f}".format(d))
for lr in ("1e-4", "1e-5"):
    p = os.path.join(R, "E003_enc%s_s1234" % lr, "history.json")
    if os.path.exists(p):
        h = json.load(open(p, encoding="utf-8"))["history"]; pk = max(h, key=lambda x: x["bleu4"])
        print("探针 %s: dev 峰值 ep%d %.2f | 末 5 轮 %s" % (lr, pk["epoch"], pk["bleu4"], " ".join("%.2f" % x["bleu4"] for x in h[-5:])))
print("-" * 70); print("E-004 留一手语者 E")
for dec in ("greedy", "beam4"):
    p = os.path.join(R, "E004_holdoutE_s1234", "eval_signerE_%s.json" % dec)
    if os.path.exists(p):
        e = json.load(open(p, encoding="utf-8"))
        print("  %s  E 全部 %d 条: BLEU-4 %.2f  chrF %.2f  R-L %.2f  | 其中 test %d 条: BLEU-4 %.2f  R-L %.2f  | 参照 E-002 见过 E 的 test-54: 15.88" % (
            dec, e["_meta"]["n_all"], e["all"]["bleu4"], e["all"]["chrf"], e["all"]["rouge-l"],
            e["_meta"]["n_test_only"], e["test_only"]["bleu4"], e["test_only"]["rouge-l"]))
p = os.path.join(R, "E004_holdoutE_s1234", "eval_test_greedy.json")
if os.path.exists(p):
    e = json.load(open(p, encoding="utf-8"))
    print("  E-004 标准 test 500 条 greedy: BLEU-4 %.2f（含 E 的 54 条未见样本）" % e["bleu4"])
PY
log "================ STAGE3 CHAIN DONE ================"
