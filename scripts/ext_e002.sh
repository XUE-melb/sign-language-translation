#!/bin/bash
# E-002 延长跑：D-024 预登记规则——"dev 曲线到 30 轮仍在涨则延长"。
# 实测 s1234 峰值 ep29、s2345 峰值 ep30，规则触发。同配置改 epochs=60，3 seed，s1234 先跑。
# 等 POST DONE 后再起，不与主链抢 GPU。产物 runs/E002_lora16_ep60_s*，日志 logs/ext_e002.log
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/slt/logs/ext_e002.log
CK=weights/unisign/csl_daily_pose_only_slt.pth
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

log "等待 POST DONE ..."
while ! grep -q "POST DONE" logs/post_stage2.log 2>/dev/null; do sleep 120; done
log "主链后处理已完成，开始 E-002 延长跑（60 epoch）"

for SEED in 1234 2345 3456; do
    D=runs/E002_lora16_ep60_s$SEED
    if [ -f "$D/best/meta.json" ]; then log "跳过 s$SEED（已有 best）"; continue; fi
    log ">>> E-002-ext s$SEED（同 E-002 配置，epochs 60）"
    $PY -m slt.train_stage2 --out "$D" --unisign-ckpt "$CK" --lora-r 16 \
        --lr 1e-4 --epochs 60 --batch-size 8 --label-smoothing 0.0 --seed $SEED >> "$LOG" 2>&1
    [ -f "$D/best/meta.json" ] || { log "!! s$SEED 没有产出 best，检查日志"; continue; }
    log ">>> test s$SEED（greedy + beam4）"
    $PY -m slt.train_stage2 --test-only "$D" --out "$D" 2>&1 | tee -a "$LOG" | grep -E "best:|n=|chrF"
    ln -sf predictions_test_greedy.csv "$D/predictions_test.csv"
    $PY - "$D" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
h = json.load(open(sys.argv[1] + "/history.json", encoding="utf-8"))["history"]
pk = max(h, key=lambda x: x["bleu4"])
e30 = h[29]["bleu4"] if len(h) > 29 else float("nan")
print("  曲线: 峰值 ep%d dev B4 %.2f | ep30 %.2f | 末 5 轮 %s" % (
    pk["epoch"], pk["bleu4"], e30, " ".join("%.2f" % x["bleu4"] for x in h[-5:])))
PY
done

log "---- 汇总 ext vs 30ep ----"
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, statistics as st
R = "/root/autodl-tmp/slt/runs"
def rows(pat, fn):
    out = []
    for s in (1234, 2345, 3456):
        p = os.path.join(R, pat % s, fn)
        if os.path.exists(p):
            e = json.load(open(p, encoding="utf-8")); out.append((s, e["bleu4"], e["chrf"], e["rouge-l"]))
    return out
for name, pat in (("E-002 30ep", "E002_lora16_s%d"), ("E-002 60ep", "E002_lora16_ep60_s%d")):
    for dec in ("greedy", "beam4"):
        rs = rows(pat, "eval_test_%s.json" % dec)
        if not rs: continue
        b = [x[1] for x in rs]; r = [x[3] for x in rs]
        print("%s %s (%d seed)  BLEU-4 %.2f±%.2f  R-L %.2f±%.2f  | %s" % (
            name, dec, len(rs), st.mean(b), st.pstdev(b), st.mean(r), st.pstdev(r),
            " ".join("s%d=%.2f" % (s, x) for s, x, _, _ in rs)))
PY
log "================ EXT DONE ================"
