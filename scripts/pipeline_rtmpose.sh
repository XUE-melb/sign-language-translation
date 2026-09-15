#!/bin/bash
# RTMPose 全量提取：dev -> test -> train（小的先跑，早暴露问题）
#
# 必须经 run_rtmpose.sh 启动，否则 onnxruntime 找不到 cuDNN 会静默退回 CPU
# （CPU 慢 90 倍，全量要 665 小时）。
# mode 固定 lightweight —— 见 DECISIONS.md D-020。

LOG=/root/autodl-tmp/slt/logs/rtmpose_extract.log
WORKERS=${1:-3}

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

log "============ RTMPose 全量提取开始（workers=$WORKERS）============"

for SPLIT in dev test train; do
    log ">>> 开始 $SPLIT"
    bash /root/autodl-tmp/slt/scripts/run_rtmpose.sh -u \
        /root/autodl-tmp/slt/scripts/extract_rtmpose.py \
        --split "$SPLIT" --workers "$WORKERS" 2>&1 \
        | grep -avE "^[[:space:]]*[0-9.]+%|onnxruntime:|CleanUnused|VerifyEachNode" \
        | tee -a "$LOG"
    n=$(find /root/autodl-tmp/slt/CE-CSL/pose_rtm/"$SPLIT" -name "*.pkl" 2>/dev/null | wc -l)
    log "<<< $SPLIT 完成，产出 $n 个 pkl"
done

log "============ 各 split 产出 ============"
for SPLIT in train dev test; do
    n=$(find /root/autodl-tmp/slt/CE-CSL/pose_rtm/"$SPLIT" -name "*.pkl" 2>/dev/null | wc -l)
    log "  $SPLIT: $n"
done
log "总体积: $(du -sh /root/autodl-tmp/slt/CE-CSL/pose_rtm 2>/dev/null | cut -f1)"
log "============ RTMPOSE DONE ============"
