#!/bin/bash
# train split 全自动链：边下边提 -> 核对 -> 质检
#
# 不等下载完成才开始提取：下载是网络瓶颈，CPU 全闲着。
# extract_pose.py 会跳过已提取的 .npy、也会跳过尚未下载的视频，
# 所以可以反复运行，每轮把新到的视频补上。
#
# 用法：tmux new-window -n pipeline; bash scripts/pipeline_train.sh

cd /root/autodl-tmp/slt || exit 1
PY=/root/autodl-tmp/slt/.venv-pose/bin/python
LOG=/root/autodl-tmp/slt/logs/pipeline_train.log
TOTAL=4973

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

count_dl()  { find CE-CSL/video/train -name "*.mp4" 2>/dev/null | wc -l; }
count_ext() { find CE-CSL/pose/train  -name "*.npy" 2>/dev/null | wc -l; }
dl_done()   { grep -q "ALL DONE" logs/download.log 2>/dev/null && echo 1 || echo 0; }

log "=========== pipeline 启动 ==========="
round=0
while true; do
    round=$((round + 1))
    have=$(count_dl)
    fin=$(dl_done)
    log "第 ${round} 轮开始：已下载 ${have}/${TOTAL}，下载完成标志=${fin}"

    $PY scripts/extract_pose.py --split train --workers 12 >> "$LOG" 2>&1
    ext=$(count_ext)
    log "第 ${round} 轮结束：已提取 ${ext}/${TOTAL}"

    # 下载已完成，且当前可见的视频都提完了 -> 收工
    if [ "$fin" = "1" ] && [ "$ext" -ge "$have" ]; then
        log "下载已完成且提取已追平（${ext}/${have}）"
        break
    fi
    # 下载还没完但这一轮没新东西可提 -> 歇一会再看
    if [ "$ext" -ge "$have" ]; then
        log "暂无新视频可提，等待下载..."
        sleep 120
    fi
done

log "=========== 核对完整性 ==========="
$PY scripts/check_split.py train 2>&1 | tee -a "$LOG"

log "=========== 提取质检 ==========="
$PY scripts/qc_pose.py --split train 2>&1 | tee -a "$LOG"

log "=========== PIPELINE DONE ==========="
log "下一步：PYTHONPATH=src python -m slt.train --train-split train --eval-split dev"
