#!/bin/bash
cd /root/autodl-tmp/slt || exit 1
LOG=/root/autodl-tmp/slt/logs/download.log
echo "[$(date +%F_%T)] START test" | tee -a $LOG
bypy downdir CE-CSL/CE-CSL/video/test ./CE-CSL/video/test 2>&1 | tee -a $LOG
echo "[$(date +%F_%T)] test done: $(find ./CE-CSL/video/test -name "*.mp4" | wc -l) mp4" | tee -a $LOG
echo "[$(date +%F_%T)] START train" | tee -a $LOG
bypy downdir CE-CSL/CE-CSL/video/train ./CE-CSL/video/train 2>&1 | tee -a $LOG
echo "[$(date +%F_%T)] train done: $(find ./CE-CSL/video/train -name "*.mp4" | wc -l) mp4" | tee -a $LOG
echo "[$(date +%F_%T)] ALL DONE" | tee -a $LOG
