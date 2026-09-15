#!/bin/bash
# 停掉三 mode 对比（下载太慢），只测 Uni-Sign 默认的 lightweight。
# 用 [s] 括号技巧避免 pkill 匹配到自己的命令行。
pkill -f 'compare_mode[s]' 2>/dev/null && echo "已停止 mode 对比" || echo "对比进程已不在"
sleep 2
pgrep -f 'compare_mode[s]' >/dev/null && echo "警告：仍在运行" || echo "确认已停"
echo

# -u 关闭缓冲，否则输出会憋到进程结束才出现
exec bash /root/autodl-tmp/slt/scripts/run_rtmpose.sh -u /root/autodl-tmp/slt/tests/test_lightweight.py
