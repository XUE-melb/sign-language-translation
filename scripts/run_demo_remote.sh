#!/bin/bash
# 在 AutoDL 服务器上起 demo 服务（本人回国后台式机不在身边，D-030）。
# 用法：bash scripts/run_demo_remote.sh [start|stop|status|restart]
#   端口 6006：AutoDL 控制台"自定义服务"映射的端口，可得到公网 https 链接；也可 ssh 隧道：
#     ssh -N -L 8000:127.0.0.1:6006 -p <端口> root@<主机>   然后浏览器开 http://127.0.0.1:8000
# 裁判：有 DEEPSEEK_API_KEY 用 DeepSeek（国内服务器连不上 Claude API），否则规则版；key 放 ~/.slt_env（不入库）。
# 上传视频的关键点提取走 .venv-rtmpose 的 GPU onnxruntime 子进程（SLT_RTMPOSE_CMD），4 秒/段；进程内 CPU 版要 3 分钟。
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=/root/autodl-tmp/slt/src
export SLT_RUN=${SLT_RUN:-runs/E003_enc1e-4_s3456}
export SLT_RTMPOSE_CMD="bash /root/autodl-tmp/slt/scripts/run_rtmpose.sh /root/autodl-tmp/slt/scripts/extract_one.py"
[ -f ~/.slt_env ] && source ~/.slt_env            # export DEEPSEEK_API_KEY=... / ANTHROPIC_API_KEY=... / SLT_JUDGE=...
PORT=${PORT:-6006}
LOG=/root/autodl-tmp/slt/logs/demo_remote.log
PIDF=/root/autodl-tmp/slt/logs/demo.pid
PAT="uvicorn server.app[:]app --host 0.0.0.0 --port $PORT"     # 方括号防 pgrep 自匹配

running() { pgrep -f "$PAT"; }
do_stop() { P=$(running); if [ -n "$P" ]; then kill $P; sleep 2; echo "stopped pid $P"; else echo "not running"; fi; rm -f "$PIDF"; }
do_start() {
    P=$(running); if [ -n "$P" ]; then echo "already running on $PORT (pid $P)"; return 0; fi
    echo "[$(date +%F_%T)] start demo on 0.0.0.0:$PORT run=$SLT_RUN judge=${SLT_JUDGE:-auto}" | tee -a "$LOG"
    nohup /root/miniconda3/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port "$PORT" >> "$LOG" 2>&1 &
    echo $! > "$PIDF"
    for i in $(seq 1 60); do sleep 2; curl -s -m 3 "http://127.0.0.1:$PORT/api/info" >/dev/null 2>&1 && break; done
    curl -s -m 5 "http://127.0.0.1:$PORT/api/info"; echo
    curl -s -m 5 "http://127.0.0.1:$PORT/api/agent/info"; echo
}
case "${1:-start}" in
    stop)    do_stop ;;
    status)  P=$(running); if [ -n "$P" ]; then echo "pid $P"; curl -s -m 5 "http://127.0.0.1:$PORT/api/info" | head -c 300; echo; else echo "not running"; fi ;;
    restart) do_stop; do_start ;;
    *)       do_start ;;
esac
