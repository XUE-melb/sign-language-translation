#!/bin/bash
# RTMPose 环境的统一入口。
#
# 为什么需要这层包装：onnxruntime-gpu 要在运行时加载 libcudnn / libcublas，
# 而这些是通过 pip 装在 venv 的 site-packages/nvidia/*/lib 下的，
# 不在系统库搜索路径里。必须在 python 启动**之前**把它们加进
# LD_LIBRARY_PATH，进程起来之后再设是无效的。
#
# 用法：bash scripts/run_rtmpose.sh <python 脚本> [参数...]

VENV=/root/autodl-tmp/slt/.venv-rtmpose
SITE=$("$VENV/bin/python" -c "import site;print(site.getsitepackages()[0])")

# 可用组合（实测，见 DECISIONS.md D-020）：
#   onnxruntime-gpu 1.19.2 + pip 装的 cuDNN 9 + 系统 CUDA 12.1 的 cublas
# 注意 1.30 要 CUDA 13（本机是 12.1），1.18.1 配系统 cuDNN 8 也起不来。
NV_LIBS=""
for d in "$SITE"/nvidia/*/lib; do
    [ -d "$d" ] && NV_LIBS="$NV_LIBS:$d"
done
SYS_CUDA=/usr/local/cuda/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${NV_LIBS#:}:$SYS_CUDA${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 模型首次下载需要走加速
source /etc/network_turbo >/dev/null 2>&1

cd /root/autodl-tmp/slt || exit 1
exec "$VENV/bin/python" "$@"
