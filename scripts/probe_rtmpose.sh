#!/bin/bash
# 可行性探测：不装任何东西，只看能不能装、网络通不通。

echo "===== 1. HuggingFace 连通性（需要 network_turbo）====="
source /etc/network_turbo 2>/dev/null && echo "network_turbo 已开启" || echo "network_turbo 不可用"
curl -sS -o /dev/null -w "  huggingface.co  HTTP %{http_code}  (%{time_total}s)\n" \
     --max-time 20 https://huggingface.co/ZechengLi19/Uni-Sign 2>&1 | head -2
curl -sS -o /dev/null -w "  github.com      HTTP %{http_code}  (%{time_total}s)\n" \
     --max-time 20 https://github.com/ZechengLi19/Uni-Sign 2>&1 | head -2

echo
echo "===== 2. pip 源上有哪些候选包（关掉代理）====="
unset http_proxy https_proxy all_proxy
for pkg in rtmlib onnxruntime-gpu onnxruntime mmpose mmcv mmengine mmdet; do
    V=$(pip index versions "$pkg" 2>/dev/null | head -1)
    if [ -z "$V" ]; then
        echo "  $pkg: 源上没有 / 查询失败"
    else
        echo "  $V"
    fi
done

echo
echo "===== 3. 现有环境 ====="
echo "  显卡: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo '(无 nvcc)')"
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.version.cuda)"
echo "  磁盘剩余: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"
