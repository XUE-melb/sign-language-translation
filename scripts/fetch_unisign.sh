#!/bin/bash
# 下载 Uni-Sign 的 pose-only 预训练权重（阶段 1 的冻结编码器）。
#
# csl_stage1_weight.pth = 他们的 Stage 1（pose-only 在 CSL-News 上预训练），
# 没有任何下游任务微调 —— 正是"预训练时空编码器"该有的样子。
#
# 许可证 CC-BY-NC-4.0（非商用）。权重放 weights/ 并已在 .gitignore 里排除，
# 不进公开仓库。
source /etc/network_turbo >/dev/null 2>&1
DIR=/root/autodl-tmp/slt/weights/unisign
mkdir -p "$DIR"
BASE=https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main

echo "===== 文件大小 ====="
for f in csl_stage1_weight.pth csl_stage2_weight.pth csl_daily_pose_only_slt.pth; do
    sz=$(curl -sSL -I --max-time 60 "$BASE/$f" 2>/dev/null \
         | grep -i '^content-length' | tail -1 | tr -dc '0-9')
    if [ -n "$sz" ]; then
        printf "  %-32s %6d MB\n" "$f" $((sz / 1048576))
    else
        printf "  %-32s (未知)\n" "$f"
    fi
done

echo
echo "===== 下载 csl_stage1_weight.pth ====="
if [ -f "$DIR/csl_stage1_weight.pth" ]; then
    echo "  已存在，跳过"
else
    curl -sSL --max-time 3600 -o "$DIR/csl_stage1_weight.pth.part" \
         "$BASE/csl_stage1_weight.pth" \
      && mv "$DIR/csl_stage1_weight.pth.part" "$DIR/csl_stage1_weight.pth" \
      && echo "  下载完成"
fi
ls -lh "$DIR"
