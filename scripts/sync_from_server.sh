#!/bin/bash
# 把服务器上的文档和脚本拉回本地（不同步数据和 .npy）
# 用法：bash scripts/sync_from_server.sh
set -e
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=/root/autodl-tmp/slt

mkdir -p "$LOCAL/docs" "$LOCAL/scripts"
scp -o BatchMode=yes autodl:$REMOTE/CLAUDE.md   "$LOCAL/CLAUDE.md"
scp -o BatchMode=yes "autodl:$REMOTE/docs/*"    "$LOCAL/docs/"
scp -o BatchMode=yes "autodl:$REMOTE/scripts/*" "$LOCAL/scripts/"
echo "已同步到 $LOCAL"
