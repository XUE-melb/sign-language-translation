# -*- coding: utf-8 -*-
"""在 CLAUDE.md 的文档纪律一节里加上 STATE.md 的指引。"""
import io
import shutil
import sys

P = "/root/autodl-tmp/slt/CLAUDE.md"
OLD = "- 每个新会话开始时先读这两份文件"
NEW = ("- docs/STATE.md：当前状态、正在跑什么、下一步、未解决的阻塞\n"
       "- **每个新会话最先读 docs/STATE.md**，再读本文件与上述两份文档")

s = io.open(P, encoding="utf-8").read()
n = s.count(OLD)
if n != 1:
    sys.exit("中止：原文命中 %d 次（应为 1 次），未做修改" % n)
shutil.copy(P, P + ".bak3")
io.open(P, "w", encoding="utf-8").write(s.replace(OLD, NEW))
print("OK  CLAUDE.md 已指向 STATE.md（备份 .bak3）")
