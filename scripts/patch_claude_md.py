#!/usr/bin/env python
"""按 DECISIONS.md D-008 修订 CLAUDE.md。每处替换都断言命中且唯一。"""
import io
import shutil
import sys

P = "/root/autodl-tmp/slt/CLAUDE.md"

PATCHES = [
    # (旧串, 新串, 说明)
    ("平均约 5.3 秒约 158 帧",
     "平均约 6.2 秒约 186 帧（dev 全量 515 条实测：均值 186，中位 181，范围 81-464）",
     "帧数与时长"),
    ("直接从视频提关键点，每个视频存一个 .npy，总量几百 MB。",
     "直接从视频提关键点，每个视频存一个 .npy，全量约 1.2 GB\n"
     "（dev 实测 103 MB / 515 条；相比 RGB 路线的约 22 GB 小约 18 倍）。",
     "存储量"),
    ("实例到期 2026-09-21，需续费",
     "实例到期 2026-09-28（已于 2026-09-14 续费一周）",
     "到期日"),
]

src = io.open(P, encoding="utf-8").read()
shutil.copy(P, P + ".bak")

for old, new, label in PATCHES:
    n = src.count(old)
    if n != 1:
        sys.exit("中止：{} 的原文命中 {} 次（应为 1 次），未做任何修改".format(label, n))
    src = src.replace(old, new)
    print("OK  {}".format(label))

io.open(P, "w", encoding="utf-8").write(src)
print("\n已写入 {}（备份 {}.bak）".format(P, P))
