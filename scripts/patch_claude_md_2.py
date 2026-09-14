# -*- coding: utf-8 -*-
"""修正 CLAUDE.md 里凭印象写的 BLEU 参考值。"""
import io, shutil, sys

P = "/root/autodl-tmp/slt/CLAUDE.md"
OLD = "- 参考量级：gloss-based SOTA 在 Phoenix-2014T 约 29 BLEU-4，CSL-Daily 约 25.5"
NEW = ("- 参考量级（gloss-based test BLEU-4，2026-09-14 核定）：\n"
       "  PHOENIX-2014T 最高 28.42 (TextCTC-SLT)，CSL-Daily 最高 25.79 (TwoStream-SLT)\n"
       "  注意：这些是 gloss-based 且跑在别的数据集上，与本项目的 CE-CSL gloss-free\n"
       "  数字不可直接比较，只作量级参照。完整榜单见 docs/EXPERIMENTS.md")

src = io.open(P, encoding="utf-8").read()
n = src.count(OLD)
if n != 1:
    sys.exit("中止：原文命中 %d 次（应为 1 次），未做修改" % n)
shutil.copy(P, P + ".bak2")
io.open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("OK  BLEU 参考值已修正（备份 CLAUDE.md.bak2）")
