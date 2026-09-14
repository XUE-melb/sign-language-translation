# -*- coding: utf-8 -*-
"""把 D-016 追加到服务器上的 DECISIONS.md，并更新 STATE.md 的状态段。"""
import io
import shutil
import sys

DEC = "/root/autodl-tmp/slt/docs/DECISIONS.md"
ADD = "/root/autodl-tmp/slt/docs/_d016.md"

dec = io.open(DEC, encoding="utf-8").read()
if "## D-016" in dec:
    print("D-016 已存在，跳过追加")
else:
    add = io.open(ADD, encoding="utf-8").read()
    shutil.copy(DEC, DEC + ".bak")
    io.open(DEC, "w", encoding="utf-8").write(dec.rstrip() + "\n" + add)
    print("D-016 已追加")

# ---- STATE.md：更新数据现状与"正在跑什么" ----
ST = "/root/autodl-tmp/slt/docs/STATE.md"
s = io.open(ST, encoding="utf-8").read()

OLD_TABLE = """| split | 视频下载 | 关键点提取 | 说明 |
|-------|---------|-----------|------|
| dev | 515/515 ✅ | **515/515 ✅** | 已核对，零缺失 |
| test | 500/500 ✅ | **500/500 ✅** | 已核对，零缺失 |
| train | 进行中 | 进行中 | 见下方"正在跑什么" |"""

NEW_TABLE = """| split | 视频下载 | 关键点提取 | 说明 |
|-------|---------|-----------|------|
| train | 4973/4973 ✅ | **4973/4973 ✅** | 已核对，零缺失零失败 |
| dev | 515/515 ✅ | **515/515 ✅** | 已核对，零缺失 |
| test | 500/500 ✅ | **500/500 ✅** | 已核对，零缺失 |

**数据准备阶段全部完成（2026-09-14 23:41）。** 全量 5988 条，pose 共 1.2 GB。
质检结论见 DECISIONS.md D-016：不剔除任何样本，train 与 dev 分布一致
（帧数均值 187 vs 186，手部检出率 train 略好）。"""

if OLD_TABLE in s:
    s = s.replace(OLD_TABLE, NEW_TABLE)
    print("STATE.md 数据表已更新")
else:
    print("警告：STATE.md 数据表未命中，未更新该段")

OLD_RUN = s[s.find("## 三、正在跑什么"):s.find("## 四、下一步")]
NEW_RUN = """## 三、正在跑什么

**没有后台任务在跑。** 数据准备已于 2026-09-14 23:41 全部完成
（`logs/pipeline_train.log` 末尾是 `PIPELINE DONE`）。

tmux 会话 `work` 里的 `download` / `pose` / `pipeline` 三个窗口均已结束，
可以复用。

"""
if OLD_RUN:
    s = s.replace(OLD_RUN, NEW_RUN)
    print("STATE.md 运行状态已更新")
else:
    print("警告：STATE.md 运行状态段未命中")

s = s.replace("最后更新：2026-09-14 20:50", "最后更新：2026-09-14 23:45")

io.open(ST, "w", encoding="utf-8").write(s)
print("STATE.md 已写入")
