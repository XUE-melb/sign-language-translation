# -*- coding: utf-8 -*-
"""追加 D-017 到 DECISIONS.md，并把 STATE.md 的运行状态改成 stage0 进行中。"""
import io
import shutil

DEC = "/root/autodl-tmp/slt/docs/DECISIONS.md"
ADD = "/root/autodl-tmp/slt/docs/_d017.md"

dec = io.open(DEC, encoding="utf-8").read()
if "## D-017" in dec:
    print("D-017 已存在，跳过")
else:
    shutil.copy(DEC, DEC + ".bak")
    io.open(DEC, "w", encoding="utf-8").write(
        dec.rstrip() + "\n" + io.open(ADD, encoding="utf-8").read())
    print("D-017 已追加")

ST = "/root/autodl-tmp/slt/docs/STATE.md"
s = io.open(ST, encoding="utf-8").read()
start = s.find("## 三、正在跑什么")
end = s.find("## 四、下一步")
NEW = """## 三、正在跑什么

**阶段 0 正式实验（E-000）进行中**，2026-09-14 23:59 启动，
跑在 tmux `work:stage0`，脚本 `scripts/run_stage0.sh`。

协议见 DECISIONS.md D-017：lr 搜索 3 趟（3e-4 / 1e-3 / 3e-3，seed 1234）
-> 选定 lr 后补 seed 2345 / 3456 -> 每个 seed 用 best checkpoint 在 test 上
各评一次 -> 汇总均值±std。

实测 25.2 秒/epoch，100 epoch 约 42 分钟一趟，**5 趟合计约 3.5 小时**，
预计 2026-09-15 03:30 前后完成。

查进度：
```bash
ssh autodl 'tail -5 /root/autodl-tmp/slt/logs/stage0.log'
ssh autodl 'grep "^\\[" /root/autodl-tmp/slt/logs/stage0.log | tail -5'
```

跑完日志末尾是 `STAGE0 DONE`，前面跟着三个 seed 的 test 结果汇总表。

**跑完后要做的事：把汇总结果写进 `docs/EXPERIMENTS.md` 的 E-000 行**
（备注注明 signer-dependent、官方划分、字级 sacrebleu），
并检查 100 epoch 时 dev 曲线是否还在上升（若是则说明欠训练，需加 epoch 重跑）。

"""
if start != -1 and end != -1:
    s = s[:start] + NEW + s[end:]
    print("STATE.md 运行状态已更新")
else:
    print("警告：STATE.md 段落未命中")

import re
s = re.sub(r"最后更新：[0-9\- :]+", "最后更新：2026-09-15 00:05", s, count=1)
io.open(ST, "w", encoding="utf-8").write(s)
print("STATE.md 已写入")
