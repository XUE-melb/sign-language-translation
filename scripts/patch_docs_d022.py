# -*- coding: utf-8 -*-
"""EXPERIMENTS.md：E-000 加 D-022 caveat；补 E-000 vs E-000b 关系；补 test 略难。"""
import io
import shutil

P = "/root/autodl-tmp/slt/docs/EXPERIMENTS.md"
s = io.open(P, encoding="utf-8").read()

old = ("| **E-000** | 0 | 老架构 baseline：逐帧 MLP（不跨帧）+ BiLSTM 编码器 + "
       "带 Bahdanau attention 的 LSTM 解码器；输入 pose 538 维 |")
new = ("| **E-000** | 0 | 老架构 baseline：逐帧 MLP（不跨帧）+ BiLSTM 编码器 + "
       "带 Bahdanau attention 的 LSTM 解码器；输入 **MediaPipe 538 维** |")
assert s.count(old) == 1, "E-000 行未命中"
s = s.replace(old, new)

old2 = "| 3 seed 均值±std；lr=3e-4 epochs=100 batch=16；signer-dependent 官方划分 |"
new2 = ("| 3 seed 均值±std；lr=3e-4 epochs=100 batch=16；signer-dependent 官方划分。"
        "**⚠️ 在 D-022 之前的协议下跑的（scheduler 把 lr 压到 ~1/10⁶，实际只训约 20 轮；"
        "stride=2）。保留作历史记录，不是消融表的第 0 行——第 0 行是 E-000b（RTMPose 输入）** |")
assert s.count(old2) == 1, "E-000 备注未命中"
s = s.replace(old2, new2)

anchor = "## 每行必须能回答"
note = """## 关于 E-000 与 E-000b 的关系

两者都是阶段 0 的架构，但**不可直接比较**：输入不同（MediaPipe 538 维 vs RTMPose 69 点 207 维）、
预处理不同（D-011 全身统一归一化 vs D-021 的 Uni-Sign 分组归一化）、协议不同（E-000 在 D-022 之前）。
它们的差异是**提取器 + 特征集 + 归一化 + 协议**的合并效应，标清楚就能当一行附加信息用，
但不做单变量归因。主消融表的四行全部建在 RTMPose 输入上，从 E-000b 起算。

## test 比 train / dev 略难（RTMPose 质检，2026-09-15）

| split | 平均帧数 | 手部置信度均值（左/右） |
|-------|--------:|----------------------:|
| train | 187 | 0.727 / 0.687 |
| dev | 186 | 0.697 / 0.674 |
| test | **199** | 0.683 / 0.679 |

test 更长、手部置信度略低。用 dev 选的 checkpoint 在 test 上的数字可能略偏悲观。
属于官方划分的固有属性，不做处理，只在解读时记住。

"""
assert s.count(anchor) == 1, "锚点未命中"
s = s.replace(anchor, note + anchor)

shutil.copy(P, P + ".bak")
io.open(P, "w", encoding="utf-8").write(s)
print("EXPERIMENTS.md 已更新")
