# -*- coding: utf-8 -*-
"""追加 D-018 / D-019，更新 EXPERIMENTS.md 的 lr 缺口说明与 STATE.md。"""
import io
import re
import shutil

DEC = "/root/autodl-tmp/slt/docs/DECISIONS.md"
ADD = "/root/autodl-tmp/slt/docs/_d018_d019.md"
dec = io.open(DEC, encoding="utf-8").read()
if "## D-018" in dec:
    print("D-018/D-019 已存在，跳过")
else:
    shutil.copy(DEC, DEC + ".bak")
    io.open(DEC, "w", encoding="utf-8").write(
        dec.rstrip() + "\n" + io.open(ADD, encoding="utf-8").read())
    print("D-018 / D-019 已追加")

# ---- EXPERIMENTS.md：lr 缺口已结案 ----
EXP = "/root/autodl-tmp/slt/docs/EXPERIMENTS.md"
s = io.open(EXP, encoding="utf-8").read()

OLD = """⚠️ **E-000 有一处方法论缺口**：选中的 lr=3e-4 是搜索范围的**下边界**，
最优值可能在范围之外（见下方实验日志与 DECISIONS.md D-018）。"""
NEW = """✅ **E-000 的 lr 缺口已结案**（DECISIONS.md D-018）：搜索已向下扩展到
1e-4 / 3e-5，最优值落在**区间内部**，边界问题解决。最优的 1e-4（dev 2.2454）
与所用的 3e-4（dev 2.1832）差距 0.0622，**小于 seed 标准差 0.1096**，
统计上不可区分 → 保留 3e-4，E-000 原样成立，test 未二次曝光。"""
if OLD in s:
    s = s.replace(OLD, NEW)
    print("EXPERIMENTS.md lr 警告已更新为结案")
else:
    print("警告：EXPERIMENTS.md lr 警告段未命中")

OLD2 = """- **lr 搜索触到下边界**：3e-4 是搜索范围里最小的值且它胜出，趋势指向更小的 lr。
  最优值可能在范围之外，当前 baseline 可能被低估。"""
NEW2 = """- ~~lr 搜索触到下边界~~ **已结案**：扩展到 1e-4 / 3e-5 后最优值落在区间内部。
  完整曲线 3e-5 1.6700 / **1e-4 2.2454** / 3e-4 2.1832 / 1e-3 1.1718 / 3e-3 0.8371。
  1e-4 与 3e-4 差 0.0622 < seed 标准差 0.1096，判定等价，保留 3e-4。见 D-018。"""
if OLD2 in s:
    s = s.replace(OLD2, NEW2)
    print("EXPERIMENTS.md 缺口条目已更新")
else:
    print("警告：EXPERIMENTS.md 缺口条目未命中")

# 补记指标定标
ANCHOR = "### 已知缺口（写入 DECISIONS.md D-018）"
CALIB = """### 指标定标：1.41 到底算什么水平

把模型的预测句**原样保留、只打乱它和参考的配对关系**（流畅度不变，
只抹掉内容对应），得到"通顺但无关"的地板分。`scripts/calibrate_metrics.py`：

| 配对方式 | BLEU-4 | R-1 | R-2 | R-L |
|----------|-------:|----:|----:|----:|
| ① 真实配对（模型，seed 1234） | **1.55** | 22.89 | 3.67 | **21.83** |
| ② 打乱配对（5 次均值，地板） | 0.48 | 15.08 | 1.24 | 14.25 |
| ③ 真人参考句互相打乱 | 1.21 | 15.30 | 1.63 | 14.36 |
| ④ 完美预测（上限） | 100 | 100 | 100 | 100 |

**结论是双面的，报告时两面都要讲：**

- ✅ 模型确实在用视频信息 —— 各项都稳定高于地板（BLEU 3.2 倍，R-L +7.6），
  与 D-015 的全零消融互相印证
- ⚠️ **ROUGE-L 20.86 的绝对值有误导性**：地板就有 14.25。
  报告 ROUGE 时必须同时给出地板值，否则读者会高估
- ⚠️ BLEU 上模型（1.55）只略高于"随机拿一句真人句子"（1.21）

被问"1.41 算什么水平"时的答案：**是地板的 3.2 倍，但离可用还很远** —— 有数据，不是感觉。

"""
if ANCHOR in s and "指标定标" not in s:
    s = s.replace(ANCHOR, CALIB + ANCHOR)
    print("EXPERIMENTS.md 已补记指标定标")

shutil.copy(EXP, EXP + ".bak")
io.open(EXP, "w", encoding="utf-8").write(s)

# ---- STATE.md ----
ST = "/root/autodl-tmp/slt/docs/STATE.md"
t = io.open(ST, encoding="utf-8").read()
a, b = t.find("## 三、正在跑什么"), t.find("## 四、下一步")
NEWRUN = """## 三、正在跑什么

**没有后台任务在跑。**

阶段 0（E-000）已完成并结案：test **BLEU-4 1.41 ± 0.15**，
ROUGE-L 20.86 ± 0.69（lr=3e-4，epochs=100，batch=16，3 seed）。
lr 搜索的边界缺口已于 2026-09-15 扩展并结案（D-018），结论是保留 3e-4，
test 未二次曝光。

详见 `docs/EXPERIMENTS.md` 的 E-000 行与实验日志明细。

"""
if a != -1 and b != -1:
    t = t[:a] + NEWRUN + t[b:]
    print("STATE.md 运行状态已更新")

OLDBLOCK = """### 🔴 阻塞阶段 1：时空编码器吃 RGB，我们的输入是关键点"""
NEWBLOCK = """### ✅ 已解决：模态路线（原阻塞阶段 1）

**2026-09-15 由本人拍板（D-019）：主线走全程 pose（路线 A）**，
四阶段输入模态一律是关键点；阶段 3 重新解释为"把 538 维拆成多条流
（身体/双手/面部）各自编码 + 跨模态融合"。
四阶段跑完后，再在**阶段 2 的配置下**做一次 RGB 附加实验（可砍项），
明确标注为模态与编码器的合并效应，不做单变量归因。

**仍待确定**：阶段 1 具体用哪个骨架预训练编码器，需调研。

### 原记录（供溯源）：时空编码器吃 RGB，我们的输入是关键点"""
if OLDBLOCK in t:
    t = t.replace(OLDBLOCK, NEWBLOCK)
    print("STATE.md 阻塞条目已更新")

t = re.sub(r"最后更新：[0-9\- :]+", "最后更新：2026-09-15 12:00", t, count=1)
shutil.copy(ST, ST + ".bak")
io.open(ST, "w", encoding="utf-8").write(t)
print("STATE.md 已写入")
