# 当前状态与交接

**新会话从这里开始读。** 然后读 `CLAUDE.md`、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md`。

最后更新：2026-09-14 23:45

---

## 一、怎么连上去干活

```bash
ssh autodl                      # 已配置免密（公钥已装到服务器）
```

| 位置 | 路径 |
|------|------|
| 服务器项目 | `/root/autodl-tmp/slt` |
| 本地项目 | `D:\AI\slt-project` |
| GitHub | https://github.com/XUE-melb/sign-language-translation （公开） |

本地 → 服务器：用 `scp`。服务器 → 本地：`bash scripts/sync_from_server.sh`
（只同步文档和脚本，不同步数据）。

**实例 2026-09-28 到期。**

### ⚠️ 两个 Python 环境，别用混

| 用途 | 解释器 |
|------|--------|
| pose 提取（mediapipe） | `/root/autodl-tmp/slt/.venv-pose/bin/python` |
| 训练 / 评测（torch） | `/root/miniconda3/bin/python`（登录 shell 里就是 `python`） |

分开的原因见 `DECISIONS.md` D-003（mediapipe 要 numpy≥2，torch 2.3.0 要 numpy<2）。
两侧只通过 `.npy` 文件交互。

### ⚠️ ssh 执行命令必须用登录 shell

```bash
ssh autodl 'bash -lc "python -V"'      # 对
ssh autodl 'python -V'                 # 错：conda 不在 PATH，报 command not found
```

### 长任务一律放 tmux

会话 `work`，现有窗口：`download`、`pose`、`pipeline`。
SSH 从墨尔本连北京，断线是常态。

---

## 二、数据现状

| split | 视频下载 | 关键点提取 | 说明 |
|-------|---------|-----------|------|
| train | 4973/4973 ✅ | **4973/4973 ✅** | 已核对，零缺失零失败 |
| dev | 515/515 ✅ | **515/515 ✅** | 已核对，零缺失 |
| test | 500/500 ✅ | **500/500 ✅** | 已核对，零缺失 |

**数据准备阶段全部完成（2026-09-14 23:41）。** 全量 5988 条，pose 共 1.2 GB。
质检结论见 DECISIONS.md D-016：不剔除任何样本，train 与 dev 分布一致
（帧数均值 187 vs 186，手部检出率 train 略好）。

特征：`CE-CSL/pose/{split}/{translator}/{number}.npy`，`(T, 538)` float16。
分段布局在 `CE-CSL/pose/layout.json`，**dataloader 按它切片，不要硬编码偏移**。

```
pose       [  0:132]  33 点 × (x,y,z,visibility)
left_hand  [132:195]  21 点 × (x,y,z)
right_hand [195:258]  21 点 × (x,y,z)
face       [258:534]  92 点 × (x,y,z)
mask       [534:538]  4 个 0/1 标志（pose/左手/右手/face 是否检出）
```

---

## 三、正在跑什么

**没有后台任务在跑。** 数据准备已于 2026-09-14 23:41 全部完成
（`logs/pipeline_train.log` 末尾是 `PIPELINE DONE`）。

tmux 会话 `work` 里的 `download` / `pose` / `pipeline` 三个窗口均已结束，
可以复用。

## 四、下一步（按顺序）

### 1. 等 PIPELINE DONE，看质检报告
重点看 train 的手部检出率分布。dev 的基线是左手均值 0.550 / 右手 0.501。
**如果 train 差很多，超参和预期都要跟着调。**

### 2. 跑阶段 0 正式实验

```bash
ssh autodl 'bash -lc "cd /root/autodl-tmp/slt && \
  PYTHONPATH=src python -m slt.train \
    --train-split train --eval-split dev --epochs 60"'
```

**这一跑就是 `EXPERIMENTS.md` 的第一行 E-000。** 跑完必须：
- 用 best checkpoint 在 **test** 上评测（不是 dev）
- 追加一行到 `docs/EXPERIMENTS.md`，备注写明 **signer-dependent，官方划分**
- 把超参、曲线观察写进 EXPERIMENTS.md 的"实验日志明细"

超参尚未定（epoch / lr / batch）。冒烟用的是 `lr=1e-3, batch=16, epochs=80`，
仅供参考，不是调过的。

### 3. 在跑阶段 1 之前，必须先解决下面那个遗留冲突

---

## 五、未解决的问题（欠着的债）

### 🔴 阻塞阶段 1：时空编码器吃 RGB，我们的输入是关键点

阶段 1 要把视觉编码器换成"冻结的预训练时空编码器"，但主流的（VideoMAE 等）
吃 RGB 视频，而 D-002 已决定不落地 RGB 帧。两条路都有代价：

- 回头解码 RGB → 推翻 D-002 的存储结论，且阶段 0→1 同时变了**输入模态**和
  **编码器**两个变量，消融失去单变量性
- 找基于骨架的预训练编码器 → 保住单变量性，但这类模型的预训练规模和通用性
  远不如 RGB 视频模型

**没有答案。阶段 1 开工前必须定。** 详见 DECISIONS.md D-002、D-013。

### 🟡 该做而未做（成本低，价值高）

| 欠的事 | 出处 | 成本 |
|--------|------|------|
| 评测结果按 Translator 分层拆解 | D-009 | 已实现在 `metrics.py`，跑评测时记得看 |
| 统计 dev/test 的 OOV 率 | D-014 | 很低，几行代码 |
| `--normalize none` 的对照实验 | D-011 | 一次训练 |
| `frame_stride=1` 的对照实验 | D-011 | 一次训练 |

### 🟡 面试准备上的缺口

- **D-001 的追问「那你申请 CSL-Daily 了吗？」** 本人需想好怎么答
- `DECISIONS.md` 里 CSL-News 一行仍是 `[待核]`

---

## 六、这个项目的规矩（不是可选项）

1. **`docs/DECISIONS.md`：每个技术决策追加一条**，含数据依据（写具体数字，
   不写"效果更好"）、否决了哪些方案及理由、代价与风险、面试可能的追问。
2. **`docs/EXPERIMENTS.md`：每次训练评测追加一行。** 这张表的最终形态就是
   四阶段消融表，是本项目的核心产出。
3. **冒烟/调试的数字绝不写进 EXPERIMENTS.md。** `train.py` 在
   `train_split == eval_split` 时会强制告警。
4. **不确定的外部数字标 `[待核]`**，不要凭印象写。这份文档是用来面试溯源的，
   编一个听起来合理的理由比没有更糟。
5. **模型定义、训练循环与 loss、评测指标计算、数据对齐方式** —— 这四块
   本人要能逐行讲清楚，写完要讲解。其余（dataloader、预处理、评测脚本、
   C 层、D 层、文档）由 Claude 全权负责。

---

## 七、踩过的坑（省时间用）

| 坑 | 解法 |
|----|------|
| ssh 里套多层引号必挂 | 本地写好脚本 → `scp` → 远程执行。不要在 ssh 命令行里嵌套引号 |
| 长文件用 heredoc 会被截断 | 用 Write 工具落地本地文件再 scp |
| Bash 工具 10 分钟超时 | 长任务丢进 tmux，用轮询查状态 |
| `pgrep -f xxx` 会匹配到自己 | 结果要排除自身命令行 |
| pip 装 mediapipe 陷入版本回溯 | `--no-deps` 装，再手工补运行时依赖（D-003） |
| 改 CLAUDE.md 等关键文件 | 用带断言的脚本（原文必须唯一命中），并留 `.bak` |
