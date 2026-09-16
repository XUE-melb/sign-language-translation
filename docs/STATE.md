# 当前状态与交接

**新会话从这里开始读。** 然后读 `CLAUDE.md`、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md`。

最后更新：2026-09-16 11:00（北京时间）

---

## 一、怎么连上去干活

```bash
ssh autodl                      # 已配置免密
```

| 位置 | 路径 |
|------|------|
| 服务器项目 | `/root/autodl-tmp/slt` |
| 本地项目 | `D:\AI\slt-project` |
| GitHub | https://github.com/XUE-melb/sign-language-translation （公开） |

本地 → 服务器：`scp`。服务器 → 本地：`bash scripts/sync_from_server.sh`（只同步文档和脚本）。

**实例 2026-09-28 到期。**

### 三个 Python 环境，别用混

| 用途 | 解释器 | 启动方式 |
|------|--------|---------|
| MediaPipe 提取 | `.venv-pose/bin/python` | 直接调 |
| RTMPose 提取 | `.venv-rtmpose/bin/python` | **必须经 `scripts/run_rtmpose.sh`** |
| 训练 / 评测 | `/root/miniconda3/bin/python` | 登录 shell 里就是 `python`，需 `PYTHONPATH=src` |

训练环境已装 **transformers 4.57.6 / peft 0.20.0 / sentencepiece**。
**transformers 钉 <5**：5.x 要 torch ≥ 2.5，torch 2.3 不动（D-003）。

### ssh 执行命令必须用登录 shell；长任务一律放 tmux 会话 `work`

---

## 二、消融表现状（核心产出）

| 行 | 阶段 | test BLEU-4 | ROUGE-L | 状态 |
|----|------|---:|---:|------|
| E-000 | 0（MediaPipe 输入，旧协议） | 1.41 ± 0.15 | 20.86 | 历史记录 |
| **E-000b** | 0（RTMPose 输入） | **1.13 ± 0.15** | 20.38 | ✅ 第 0 行 |
| E-001 | 1 冻结 stage-1 编码器 | 3.28 ± 0.36 | 27.44 | ✅ bootstrap 3/3 p<0.001 |
| **E-001b** | 1 冻结 **CSL-Daily** 编码器 | — | — | 🔄 跑中（路线 B 要求编码器与 E-002 一致） |
| **E-002** | 2 只换解码器：CSL-Daily mT5 + LoRA r=16 + pose_proj | — | — | 🔄 排队 |
| E-003 | 3 | — | — | **定义待重议**（见 D-024 第五节） |

参照：Uni-Sign CSL-Daily 版**零样本**（不训练）dev BLEU-4 **3.09**；stage-1 版零样本 **0.27**（贴地板，只会说新闻）。

---

## 三、正在跑什么

**`tmux work:s2chain` 跑 `scripts/run_stage2_chain.sh`**（09-16 11:00 启动），
接管手动启动的 E-001b s1234（`tmux work:s2`）后依次：E-001b s2345/s3456 → 3 个 test 评测
→ E-002 3 seed（30 epoch，约 66 分钟/趟）→ test（greedy + beam4）→ 汇总。

预计 **09-16 16:30** 打出 `STAGE2 CHAIN DONE`。
查进度：`ssh autodl 'grep "^\[" /root/autodl-tmp/slt/logs/stage2_chain.log | tail'`

**跑完后要做**：E-001b / E-002 入 EXPERIMENTS.md；配对 bootstrap（`scripts/paired_bootstrap.py`，
注意 E-002 的预测文件名是 `predictions_test_greedy.csv`）；**单独报"零样本 3.09 → E-002"的差值**
（那是我们适配训练的贡献）；检查 E-002 dev 曲线到 30 轮是否还在涨。

---

## 四、D-024 的要点（新会话必读）

- 本人选定阶段 2 用 Uni-Sign 的 mT5（路线 (a)）。零样本探针表明 **stage-1 checkpoint 拿来贴地板**
  （域偏移，输出新闻），**CSL-Daily 微调版零样本 3.09**，故整套用 CSL-Daily 版 = 路线 B。
- 编码器因此也换成 CSL-Daily 版，**E-001 需重跑为 E-001b**，否则 E-001→E-002 同时换两样。
- 混搭（stage-1 编码器 + CSL-Daily 解码器）零样本 1.08，出局。
- E-002 与 D-017 协议的三处偏离已写明：epochs 30、batch 8、label_smoothing 0。
- **面试表述**：阶段 1、2 借了同一个同域微调过的预训练系统的两半；零样本→E-002 的差值才是我们的贡献。

---

## 五、代码资产

| 文件 | 作用 |
|------|------|
| `src/slt/models/unisign_full.py` | Uni-Sign 完整管线（编码器 + pose_proj + mT5），可指定编码器来源 |
| `src/slt/train_stage2.py` | 阶段 2 训练 / `--test-only` 评测（greedy + beam4） |
| `tests/probe_zeroshot.py` / `probe_control.py` | 零样本探针 / 全零-打乱对照 |
| `scripts/run_stage2_chain.sh` | 路线 B 无人值守链 |
| `scripts/paired_bootstrap.py` | 配对 bootstrap |
| 其余 | 见 D-021/D-022，未变 |

---

## 六、未解决的问题

### 阶段 3 定义（阻塞阶段 3）
Uni-Sign 编码器已含多流分组与融合，D-019 的阶段 3 定义失效。候选：放开编码器端到端适配 /
加 MediaPipe 92 点面部流 / RGB 流。等 E-002 出结果再定。

### 记录在案、暂不修
- 原版 mT5（路线 (b)）未实测，是一个可补的对照行
- E-000b/E-001 的 LayerNorm 单独效应未测；`--normalize none` / stride 对照未做
- F、I 在阶段 1 BLEU-4 持平、ROUGE-L 上升，原因未查
- 零填充 vs Uni-Sign"重复末帧"（只污染阶段 1 每序列末 2 帧）
- C 层（端侧 MediaPipe）与 B 层输入（RTMPose）不一致
- `weights/mt5-base` 拉了 6.6G（含 tf/flax），只用了 config + spiece.model，可清

### 欠的账
| 事项 | 状态 |
|------|------|
| OOV dev 2/4880、test 6/5383 | 待本人复核后关 D-014 |
| CSL-News 均 9.5s / 40 字 | [待核] |
| 「申请 CSL-Daily 了吗」 | 本人待答 |
| 三个环境无 requirements 文件 | 待补 |

---

## 七、规矩（不是可选项）

1. 每个技术决策进 `DECISIONS.md`，每次实验进 `EXPERIMENTS.md`；冒烟数字不进表。
2. 外部数字标 `[待核]`；不替本人编决策理由。
3. 模型定义、训练循环与 loss、评测指标、数据对齐 —— 本人要能逐行讲。
4. 预判先写下再看结果（D-020→D-023）。
5. **借来的东西接进流水线前先零样本测一次**（D-024）。

---

## 八、踩过的坑

| 坑 | 解法 |
|----|------|
| ssh 套多层引号 / heredoc 超长 | Write 落脚本 → scp → 远程执行 |
| Bash 工具 10 分钟超时 | 长任务进 tmux |
| `pgrep -f` / `pkill -f` 自匹配 | `'xx[x]'` 或按 PID |
| onnxruntime 报 CUDA 可用却跑 CPU | 看 `session.get_providers()` |
| 对齐第三方模型只读论文/摘要 | 读 dataloader 源码（D-020/021/022） |
| 挂在噪声指标上的 scheduler | 不用；lr 落盘（D-022） |
| 用参数量代理能力 | 预训练数据量才是变量（D-023） |
| 装 transformers 5.x 静默禁用 torch 2.3 | 钉 `transformers<5`（D-024） |
| 借来的 checkpoint 默认能用 | 先零样本测；两个 checkpoint 差 10 倍（D-024） |
