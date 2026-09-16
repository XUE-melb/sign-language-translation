# 当前状态与交接

**新会话从这里开始读。** 然后读 `CLAUDE.md`、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md`。

最后更新：2026-09-16 17:00（北京时间）

---

## 一、怎么连上去干活

```bash
ssh autodl                      # 已配置免密
```

| 位置 | 路径 |
|------|------|
| 服务器项目 | `/root/autodl-tmp/slt` |
| 本地项目 | `D:\AI\slt-project`（git 仓库在本地；README 只在本地） |
| GitHub | https://github.com/XUE-melb/sign-language-translation （公开） |

本地 → 服务器：`scp`。服务器 → 本地：`bash scripts/sync_from_server.sh`（只同步文档和脚本；
`scripts/__pycache__` 会报一条无害错误）。

**实例 2026-09-28 到期。** 所有产物必须在 09-27 前撤到本地。

### 三个 Python 环境，别用混

| 用途 | 解释器 | 启动方式 |
|------|--------|---------|
| MediaPipe 提取 | `.venv-pose/bin/python` | 直接调 |
| RTMPose 提取 | `.venv-rtmpose/bin/python` | **必须经 `scripts/run_rtmpose.sh`** |
| 训练 / 评测 | `/root/miniconda3/bin/python` | 登录 shell 里就是 `python`，需 `PYTHONPATH=src` |

训练环境：transformers **4.57.6**（钉 <5，5.x 要 torch ≥ 2.5）/ peft 0.20.0 / sentencepiece。

### 本地机（demo 部署目标）

RTX 5060 8 GB、Core Ultra 7 265K、47 GB 内存、Python 3.13。需要单独的推理 venv（3.11/3.12，torch ≥ 2.7）。
已传回本地：`weights/unisign/csl_daily_pose_only_slt.pth`（1.2 GB）、`weights/mt5-base/` 的 config 与 spiece、
`runs/E002_lora16_s1234/best/`（17 MB）。均 gitignored。

### ssh 执行命令必须用登录 shell；长任务一律放 tmux 会话 `work`

---

## 二、消融表现状（核心产出）

| 行 | 阶段 | test BLEU-4 | ROUGE-L | 状态 |
|----|------|---:|---:|------|
| E-000 | 0（MediaPipe 输入，旧协议） | 1.41 ± 0.15 | 20.86 | 历史记录 |
| **E-000b** | 0（RTMPose 输入） | **1.13 ± 0.15** | 20.38 | ✅ 第 0 行 |
| E-001 | 1 冻结 stage-1 编码器 | 3.28 ± 0.36 | 27.44 | 附加行："同域微调对编码器值多少" |
| **E-001b** | 1 冻结 **CSL-Daily** 编码器 | **3.76 ± 0.29** | 29.61 | ✅ 第 1 行，bootstrap 3/3 p<0.001 |
| **E-002** | 2 只换解码器：CSL-Daily mT5 + LoRA r=16 + pose_proj，30 轮 | **16.40 ± 0.38** | 46.92 | ✅ 第 2 行，bootstrap 3/3 p<0.001；beam4 17.40 |
| E-002 60 轮 | 同上，延长跑 | — | — | 🔄 跑中，约 09-16 22:00；表的替换规则见 D-025 |
| E-003 | 3 放开编码器端到端 | — | — | 09-17 起，D-026 |
| E-004 | 留一手语者 E（不进消融表） | — | — | 09-17，D-026 |

参照：Uni-Sign CSL-Daily 版**零样本** dev 3.09 → E-002 dev 峰值 16.2，差值是本项目的适配贡献。

---

## 三、正在跑什么

**`tmux work:ext` 跑 `scripts/ext_e002.sh`**（09-16 15:48 启动）：E-002 同配置 60 轮 × 3 seed，
顺序 1234 → 2345 → 3456，每 seed 约 2 小时 + test（greedy + beam4）。日志 `logs/ext_e002.log`，
产物 `runs/E002_lora16_ep60_s*`，结束打 `EXT DONE`（约 22:00）。要停：`tmux kill-window -t work:ext`。

查进度：`ssh autodl 'grep "^\[" /root/autodl-tmp/slt/logs/ext_e002.log | tail -3'`

**跑完后要做**：按 D-025 的规则决定 60 轮行是否替换 30 轮行（均值高出 > 0.38 则替换，30 轮行保留标注）；
入 EXPERIMENTS.md；定 E-003 的 epochs。

主链与后处理已完成：`logs/stage2_chain.log`（STAGE2 CHAIN DONE 15:37）、`logs/post_stage2.log`（POST DONE 15:48）、
`runs/stage2_summary.md`。

---

## 四、双线计划（D-027）

**接口契约**：`translate(keypoints: (T, 207)) -> [(text, logprob), ...]`。软件线以此开工，不等模型线。

| 日期 | 模型线（远程 4090） | 软件线（本地 5060） |
|---|---|---|
| 09-17 | E-003 lr 探针 {1e-4, 1e-5} → 3 seed；E-004 留 E | 推理封装：合并 LoRA、导出置信度、新 torch；FastAPI |
| 09-18 | 四行表定稿、文档 | WebSocket + 骨架页面 + 上传视频模式 |
| 09-19 至 09-22 | 可补 E-004 变体（E-003 配置 / 留 A） | agent 工具循环（Anthropic SDK 手写）+ 两行评测 |
| 09-23 至 09-25 | | 本人自录三五句、跟打模式、README |
| 09-26 至 09-27 | 撤出实例 | 录像、收尾 |

本人并行做：跟数据集视频学三五句手语（句子待 E-002 预测里挑）；读模型讲解；过 DECISIONS 的追问清单。

---

## 五、D-025 / 026 / 027 要点（新会话必读）

- **D-025**：E-002 30 轮未收敛（峰值在 ep29/30），预登记规则触发 → 60 轮延长跑。表的规则写在结果前。
  原版 mT5 对照行本人砍掉（秋招时间压力）。
- **D-026**：阶段 3 = 放开编码器（单变量：编码器是否更新）。E-004 留一手语者 E，三格表拆开
  "陌生人"与"陌生句子"两个效应；决定 demo 摄像头模式开不开。数据集是同一套约 600 句每人打一遍、划分按句子切。
- **D-027**：交付 = README + 录像 + 本地 Web 应用（FastAPI + 单页 HTML）。端侧提取器改 rtmlib。
  agent 只做"管理不确定性"：LLM 只在 n-best 里选；两行评测（重排 BLEU、门控曲线）。Claude API。

---

## 六、代码资产

| 文件 | 作用 |
|------|------|
| `src/slt/models/unisign_full.py` | Uni-Sign 完整管线（编码器 + pose_proj + mT5），可指定编码器来源 |
| `src/slt/train_stage2.py` | 阶段 2 训练 / `--test-only` 评测（greedy + beam4）；无 resume |
| `scripts/run_stage2_chain.sh` / `post_stage2.sh` / `ext_e002.sh` | 路线 B 主链 / 后处理 / 60 轮延长 |
| `scripts/paired_bootstrap.py` | 配对 bootstrap（E-002 的预测文件名带 `_greedy` 后缀，post 脚本已建软链） |
| `tests/probe_zeroshot.py` / `probe_control.py` | 零样本探针 / 全零-打乱对照 |
| 其余 | 见 D-021/D-022，未变 |

待写（软件线）：`src/slt/infer.py`（合并 LoRA、置信度）、`server/`（FastAPI + WebSocket + 页面）、
`client/`（rtmlib 关键点流）、`agent/`（工具循环 + 评测脚本）。

---

## 七、未解决的问题

### 阻塞
- E-003 编码器 lr 未定（探针后定）；epochs 跟随 D-025 结论。

### 记录在案、暂不修
- 原版 mT5（路线 (b)）已砍；只训 pose_proj 无 LoRA、label_smoothing 0.2 的对照未做
- E-000b/E-001 的 LayerNorm 单独效应未测；`--normalize none` / stride 对照未做
- 手语者 D 在两个阶段都最高（E-002 32.8），句子是否更短未查；H/I n=9 不下结论
- 零填充 vs Uni-Sign"重复末帧"（只污染阶段 1 每序列末 2 帧）
- `weights/mt5-base` 6.6G（含 tf/flax），只用 config + spiece.model，可清
- 隧道（Cloudflare/ngrok）国内可达性未测

### 欠的账
| 事项 | 状态 |
|------|------|
| OOV dev 2/4880、test 6/5383 | 待本人复核后关 D-014 |
| CSL-News 均 9.5s / 40 字；PHOENIX/CSL-Daily 划分 signer-dependent | [待核] |
| 「申请 CSL-Daily 了吗」 | 本人待答 |
| 三个环境无 requirements 文件 | 待补，软件线一并做 |

---

## 八、规矩（不是可选项）

1. 每个技术决策进 `DECISIONS.md`，每次实验进 `EXPERIMENTS.md`；冒烟数字不进表。
2. 外部数字标 `[待核]`；不替本人编决策理由。
3. 模型定义、训练循环与 loss、评测指标、数据对齐 —— 本人要能逐行讲。
4. 预判和表的处理规则先写下再看结果（D-020→D-023，D-025）。
5. 借来的东西接进流水线前先零样本测一次（D-024）。
6. 有新想法先问"对秋招展示有没有直接价值"；不主动扩 scope。

---

## 九、踩过的坑

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
| 训练预算拍脑袋 | 先跑短的看曲线，延长规则预先写下（D-025） |
