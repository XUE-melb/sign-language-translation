# 当前状态与交接

**新会话从这里开始读。** 然后读 `CLAUDE.md`、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md`。

最后更新：2026-09-16 10:30（北京时间）

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
| RTMPose 提取 | `.venv-rtmpose/bin/python` | **必须经 `scripts/run_rtmpose.sh`**（设 LD_LIBRARY_PATH，否则 onnxruntime 静默退回 CPU，慢 90 倍） |
| 训练 / 评测 | `/root/miniconda3/bin/python` | 登录 shell 里就是 `python`，需 `PYTHONPATH=src` |

### ssh 执行命令必须用登录 shell

`ssh autodl 'bash -lc "python -V"'` 对；`ssh autodl 'python -V'` 错（conda 不在 PATH）。

### 长任务一律放 tmux 会话 `work`

---

## 二、消融表现状（核心产出）

| 行 | 阶段 | test BLEU-4 | ROUGE-L | 状态 |
|----|------|---:|---:|------|
| E-000 | 0（MediaPipe 输入，旧协议） | 1.41 ± 0.15 | 20.86 | 历史记录，**不是第 0 行** |
| **E-000b** | 0（RTMPose 输入） | **1.13 ± 0.15** | 20.38 | ✅ 第 0 行 |
| **E-001** | 1 只换编码器 | **3.28 ± 0.36** | 27.44 | ✅ 配对 bootstrap 3/3 p<0.001 |
| E-002 | 2 只换解码器 | — | — | **设计未定，阻塞** |
| E-003 | 3 多流融合 | — | — | 未开始 |

统一协议（D-017 + D-022 + D-023）：RTMPose lightweight 69 点，Uni-Sign 预处理，
frame_stride=1，max_frames=256，**lr=1e-4 恒定、无 scheduler**，epochs=100，batch=16，
3 seed（1234/2345/3456），dev 选 best.pt，test 只评一次，greedy max_len=60。

---

## 三、正在跑什么

**没有后台任务在跑。** 链路 09-16 04:00 完成（`logs/chain.log` 末尾 `CHAIN DONE`），
结果已全部写入 EXPERIMENTS.md / DECISIONS.md D-023。GPU 空闲。

---

## 四、下一步：阶段 2 设计（需要本人拍板的三件事）

阶段 2 = 只换解码器（LSTM → mT5 + 可训练投影层），视觉端不动（= E-001 的冻结 Uni-Sign 编码器）。
投影层是 CLAUDE.md 点名要本人逐行讲的部分。

| 决定 | 选项 | 倾向与理由 |
|------|------|-----------|
| ① mT5 来源 | (a) Uni-Sign checkpoint 里的 mT5（已在 CSL-News 手语→文本上调过）<br>(b) HF 原版 mT5-base | 倾向 (b)：项目论点是"两代技术对比"，不是复现 Uni-Sign；用 (a) 则"换了解码器"和"借了别人训好的翻译器"分不开。但 (a) 数字会好看得多，本人拍板 |
| ② 冻结还是 LoRA | (a) 冻结 mT5 只训投影层（CLAUDE.md 原文）<br>(b) LoRA | 先跑 (a) 作 E-002，不行再加 (b) 作 E-002b，不需要现在定死 |
| ③ 投影层接法 | 阶段 1 编码器输出 1024 → 投影 → mT5 d_model 768 | Uni-Sign 的 `pose_proj` Linear(1024→768) 已在权重里可作起点（若选 ① (b) 则只借结构不借权重） |

前置工程：训练环境要装 `transformers`（注意 pip 前 unset proxy，D-003 的教训）；
HF 下载 mT5-base 要 `source /etc/network_turbo`。

**解码方式**（greedy / beam）要在阶段 2 前专门定一次，写 D 条目。目前所有结果都是 greedy max_len=60。

---

## 五、已完成的代码资产

| 文件 | 作用 |
|------|------|
| `src/slt/models/decoder.py` | 所有阶段共用的 LSTM+attention 解码器（D-021） |
| `src/slt/models/stage0.py` | 逐帧 MLP（含 LayerNorm，D-022）+ BiLSTM |
| `src/slt/models/stage1.py` | 冻结 Uni-Sign 编码器 + 同一解码器；均值池化初始化 |
| `src/slt/models/unisign_encoder.py` | 只搭 Uni-Sign pose 分支（4.56M），不加载 mT5 |
| `src/slt/data_rtm.py` | Uni-Sign 预处理复刻，六项测试 |
| `src/slt/metrics.py` | BLEU-1..4 / chrF / ROUGE / 打乱配对地板 / 按手语者分层 |
| `src/slt/train.py` / `evaluate.py` | `--stage {0,1} --input {mediapipe,rtm}`；lr 逐 epoch 落盘 |
| `scripts/run_chain.sh` | 无人值守链（lr 复查 → 3 seed → test → 汇总） |
| `scripts/paired_bootstrap.py` | 配对 bootstrap 显著性 |

---

## 六、未解决的问题

### 阻塞阶段 2 开工
见第四节的三个决定。

### 记录在案、暂不修
- 零填充 vs Uni-Sign"重复末帧"：只污染阶段 1 每序列末 2 帧（D-022）
- 超长抽样确定性 linspace vs Uni-Sign 每轮随机（D-022，有意偏离）
- LayerNorm 的单独效应未测（D-023）；`--normalize none` / stride 对照未做
- F、I 两位手语者 BLEU-4 在阶段 1 持平而 ROUGE-L 上升，原因未查（D-023）
- C 层（端侧 MediaPipe）与 B 层输入（RTMPose 格式）不一致，未定
- RTMPose 对画外手会外推 conf>0.3，D-005 的叙事应改成"依赖提取器"

### 欠的账
| 事项 | 出处 | 状态 |
|------|------|------|
| OOV 率 dev 2/4880、test 6/5383 | D-014 | 待本人复核后关账 |
| CSL-News 均 9.5s / 40 字 | D-001 | **[待核]**，本人未核 |
| 「申请 CSL-Daily 了吗」 | D-001 | 本人待答 |
| 三个环境无 requirements 文件 | — | 待补 |

---

## 七、这个项目的规矩（不是可选项）

1. `docs/DECISIONS.md`：每个技术决策追加一条，含数据依据、否决方案、代价、面试追问。
2. `docs/EXPERIMENTS.md`：每次训练评测追加一行。
3. 冒烟数字绝不进 EXPERIMENTS.md。
4. 外部数字标 `[待核]`，不凭印象写；**不替本人编决策理由**。
5. 模型定义、训练循环与 loss、评测指标、数据对齐 —— 本人要能逐行讲。
6. **预判先写下再看结果**（D-020 → D-023 的教训：被推翻的预判比事后解释可信）。

---

## 八、踩过的坑

| 坑 | 解法 |
|----|------|
| ssh 套多层引号 / heredoc 超长 | 本地用 Write 落脚本 → scp → 远程执行 |
| Bash 工具 10 分钟超时 | 长任务进 tmux，轮询查 |
| `pgrep -f` / `pkill -f` 自匹配 | `pkill -f 'xx[x]'` 或按 PID 杀 |
| onnxruntime 报 CUDA 可用却跑 CPU | 看 `session.get_providers()` 或直接看耗时 |
| 对齐第三方模型只读论文/摘要 | 读它的 dataloader 源码：mode、预处理、帧采样各踩一次（D-020/021/022） |
| 挂在噪声指标上的 scheduler | 不用；lr 逐 epoch 落盘（D-022） |
| 用参数量代理模型能力 | 预训练数据量才是变量（D-023） |
