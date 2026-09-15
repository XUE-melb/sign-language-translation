# 当前状态与交接

**新会话从这里开始读。** 然后读 `CLAUDE.md`、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md`。

最后更新：2026-09-15 22:30（北京时间）

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

原因：D-003（mediapipe 与 torch 的 numpy 冲突）、D-020（onnxruntime-gpu 1.19.2 + cuDNN 9 是唯一能走 GPU 的组合）。

### ssh 执行命令必须用登录 shell

`ssh autodl 'bash -lc "python -V"'` 对；`ssh autodl 'python -V'` 错（conda 不在 PATH）。

### 长任务一律放 tmux 会话 `work`

---

## 二、数据现状：两套关键点，用途不同

| 数据 | 路径 | 格式 | 用途 |
|------|------|------|------|
| MediaPipe | `CE-CSL/pose/` | `(T, 538)` float16 `.npy`，布局在 `pose/layout.json` | E-000（历史基线）、**C 层端侧** |
| **RTMPose** | `CE-CSL/pose_rtm/` | `(T, 133, 2)` + `(T, 133)` `.pkl`，Uni-Sign 格式 | **主消融表全部四行**（E-000b 起） |

两套都是全量 5988 条，零缺失（train 4973 / dev 515 / test 500）。
RTMPose 用 `lightweight` mode（不是论文写的 RTMPose-x，见 D-020 的证据）。
`src/slt/data_rtm.py` 复刻 Uni-Sign 的预处理：69 点分组、**只有 body 走 crop_scale**、
手/脸共享其 scale（D-021 修正过一次实质错误，现已逐行对齐源码）。

---

## 三、正在跑什么

**`tmux work:chain` 正在跑 `scripts/run_chain.sh`，2026-09-15 22:04 启动**，
这是 **D-022 纠错后**的重跑。链路：lr 复查（3 个 lr，dev-only）→ E-000b（阶段 0，3 seed）
→ E-001（阶段 1，3 seed）→ test 各评一次 → 汇总。

实测单 epoch 约 29s，全链约 **6.5 小时**，预计北京时间 **09-16 04:30** 出结果。
查进度：`ssh autodl 'grep "^\[" /root/autodl-tmp/slt/logs/chain.log | tail'`

跑完日志末尾是 `CHAIN DONE`，前面有可直接粘进 EXPERIMENTS.md 的两行。
**跑完后要做**：把 E-000b / E-001 写进 EXPERIMENTS.md；检查 `summarize_chain.py`
输出的"差值 vs 合成标准差"判定，**不许把噪声内的差异写成提升**。

---

## 四、D-022：审视后停链重跑（重要，新会话必读）

2026-09-15 晚用另一个模型对项目做独立审视，发现三处协议缺陷，全部经实测复核成立：

1. **ReduceLROnPlateau 把 lr 吃到 ~1/10⁶**，100 个 epoch 只有约 20 个在真正优化。已删掉 scheduler，lr 逐 epoch 落盘。
2. **rtm 输入无条件 `frame_stride=2`**，Uni-Sign 训练时是原生帧率。已改为 rtm 默认 stride=1。
3. **阶段 0 FrameEncoder 无输入归一化**，而 rtm 各段尺度差 5-8 倍、阶段 1 内置 BN。已加 LayerNorm。

E-000b 三 seed + E-001 部分结果**已作废删除**，在新协议下重跑（即上面正在跑的链）。
**E-000（MediaPipe 基线）也受缺陷 ① 影响**，保留但已加注，它本就不是消融表的第 0 行。

---

## 五、已完成的代码资产

| 文件 | 作用 | 状态 |
|------|------|------|
| `src/slt/models/decoder.py` | 所有阶段共用的 LSTM+attention 解码器 | D-021，两阶段解码器逐张量验证相同 |
| `src/slt/models/stage0.py` | 逐帧 MLP（**含 LayerNorm**，D-022）+ BiLSTM | 就绪 |
| `src/slt/models/stage1.py` | 冻结 Uni-Sign 编码器 + 同一解码器；均值池化初始化 | 权重 missing 0 / unexpected 0 |
| `src/slt/models/unisign_encoder.py` | 只搭 Uni-Sign pose 分支，不加载 966M 的 mT5 | 就绪 |
| `src/slt/data_rtm.py` | Uni-Sign 预处理复刻 | 六项测试通过 |
| `src/slt/train.py` / `evaluate.py` | `--stage {0,1} --input {mediapipe,rtm}` | 就绪 |
| `scripts/run_chain.sh` | 无人值守链 | 运行中 |

---

## 六、下一步（链跑完之后，按顺序）

1. 填 E-000b / E-001 进 EXPERIMENTS.md，判显著性
2. 便宜的评测增强：evaluate.py 加 chrF / BLEU-1..3 / 打乱配对地板；配对 bootstrap 脚本
3. **阶段 2 设计**（未定）：用 Uni-Sign checkpoint 里的 mT5 还是原版 mT5；冻结还是 LoRA；训练环境要装 transformers
4. 阶段 3、RGB 收尾实验（D-019）

---

## 七、未解决的问题

### 阶段 2 的设计还没定（阻塞阶段 2 开工）
mT5 来源（Uni-Sign 已调过的 vs 原版）、冻结 vs LoRA、投影层怎么接。`pose_proj`（Linear 1024→768）已在 Uni-Sign 权重里，可作起点。

### 记录在案、暂不修
- 零填充 vs Uni-Sign 的"重复末帧"：只污染阶段 1 每条序列最后约 2 帧（D-022）
- 超长序列抽样：我们确定性 `linspace`，Uni-Sign 每轮随机（D-022，有意偏离）
- 解码方式（greedy、max_len=60）**未写进任何 D 条目**，阶段 2 前要定 beam 与否
- C 层（端侧 MediaPipe）与 B 层输入（RTMPose 格式）不一致，未定
- RTMPose 对画外的手会外推出 conf>0.3 的点（K：MediaPipe 0.157 → RTMPose 0.79），D-005 的叙事应改成"依赖提取器"

### 欠的账
| 事项 | 出处 | 状态 |
|------|------|------|
| OOV 率 | D-014 | 审视时已量：dev 2/4880 (0.04%)，test 6/5383 (0.11%)，**待本人复核后关账** |
| test 比 train 略难（199 帧 vs 187；手部置信度 0.68 vs 0.71） | 质检 | 已写进 EXPERIMENTS |
| `--normalize none` / stride 对照 | D-011 | 未做 |
| CSL-News 规模 751,320 条 / 均 9.5s / 40 字 | D-001 | **[待核]**，本人未核 |
| D-001 追问「申请 CSL-Daily 了吗」 | D-001 | 本人待答 |
| 三个环境无 requirements 文件 | — | 待补 |

---

## 八、这个项目的规矩（不是可选项）

1. `docs/DECISIONS.md`：每个技术决策追加一条，含数据依据、否决方案、代价、面试追问。
2. `docs/EXPERIMENTS.md`：每次训练评测追加一行。
3. 冒烟数字绝不进 EXPERIMENTS.md。
4. 外部数字标 `[待核]`，不凭印象写；**不替本人编决策理由**。
5. 模型定义、训练循环与 loss、评测指标、数据对齐 —— 本人要能逐行讲。

---

## 九、踩过的坑

| 坑 | 解法 |
|----|------|
| ssh 套多层引号 / heredoc 超长 | 本地用 Write 落脚本 → scp → 远程执行；长文件绝不用 heredoc |
| Bash 工具 10 分钟超时 | 长任务进 tmux，轮询查 |
| `pgrep -f` / **`pkill -f` 自匹配** | `pkill -f 'xx[x]'` 或按 PID 杀；pkill 自匹配会把自己的 ssh 一起杀掉 |
| onnxruntime `get_available_providers()` 报 CUDA 可用却跑 CPU | 看 `session.get_providers()` 或直接看耗时 |
| 对齐第三方模型只读论文/摘要 | **读它的 dataloader 源码**：mode、预处理、帧采样每一步都可能有约定（D-020/021/022 各踩一次） |
| 挂在噪声指标上的 scheduler | 不用；lr 逐 epoch 落盘（D-022） |
