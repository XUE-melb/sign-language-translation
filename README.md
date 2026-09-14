# 手语翻译（SLT）：同一问题，隔六年，两代技术

把六年前的本科手语翻译项目用当前技术栈重做一遍。

**核心产出不是最终 BLEU 分数，而是一张四阶段消融表** —— 从 2019 年的
CNN + LSTM seq2seq，到 gloss-free + 冻结 LLM 解码器，每次只换一个组件，
量化每一代技术各自贡献了多少。

> 状态：**进行中**。数据预处理已完成，模型尚未开始训练。
> 下表为空即表示尚无结果 —— 不会先填数字再补实验。

## 任务

中文手语视频 → 中文句子。**SLT（翻译）而非 CSLR（识别）**，
用 BLEU-4 / ROUGE 评估，不用 WER。走 **gloss-free** 路线，不依赖 gloss 中间标注。

## 四阶段消融

| 阶段 | 改动 | BLEU-4 | ROUGE |
|------|------|--------|-------|
| 0 | 老架构 baseline：逐帧 CNN + LSTM encoder-decoder | — | — |
| 1 | 只换视觉编码器：CNN → 冻结预训练时空编码器 | — | — |
| 2 | 只换解码器：LSTM → LLM + 可训练投影层 | — | — |
| 3 | 加 pose 第二条流 + 跨模态融合 | — | — |

完整记录见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## 数据

**CE-CSL**，5,988 条（train 4,973 / dev 515 / test 500），12 位手语者，
1920×1080 / 30fps。

⚠️ **本仓库不包含数据集视频**，数据集不得再分发。

**评估协议**：采用官方划分，这是 **signer-dependent** 的 ——
12 位手语者在 train/dev/test 全部出现，测试集中训练未见过的人为 0/12。
分数因此天然偏乐观，理由与取舍见
[`docs/DECISIONS.md`](docs/DECISIONS.md) D-009。

## 预处理：关键点而非 RGB 帧

不解码 RGB 帧（约 111 万帧、约 22 GB），直接提 MediaPipe 关键点，
每视频一个 `.npy`，全量约 1.2 GB（约 18 倍压缩）。

每帧 538 维：pose 33×4 + 左手 21×3 + 右手 21×3 + 面部子集 92×3 + mask 4。
布局写在 `pose/layout.json`，下游按它切片。

两个已验证的坑：
- **通道顺序**：喂 BGR 而非 RGB，手部检出率掉 18 个百分点，
  而 pose / face 毫无变化 —— 不会报错，只会表现为"效果不好"（D-004）
- **手部检出率低不是 bug**：部分手语者的拍摄取景导致手在静止时垂出画面，
  是数据固有属性，调模型参数修不动（D-005）

## 文档

| 文件 | 内容 |
|------|------|
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 决策日志。每条含数据依据、否决的方案、代价与风险 |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | 实验记录，最终形态即四阶段消融表 |
| [`docs/preprocessing_notes.md`](docs/preprocessing_notes.md) | 预处理实验记录 |
| `CLAUDE.md` | 项目约定与环境说明 |

决策日志是**边做边记**的，不是事后补写。

## 脚本

```bash
# 关键点提取（12 并行，可断点续跑）
python scripts/extract_pose.py --split dev --workers 12

# 验证喂给 MediaPipe 的通道顺序
python scripts/extract_pose.py --split dev --verify-channel-order

# 提取质检：帧数 / 各部位检出率 / 存储量 / 按手语者聚合
python scripts/qc_pose.py --split dev

# 核对划分完整性与手语者重叠
python scripts/check_split.py dev
python scripts/check_signer_overlap.py
```

pose 提取跑在独立 venv（mediapipe 与训练环境的 numpy 版本冲突，
原因见 D-003），与训练环境只通过 `.npy` 交互。

## 环境

RTX 4090 24GB / PyTorch 2.3.0 / CUDA 12.1 / Python 3.12
