# 手语翻译（SLT）：同一问题，隔六年，两代技术

把六年前的本科手语翻译项目用当前技术栈重做一遍。

**核心产出不是最终 BLEU 分数，而是一张四阶段消融表** —— 从 2019 年的
CNN + LSTM seq2seq，到 gloss-free + 冻结 LLM 解码器，每次只换一个组件，
量化每一代技术各自贡献了多少。

> 状态：**四阶段消融完成**（各 3 seed，test 集 500 条），本地 demo 可运行；agent 层与录像进行中（见 docs/STATE.md）。

## 任务

中文手语视频 → 中文句子。**SLT（翻译）而非 CSLR（识别）**，
用 BLEU-4 / ROUGE 评估，不用 WER。走 **gloss-free** 路线，不依赖 gloss 中间标注。

## 四阶段消融

| 阶段 | 改动 | BLEU-4 | ROUGE |
|------|------|--------|-------|
| 0 | 老架构 baseline：逐帧 MLP（关键点输入，见 D-013）+ BiLSTM + attention-LSTM 解码器 | 1.13 ± 0.15 | 20.38 |
| 1 | 只换视觉编码器：→ 冻结的 Uni-Sign pose 编码器（CSL-Daily 微调版，D-020 / D-024） | 3.76 ± 0.29 | 29.61 |
| 2 | 只换解码器：LSTM → mT5-base（冻结）+ LoRA r=16 + 可训练投影层 | **16.79 ± 0.23** | **47.50** |
| 3 | 放开视觉编码器端到端适配（D-026；原多流定义因编码器已含多流融合而失效） | **18.93 ± 0.58** | **50.15** |

BLEU-4 为 3 seed 均值 ± std，ROUGE 列为 ROUGE-L，greedy 解码；beam=4 下阶段 2 为 17.74 ± 0.20、阶段 3 为 20.17 ± 0.58
（阶段 2、3 为 60 epoch；预登记协议 30 epoch 的 16.40 ± 0.38 也保留在表中，见 D-025）。
相邻两行配对 bootstrap：0→1、1→2 三 seed 均 p<0.001，2→3 两个 seed 显著（D-026）。
**留一手语者实验（E-004，D-026/D-028）**：训练时去掉一位手语者再在其片段上测。同片段对比，陌生手语者保留 43%（E）/ 78%（A）/ 83%（D）；
解冻编码器把 E 从 43% 提到 73%，关键点增广无效。去掉 LoRA 只训投影层：13.41（LoRA 值 +3.4）。
相邻两行配对 bootstrap（1000 次）三 seed 均 p<0.001。**阶段 1、2 借用了 Uni-Sign 在 CSL-Daily 上微调过的
编码器与 mT5**（D-024）：该系统不训练直接跑 CE-CSL 的零样本 dev BLEU-4 为 3.09，适配训练后 dev 16.2，
差值是本项目自己的贡献。gloss-free、单数据集、signer-dependent 官方划分，与文献榜单不可直接比较。

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

## 本地 demo（B / C / D 三层接线，D-027）

```
set PYTHONPATH=src
.venv-infer\Scripts\python -m uvicorn server.app:app --port 8000     # 浏览器打开 http://127.0.0.1:8000
```

| 层 | 实现 | 说明 |
|----|------|------|
| B 服务端 | `server/app.py`（FastAPI） | 常驻进程持有模型；`/api/translate`、`/api/translate_video`、WebSocket `/ws/stream` |
| C 端侧 | 页面内置回放 + 浏览器录制 | 关键点逐帧经 WebSocket 推送，**传坐标不传画面**；上传/录制模式由服务端 rtmlib 提关键点 |
| D agent | `agent/judge.py` + 页面 Agent 面板 | LLM 裁判只在 n-best 里选并判定是否追问（Claude / DeepSeek 双后端，规则版兜底）。test 集评测：重排 +2.07 BLEU-4（上限 +7.22），门控直接输出部分 30.5 vs 追问部分 15.1（D-027） |

页面三种输入：测试集骨架回放（私下展示用，只播关键点不播视频）、上传视频、摄像头录制。
推理封装 `src/slt/infer.py` 把 LoRA 合并进 mT5 主干，接口 `translate((T,207)) -> [(text, logprob)]`，
本地 RTX 5060 上每条约 0.2 s。设计系统由 UI/UX Pro Max 生成，见 `design-system/slt-demo/MASTER.md`。

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
