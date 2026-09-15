# 手语翻译项目（SLT）

## 项目目标
重做六年前的手语翻译本科项目，用当前技术栈（gloss-free + LLM 解码器）。
用途：求职作品集。核心价值是"同一问题隔六年用两代技术各做一次"的纵向对比。

## 数据集：CE-CSL
- 来源仓库：/root/autodl-tmp/slt/TFNet （https://github.com/woshisad159/TFNet）
- 视频：1920x1080 / 30fps / 平均约 6.2 秒约 186 帧（dev 全量 515 条实测：均值 186，中位 181，范围 81-464） / H.264 mp4
- 目录结构：CE-CSL/video/{train,dev,test}/{translator}/{视频名}.mp4
- translator 对应 CSV 的 Translator 列（A/B/...），pose 输出必须沿用同一结构
- 标注：TFNet/data/CE-CSL/{train,dev,test}.csv
  列：Number,Translator,Chinese Sentences,Gloss,Note
- 划分：train 4973 / dev 515 / test 500
- 有 Chinese Sentences 列，可支撑 SLT（不只是 CSLR）

## 预处理路线（重要）
不要走 TFNet 原脚本的 RGB 路线（解码成 256x256 jpg，95 万帧、几十 GB）。
直接从视频提关键点，每个视频存一个 .npy，全量约 1.2 GB
（dev 实测 103 MB / 515 条；相比 RGB 路线的约 22 GB 小约 18 倍）。
参考 CE-CSLDataPreProcess.py 的目录遍历和统计逻辑，但替换核心处理步骤。
原脚本两个问题：内外层循环变量 i 重名；imageio 读出已是 RGB 却又做了
BGR2RGB 转换，通道顺序需自行验证。

## 架构：三层
- B 层（核心）：服务端 gloss-free 翻译。冻结视觉编码器 + 冻结 LLM，
  只训中间投影层（LLaVA 式）。解码器目标量级 mT5-base（24GB 显存可承载）
- C 层：端侧 MediaPipe 提关键点 → WebSocket 传关键点流（非视频）→
  服务端推理。卖点是带宽与隐私
- D 层：agent 壳。流式部分结果、上下文语义纠错、轮次管理

## 四阶段消融计划（核心产出）
不要一步到位替换，按阶段改，每阶段测 BLEU-4 + ROUGE：
- 阶段 0：老架构 baseline（CNN 逐帧特征 + LSTM encoder-decoder seq2seq）
- 阶段 1：只换视觉编码器（CNN → 冻结预训练时空编码器），解码器不动
- 阶段 2：只换解码器（LSTM → LLM + 可训练投影层），视觉端不动
- 阶段 3：加 pose 第二条流 + 跨模态融合
最终产出一张四行消融表。这比最终数字更重要。

注意：TFNet 本身是 CSLR（识别）取向，用 WER 评估、CTC 解码。
本项目是 SLT（翻译），用 BLEU/ROUGE。TFNet 的模型部分基本要重写，
只复用数据处理和目录约定。

## 老架构的四个已知缺陷（面试要讲的内容）
1. LSTM 长程依赖弱，而手语语法线索（眉毛、摇头等非手部特征）跨越整句
2. 逐帧 CNN 是晚融合，空间信息先被压扁才轮到时序
3. LSTM 解码器无语言先验，小数据量学不动目标语言语法
4. 单 RGB 流吃不下手/身/面的不同粒度

## 评估纪律
- 必须用标准划分的测试集，严禁拿训练分布内样本做演示
- 指标：BLEU-4、ROUGE
- gloss-free 普遍打不过 gloss-based，数字偏低是正常的，如实报告
- 参考量级（gloss-based test BLEU-4，2026-09-14 核定）：
  PHOENIX-2014T 最高 28.42 (TextCTC-SLT)，CSL-Daily 最高 25.79 (TwoStream-SLT)
  注意：这些是 gloss-based 且跑在别的数据集上，与本项目的 CE-CSL gloss-free
  数字不可直接比较，只作量级参照。完整榜单见 docs/EXPERIMENTS.md

## 分工约定（重要）
Codex 全权负责：dataloader、预处理、pose 提取管线、评测脚本、
C 层全部、D 层全部、文档、实验记录整理

用户必须逐行读懂（写完要讲解，不懂就解释到懂）：
- 模型定义，尤其投影层如何把视觉表征映射进 LLM 语义空间
- 训练循环与 loss
- 评测指标的计算逻辑
- 数据对齐方式
理由：项目用于面试，答不上来的项目比没有项目更糟。

## 环境
- AutoDL 北京B区，RTX 4090 24GB，16 核，90GB 内存
- PyTorch 2.3.0 / Python 3.12 / CUDA 12.1
- 工作目录 /root/autodl-tmp/slt（数据盘 250GB）
- 每次新会话先 `source /etc/network_turbo` 才能访问 GitHub/HuggingFace
- 注意：开了加速后 pip 会变慢，装包前先 unset http_proxy https_proxy
- 长任务一律在 tmux 中运行（SSH 从墨尔本连北京，断线是常态）
- 实例到期 2026-09-28（已于 2026-09-14 续费一周）

## 硬约束
- 数据集视频不得再分发，公开 demo 只能用自录片段
- 依赖版本偏保守（TFNet 锁 torch 1.11 + ctcdecode），遇冲突优先降级适配
  而非升级；但不必跑通 TFNet 本身，只取其数据处理部分

## 文档纪律
- docs/DECISIONS.md：每个技术决策追加一条，含数据依据、代价、可能的追问
- docs/EXPERIMENTS.md：每次训练评测追加一行
- 这两份文件的更新是任务完成的一部分，不是可选项
- docs/STATE.md：当前状态、正在跑什么、下一步、未解决的阻塞
- **每个新会话最先读 docs/STATE.md**，再读本文件与上述两份文档
