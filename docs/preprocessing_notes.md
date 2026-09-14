# 预处理实验记录

2026-09-14

## 环境隔离

pose 提取用独立 venv：`/root/autodl-tmp/slt/.venv-pose`，**不要装进训练环境**。

原因：mediapipe 最新版（1.0.1）会把 numpy 从 1.26.4 升到 2.5.3，而 torch 2.3.0
是按 numpy 1.x 编译的，升上去训练环境直接废。按 CLAUDE.md「优先降级适配」的
约束，pose 侧单独隔离。两边只通过 `.npy` 文件交互，不需要共存。

装法（`scripts/fix_pose_env.sh`）：
```
pip install --no-deps mediapipe==0.10.14
pip install "numpy<2" "protobuf>=4.25.3,<5" absl-py attrs flatbuffers \
            "opencv-contrib-python<5" sounddevice matplotlib
```

`--no-deps` 是关键。mediapipe 声明依赖 jax/jaxlib，但 legacy
`mp.solutions.holistic` 根本不 import 它们（只有 genai/model_maker 路径才用）。
不加 `--no-deps` 时 pip 会陷入版本回溯，反复下载上百 MB 的 jaxlib 轮子，
15 分钟没跑完，缓存涨到 1.1G。实际只需要补 matplotlib——`drawing_utils`
在模块级 import 了它。

## 已验证：通道顺序

CLAUDE.md 提到原脚本 `CE-CSLDataPreProcess.py` 用 imageio 读出已是 RGB
却又做了一次 BGR2RGB。实测（`extract_pose.py --verify-channel-order`，
dev-00001 前 60 帧）：

| 喂法 | pose | 左手 | 右手 | face |
|------|------|------|------|------|
| RGB（正确） | 1.000 | 0.933 | 0.883 | 1.000 |
| BGR（错误） | 1.000 | 0.750 | 0.700 | 1.000 |

**pose 和 face 对通道顺序不敏感，手部差 18 个百分点。** 手恰恰是手语的核心
信号。这个错误不会报错，只会表现为「模型效果不好」，极难回溯到预处理。

本项目用 `cv2.VideoCapture`（确实返回 BGR），转换一次是正确的。

## 特征布局

`CE-CSL/pose/{split}/{translator}/{number}.npy`，形状 `(T, 538)` float16。
分段定义写在 `CE-CSL/pose/layout.json`，dataloader 按它切片，不要硬编码偏移。

| 段 | 区间 | 内容 |
|----|------|------|
| pose | [0:132] | 33 点 × (x, y, z, visibility) |
| left_hand | [132:195] | 21 点 × (x, y, z) |
| right_hand | [195:258] | 21 点 × (x, y, z) |
| face | [258:534] | 92 点 × (x, y, z) |
| mask | [534:538] | 4 个 0/1 标志 |

面部不取全部 468 点，只取眉毛/眼/唇（对应 CLAUDE.md 缺陷 #1 提到的非手部
语法线索）。索引从 mediapipe 的 `FACEMESH_*` 常量推导，不硬编码。

### mask 段为什么必须有

未检出的部位填 0，但 0 是合法的归一化坐标（图像左上角）。没有 mask，模型
无法区分「手不在画面里」和「手在左上角」。而「手垂下休息」本身携带语义
（句子边界、非签名段），不能和真实坐标混为一谈。

## 重要发现：手部检出率低不是 bug

dev 全量提取后，手部检出率均值只有 0.550 / 0.501，56% 的样本双手均值 < 0.5。
按 translator 拆开差异极大：

| T | 左手 | 右手 | 左腕在画内 | 右腕在画内 |
|---|------|------|-----------|-----------|
| A | 0.961 | 0.940 | 1.000 | 1.000 |
| F | 0.959 | 0.975 | 0.985 | 0.991 |
| K | 0.157 | 0.318 | 0.217 | 0.386 |
| H | 0.417 | 0.111 | 0.507 | 0.166 |

诊断（`scripts/diag_hands.py`，纯用已提取的 .npy，不重跑视频）：

- 肩宽与手部检出相关系数 **-0.319**（负相关）——「人太小所以手检不到」是错的
- 手腕 visibility 相关 **+0.863 / +0.793**
- 躯干高相关 **-0.462**——人在画面里越大，手越容易抬出画面

抽帧肉眼确认（`scripts/peek_frames.py`）：**K 是站姿 + 腰部以上取景，静止时
双手垂在画面下边缘之外**，手腕坐标恒定 y≈1.05（MediaPipe 在往画外外推）。
抬手入画的帧就能检到。A 是坐姿，手全程在签名空间内。

结论：低检出率**大部分是真实的「手不在画面里」**。调 `model_complexity`
或放大 ROI 修不动——手根本不在。不要为了刷高这个数字去改管线。

第二个失败模式是运动模糊（快速打手语时单帧糊掉）。

### 对建模的影响

- 拍摄取景在 translator 之间差异很大，这是数据集的固有属性
- 若要做 signer-independent 切分，H 和 I 各只有 9 条（train/dev/test 都是），
  样本量撑不起来
- 这正好是 CLAUDE.md 缺陷 #4「单 RGB 流吃不下手/身/面的不同粒度」的具体实例，
  阶段 3 加 pose 流时可以引用

## 与 CLAUDE.md 记载不符的地方

**帧数**：CLAUDE.md 记「平均约 158 帧」，dev 实测均值 **186**，中位 181，
范围 81–464。差异不小，建模时按实测值规划序列长度。

**存储量**：CLAUDE.md 预期「几百 MB」，实测 dev 103 MB / 515 条，
按 5988 条全量外推约 **1.2 GB**。仍远小于 RGB 路线的几十 GB，但不是几百 MB。
要压可以砍 face 段（-51%）或改 `--face-set none`。

## 性能

12 worker 并行，dev 515 条用时 19.8 分钟（单视频中位 26 秒）。
train 4973 条外推约 **3.2 小时**。
