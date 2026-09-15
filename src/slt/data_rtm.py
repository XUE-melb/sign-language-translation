"""RTMPose (COCO-WholeBody 133) 数据加载，预处理严格复刻 Uni-Sign。

为什么要逐行复刻而不是"差不多就行"：阶段 1 用的是 Uni-Sign 的**冻结**
编码器，它见过的输入分布由这套预处理定义。冻结模型没有适应能力，
预处理差一点就是分布偏移。E-000b 也必须用同一套，否则阶段 0→1
会同时变动输入和编码器，单变量性破掉。

分组与中心化（来自 Uni-Sign datasets.py 的 load_part_kp）：
    body       [0] + [3..10]                      9 点，不做中心化
    left_hand  [91:112]                           21 点，减自身第 0 点（手腕）
    right_hand [112:133]                          21 点，减自身第 0 点（手腕）
    face       [23,25,...,39] + [83..90] + [53]   18 点，减自身最后一点（即 53）
                                                  ------
                                                  69 点

随后每组各自过 crop_scale 归一化到 [-1, 1]。
"""
import copy
import csv
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from slt.data import CharVocab            # 词表复用，保证与 E-000 同一份

# --- Uni-Sign 的 69 点索引（与 datasets.py 逐字对应）---
BODY_IDX = [0] + list(range(3, 11))                                   # 9
LH_IDX = list(range(91, 112))                                         # 21
RH_IDX = list(range(112, 133))                                        # 21
FACE_IDX = (list(range(23, 23 + 17))[::2]                             # 9
            + list(range(83, 83 + 8))                                 # 8
            + [53])                                                   # 1  -> 18

GROUPS = [("body", BODY_IDX, None),          # None = 不中心化
          ("left_hand", LH_IDX, 0),          # 0 = 减分组内第 0 点
          ("right_hand", RH_IDX, 0),
          ("face", FACE_IDX, -1)]            # -1 = 减分组内最后一点（索引 53）

N_KP = sum(len(g[1]) for g in GROUPS)        # 69
CONF_THR = 0.3                               # 与 Uni-Sign 一致


def crop_scale(motion, thr=CONF_THR):
    """复刻 Uni-Sign 的 crop_scale（源自 MotionBERT）。

    注意三点，都和直觉不同：
      1. bbox 在**整段序列**上算（跨所有帧、所有关键点），不是逐帧
      2. 低置信度的点**整行清零**（x, y, conf 全为 0），不只是坐标
      3. 先 clip 到 [-1,1] 再清零，顺序不能反
    """
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape, dtype=np.float32)
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=np.float32)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - np.array([xs, ys])) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result.astype(np.float32)


def build_groups(K, S, thr=CONF_THR):
    """(T,133,2)+(T,133) -> (T,69,3)，按 Uni-Sign 的分组、中心化、归一化。

    返回的点顺序固定为 body / left_hand / right_hand / face，
    下游（阶段 0 展平、阶段 1 分组喂 GCN）都按这个顺序切片。
    """
    out, spans, off = [], {}, 0
    for name, idx, anchor in GROUPS:
        kp = K[:, idx, :].astype(np.float32)          # (T, n, 2)
        cf = S[:, idx].astype(np.float32)             # (T, n)
        if anchor is not None:
            # 中心化只作用于坐标，置信度不参与（Uni-Sign 也是分开处理后再拼接）
            a = anchor if anchor >= 0 else len(idx) + anchor   # -1 -> 最后一点
            kp = kp - kp[:, a:a + 1, :]
        part = np.concatenate([kp, cf[..., None]], axis=-1)   # (T, n, 3)
        out.append(crop_scale(part, thr))
        spans[name] = (off, off + len(idx))
        off += len(idx)
    return np.concatenate(out, axis=1), spans          # (T, 69, 3)


def load_pose_pkl(path, thr=CONF_THR):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return build_groups(d["keypoints"], d["scores"], thr)


class RTMPoseSLTDataset(Dataset):
    """与 PoseSLTDataset 接口一致，便于 train.py 直接切换。"""

    def __init__(self, root, csv_dir, split, vocab, frame_stride=2,
                 max_frames=256, conf_thr=CONF_THR, **_ignored):
        self.pose_root = os.path.join(root, "pose_rtm")
        self.split = split
        self.vocab = vocab
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.conf_thr = conf_thr
        self.dim = N_KP * 3                            # 207，给阶段 0 展平用
        self.n_kp = N_KP
        self.spans = None                              # 首次 __getitem__ 后填上

        rows = list(csv.DictReader(
            open(os.path.join(csv_dir, split + ".csv"), encoding="utf-8")))
        self.items, self.missing = [], []
        for r in rows:
            num, tr = r["Number"].strip(), r["Translator"].strip()
            p = os.path.join(self.pose_root, split, tr, num + ".pkl")
            if not os.path.exists(p):
                self.missing.append(num)
                continue
            self.items.append({"path": p, "number": num, "translator": tr,
                               "text": r["Chinese Sentences"].strip()})

    def __len__(self):
        return len(self.items)

    def texts(self):
        return [it["text"] for it in self.items]

    def __getitem__(self, i):
        it = self.items[i]
        with open(it["path"], "rb") as f:
            d = pickle.load(f)
        K, S = d["keypoints"], d["scores"]

        # 先抽帧再归一化：crop_scale 的 bbox 应当反映真正喂进模型的那些帧
        if self.frame_stride > 1:
            K, S = K[::self.frame_stride], S[::self.frame_stride]
        if self.max_frames and len(K) > self.max_frames:
            idx = np.linspace(0, len(K) - 1, self.max_frames).round().astype(int)
            K, S = K[idx], S[idx]

        feat, spans = build_groups(K, S, self.conf_thr)     # (T, 69, 3)
        self.spans = spans

        tokens = self.vocab.encode(it["text"])
        return {"feat": torch.from_numpy(feat.reshape(len(feat), -1)),  # (T, 207)
                "tokens": torch.tensor(tokens, dtype=torch.long),
                "number": it["number"], "translator": it["translator"],
                "text": it["text"]}
