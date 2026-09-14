"""CE-CSL 关键点数据加载。

对齐方式（这是要能讲清楚的部分）：
  标注 CSV 的一行  ->  (Translator, Number)  ->  pose/{split}/{T}/{Number}.npy
  一个样本 = 一整段视频的关键点序列 (T, D) + 一个中文句子
  没有帧级对齐，没有 gloss 监督 —— 这就是 gloss-free：
  监督信号只有"整段视频 -> 整句中文"这一对，中间怎么对齐由模型自己学。
"""
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# MediaPipe Pose 的左右肩索引，用于身体中心归一化
L_SHOULDER, R_SHOULDER = 11, 12


# ----------------------------------------------------------------- 词表

class CharVocab:
    """字级词表。中文 SLT 在这个数据量下按字比按词稳（见 DECISIONS.md D-010）。"""

    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, sentences, min_freq=1):
        freq = {}
        for s in sentences:
            for ch in s:
                freq[ch] = freq.get(ch, 0) + 1
        chars = sorted([c for c, n in freq.items() if n >= min_freq],
                       key=lambda c: (-freq[c], c))
        return cls(cls.SPECIALS + chars)

    def encode(self, text, add_bos_eos=True):
        ids = [self.stoi.get(ch, self.UNK) for ch in text]
        if add_bos_eos:
            ids = [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids, strip_specials=True):
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if strip_specials and i in (self.PAD, self.BOS):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return "".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf-8"))["itos"])


# ----------------------------------------------------------------- 归一化

def parse_layout(layout):
    """把 layout.json 的 segments 转成 {name: (start, end, n_points, n_ch)}。"""
    return {s["name"]: (s["start"], s["end"], s["n_points"], s["n_channels"])
            for s in layout["segments"]}


def normalize_body(feat, segs):
    """身体中心 + 肩宽缩放归一化。

    为什么要做：D-005 已确认取景在 signer 之间差异巨大（站姿/坐姿、远近、
    人在画面里的位置都不同）。原始坐标是相对整幅图像的，直接喂进去等于把
    "这是谁、他站在哪"一起喂给模型，助长 signer 泄漏。
    减去肩中点、除以肩宽之后，剩下的是相对身体的几何关系。

    ⚠️ 最容易写错的地方：未检出的部位是全 0 填充，而 0 是合法坐标。
    如果对全 0 块也做 (0 - center)/scale，它会变成一个非零的虚假坐标，
    等于凭空捏造了一只手。所以必须借 mask 段只归一化真正检出的部位。
    """
    T = feat.shape[0]
    out = feat.copy()

    ps, pe, pn, pc = segs["pose"]
    pose = feat[:, ps:pe].reshape(T, pn, pc)
    center = (pose[:, L_SHOULDER, :2] + pose[:, R_SHOULDER, :2]) / 2.0   # (T, 2)
    scale = np.linalg.norm(pose[:, L_SHOULDER, :2] - pose[:, R_SHOULDER, :2],
                           axis=-1)                                       # (T,)
    scale = np.where(scale < 1e-3, 1.0, scale)[:, None]                   # 退化保护

    ms, me, _, _ = segs["mask"]
    mask = feat[:, ms:me]            # (T, 4): pose, left_hand, right_hand, face

    for name, mi in (("pose", 0), ("left_hand", 1), ("right_hand", 2), ("face", 3)):
        if name not in segs:
            continue
        s, e, n, c = segs[name]
        blk = out[:, s:e].reshape(T, n, c).copy()
        xy = (blk[:, :, :2] - center[:, None, :]) / scale[:, None, :]
        blk[:, :, :2] = xy
        if c >= 3:                    # z 只缩放不平移（它本就是相对量）
            blk[:, :, 2] = blk[:, :, 2] / scale
        # 只把归一化结果写回"该帧该部位确实检出"的位置，其余保持全 0
        keep = mask[:, mi] > 0.5                                          # (T,)
        blk[~keep] = 0.0
        out[:, s:e] = blk.reshape(T, n * c)

    return out


# ----------------------------------------------------------------- 数据集

class PoseSLTDataset(Dataset):
    def __init__(self, root, csv_dir, split, vocab, frame_stride=2,
                 max_frames=256, normalize="body"):
        self.pose_root = os.path.join(root, "pose")
        self.split = split
        self.vocab = vocab
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.normalize = normalize

        layout = json.load(open(os.path.join(self.pose_root, "layout.json"),
                                encoding="utf-8"))
        self.layout = layout
        self.segs = parse_layout(layout)
        self.dim = layout["dim"]

        rows = list(csv.DictReader(
            open(os.path.join(csv_dir, split + ".csv"), encoding="utf-8")))
        self.items, self.missing = [], []
        for r in rows:
            number = r["Number"].strip()
            translator = r["Translator"].strip()
            p = os.path.join(self.pose_root, split, translator, number + ".npy")
            if not os.path.exists(p):
                self.missing.append(number)
                continue
            self.items.append({"path": p, "number": number,
                               "translator": translator,
                               "text": r["Chinese Sentences"].strip()})

    def __len__(self):
        return len(self.items)

    def texts(self):
        return [it["text"] for it in self.items]

    def __getitem__(self, i):
        it = self.items[i]
        feat = np.load(it["path"]).astype(np.float32)      # (T, D) float16 -> f32

        if self.frame_stride > 1:
            feat = feat[::self.frame_stride]
        if self.max_frames and len(feat) > self.max_frames:
            # 均匀抽帧而非截断：截断会直接丢掉句子后半段的手语
            idx = np.linspace(0, len(feat) - 1, self.max_frames).round().astype(int)
            feat = feat[idx]

        if self.normalize == "body":
            feat = normalize_body(feat, self.segs)

        tokens = self.vocab.encode(it["text"])
        return {"feat": torch.from_numpy(feat),
                "tokens": torch.tensor(tokens, dtype=torch.long),
                "number": it["number"], "translator": it["translator"],
                "text": it["text"]}


def collate_fn(batch, pad_id=CharVocab.PAD):
    """按最长样本补齐，同时返回真实长度供 pack_padded_sequence 与 mask 使用。"""
    B = len(batch)
    fl = [b["feat"].shape[0] for b in batch]
    tl = [b["tokens"].shape[0] for b in batch]
    D = batch[0]["feat"].shape[1]

    feats = torch.zeros(B, max(fl), D)
    tokens = torch.full((B, max(tl)), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        feats[i, :fl[i]] = b["feat"]
        tokens[i, :tl[i]] = b["tokens"]

    return {"feats": feats,
            "feat_lens": torch.tensor(fl, dtype=torch.long),
            "tokens": tokens,
            "token_lens": torch.tensor(tl, dtype=torch.long),
            "numbers": [b["number"] for b in batch],
            "translators": [b["translator"] for b in batch],
            "texts": [b["text"] for b in batch]}
