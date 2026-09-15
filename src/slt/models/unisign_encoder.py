"""把 Uni-Sign 的 pose 编码器单独拿出来用（阶段 1 的冻结视觉编码器）。

只搭 pose 分支，不实例化那 966M 的 mT5 —— 我们阶段 1 的解码器仍是自己的
LSTM（CLAUDE.md：只换视觉编码器，解码器不动）。

结构（逐行对照 third_party/Uni-Sign/models.py 的 Uni_Sign.__init__ 与
pose branch forward）：

    每组 (B,T,V,3)
      -> proj_linear[part]          Linear(3, 64)
      -> gcn_modules[part]          3 层空间 GCN      -> (B, C, T, V)
      -> 身体先验注入（非 body 组）  见下
      -> fusion_gcn_modules[part]   3 层时序 ST-GCN (kernel 5)
      -> 在 V 维平均池化            -> (B, T, 256)
    4 组拼接 -> (B, T, 1024) + part_para

身体先验注入：手减去了自身手腕，绝对位置丢失；这里把身体分支上对应
关节的特征加回来补上"手在身体哪个位置"。
    left     += body_feat[..., -2]   body 索引 [0,3..10] 的倒数第二个 = 9  = 左手腕
    right    += body_feat[..., -1]   最后一个 = 10 = 右手腕
    face_all += body_feat[...,  0]   第一个   = 0  = 鼻子
都带 .detach()，梯度不回流 body 分支。

左右手共享权重（gcn/fusion/proj_linear 三者都是同一对象）。

输出取 **pose_proj 之前的 1024 维**：pose_proj 是接进 mT5 语义空间的接口，
属于阶段 2 的"可训练投影层"，不应算进阶段 1 的视觉编码器。
"""
import os
import sys

import torch
import torch.nn as nn

UNISIGN_DIR = "/root/autodl-tmp/slt/third_party/Uni-Sign"
if UNISIGN_DIR not in sys.path:
    sys.path.insert(0, UNISIGN_DIR)

from stgcn_layers import Graph, get_stgcn_chain          # noqa: E402

MODES = ["body", "left", "right", "face_all"]
# 与 slt.data_rtm.GROUPS 的顺序一一对应（body / left_hand / right_hand / face）
GROUP_SIZES = {"body": 9, "left": 21, "right": 21, "face_all": 18}
HIDDEN_DIM = 256
OUT_DIM = HIDDEN_DIM * len(MODES)                        # 1024


class UniSignPoseEncoder(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.modes = list(MODES)
        self.graph, A = {}, []
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            self.graph[mode] = Graph(layout=mode, strategy="distance", max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32,
                                  requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)

        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        sks = A[0].size(0)                               # spatial kernel size
        for i, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(
                64, "spatial", (1, sks), A[i].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                final_dim, "temporal", (5, sks), A[i].clone(), True)

        # 左右手共享权重 —— 必须在加载 state_dict 之前就建立共享关系
        self.gcn_modules["left"] = self.gcn_modules["right"]
        self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
        self.proj_linear["left"] = self.proj_linear["right"]

        self.part_para = nn.Parameter(torch.zeros(hidden_dim * len(self.modes)))
        self.out_dim = hidden_dim * len(self.modes)

    def forward(self, groups):
        """groups: {mode: (B, T, V, 3)} -> (B, T, 1024)"""
        feats, body_feat = [], None
        for part in self.modes:
            x = self.proj_linear[part](groups[part]).permute(0, 3, 1, 2)  # B,C,T,V
            g = self.gcn_modules[part](x)
            if part == "body":
                body_feat = g
            elif part == "left":
                g = g + body_feat[..., -2][..., None].detach()
            elif part == "right":
                g = g + body_feat[..., -1][..., None].detach()
            elif part == "face_all":
                g = g + body_feat[..., 0][..., None].detach()
            g = self.fusion_gcn_modules[part](g)
            feats.append(g.mean(-1).transpose(1, 2))                      # B,T,C
        return torch.cat(feats, dim=-1) + self.part_para                  # B,T,1024


def split_groups(feat_flat):
    """(B, T, 207) -> {mode: (B, T, V, 3)}，切分顺序与 data_rtm.GROUPS 一致。"""
    B, T, _ = feat_flat.shape
    x = feat_flat.view(B, T, 69, 3)
    out, off = {}, 0
    for mode in MODES:
        n = GROUP_SIZES[mode]
        out[mode] = x[:, :, off:off + n, :]
        off += n
    assert off == 69, off
    return out


def load_pretrained(ckpt_path, device="cpu", strict=True):
    """只取 pose 分支的权重，mT5 的部分直接丢掉。"""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    keep_prefixes = ("proj_linear.", "gcn_modules.", "fusion_gcn_modules.",
                     "part_para")
    pose_sd = {k: v for k, v in sd.items() if k.startswith(keep_prefixes)}

    enc = UniSignPoseEncoder().to(device)
    missing, unexpected = enc.load_state_dict(pose_sd, strict=False)
    info = {"loaded": len(pose_sd), "missing": list(missing),
            "unexpected": list(unexpected),
            "dropped_mt5": sum(1 for k in sd if k.startswith("mt5_model")),
            "dropped_pose_proj": sum(1 for k in sd if k.startswith("pose_proj"))}
    if strict and (missing or unexpected):
        raise RuntimeError("权重不匹配 missing={} unexpected={}".format(
            missing[:5], unexpected[:5]))
    return enc, info
