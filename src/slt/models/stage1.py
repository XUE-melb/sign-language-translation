"""阶段 1：只换视觉编码器 —— 冻结的 Uni-Sign pose 编码器 + 原样的 LSTM 解码器。

相对阶段 0 **只改了一件事**：
    阶段 0   FrameEncoder(逐帧 MLP) + BiLSTM   -> (B, T, 1024)
    阶段 1   冻结的 Uni-Sign pose 编码器        -> (B, T, 1024)

解码器来自 slt.models.decoder，与阶段 0 是同一个类、同一份代码。
输入也完全相同（RTMPose 69 点，见 D-020），所以 E-000b -> E-001 是严格单变量。

编码器输出维度恰好也是 1024，与阶段 0 的 BiLSTM 双向输出一致 ——
这是巧合但很方便：解码器的 enc_dim 不用改，注意力结构完全相同。

## 一处必要的适配（要在报告里讲明）

阶段 0 的解码器初始状态来自 BiLSTM 的**末状态**。Uni-Sign 编码器不是
循环结构，没有"末状态"可用。这里改用**在时间维上对有效帧做平均池化**，
再经同样的 init_h / init_c 线性层。

这属于"换编码器"的必要适配，不是对解码器的改动 —— 解码器本体一行未动。
"""
import torch
import torch.nn as nn

from slt.models.decoder import AttnLSTMDecoder, build_loss, lengths_to_mask  # noqa: F401
from slt.models.unisign_encoder import (OUT_DIM, UniSignPoseEncoder,
                                        load_pretrained, split_groups)


class Stage1Model(nn.Module):
    def __init__(self, vocab_size, ckpt_path=None, dec_hidden=512, emb_dim=256,
                 dropout=0.3, freeze_encoder=True,
                 pad_id=0, bos_id=1, eos_id=2, device="cpu"):
        super().__init__()
        if ckpt_path:
            self.encoder, self.load_info = load_pretrained(ckpt_path, device="cpu",
                                                           strict=True)
        else:
            self.encoder, self.load_info = UniSignPoseEncoder(), {"loaded": 0}

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        enc_dim = OUT_DIM                               # 1024
        self.init_h = nn.Linear(enc_dim, dec_hidden)
        self.init_c = nn.Linear(enc_dim, dec_hidden)
        self.decoder = AttnLSTMDecoder(vocab_size, enc_dim, dec_hidden,
                                       emb_dim, dropout, pad_id, bos_id, eos_id)

    def train(self, mode=True):
        """冻结的编码器始终保持 eval —— 否则 BatchNorm 的 running stats 会被
        训练数据悄悄更新，等于偷偷改动了"冻结"的编码器。"""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, feats, feat_lens):
        groups = split_groups(feats)                    # (B,T,207) -> 4 组
        if self.freeze_encoder:
            with torch.no_grad():
                enc_out = self.encoder(groups)          # (B, T, 1024)
        else:
            enc_out = self.encoder(groups)

        enc_mask = lengths_to_mask(feat_lens, enc_out.size(1))
        # 非循环编码器没有"末状态"：在有效帧上做平均池化
        m = enc_mask.unsqueeze(-1).to(enc_out.dtype)
        pooled = (enc_out * m).sum(1) / m.sum(1).clamp(min=1.0)
        dec_h = torch.tanh(self.init_h(pooled))
        dec_c = torch.tanh(self.init_c(pooled))
        return enc_out, enc_mask, dec_h, dec_c

    def forward(self, feats, feat_lens, tokens):
        enc_out, enc_mask, h, c = self.encode(feats, feat_lens)
        return self.decoder(enc_out, enc_mask, h, c, tokens)

    @torch.no_grad()
    def greedy_decode(self, feats, feat_lens, max_len=60):
        enc_out, enc_mask, h, c = self.encode(feats, feat_lens)
        return self.decoder.greedy(enc_out, enc_mask, h, c, max_len)
