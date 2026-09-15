"""阶段 0：老架构 baseline —— 逐帧特征提取 + LSTM encoder-decoder seq2seq。

这是消融表的对照组，复刻六年前的架构思路。三个刻意的设计取舍：

1. **逐帧编码器不跨帧。** FrameEncoder 是逐帧共享权重的 MLP，只做空间编码，
   时序完全交给后面的 LSTM。这正是 CLAUDE.md 缺陷 #2「逐帧 CNN 是晚融合，
   空间信息先被压扁才轮到时序」—— 如果在这里加时序卷积，缺陷就被掩盖了，
   后续阶段的提升也就无从归因。

2. **解码器带 Bahdanau attention。** 2014 年就有的技术，2019 年的 seq2seq
   理应有。故意做一个无 attention 的弱 baseline 会让后面每一阶段的提升
   看起来都很大 —— 那是假的提升。见 DECISIONS.md D-012。

3. **保留 LSTM 的真实短板。** 缺陷 #1（长程依赖弱）和 #3（解码器无语言先验）
   不做任何补救，它们正是阶段 1 / 阶段 2 要解决的东西。

解码器来自 slt.models.decoder，与阶段 1 共用同一份代码 —— "解码器不动"
因此是结构上保证的，不是口头承诺。
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slt.models.decoder import AttnLSTMDecoder, build_loss, lengths_to_mask  # noqa: F401


class FrameEncoder(nn.Module):
    """逐帧空间编码。共享权重、逐帧独立，**不跨帧混合信息**。"""

    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):                       # (B, T, D) -> (B, T, out)
        return self.net(x)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, in_dim, vocab_size, frame_hidden=512, feat_dim=512,
                 enc_hidden=512, dec_hidden=512, emb_dim=256,
                 enc_layers=2, dropout=0.3, pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id

        self.frame_enc = FrameEncoder(in_dim, frame_hidden, feat_dim, dropout)
        self.encoder = nn.LSTM(feat_dim, enc_hidden, num_layers=enc_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if enc_layers > 1 else 0.0)
        enc_out_dim = enc_hidden * 2

        # 用编码器最后一层双向状态初始化解码器
        self.init_h = nn.Linear(enc_out_dim, dec_hidden)
        self.init_c = nn.Linear(enc_out_dim, dec_hidden)

        self.decoder = AttnLSTMDecoder(vocab_size, enc_out_dim, dec_hidden,
                                       emb_dim, dropout, pad_id, bos_id, eos_id)

    def encode(self, feats, feat_lens):
        x = self.frame_enc(feats)                       # (B, T, feat_dim)
        packed = pack_padded_sequence(x, feat_lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, (h, c) = self.encoder(packed)
        enc_out, _ = pad_packed_sequence(out, batch_first=True)   # (B, T, 2H)

        # h: (layers*2, B, H) -> 取最后一层的前后向拼接
        h_last = torch.cat([h[-2], h[-1]], dim=-1)      # (B, 2H)
        c_last = torch.cat([c[-2], c[-1]], dim=-1)
        dec_h = torch.tanh(self.init_h(h_last))
        dec_c = torch.tanh(self.init_c(c_last))
        enc_mask = lengths_to_mask(feat_lens, enc_out.size(1))
        return enc_out, enc_mask, dec_h, dec_c

    def forward(self, feats, feat_lens, tokens):
        enc_out, enc_mask, h, c = self.encode(feats, feat_lens)
        return self.decoder(enc_out, enc_mask, h, c, tokens)

    @torch.no_grad()
    def greedy_decode(self, feats, feat_lens, max_len=60):
        enc_out, enc_mask, h, c = self.encode(feats, feat_lens)
        return self.decoder.greedy(enc_out, enc_mask, h, c, max_len)
