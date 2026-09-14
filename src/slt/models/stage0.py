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
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def lengths_to_mask(lengths, max_len):
    """(B,) -> (B, max_len) 的 bool，True 表示真实帧。"""
    ar = torch.arange(max_len, device=lengths.device)[None, :]
    return ar < lengths[:, None]


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


class BahdanauAttention(nn.Module):
    """加性注意力。query=解码器隐状态，keys/values=编码器每帧输出。"""

    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_h, enc_out, enc_mask):
        # dec_h (B, dec_dim) / enc_out (B, T, enc_dim) / enc_mask (B, T)
        score = self.v(torch.tanh(
            self.W_enc(enc_out) + self.W_dec(dec_h)[:, None, :])).squeeze(-1)
        score = score.masked_fill(~enc_mask, float("-inf"))   # padding 不参与
        attn = torch.softmax(score, dim=-1)                   # (B, T)
        ctx = torch.bmm(attn[:, None, :], enc_out).squeeze(1)  # (B, enc_dim)
        return ctx, attn


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

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attn = BahdanauAttention(enc_out_dim, dec_hidden, dec_hidden)
        self.dec_cell = nn.LSTMCell(emb_dim + enc_out_dim, dec_hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_hidden + enc_out_dim, vocab_size)

    # ---------------------------------------------------------- 编码

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

    # ---------------------------------------------------------- 单步解码

    def _step(self, y_prev, dec_h, dec_c, enc_out, enc_mask):
        ctx, attn = self.attn(dec_h, enc_out, enc_mask)
        emb = self.emb(y_prev)                                  # (B, emb)
        dec_h, dec_c = self.dec_cell(torch.cat([emb, ctx], -1), (dec_h, dec_c))
        logit = self.out(self.dropout(torch.cat([dec_h, ctx], -1)))
        return logit, dec_h, dec_c, attn

    # ---------------------------------------------------------- 训练前向

    def forward(self, feats, feat_lens, tokens):
        """teacher forcing。tokens 形如 [BOS, c1, ..., cn, EOS]。

        输入给解码器的是 tokens[:, :-1]，预测目标是 tokens[:, 1:] ——
        即每一步都用"真实的前一个字"去预测"下一个字"，错位一位。
        """
        enc_out, enc_mask, dec_h, dec_c = self.encode(feats, feat_lens)
        inp = tokens[:, :-1]
        logits = []
        for t in range(inp.size(1)):
            logit, dec_h, dec_c, _ = self._step(inp[:, t], dec_h, dec_c,
                                                enc_out, enc_mask)
            logits.append(logit)
        return torch.stack(logits, dim=1)          # (B, L-1, V)

    # ---------------------------------------------------------- 推理

    @torch.no_grad()
    def greedy_decode(self, feats, feat_lens, max_len=60):
        enc_out, enc_mask, dec_h, dec_c = self.encode(feats, feat_lens)
        B = feats.size(0)
        y = torch.full((B,), self.bos_id, dtype=torch.long, device=feats.device)
        done = torch.zeros(B, dtype=torch.bool, device=feats.device)
        seqs = []
        for _ in range(max_len):
            logit, dec_h, dec_c, _ = self._step(y, dec_h, dec_c, enc_out, enc_mask)
            y = logit.argmax(-1)
            y = torch.where(done, torch.full_like(y, self.pad_id), y)
            seqs.append(y)
            done = done | (y == self.eos_id)
            if bool(done.all()):
                break
        return torch.stack(seqs, dim=1)             # (B, <=max_len)


def build_loss(pad_id, label_smoothing=0.0):
    """padding 位不计入 loss —— 否则模型会因为'学会输出 pad'而虚假降 loss。"""
    return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
