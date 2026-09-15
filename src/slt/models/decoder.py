"""共用的带注意力 LSTM 解码器。

为什么抽出来：CLAUDE.md 阶段 1 要求"只换视觉编码器，解码器不动"。
如果每个阶段各写一份解码器，"不动"就只是口头承诺 —— 任何一处笔误或
无意改动都不会报错，却会让消融表的归因失效。
抽成同一个类之后，"解码器不动"是**结构上保证的事实**，不是承诺。

阶段 0 和阶段 1 的差异因此被压缩到一处：谁来产生 (enc_out, enc_mask,
初始 h/c)。解码器本身逐字节相同。
"""
import torch
import torch.nn as nn


def lengths_to_mask(lengths, max_len):
    """(B,) -> (B, max_len) 的 bool，True 表示真实帧。"""
    ar = torch.arange(max_len, device=lengths.device)[None, :]
    return ar < lengths[:, None]


class BahdanauAttention(nn.Module):
    """加性注意力。query=解码器隐状态，keys/values=编码器每帧输出。"""

    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_h, enc_out, enc_mask):
        score = self.v(torch.tanh(
            self.W_enc(enc_out) + self.W_dec(dec_h)[:, None, :])).squeeze(-1)
        score = score.masked_fill(~enc_mask, float("-inf"))   # padding 不参与
        attn = torch.softmax(score, dim=-1)                   # (B, T)
        ctx = torch.bmm(attn[:, None, :], enc_out).squeeze(1)  # (B, enc_dim)
        return ctx, attn


class AttnLSTMDecoder(nn.Module):
    """逐字自回归解码。所有阶段共用，不允许按阶段改。"""

    def __init__(self, vocab_size, enc_dim, dec_hidden=512, emb_dim=256,
                 dropout=0.3, pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attn = BahdanauAttention(enc_dim, dec_hidden, dec_hidden)
        self.cell = nn.LSTMCell(emb_dim + enc_dim, dec_hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_hidden + enc_dim, vocab_size)

    def step(self, y_prev, h, c, enc_out, enc_mask):
        ctx, attn = self.attn(h, enc_out, enc_mask)
        emb = self.emb(y_prev)
        h, c = self.cell(torch.cat([emb, ctx], -1), (h, c))
        logit = self.out(self.dropout(torch.cat([h, ctx], -1)))
        return logit, h, c, attn

    def forward(self, enc_out, enc_mask, h, c, tokens):
        """teacher forcing。tokens = [BOS, c1..cn, EOS]，输入取 [:-1]。"""
        inp = tokens[:, :-1]
        logits = []
        for t in range(inp.size(1)):
            logit, h, c, _ = self.step(inp[:, t], h, c, enc_out, enc_mask)
            logits.append(logit)
        return torch.stack(logits, dim=1)            # (B, L-1, V)

    @torch.no_grad()
    def greedy(self, enc_out, enc_mask, h, c, max_len=60):
        B = enc_out.size(0)
        y = torch.full((B,), self.bos_id, dtype=torch.long, device=enc_out.device)
        done = torch.zeros(B, dtype=torch.bool, device=enc_out.device)
        seqs = []
        for _ in range(max_len):
            logit, h, c, _ = self.step(y, h, c, enc_out, enc_mask)
            y = logit.argmax(-1)
            y = torch.where(done, torch.full_like(y, self.pad_id), y)
            seqs.append(y)
            done = done | (y == self.eos_id)
            if bool(done.all()):
                break
        return torch.stack(seqs, dim=1)


def build_loss(pad_id, label_smoothing=0.0):
    """padding 位不计入 loss —— 否则模型会因为"学会输出 pad"而虚假降 loss。"""
    return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
