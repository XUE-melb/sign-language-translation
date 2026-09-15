"""验证重构后的阶段 0 行为不变，且阶段 0 / 阶段 1 的解码器逐张量相同。"""
import torch

from slt.data import CharVocab
from slt.data_rtm import RTMPoseSLTDataset
from slt.models.stage0 import Seq2SeqLSTM
from slt.models.stage1 import Stage1Model

CKPT = "/root/autodl-tmp/slt/weights/unisign/csl_stage1_weight.pth"
ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"
V = 1974
ok = True

print("=" * 74)
print("检查 1：重构后阶段 0 的参数量与重构前一致")
m0_mp = Seq2SeqLSTM(in_dim=538, vocab_size=V)
n = sum(p.numel() for p in m0_mp.parameters())
print("  in_dim=538（MediaPipe）: {:.2f}M  （重构前实测 20.09M）".format(n / 1e6))
ok &= abs(n / 1e6 - 20.09) < 0.01
m0 = Seq2SeqLSTM(in_dim=207, vocab_size=V)
n207 = sum(p.numel() for p in m0.parameters())
print("  in_dim=207（RTMPose）  : {:.2f}M  （E-000b 用这个）".format(n207 / 1e6))

print("=" * 74)
print("检查 2：阶段 1 实例化 + 权重严格加载")
m1 = Stage1Model(vocab_size=V, ckpt_path=CKPT)
print("  编码器权重: 载入 {} 个张量，missing {}，unexpected {}".format(
    m1.load_info["loaded"], len(m1.load_info["missing"]),
    len(m1.load_info["unexpected"])))
ok &= not m1.load_info["missing"] and not m1.load_info["unexpected"]
tot = sum(p.numel() for p in m1.parameters())
trn = sum(p.numel() for p in m1.parameters() if p.requires_grad)
enc = sum(p.numel() for p in m1.encoder.parameters())
print("  总参数 {:.2f}M | 可训练 {:.2f}M | 冻结编码器 {:.2f}M".format(
    tot / 1e6, trn / 1e6, enc / 1e6))
print("  编码器参数全部冻结:", all(not p.requires_grad for p in m1.encoder.parameters()))
ok &= all(not p.requires_grad for p in m1.encoder.parameters())

print("=" * 74)
print("检查 3：两个阶段的解码器逐张量相同（这是'解码器不动'的硬证据）")
d0 = dict(m0.decoder.named_parameters())
d1 = dict(m1.decoder.named_parameters())
same_keys = set(d0) == set(d1)
print("  参数名集合相同:", same_keys)
shape_ok = same_keys and all(d0[k].shape == d1[k].shape for k in d0)
print("  每个张量形状相同:", shape_ok)
print("  解码器参数量: 阶段0 {:,} | 阶段1 {:,}".format(
    sum(v.numel() for v in d0.values()), sum(v.numel() for v in d1.values())))
print("  两者是同一个类:", type(m0.decoder) is type(m1.decoder))
ok &= same_keys and shape_ok
for k in list(d0)[:4]:
    print("    {:<28} {} | {}".format(k, tuple(d0[k].shape), tuple(d1[k].shape)))

print("=" * 74)
print("检查 4：两个阶段在同一批真实数据上前向")
vocab = CharVocab.build(["测试句子"])
ds = RTMPoseSLTDataset(ROOT, CSVD, "dev", vocab, frame_stride=2, max_frames=256)
fs = [ds[i]["feat"] for i in range(3)]
L = min(f.shape[0] for f in fs)
x = torch.stack([f[:L] for f in fs])
lens = torch.full((3,), L, dtype=torch.long)
tok = torch.randint(4, V, (3, 10))
tok[:, 0] = CharVocab.BOS

for name, m in (("阶段0", m0), ("阶段1", m1)):
    m.eval()
    with torch.no_grad():
        logits = m(x, lens, tok)
        ids = m.greedy_decode(x, lens, max_len=12)
    print("  {}  输入 {} -> logits {} | greedy {}".format(
        name, tuple(x.shape), tuple(logits.shape), tuple(ids.shape)))
    ok &= logits.shape == (3, 9, V)

print("=" * 74)
print("检查 5：阶段 1 的 train() 不会把冻结编码器切回 train 模式")
m1.train()
print("  model.training =", m1.training, "| encoder.training =", m1.encoder.training,
      "（编码器应为 False，否则 BatchNorm 统计量会被偷偷更新）")
ok &= m1.training and not m1.encoder.training

print("=" * 74)
print("总判定:", "全部通过" if ok else "有失败项")
