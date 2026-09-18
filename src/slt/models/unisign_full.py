"""把 Uni-Sign 的完整 pose-only 管线装回来：编码器 + pose_proj + mT5。

用途：
  1. 零样本探针 —— 不训练，直接在 CE-CSL dev 上评，量"借来的系统"的起点
  2. 阶段 2（E-002）的骨架 —— 在这上面训 pose_proj（+ 可选 LoRA）

接线逐行对照 third_party/Uni-Sign/models.py 的 forward / generate：
    pose_embeds = pose_proj( encoder(groups) )                  # (B, T, 768)
    prefix      = embed_tokens( tok("Translate sign language video to Chinese: ") )
    inputs_embeds  = cat([prefix, pose_embeds], dim=1)          # prefix 在前
    attention_mask = cat([prefix_mask, pose_mask], dim=1)
    训练：labels = tok(句子, max_length=50)，pad -> -100，CE
    生成：mt5.generate(inputs_embeds, attention_mask, max_new_tokens=100, num_beams=4)

mT5 权重从 Uni-Sign checkpoint 的 mt5_model.* 覆盖到 HF 的 MT5ForConditionalGeneration；
tokenizer / config 来自 google/mt5-base（checkpoint 里没有）。
"""
import torch
import torch.nn as nn
from transformers import MT5Config, MT5ForConditionalGeneration

try:                                   # transformers 4.x
    from transformers import T5Tokenizer as _Tok
    _TOK_KW = {"legacy": False}
except ImportError:                    # transformers 5.x 砍了慢速 tokenizer
    from transformers import AutoTokenizer as _Tok
    _TOK_KW = {}

from slt.models.decoder import lengths_to_mask
from slt.models.unisign_encoder import (OUT_DIM, UniSignPoseEncoder,
                                        split_groups)

MT5_DIR = "/root/autodl-tmp/slt/weights/mt5-base"
CKPT = "/root/autodl-tmp/slt/weights/unisign/csl_stage1_weight.pth"
PREFIX = "Translate sign language video to Chinese: "
PREFIXES = {"zh": PREFIX, "en": "Translate sign language video to English: "}   # D-029
LABEL_MAX = {"zh": 50, "en": 64}      # 英文句子 token 数更多（How2Sign 均 17 词）


def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class UniSignFull(nn.Module):
    def __init__(self, ckpt_path=CKPT, mt5_dir=MT5_DIR, load_mt5_weights=True,
                 encoder_ckpt=None, lang="zh"):
        """encoder_ckpt: 若给出，编码器从这个文件读，pose_proj/mT5 仍从 ckpt_path 读。
        用于混搭：E-001 的 stage-1 编码器 + CSL-Daily 的 mT5。"""
        super().__init__()
        self.lang = lang
        self.prefix = PREFIXES[lang]
        self.label_max_length = LABEL_MAX[lang]
        self.tokenizer = _Tok.from_pretrained(mt5_dir, **_TOK_KW)
        if load_mt5_weights:
            # 权重全在 Uni-Sign checkpoint 里，HF 目录只借 config + tokenizer，
            # 不读 HF 的权重文件（也就不依赖 2.3G 下载完成）
            self.mt5 = MT5ForConditionalGeneration(MT5Config.from_pretrained(mt5_dir))
        else:
            self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_dir)
        self.encoder = UniSignPoseEncoder()
        self.pose_proj = nn.Linear(OUT_DIM, self.mt5.config.d_model)   # 1024 -> 768

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for k in ("model", "state_dict"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break

        esd = sd
        if encoder_ckpt:
            esd = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            for k in ("model", "state_dict"):
                if isinstance(esd, dict) and k in esd and isinstance(esd[k], dict):
                    esd = esd[k]
                    break
        enc_sd = {k: v for k, v in esd.items() if k.startswith(
            ("proj_linear.", "gcn_modules.", "fusion_gcn_modules.", "part_para"))}
        m, u = self.encoder.load_state_dict(enc_sd, strict=False)
        assert not m and not u, "编码器权重不匹配 m={} u={}".format(m[:3], u[:3])

        proj_sd = _strip_prefix(sd, "pose_proj.")
        self.pose_proj.load_state_dict(proj_sd, strict=True)

        self.load_info = {"encoder": len(enc_sd), "pose_proj": len(proj_sd),
                          "encoder_from": (encoder_ckpt or ckpt_path).split("/")[-1]}
        if load_mt5_weights:
            mt5_sd = _strip_prefix(sd, "mt5_model.")
            m, u = self.mt5.load_state_dict(mt5_sd, strict=False)
            # HF 会把 lm_head 与 shared 绑权重；checkpoint 里若只存其一，
            # 另一个会出现在 missing 里，属于正常。其余必须为空。
            m = [k for k in m if k not in ("lm_head.weight",)]
            assert not m and not u, "mT5 权重不匹配 missing={} unexpected={}".format(m[:5], u[:5])
            self.load_info["mt5"] = len(mt5_sd)
        else:
            self.load_info["mt5"] = "HF 原版"

    # ---------------------------------------------------------------- 编码

    def build_inputs(self, feats, feat_lens):
        """(B,T,207) -> mT5 的 inputs_embeds / attention_mask，prefix 在前。"""
        dev = feats.device
        pose = self.pose_proj(self.encoder(split_groups(feats)))            # (B,T,768)
        pose_mask = lengths_to_mask(feat_lens, pose.size(1)).long()

        B = feats.size(0)
        tok = self.tokenizer([self.prefix] * B, padding="longest", truncation=True,
                             return_tensors="pt").to(dev)
        prefix = self.mt5.encoder.embed_tokens(tok["input_ids"])           # (B,P,768)

        inputs_embeds = torch.cat([prefix, pose], dim=1)
        attention_mask = torch.cat([tok["attention_mask"], pose_mask], dim=1)
        return inputs_embeds, attention_mask

    # ---------------------------------------------------------------- 训练

    def forward(self, feats, feat_lens, sentences, label_smoothing=0.0):
        inputs_embeds, attention_mask = self.build_inputs(feats, feat_lens)
        lab = self.tokenizer(list(sentences), return_tensors="pt", padding=True,
                             truncation=True, max_length=self.label_max_length)["input_ids"].to(feats.device)
        lab[lab == self.tokenizer.pad_token_id] = -100
        out = self.mt5(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       labels=lab, return_dict=True)
        if label_smoothing > 0:
            lf = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
            return lf(out.logits.reshape(-1, out.logits.size(-1)), lab.reshape(-1))
        return out.loss

    # ---------------------------------------------------------------- 生成

    @torch.no_grad()
    def generate(self, feats, feat_lens, num_beams=4, max_new_tokens=100):
        inputs_embeds, attention_mask = self.build_inputs(feats, feat_lens)
        ids = self.mt5.generate(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                num_beams=num_beams)
        txt = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        # zh：参考句无空格，去掉 sentencepiece 可能带出的空格以与字级指标对齐；en：保留词间单空格。
        # 原版 mT5 训练早期会吐 span-corruption 的哨兵符 <extra_id_N>，skip_special_tokens 不一定剥掉，这里兜底
        import re
        txt = [re.sub(r"<extra_id_\d+>", " ", t) for t in txt]
        if self.lang == "zh":
            return ["".join(t.split()) for t in txt]
        return [" ".join(t.split()) for t in txt]
