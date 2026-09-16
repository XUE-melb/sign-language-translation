"""推理封装（D-027）。B 层对软件层的全部承诺就是一个函数：

    translate(feats: (T, 207)) -> [(text, score), ...]      # n-best，score 降序

加载顺序：Uni-Sign 底座 checkpoint（编码器 + pose_proj + mT5）→ 用 run_dir/best 里的训练产物覆盖
（pose_proj.pt；lora/ 合并进 mT5 主干，运行时不再需要 peft 的包装；阶段 3 还有 encoder.pt）→ eval。
合并 LoRA 的意思：W' = W + B·A，把 14 MB 的低秩补丁加回 mT5 的 q/k/v/o 权重，前向和训练时逐位等价。

输入三种形式：
  (a) 已预处理的 (T, 207) 特征 —— 训练时 dataloader 给模型的东西
  (b) RTMPose 原始输出 (T, 133, 2) 归一化坐标 + (T, 133) 置信度 —— 端侧 rtmlib 传来的东西
  (c) pkl 文件（CE-CSL/pose_rtm 里的格式）
(b)(c) 走与训练完全相同的预处理：超过 max_frames 时 linspace 抽帧，然后 data_rtm.build_groups。

置信度：beam search 的 sequences_scores = 该候选的对数概率 / 长度（HF 默认 length_penalty=1.0），
越接近 0 越自信。greedy（num_beams=1）时用 compute_transition_scores 逐 token 算出同样口径的分数。

用法：
  python -m slt.infer --run runs/E002_lora16_s1234 --pkl CE-CSL/pose_rtm/test/E/test-00202.pkl
"""
import argparse
import csv
import os
import pickle
import time

import numpy as np
import torch

from slt.data_rtm import CONF_THR, build_groups
from slt.models.unisign_full import UniSignFull

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_BASE = os.path.join(REPO, "weights", "unisign", "csl_daily_pose_only_slt.pth")
DEFAULT_MT5 = os.path.join(REPO, "weights", "mt5-base")


class Translator:
    def __init__(self, run_dir=None, base_ckpt=DEFAULT_BASE, mt5_dir=DEFAULT_MT5,
                 device=None, max_frames=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_frames = max_frames
        self.model = UniSignFull(ckpt_path=base_ckpt, mt5_dir=mt5_dir)
        self.info = dict(self.model.load_info)
        self.info["run"] = run_dir or "零样本（只有底座）"
        if run_dir:
            best = os.path.join(run_dir, "best")
            self.model.pose_proj.load_state_dict(
                torch.load(os.path.join(best, "pose_proj.pt"), map_location="cpu"))
            enc = os.path.join(best, "encoder.pt")
            if os.path.exists(enc):                          # 阶段 3
                self.model.encoder.load_state_dict(torch.load(enc, map_location="cpu"))
                self.info["encoder_from"] = "encoder.pt（阶段 3 端到端）"
            lora = os.path.join(best, "lora")
            if os.path.isdir(lora):
                from peft import PeftModel
                self.model.mt5 = PeftModel.from_pretrained(self.model.mt5, lora).merge_and_unload()
                self.info["lora"] = "已合并进主干"
        self.model.eval().to(self.device)

    # ------------------------------------------------------------ 预处理

    def features_from_keypoints(self, keypoints, scores):
        """RTMPose 输出 -> (T, 207)。keypoints (T,133,2) 已按 [w,h] 归一化到 [0,1]，scores (T,133)。"""
        K = np.asarray(keypoints, dtype=np.float32)
        S = np.asarray(scores, dtype=np.float32)
        if self.max_frames and len(K) > self.max_frames:
            idx = np.linspace(0, len(K) - 1, self.max_frames).round().astype(int)
            K, S = K[idx], S[idx]
        feat, _ = build_groups(K, S, CONF_THR)                # (T, 69, 3)
        return feat.reshape(len(feat), -1).astype(np.float32)

    def features_from_pkl(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return self.features_from_keypoints(d["keypoints"], d["scores"])

    # ------------------------------------------------------------ 推理

    @torch.no_grad()
    def translate(self, feats, n_best=4, num_beams=4, max_new_tokens=100):
        """feats: (T,207) 或 [(T_i,207), ...]。返回 [(text, score), ...]（单样本）或其列表（批）。"""
        single = not isinstance(feats, (list, tuple))
        items = [feats] if single else list(feats)
        lens = torch.tensor([len(f) for f in items])
        batch = torch.zeros(len(items), int(lens.max()), items[0].shape[-1])
        for i, f in enumerate(items):                       # 零填充，与训练 collate 一致
            batch[i, :len(f)] = torch.as_tensor(np.asarray(f), dtype=torch.float32)
        batch, lens = batch.to(self.device), lens.to(self.device)

        emb, mask = self.model.build_inputs(batch, lens)
        nb = max(num_beams, n_best)
        out = self.model.mt5.generate(inputs_embeds=emb, attention_mask=mask,
                                      max_new_tokens=max_new_tokens, num_beams=nb,
                                      num_return_sequences=n_best,
                                      output_scores=True, return_dict_in_generate=True)
        txt = self.model.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        txt = ["".join(t.split()) for t in txt]              # 与字级指标对齐
        if getattr(out, "sequences_scores", None) is not None:
            scores = out.sequences_scores.tolist()
        else:                                                # greedy：逐 token 对数概率求平均
            tr = self.model.mt5.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            valid = (out.sequences[:, 1:] != self.model.tokenizer.pad_token_id).float()
            scores = ((tr * valid).sum(1) / valid.sum(1).clamp(min=1)).tolist()
        res = [list(zip(txt[i * n_best:(i + 1) * n_best], scores[i * n_best:(i + 1) * n_best]))
               for i in range(len(items))]
        return res[0] if single else res

    def translate_pkl(self, path, **kw):
        return self.translate(self.features_from_pkl(path), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="训练产物目录（含 best/）；不给 = 零样本底座")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--mt5", default=DEFAULT_MT5)
    ap.add_argument("--pkl", nargs="+", required=True)
    ap.add_argument("--csv", default="", help="给 test.csv 可顺带打印参考句")
    ap.add_argument("--n-best", type=int, default=4)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    t0 = time.time()
    tr = Translator(a.run or None, a.base, a.mt5, device=a.device)
    print("加载 {:.1f}s | device {} | {}".format(time.time() - t0, tr.device, tr.info))
    refs = {}
    if a.csv:
        for r in csv.DictReader(open(a.csv, encoding="utf-8")):
            refs[r["Number"].strip()] = r["Chinese Sentences"].strip()
    for p in a.pkl:
        f = tr.features_from_pkl(p)
        t0 = time.time()
        nb = tr.translate(f, n_best=a.n_best, num_beams=a.beams)
        dt = time.time() - t0
        num = os.path.splitext(os.path.basename(p))[0]
        print("\n{}  T={}  {:.2f}s{}".format(num, len(f), dt, "  参考: " + refs[num] if num in refs else ""))
        for i, (t, s) in enumerate(nb):
            print("  {} {:>7.3f}  {}".format("*" if i == 0 else " ", s, t))


if __name__ == "__main__":
    main()
