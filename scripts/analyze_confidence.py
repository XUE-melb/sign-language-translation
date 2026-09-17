#!/usr/bin/env python
"""置信度校准 / 前缀翻译曲线 / 重排上限 / 分桶分析（D-027 agent 门控与重排的依据）。

只做推理，不训练。用法：
    PYTHONPATH=src python scripts/analyze_confidence.py --run runs/E003_enc1e-4_s3456 [--split test] [--batch 16]
产物：<run>/analysis_confidence.json 与 analysis_confidence.md

四个问题：
  1. 前缀曲线：只看前 25 / 50 / 75 / 100% 的帧，译文有多好？——"边打边译"多早能出对。
  2. 校准：beam4 首选的长度归一化对数概率 exp(score) 分 5 个分位桶，每桶 BLEU / 完全匹配率 / 句级 chrF 是否单调。
     单调，置信度门控就有依据；不单调，agent 设计要改。
  3. 重排上限：4 条候选里按句级 chrF 挑最好的那条（oracle），corpus BLEU 能到多少。这是 LLM 重排的天花板，
     也是"重排值不值得做"的先验。
  4. 分桶：按参考句字数、按帧数分桶的 BLEU。
"""
import argparse
import json
import math
import os
import statistics as st

import numpy as np
import sacrebleu
import torch

from slt.data import CharVocab
from slt.data_rtm import RTMPoseSLTDataset
from slt.infer import Translator

ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"


def corpus_bleu(h, r):
    return sacrebleu.corpus_bleu(h, [r], tokenize="zh").score if h else float("nan")


def corpus_chrf(h, r):
    return sacrebleu.corpus_chrf(h, [r]).score if h else float("nan")


def sent_chrf(h, r):
    return sacrebleu.sentence_chrf(h, [r]).score


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fracs", default="0.25,0.5,0.75,1.0")
    a = ap.parse_args()
    fracs = [float(x) for x in a.fracs.split(",")]

    tr = Translator(a.run, base_ckpt="weights/unisign/csl_daily_pose_only_slt.pth", mt5_dir="weights/mt5-base")
    ds = RTMPoseSLTDataset(ROOT, CSVD, a.split, CharVocab.build(["x"]), frame_stride=1, max_frames=256)
    items = [ds[i] for i in range(len(ds))]
    feats = [it["feat"].numpy() for it in items]
    refs = [it["text"] for it in items]
    nums = [it["number"] for it in items]
    print("run {} | {} {} 条 | device {}".format(a.run, a.split, len(items), tr.device), flush=True)

    def run_batched(fn_feats, **kw):
        out = []
        for i in range(0, len(fn_feats), a.batch):
            out += tr.translate(fn_feats[i:i + a.batch], **kw)
        return out

    # ---------------- 1. 前缀曲线（greedy）
    prefix = {}
    for f in fracs:
        pf = [x[:max(8, int(round(len(x) * f)))] for x in feats]
        res = run_batched(pf, n_best=1, num_beams=1)
        hyps = [r[0][0] for r in res]; sc = [r[0][1] for r in res]
        prefix[f] = {"bleu4": corpus_bleu(hyps, refs), "chrf": corpus_chrf(hyps, refs),
                     "exact": sum(h == r for h, r in zip(hyps, refs)) / len(refs),
                     "mean_conf": float(np.mean(np.exp(sc)))}
        print("前缀 {:>4.0%}: BLEU-4 {:.2f}  chrF {:.2f}  完全匹配 {:.1%}  平均置信 {:.2f}".format(
            f, prefix[f]["bleu4"], prefix[f]["chrf"], prefix[f]["exact"], prefix[f]["mean_conf"]), flush=True)

    # ---------------- 2. beam4 n-best：校准 + 重排上限
    nb = run_batched(feats, n_best=4, num_beams=4)
    top = [x[0][0] for x in nb]; conf = [math.exp(x[0][1]) for x in nb]
    schrf = [sent_chrf(h, r) for h, r in zip(top, refs)]
    exact = [h == r for h, r in zip(top, refs)]
    full = {"bleu4": corpus_bleu(top, refs), "chrf": corpus_chrf(top, refs), "exact": sum(exact) / len(refs)}
    print("beam4 首选: BLEU-4 {:.2f}  chrF {:.2f}  完全匹配 {:.1%}".format(full["bleu4"], full["chrf"], full["exact"]), flush=True)

    order = sorted(range(len(conf)), key=lambda i: conf[i])
    nbins = 5; bins = []
    for b in range(nbins):
        idx = order[b * len(order) // nbins:(b + 1) * len(order) // nbins]
        bins.append({"bin": b + 1, "n": len(idx), "conf_lo": conf[idx[0]], "conf_hi": conf[idx[-1]],
                     "conf_mean": float(np.mean([conf[i] for i in idx])),
                     "bleu4": corpus_bleu([top[i] for i in idx], [refs[i] for i in idx]),
                     "exact": float(np.mean([exact[i] for i in idx])),
                     "sent_chrf": float(np.mean([schrf[i] for i in idx]))})
    rho = spearman(conf, schrf)
    mono = all(bins[i]["sent_chrf"] <= bins[i + 1]["sent_chrf"] for i in range(nbins - 1))
    print("校准（按置信度 5 分位桶）：")
    for b in bins:
        print("  桶{} n={} conf {:.2f}–{:.2f}  BLEU-4 {:.2f}  完全匹配 {:.1%}  句级chrF {:.1f}".format(
            b["bin"], b["n"], b["conf_lo"], b["conf_hi"], b["bleu4"], b["exact"], b["sent_chrf"]))
    print("  Spearman(置信度, 句级chrF) = {:.3f}；桶间单调: {}".format(rho, mono), flush=True)

    # 阈值扫描：低于 τ 的句子交给追问，剩下的（覆盖率）质量多高
    sweep = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        keep = [i for i in range(len(conf)) if conf[i] >= tau]
        ask = [i for i in range(len(conf)) if conf[i] < tau]
        sweep.append({"tau": tau, "coverage": len(keep) / len(conf),
                      "bleu4_kept": corpus_bleu([top[i] for i in keep], [refs[i] for i in keep]) if keep else float("nan"),
                      "bleu4_asked": corpus_bleu([top[i] for i in ask], [refs[i] for i in ask]) if ask else float("nan"),
                      "exact_kept": float(np.mean([exact[i] for i in keep])) if keep else float("nan")})
    print("门控阈值扫描（覆盖率 = 不追问直接输出的比例）：")
    for s in sweep:
        print("  τ={:.1f} 覆盖 {:.0%}  直接输出部分 BLEU-4 {:.2f} 完全匹配 {:.1%} | 追问部分 BLEU-4 {:.2f}".format(
            s["tau"], s["coverage"], s["bleu4_kept"], s["exact_kept"], s["bleu4_asked"]), flush=True)

    # 重排上限（oracle：4 候选里句级 chrF 最高的）
    oracle = []; hit = 0; rank_of_best = []
    for cands, r in zip(nb, refs):
        texts = [c[0] for c in cands]
        sc = [sent_chrf(t, r) for t in texts]
        j = int(np.argmax(sc)); oracle.append(texts[j]); rank_of_best.append(j + 1)
        hit += r in texts
    orc = {"bleu4": corpus_bleu(oracle, refs), "chrf": corpus_chrf(oracle, refs),
           "ref_in_nbest": hit / len(refs), "rank_hist": {str(k): rank_of_best.count(k) for k in (1, 2, 3, 4)}}
    print("重排上限（oracle 选 4 候选中最好）：BLEU-4 {:.2f}（首选 {:.2f}，上限差 +{:.2f}）| 参考句正好在 n-best 里 {:.1%} | 最佳候选排名分布 {}".format(
        orc["bleu4"], full["bleu4"], orc["bleu4"] - full["bleu4"], orc["ref_in_nbest"], orc["rank_hist"]), flush=True)

    # ---------------- 4. 分桶
    def bucket(keyf, edges, label):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = [i for i in range(len(refs)) if lo <= keyf(i) < hi]
            if idx:
                rows.append({"range": "{}–{}".format(lo, hi - 1 if hi != 10 ** 9 else "∞"), "n": len(idx),
                             "bleu4": corpus_bleu([top[i] for i in idx], [refs[i] for i in idx]),
                             "exact": float(np.mean([exact[i] for i in idx]))})
        print(label + "：" + " | ".join("{} n={} B4 {:.1f} 匹配 {:.0%}".format(r["range"], r["n"], r["bleu4"], r["exact"]) for r in rows), flush=True)
        return rows
    by_len = bucket(lambda i: len(refs[i].rstrip("。！？，")), [0, 7, 10, 13, 10 ** 9], "按参考句字数")
    by_T = bucket(lambda i: len(feats[i]), [0, 120, 160, 200, 10 ** 9], "按帧数")

    out = {"run": a.run, "split": a.split, "n": len(refs), "prefix_curve": {str(k): v for k, v in prefix.items()},
           "beam4_top": full, "calibration": {"bins": bins, "spearman_conf_sentchrf": rho, "monotonic": mono},
           "gating_sweep": sweep, "rerank_oracle": orc, "by_ref_len": by_len, "by_frames": by_T,
           "per_sentence": [{"number": n, "ref": r, "top": t, "conf": round(c, 4), "sent_chrf": round(s, 2),
                             "nbest": [[x[0], round(x[1], 4)] for x in cands]}
                            for n, r, t, c, s, cands in zip(nums, refs, top, conf, schrf, nb)]}
    json.dump(out, open(os.path.join(a.run, "analysis_confidence.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = ["# 置信度 / 前缀 / 重排上限分析：{}（{} {} 条）".format(a.run, a.split, len(refs)), "",
          "| 前缀比例 | BLEU-4 | chrF | 完全匹配 | 平均置信 |", "|---|---:|---:|---:|---:|"]
    md += ["| {:.0%} | {:.2f} | {:.2f} | {:.1%} | {:.2f} |".format(f, v["bleu4"], v["chrf"], v["exact"], v["mean_conf"]) for f, v in prefix.items()]
    md += ["", "beam4 首选：BLEU-4 {:.2f}，chrF {:.2f}，完全匹配 {:.1%}".format(full["bleu4"], full["chrf"], full["exact"]), "",
           "| 置信度桶 | n | conf 范围 | BLEU-4 | 完全匹配 | 句级 chrF |", "|---|---:|---|---:|---:|---:|"]
    md += ["| {} | {} | {:.2f}–{:.2f} | {:.2f} | {:.1%} | {:.1f} |".format(b["bin"], b["n"], b["conf_lo"], b["conf_hi"], b["bleu4"], b["exact"], b["sent_chrf"]) for b in bins]
    md += ["", "Spearman(置信度, 句级 chrF) = {:.3f}，桶间单调 = {}".format(rho, mono), "",
           "| τ | 覆盖率 | 直接输出 BLEU-4 | 直接输出完全匹配 | 追问部分 BLEU-4 |", "|---|---:|---:|---:|---:|"]
    md += ["| {:.1f} | {:.0%} | {:.2f} | {:.1%} | {:.2f} |".format(s["tau"], s["coverage"], s["bleu4_kept"], s["exact_kept"], s["bleu4_asked"]) for s in sweep]
    md += ["", "重排上限（oracle）：BLEU-4 {:.2f}（首选 {:.2f}，差 +{:.2f}）；参考句在 n-best 中 {:.1%}；最佳候选排名分布 {}".format(
        orc["bleu4"], full["bleu4"], orc["bleu4"] - full["bleu4"], orc["ref_in_nbest"], orc["rank_hist"]), "",
        "按参考句字数：" + "；".join("{} n={} B4 {:.1f}".format(r["range"], r["n"], r["bleu4"]) for r in by_len),
        "按帧数：" + "；".join("{} n={} B4 {:.1f}".format(r["range"], r["n"], r["bleu4"]) for r in by_T)]
    open(os.path.join(a.run, "analysis_confidence.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("已写 {}/analysis_confidence.{{json,md}}".format(a.run))


if __name__ == "__main__":
    main()
