"""阶段 0 训练与评测。

词表只从 **train.csv** 构建，即使 train 的关键点还没提完 —— 用 dev/test 的
句子建词表等于把测试集的字表泄漏进模型，是隐蔽但真实的泄漏。

用法：
  # 冒烟：在 dev 上训练+评测，只验证流程能跑通（不是结果！）
  python -m slt.train --train-split dev --eval-split dev --epochs 3 --smoke

  # 正式：train 训练，dev 选 checkpoint
  python -m slt.train --train-split train --eval-split dev --epochs 60
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from slt.data import CharVocab, PoseSLTDataset, collate_fn
from slt.data_rtm import RTMPoseSLTDataset
from slt.metrics import evaluate, format_report
from slt.models.decoder import build_loss
from slt.models.stage0 import Seq2SeqLSTM
from slt.models.stage1 import Stage1Model


def pick_dataset(name):
    """输入源选择。mediapipe = E-000 用的 538 维；rtm = Uni-Sign 对齐的 207 维。

    两者不可混用：阶段 1 的冻结编码器只认 rtm 这一套（见 D-020）。
    """
    return {"mediapipe": PoseSLTDataset, "rtm": RTMPoseSLTDataset}[name]


def build_vocab(csv_dir, cache, min_freq=1):
    """始终从 train.csv 建词表。"""
    if cache and os.path.exists(cache):
        return CharVocab.load(cache)
    rows = list(csv.DictReader(open(os.path.join(csv_dir, "train.csv"),
                                    encoding="utf-8")))
    vocab = CharVocab.build([r["Chinese Sentences"].strip() for r in rows], min_freq)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        vocab.save(cache)
    return vocab


@torch.no_grad()
def run_eval(model, loader, vocab, device, max_len=60):
    model.eval()
    hyps, refs, trs = [], [], []
    for batch in loader:
        feats = batch["feats"].to(device)
        lens = batch["feat_lens"].to(device)
        ids = model.greedy_decode(feats, lens, max_len=max_len)
        for i in range(ids.size(0)):
            hyps.append(vocab.decode(ids[i].tolist()))
        refs.extend(batch["texts"])
        trs.extend(batch["translators"])
    return evaluate(hyps, refs, trs), hyps, refs, trs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/slt/CE-CSL")
    ap.add_argument("--csv-dir", default="/root/autodl-tmp/slt/TFNet/data/CE-CSL")
    ap.add_argument("--out", default="/root/autodl-tmp/slt/runs/stage0")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--frame-stride", type=int, default=None,
                    help="不传则按 --input 自动选：mediapipe=2（沿用 E-000 的算力折衷），"
                         "rtm=1（原生帧率，匹配 Uni-Sign 预训练时的采样约定，见 D-022）")
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--normalize", default="body", choices=["body", "none"])
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1],
                    help="0 = 逐帧 MLP + BiLSTM；1 = 冻结的 Uni-Sign 编码器")
    ap.add_argument("--unisign-ckpt",
                    default="/root/autodl-tmp/slt/weights/unisign/csl_stage1_weight.pth")
    ap.add_argument("--input", default="mediapipe", choices=["mediapipe", "rtm"],
                    help="rtm = RTMPose 69 点(207 维)，阶段 1 的冻结编码器只认这个")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：只验证流程，结果不得写入 EXPERIMENTS.md")
    args = ap.parse_args()

    if args.frame_stride is None:
        args.frame_stride = 2 if args.input == "mediapipe" else 1

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.train_split == args.eval_split:
        print("!" * 72)
        print("!! 训练集与评测集相同 ({})。这只能用于验证流程能跑通，".format(args.train_split))
        print("!! 得到的任何数字都**不是结果**，不得写入 docs/EXPERIMENTS.md。")
        print("!" * 72, flush=True)

    vocab = build_vocab(args.csv_dir, os.path.join(args.out, "vocab.json"))
    print("词表大小 {}（仅由 train.csv 构建）| 输入源 {}".format(
        len(vocab), args.input), flush=True)

    common = dict(root=args.root, csv_dir=args.csv_dir, vocab=vocab,
                  frame_stride=args.frame_stride, max_frames=args.max_frames,
                  normalize=args.normalize)
    DS = pick_dataset(args.input)
    ds_tr = DS(split=args.train_split, **common)
    ds_ev = DS(split=args.eval_split, **common)
    print("train[{}] {} 条（缺关键点 {} 条） | eval[{}] {} 条（缺 {} 条）".format(
        args.train_split, len(ds_tr), len(ds_tr.missing),
        args.eval_split, len(ds_ev), len(ds_ev.missing)), flush=True)
    if len(ds_tr) == 0:
        raise SystemExit("训练集为空：该 split 的关键点还没提取完")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_fn,
                       drop_last=False, pin_memory=True)
    dl_ev = DataLoader(ds_ev, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_fn,
                       pin_memory=True)

    # 显式写出来并存进 checkpoint，evaluate.py 据此重建同构模型，
    # 不依赖"默认值将来不变"这种隐含假设
    ids = dict(pad_id=CharVocab.PAD, bos_id=CharVocab.BOS, eos_id=CharVocab.EOS)
    if args.stage == 0:
        model_cfg = dict(in_dim=ds_tr.dim, vocab_size=len(vocab),
                         frame_hidden=512, feat_dim=512, enc_hidden=512,
                         dec_hidden=512, emb_dim=256, enc_layers=2,
                         dropout=0.3, **ids)
        model = Seq2SeqLSTM(**model_cfg).to(device)
    else:
        if args.input != "rtm":
            raise SystemExit("阶段 1 的冻结编码器只认 rtm 输入（见 D-020）")
        model_cfg = dict(vocab_size=len(vocab), ckpt_path=args.unisign_ckpt,
                         dec_hidden=512, emb_dim=256, dropout=0.3,
                         freeze_encoder=True, **ids)
        model = Stage1Model(**model_cfg).to(device)
        print("  冻结编码器载入 {} 个张量（missing {} / unexpected {}）".format(
            model.load_info["loaded"], len(model.load_info["missing"]),
            len(model.load_info["unexpected"])), flush=True)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print("阶段 {} | 可训练 {:.2f}M / 总计 {:.2f}M | 输入维度 {}".format(
        args.stage, n_par / 1e6, n_all / 1e6, ds_tr.dim), flush=True)

    crit = build_loss(CharVocab.PAD, args.label_smoothing)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # 不用 scheduler：D-017 定的协议是"搜一次 lr，全程固定"。一个挂在噪声很大的
    # dev BLEU-4 上的 ReduceLROnPlateau 会在训练中把这个"固定值"悄悄改掉，
    # 而且不落盘、事后无法追溯（见 D-022 —— 曾经因此把 100 个 epoch 的
    # 预算变成实际只有约 20 个 epoch 在真正优化，其余在 lr≈0 下空转）。
    # 固定 lr 意味着可能会晚一点过拟合甚至过拟合更狠，但 best.pt 是按
    # dev BLEU-4 选的，不依赖训练能不能"收住"，所以不需要 scheduler 兜底。

    best = -1.0
    hist = []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for batch in dl_tr:
            feats = batch["feats"].to(device, non_blocking=True)
            lens = batch["feat_lens"].to(device, non_blocking=True)
            tok = batch["tokens"].to(device, non_blocking=True)

            logits = model(feats, lens, tok)          # (B, L-1, V)
            # 目标是 tokens 右移一位：用前 t 个字预测第 t+1 个字
            loss = crit(logits.reshape(-1, logits.size(-1)),
                        tok[:, 1:].reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step()
            tot += loss.item()
            nb += 1

        msg = "epoch {:>3}  loss {:.4f}  {:.1f}s".format(ep, tot / max(nb, 1),
                                                         time.time() - t0)
        if ep % args.eval_every == 0 or ep == args.epochs:
            res, hyps, refs, _ = run_eval(model, dl_ev, vocab, device)
            msg += "  | BLEU-4 {:.2f}  ROUGE-L {:.2f}".format(
                res["bleu4"], res["rouge-l"])
            hist.append({"epoch": ep, "loss": tot / max(nb, 1),
                         "bleu4": res["bleu4"], "rouge-l": res["rouge-l"],
                         "lr": opt.param_groups[0]["lr"]})
            if res["bleu4"] > best:
                best = res["bleu4"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "model_cfg": model_cfg, "epoch": ep,
                            "dev_bleu4": best},
                           os.path.join(args.out, "best.pt"))
            if ep % (args.eval_every * 5) == 0 or ep == args.epochs:
                print("  样例  预测: {}\n        参考: {}".format(
                    hyps[0], refs[0]), flush=True)
        print(msg, flush=True)

    # 最终评测 + 落盘
    res, hyps, refs, trs = run_eval(model, dl_ev, vocab, device)
    print("\n" + format_report(res, "[最终 · {}]".format(args.eval_split)), flush=True)
    print("\n总训练耗时 {:.1f} 分钟".format((time.time() - t_start) / 60), flush=True)

    with open(os.path.join(args.out, "predictions_{}.csv".format(args.eval_split)),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["translator", "hyp", "ref"])
        w.writerows(zip(trs, hyps, refs))
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": hist, "final": res, "args": vars(args),
                   "smoke": args.smoke or args.train_split == args.eval_split},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
