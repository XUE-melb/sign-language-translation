"""阶段 2 / 阶段 3：Uni-Sign 编码器 + mT5（+LoRA）+ 可训练 pose_proj。

阶段 2（默认）相对 E-001b 只改了一件事：LSTM 解码器 -> mT5。视觉端仍是同一个冻结编码器，
输入同一批 pkl、同一套预处理、同一个 frame_stride / max_frames。

可训练的部分（--lora-r 0 时只有 pose_proj，即 CLAUDE.md 原文的 LLaVA 式）：
    pose_proj      Linear(1024 -> 768)，从 checkpoint 初始化（不是从零）
    LoRA on mT5    q/k/v/o，r=--lora-r，默认 16
编码器与 mT5 主干冻结。冻结的编码器强制 eval()（BatchNorm 统计量不能被更新，D-021）。

阶段 3（D-026）：--unfreeze-encoder。相对阶段 2 唯一的变量是"视觉编码器是否更新"：
编码器 4.56M 参数加入优化器（单独学习率 --encoder-lr），并随 model.train() 进入训练模式
（BatchNorm 统计量更新），与 Uni-Sign 自己的微调方式一致。best 目录多存一个 encoder.pt。

E-004 留一手语者（D-026）：--exclude-signer X 让 train 与 dev 都去掉手语者 X；
评测时 --test-only DIR --eval-signer X 在 X 的 train+dev+test 全部片段上评一次
（对留出者来说全是没见过的手语者 + 没见过的句子），并单独报其中原 test 划分的部分。

评测口径与前两个阶段一致：dev 选点用 greedy（和 E-001b 同口径），
test 同时报 greedy 与 beam4（后者是 Uni-Sign 的设置）。指标走 slt.metrics。

用法：
  训练：python -m slt.train_stage2 --out runs/E002_s1234 --seed 1234
  test： python -m slt.train_stage2 --test-only runs/E002_s1234
  阶段 3：python -m slt.train_stage2 --out runs/E003_s1234 --unfreeze-encoder --encoder-lr 1e-5
  E-004： python -m slt.train_stage2 --out runs/E004_s1234 --exclude-signer E
          python -m slt.train_stage2 --test-only runs/E004_s1234 --eval-signer E
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from slt.data import CharVocab, collate_fn
from slt.data_rtm import RTMPoseSLTDataset
from slt.metrics import evaluate, format_report, set_lang
from slt.models.unisign_full import CKPT, UniSignFull

ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"


def build_model(args, device):
    model = UniSignFull(ckpt_path=args.unisign_ckpt,
                        load_mt5_weights=not args.hf_mt5,
                        lang=getattr(args, "lang", "zh")).to(device)
    unfreeze = bool(getattr(args, "unfreeze_encoder", False))
    for p in model.encoder.parameters():          # 阶段 2 视觉端不动；阶段 3 放开
        p.requires_grad_(unfreeze)
    train_mt5 = bool(getattr(args, "train_mt5", False))
    for p in model.mt5.parameters():              # 主干默认冻结；--train-mt5 全量微调（How2Sign 容量实验，D-029）
        p.requires_grad_(train_mt5)
    if args.lora_r > 0 and not train_mt5:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout,
                         target_modules=["q", "k", "v", "o"], bias="none",
                         task_type="SEQ_2_SEQ_LM")
        model.mt5 = get_peft_model(model.mt5, cfg)
    for p in model.pose_proj.parameters():
        p.requires_grad_(True)
    return model


def set_train_mode(model, training, unfreeze=False):
    model.train(training)
    if not unfreeze:
        model.encoder.eval()                      # 冻结编码器永远 eval


@torch.no_grad()
def decode_split(model, loader, device, num_beams, max_new_tokens=100):
    set_train_mode(model, False)
    hyps, refs, trs, nums = [], [], [], []
    for b in loader:
        f = b["feats"].to(device, non_blocking=True)
        L = b["feat_lens"].to(device, non_blocking=True)
        hyps += model.generate(f, L, num_beams=num_beams, max_new_tokens=max_new_tokens)
        refs += b["texts"]
        trs += b["translators"]
        nums += b["numbers"]
    return hyps, refs, trs, nums


def save_best(model, out, args, epoch, dev_bleu4):
    d = os.path.join(out, "best")
    os.makedirs(d, exist_ok=True)
    torch.save(model.pose_proj.state_dict(), os.path.join(d, "pose_proj.pt"))
    if getattr(args, "train_mt5", False):
        torch.save(model.mt5.state_dict(), os.path.join(d, "mt5_full.pt"))
    elif args.lora_r > 0:
        model.mt5.save_pretrained(os.path.join(d, "lora"))
    if getattr(args, "unfreeze_encoder", False):
        torch.save(model.encoder.state_dict(), os.path.join(d, "encoder.pt"))
    json.dump({"epoch": epoch, "dev_bleu4": dev_bleu4, "args": vars(args)},
              open(os.path.join(d, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def load_best(model, out, args):
    d = os.path.join(out, "best")
    model.pose_proj.load_state_dict(torch.load(os.path.join(d, "pose_proj.pt"), map_location="cpu"))
    full = os.path.join(d, "mt5_full.pt")
    if os.path.exists(full):
        model.mt5.load_state_dict(torch.load(full, map_location="cpu"))
    elif args.lora_r > 0:
        from peft import PeftModel
        base = model.mt5.get_base_model() if hasattr(model.mt5, "get_base_model") else model.mt5
        model.mt5 = PeftModel.from_pretrained(base, os.path.join(d, "lora"))
    enc = os.path.join(d, "encoder.pt")
    if os.path.exists(enc):                       # 阶段 3 才有
        model.encoder.load_state_dict(torch.load(enc, map_location="cpu"))
    return json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))


def _meta(args, meta, split, name):
    unfreeze = bool(getattr(args, "unfreeze_encoder", False))
    excl = getattr(args, "exclude_signer", "")
    return {"split": split, "decode": "{}, max_new_tokens=100".format(name),
            "epoch": meta["epoch"], "dev_bleu4_at_select": meta["dev_bleu4"],
            "seed": args.seed, "stage": 3 if unfreeze else 2,
            "lora_r": args.lora_r, "unfreeze_encoder": unfreeze,
            "encoder_lr": getattr(args, "encoder_lr", None), "exclude_signer": excl,
            "lang": getattr(args, "lang", "zh"), "root": ROOT,
            "train_mt5": bool(getattr(args, "train_mt5", False)), "mt5_lr": getattr(args, "mt5_lr", None),
            "augment": bool(getattr(args, "augment", False)),
            "unisign_ckpt": os.path.basename(args.unisign_ckpt), "hf_mt5": args.hf_mt5,
            "protocol": ("留一手语者 {}（train/dev 均去掉）".format(excl) if excl
                         else "signer-dependent，CE-CSL 官方划分"),
            "bleu": "sacrebleu tokenize=zh 字级", "rouge": "rouge-chinese 字级"}


def _dump(out, tag, res, n, t, h, r):
    json.dump(res, open(os.path.join(out, "eval_{}.json".format(tag)), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(out, "predictions_{}.csv".format(tag)), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "translator", "hyp", "ref"])
        w.writerows(zip(n, t, h, r))


def run_test(model, args, device, out):
    """test 只评一次；同时给 greedy（与 E-001 同口径）和 beam4（Uni-Sign 设置）。"""
    vocab = CharVocab.build(["x"])
    ds = RTMPoseSLTDataset(ROOT, CSVD, "test", vocab, frame_stride=1,
                           max_frames=args.max_frames)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, collate_fn=collate_fn)
    meta = load_best(model, out, args)
    print("best: epoch {} dev BLEU-4 {:.2f}".format(meta["epoch"], meta["dev_bleu4"]))
    for name, beams in (("greedy", 1), ("beam4", 4)):
        h, r, t, n = decode_split(model, dl, device, beams)
        res = evaluate(h, r, t, with_floor=True)
        res["_meta"] = _meta(args, meta, "test", name)
        print(format_report(res, "[test · {} · {}]".format(os.path.basename(out), name)))
        print("  输出多样性: {}/{}".format(len(set(h)), len(h)))
        _dump(out, "test_{}".format(name), res, n, t, h, r)
    return meta


def run_signer_eval(model, args, device, out, meta, signer):
    """留一手语者评测：在 signer 的 train+dev+test 全部片段上评。

    对 E-004（训练时 --exclude-signer signer）这 600 条全是没见过的手语者 + 没见过的句子；
    另外单独报其中原 test 划分的那部分，与 E-002 的分手语者表同口径。
    对没留出该手语者的模型跑这个评测，train 部分是训练数据，数字只作参照。
    """
    vocab = CharVocab.build(["x"])
    parts = [RTMPoseSLTDataset(ROOT, CSVD, sp, vocab, frame_stride=1, max_frames=args.max_frames,
                               only_translators=[signer]) for sp in ("train", "dev", "test")]
    ds = ConcatDataset(parts)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    seen = getattr(args, "exclude_signer", "") != signer
    print("留一手语者评测 {}：{} 条（train/dev/test = {}）| 该模型训练时{}见过 {}".format(
        signer, len(ds), [len(p) for p in parts], "" if seen else "没", signer), flush=True)
    for name, beams in (("greedy", 1), ("beam4", 4)):
        h, r, t, n = decode_split(model, dl, device, beams)
        res = {"all": evaluate(h, r, t, with_floor=True)}
        idx = [i for i, x in enumerate(n) if x.startswith("test")]
        if idx:
            res["test_only"] = evaluate([h[i] for i in idx], [r[i] for i in idx], [t[i] for i in idx],
                                        with_floor=True)
        res["_meta"] = _meta(args, meta, "signer-{}-all".format(signer), name)
        res["_meta"]["signer_seen_in_training"] = seen
        res["_meta"]["n_all"] = len(h)
        res["_meta"]["n_test_only"] = len(idx)
        print(format_report(res["all"], "[signer {} 全部 {} 条 · {}]".format(signer, len(h), name)))
        if idx:
            print("  其中原 test 划分 {} 条：BLEU-4 {:.2f}  chrF {:.2f}  R-L {:.2f}".format(
                len(idx), res["test_only"]["bleu4"], res["test_only"]["chrf"], res["test_only"]["rouge-l"]))
        _dump(out, "signer{}_{}".format(signer, name), res, n, t, h, r)


def main():
    global ROOT, CSVD                      # --root / --csv-dir 覆盖模块级默认（How2Sign，D-029）
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="输出目录；--test-only 时可省略（默认同 run 目录）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--unisign-ckpt", default=CKPT)
    ap.add_argument("--hf-mt5", action="store_true",
                    help="用 HF 原版 mT5-base 权重而非 checkpoint 里的（选项 b）")
    ap.add_argument("--lora-r", type=int, default=16, help="0 = 不加 LoRA，只训 pose_proj")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.2,
                    help="Uni-Sign 微调默认 0.2")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--unfreeze-encoder", action="store_true",
                    help="阶段 3：视觉编码器参与训练（D-026）")
    ap.add_argument("--encoder-lr", type=float, default=None,
                    help="编码器学习率，默认等于 --lr；只在 --unfreeze-encoder 时有意义")
    ap.add_argument("--train-mt5", action="store_true",
                    help="全量微调 mT5 主干（忽略 LoRA），配 --mt5-lr；显存约 +10 GB，建议 --bf16（D-029）")
    ap.add_argument("--mt5-lr", type=float, default=None, help="mT5 主干学习率，默认等于 --lr")
    ap.add_argument("--exclude-signer", default="",
                    help="E-004：train 与 dev 都去掉这位手语者（如 E）")
    ap.add_argument("--augment", action="store_true",
                    help="训练集关键点增广（变速/旋转/手部缩放/噪声，E-004b，D-026 补救方向）")
    ap.add_argument("--eval-signer", default="",
                    help="与 --test-only 连用：在该手语者的 train+dev+test 全部片段上评")
    ap.add_argument("--test-only", default="", help="给 run 目录，只在 test 上评 best")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"],
                    help="zh：CE-CSL；en：How2Sign（prefix/标签长度/去空格/评测分词随之切换，D-029）")
    ap.add_argument("--root", default=ROOT, help="数据根目录（含 pose_rtm/）")
    ap.add_argument("--csv-dir", default=CSVD, help="{train,dev,test}.csv 所在目录")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.test_only and not args.out:
        ap.error("训练必须给 --out")

    device = "cuda"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if args.test_only:
        args.out = args.test_only
        saved = json.load(open(os.path.join(args.out, "best", "meta.json"), encoding="utf-8"))["args"]
        for k in ("lora_r", "lora_alpha", "lora_dropout", "unisign_ckpt", "hf_mt5", "max_frames", "seed"):
            setattr(args, k, saved[k])
        for k, dflt in (("unfreeze_encoder", False), ("encoder_lr", None), ("exclude_signer", ""), ("augment", False),
                        ("lang", "zh"), ("root", ROOT), ("csv_dir", CSVD), ("train_mt5", False), ("mt5_lr", None)):
            setattr(args, k, saved.get(k, dflt))      # 旧 run 的 meta 没有这些键
        ROOT, CSVD = args.root, args.csv_dir
        set_lang(args.lang)
        model = build_model(args, device)
        meta = run_test(model, args, device, args.out)
        if args.eval_signer:
            run_signer_eval(model, args, device, args.out, meta, args.eval_signer)
        return

    ROOT, CSVD = args.root, args.csv_dir
    set_lang(args.lang)
    os.makedirs(args.out, exist_ok=True)
    if args.train_split == args.eval_split:
        print("!! 训练集与评测集相同，冒烟用，数字不是结果", flush=True)
    if args.encoder_lr is None:
        args.encoder_lr = args.lr

    model = build_model(args, device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print("阶段 {} | lang={} | 可训练 {:.2f}M / 总计 {:.1f}M | LoRA r={} | 编码器 {} | mT5 权重: {} | 权重加载 {}".format(
        3 if args.unfreeze_encoder else 2, args.lang, n_tr / 1e6, n_all / 1e6, args.lora_r,
        "可训 lr={:g}".format(args.encoder_lr) if args.unfreeze_encoder else "冻结",
        "HF 原版" if args.hf_mt5 else "Uni-Sign", model.load_info), flush=True)

    excl = [args.exclude_signer] if args.exclude_signer else None
    vocab = CharVocab.build(["x"])
    ds_tr = RTMPoseSLTDataset(ROOT, CSVD, args.train_split, vocab, frame_stride=1,
                              max_frames=args.max_frames, exclude_translators=excl,
                              augment=args.augment)
    ds_ev = RTMPoseSLTDataset(ROOT, CSVD, args.eval_split, vocab, frame_stride=1,
                              max_frames=args.max_frames, exclude_translators=excl)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    dl_ev = DataLoader(ds_ev, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_fn)
    print("train {} 条 | eval {} 条 | frame_stride=1 max_frames={}{}".format(
        len(ds_tr), len(ds_ev), args.max_frames,
        " | 已去掉手语者 {}".format(args.exclude_signer) if excl else "")
        + (" | 训练增广开" if args.augment else ""), flush=True)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    mt5_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("mt5.")] if args.train_mt5 else []
    other = [p for name, p in model.named_parameters()
             if p.requires_grad and not name.startswith("encoder.") and not (args.train_mt5 and name.startswith("mt5."))]
    groups = [{"params": other, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr})
    if mt5_params:
        groups.append({"params": mt5_params, "lr": args.mt5_lr if args.mt5_lr is not None else args.lr})
    opt = torch.optim.AdamW(groups)
    best, hist = -1.0, []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        set_train_mode(model, True, args.unfreeze_encoder)
        tot, nb, t0 = 0.0, 0, time.time()
        for b in dl_tr:
            f = b["feats"].to(device, non_blocking=True)
            L = b["feat_lens"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(f, L, b["texts"], label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step()
            tot += loss.item(); nb += 1
        tr_t = time.time() - t0

        h, r, t, _ = decode_split(model, dl_ev, device, num_beams=1)
        res = evaluate(h, r, t)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "bleu4": res["bleu4"],
                     "chrf": res["chrf"], "rouge-l": res["rouge-l"],
                     "lr": opt.param_groups[0]["lr"],
                     "enc_lr": opt.param_groups[1]["lr"] if len(groups) > 1 else None})
        flag = ""
        if res["bleu4"] > best:
            best = res["bleu4"]
            save_best(model, args.out, args, ep, best)
            flag = "  *best"
        print("epoch {:>3}  loss {:.4f}  {:.0f}s | dev greedy BLEU-4 {:.2f}  chrF {:.2f}  R-L {:.2f}{}".format(
            ep, tot / max(nb, 1), tr_t, res["bleu4"], res["chrf"], res["rouge-l"], flag), flush=True)
        if ep % 5 == 0 or ep == args.epochs:
            print("  样例  预测: {}\n        参考: {}".format(h[0], r[0]), flush=True)

    print("\n总训练耗时 {:.1f} 分钟，最佳 dev BLEU-4 {:.2f}".format((time.time() - t_start) / 60, best))
    json.dump({"history": hist, "args": vars(args), "best_dev_bleu4": best,
               "smoke": args.smoke or args.train_split == args.eval_split},
              open(os.path.join(args.out, "history.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
