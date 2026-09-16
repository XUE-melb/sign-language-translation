"""阶段 2：只换解码器 —— 冻结 Uni-Sign 编码器 + mT5（+LoRA）+ 可训练 pose_proj。

相对 E-001 只改了一件事：LSTM 解码器 -> mT5。视觉端仍是同一个冻结编码器，
输入同一批 pkl、同一套预处理、同一个 frame_stride / max_frames。

可训练的部分（--lora-r 0 时只有 pose_proj，即 CLAUDE.md 原文的 LLaVA 式）：
    pose_proj      Linear(1024 -> 768)，从 checkpoint 初始化（不是从零）
    LoRA on mT5    q/k/v/o，r=--lora-r，默认 16
编码器与 mT5 主干冻结。编码器强制 eval()（BatchNorm 统计量不能被更新，D-021）。

评测口径与前两个阶段一致：dev 选点用 greedy（和 E-001 同口径），
test 同时报 greedy 与 beam4（后者是 Uni-Sign 的设置）。指标走 slt.metrics。

用法：
  训练：python -m slt.train_stage2 --out runs/E002_s1234 --seed 1234
  test： python -m slt.train_stage2 --test-only runs/E002_s1234
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

from slt.data import CharVocab, collate_fn
from slt.data_rtm import RTMPoseSLTDataset
from slt.metrics import evaluate, format_report
from slt.models.unisign_full import CKPT, UniSignFull

ROOT = "/root/autodl-tmp/slt/CE-CSL"
CSVD = "/root/autodl-tmp/slt/TFNet/data/CE-CSL"


def build_model(args, device):
    model = UniSignFull(ckpt_path=args.unisign_ckpt,
                        load_mt5_weights=not args.hf_mt5).to(device)
    for p in model.encoder.parameters():          # 视觉端不动
        p.requires_grad_(False)
    for p in model.mt5.parameters():              # 主干冻结
        p.requires_grad_(False)
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout,
                         target_modules=["q", "k", "v", "o"], bias="none",
                         task_type="SEQ_2_SEQ_LM")
        model.mt5 = get_peft_model(model.mt5, cfg)
    for p in model.pose_proj.parameters():
        p.requires_grad_(True)
    return model


def set_train_mode(model, training):
    model.train(training)
    model.encoder.eval()                          # 冻结编码器永远 eval


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
    if args.lora_r > 0:
        model.mt5.save_pretrained(os.path.join(d, "lora"))
    json.dump({"epoch": epoch, "dev_bleu4": dev_bleu4, "args": vars(args)},
              open(os.path.join(d, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def load_best(model, out, args):
    d = os.path.join(out, "best")
    model.pose_proj.load_state_dict(torch.load(os.path.join(d, "pose_proj.pt"), map_location="cpu"))
    if args.lora_r > 0:
        from peft import PeftModel
        base = model.mt5.get_base_model() if hasattr(model.mt5, "get_base_model") else model.mt5
        model.mt5 = PeftModel.from_pretrained(base, os.path.join(d, "lora"))
    return json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))


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
        res["_meta"] = {"split": "test", "decode": "{}, max_new_tokens=100".format(name),
                        "epoch": meta["epoch"], "dev_bleu4_at_select": meta["dev_bleu4"],
                        "seed": args.seed, "stage": 2, "lora_r": args.lora_r,
                        "unisign_ckpt": os.path.basename(args.unisign_ckpt),
                        "hf_mt5": args.hf_mt5,
                        "protocol": "signer-dependent，CE-CSL 官方划分",
                        "bleu": "sacrebleu tokenize=zh 字级", "rouge": "rouge-chinese 字级"}
        print(format_report(res, "[test · {} · {}]".format(os.path.basename(out), name)))
        print("  输出多样性: {}/{}".format(len(set(h)), len(h)))
        json.dump(res, open(os.path.join(out, "eval_test_{}.json".format(name)), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
        with open(os.path.join(out, "predictions_test_{}.csv".format(name)), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["number", "translator", "hyp", "ref"])
            w.writerows(zip(n, t, h, r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
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
    ap.add_argument("--test-only", default="", help="给 run 目录，只在 test 上评 best")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if args.test_only:
        args.out = args.test_only
        saved = json.load(open(os.path.join(args.out, "best", "meta.json"), encoding="utf-8"))["args"]
        for k in ("lora_r", "lora_alpha", "lora_dropout", "unisign_ckpt", "hf_mt5", "max_frames", "seed"):
            setattr(args, k, saved[k])
        model = build_model(args, device)
        run_test(model, args, device, args.out)
        return

    os.makedirs(args.out, exist_ok=True)
    if args.train_split == args.eval_split:
        print("!! 训练集与评测集相同，冒烟用，数字不是结果", flush=True)

    model = build_model(args, device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print("阶段 2 | 可训练 {:.2f}M / 总计 {:.1f}M | LoRA r={} | mT5 权重: {} | 权重加载 {}".format(
        n_tr / 1e6, n_all / 1e6, args.lora_r, "HF 原版" if args.hf_mt5 else "Uni-Sign",
        model.load_info), flush=True)

    vocab = CharVocab.build(["x"])
    ds_tr = RTMPoseSLTDataset(ROOT, CSVD, args.train_split, vocab, frame_stride=1,
                              max_frames=args.max_frames)
    ds_ev = RTMPoseSLTDataset(ROOT, CSVD, args.eval_split, vocab, frame_stride=1,
                              max_frames=args.max_frames)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    dl_ev = DataLoader(ds_ev, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_fn)
    print("train {} 条 | eval {} 条 | frame_stride=1 max_frames={}".format(
        len(ds_tr), len(ds_ev), args.max_frames), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best, hist = -1.0, []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        set_train_mode(model, True)
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
                     "lr": opt.param_groups[0]["lr"]})
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
