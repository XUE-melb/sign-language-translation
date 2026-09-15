"""用训练好的 checkpoint 在指定 split 上评测。

正式流程：用 dev 选出 best checkpoint，再用本脚本在 **test** 上评一次。

⚠️ test 只评一次。看了 test 的数字再回去调超参/改模型，test 就被污染成
第二个 dev 了，之后报出来的数字不再是"未见过数据上的表现"。

用法：
  python -m slt.evaluate --ckpt runs/stage0_s1234/best.pt --split test
"""
import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

from slt.data import CharVocab, PoseSLTDataset, collate_fn
from slt.data_rtm import RTMPoseSLTDataset
from slt.metrics import evaluate as compute_metrics, format_report
from slt.models.stage0 import Seq2SeqLSTM
from slt.models.stage1 import Stage1Model


def pick_dataset(name):
    """输入源选择。mediapipe = E-000 用的 538 维；rtm = Uni-Sign 对齐的 207 维。

    两者不可混用：阶段 1 的冻结编码器只认 rtm 这一套（见 D-020）。
    """
    return {"mediapipe": PoseSLTDataset, "rtm": RTMPoseSLTDataset}[name]


@torch.no_grad()
def decode_split(model, loader, vocab, device, max_len=60):
    model.eval()
    hyps, refs, trs, nums = [], [], [], []
    for b in loader:
        feats = b["feats"].to(device, non_blocking=True)
        lens = b["feat_lens"].to(device, non_blocking=True)
        ids = model.greedy_decode(feats, lens, max_len=max_len)
        hyps += [vocab.decode(ids[i].tolist()) for i in range(ids.size(0))]
        refs += b["texts"]
        trs += b["translators"]
        nums += b["numbers"]
    return hyps, refs, trs, nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--root", default="/root/autodl-tmp/slt/CE-CSL")
    ap.add_argument("--csv-dir", default="/root/autodl-tmp/slt/TFNet/data/CE-CSL")
    ap.add_argument("--vocab", default="",
                    help="默认取 checkpoint 同目录下的 vocab.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="输出文件名后缀，便于区分多次运行")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    vocab_path = args.vocab or os.path.join(run_dir, "vocab.json")
    vocab = CharVocab.load(vocab_path)

    ck = torch.load(args.ckpt, map_location=device)
    tr_args = ck["args"]

    # 数据侧参数必须与训练时一致，否则评的不是同一个东西。
    # 输入源也从 checkpoint 里读，不允许命令行覆盖 —— 用错输入源的模型
    # 维度对不上会直接报错，但若两者维度恰好相同就会静默评错。
    inp = tr_args.get("input", "mediapipe")
    ds = pick_dataset(inp)(root=args.root, csv_dir=args.csv_dir, split=args.split,
                           vocab=vocab, frame_stride=tr_args["frame_stride"],
                           max_frames=tr_args["max_frames"],
                           normalize=tr_args["normalize"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn,
                    pin_memory=True)

    stage = tr_args.get("stage", 0)
    Model = Seq2SeqLSTM if stage == 0 else Stage1Model
    model = Model(**ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])

    print("checkpoint: {}".format(args.ckpt))
    print("  阶段 {} | 来自 epoch {}，训练时 dev BLEU-4 {:.2f}，seed {}".format(
        stage,
        ck["epoch"], ck["dev_bleu4"], tr_args.get("seed", "?")))
    print("  训练 split {} | 本次评测 split {} | {} 条（缺关键点 {} 条）".format(
        tr_args["train_split"], args.split, len(ds), len(ds.missing)))
    print("  输入源 {} | 词表 {} | 归一化 {} | frame_stride {} | max_frames {}".format(
        inp, len(vocab), tr_args["normalize"], tr_args["frame_stride"],
        tr_args["max_frames"]))

    hyps, refs, trs, nums = decode_split(model, dl, vocab, device, args.max_len)
    res = compute_metrics(hyps, refs, trs)
    res["_meta"] = {
        "ckpt": os.path.abspath(args.ckpt), "split": args.split,
        "epoch": ck["epoch"], "dev_bleu4_at_select": ck["dev_bleu4"],
        "seed": tr_args.get("seed"), "train_split": tr_args["train_split"],
        "protocol": "signer-dependent，CE-CSL 官方划分",
        "bleu": "sacrebleu corpus_bleu tokenize=zh（字级）",
        "rouge": "rouge-chinese，字级", "input": inp,
    }

    print()
    print(format_report(res, "[{} · {}]".format(args.split, os.path.basename(run_dir))))
    print()
    print("  协议: signer-dependent（官方划分，test 中训练未见过的手语者为 0/12）")
    print("  BLEU: sacrebleu tokenize=zh 字级 | ROUGE: rouge-chinese 字级")
    uniq = len(set(hyps))
    print("  输出多样性: {} 条预测中有 {} 种不同句子".format(len(hyps), uniq))

    tag = ("_" + args.tag) if args.tag else ""
    with open(os.path.join(run_dir, "eval_{}{}.json".format(args.split, tag)),
              "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "predictions_{}{}.csv".format(args.split, tag)),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "translator", "hyp", "ref"])
        w.writerows(zip(nums, trs, hyps, refs))
    print("\n已写入 {}/eval_{}{}.json 与 predictions_{}{}.csv".format(
        run_dir, args.split, tag, args.split, tag))


if __name__ == "__main__":
    main()
