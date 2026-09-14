"""BLEU-4 与 ROUGE 计算。

tokenize 方式（面试高频陷阱，见 DECISIONS.md D-010）：
中文统一按**字**切分。BLEU 用 sacrebleu 的 tokenize="zh"，ROUGE 喂空格分隔
的单字序列。不同实现/不同切分方式之间差几个点是常事，所以：
**报告数字时必须连同实现和切分方式一起报。**
"""
import collections

import sacrebleu
from rouge_chinese import Rouge


def _char_tokens(s):
    """按字切分并用空格连接，给 rouge-chinese 用。"""
    return " ".join(list(s.strip())) or "空"


def corpus_bleu4(hyps, refs):
    """sacrebleu，中文分词器。返回 0-100 的 BLEU-4。"""
    if not hyps:
        return float("nan")
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score


def corpus_rouge(hyps, refs):
    """rouge-chinese，字级。返回 rouge-1/2/l 的 F 值（0-100）。"""
    if not hyps:
        return {"rouge-1": float("nan"), "rouge-2": float("nan"),
                "rouge-l": float("nan")}
    # 空预测会让 rouge-chinese 抛异常；用占位符顶上，保持样本数不变，
    # 这样空预测被算作 0 分而不是被悄悄丢掉（丢掉会虚高）
    h = [_char_tokens(x) for x in hyps]
    r = [_char_tokens(x) for x in refs]
    sc = Rouge().get_scores(h, r, avg=True)
    return {k: sc[k]["f"] * 100 for k in ("rouge-1", "rouge-2", "rouge-l")}


def evaluate(hyps, refs, translators=None):
    """总体指标 + 按手语者分层（D-009 派生要求：不能只给一个总分）。"""
    res = {"n": len(hyps), "bleu4": corpus_bleu4(hyps, refs)}
    res.update(corpus_rouge(hyps, refs))

    if translators is not None:
        by = collections.defaultdict(lambda: ([], []))
        for h, r, t in zip(hyps, refs, translators):
            by[t][0].append(h)
            by[t][1].append(r)
        per = {}
        for t in sorted(by):
            h, r = by[t]
            row = {"n": len(h), "bleu4": corpus_bleu4(h, r)}
            row.update(corpus_rouge(h, r))
            per[t] = row
        res["per_translator"] = per
    return res


def format_report(res, title=""):
    lines = []
    if title:
        lines.append(title)
    lines.append("  n={}  BLEU-4 {:.2f}  ROUGE-1 {:.2f}  ROUGE-2 {:.2f}  ROUGE-L {:.2f}"
                 .format(res["n"], res["bleu4"], res["rouge-1"],
                         res["rouge-2"], res["rouge-l"]))
    per = res.get("per_translator")
    if per:
        lines.append("  按手语者分层:")
        lines.append("    {:<4}{:>5}{:>10}{:>10}".format("T", "n", "BLEU-4", "ROUGE-L"))
        for t, row in per.items():
            lines.append("    {:<4}{:>5}{:>10.2f}{:>10.2f}".format(
                t, row["n"], row["bleu4"], row["rouge-l"]))
    return "\n".join(lines)
