"""BLEU / chrF / ROUGE 计算，含"通顺但无关"地板。

tokenize 方式（面试高频陷阱，见 DECISIONS.md D-010）：
中文统一按**字**切分。BLEU / chrF 用 sacrebleu（tokenize="zh"；chrF 本身按字符算），
ROUGE 喂空格分隔的单字序列。**报告数字时必须连同实现和切分方式一起报。**

为什么不只报 BLEU-4：项目现阶段 BLEU-4 在 1 上下，贴着地板，4-gram 几乎没有
分辨力。BLEU-1..3 和 chrF（字符 n-gram F 值，对中文更合适）能在低分区间区分
"完全乱说"和"用对了字但连不成词"。sacrebleu 免费给，不加白不加。

地板（floor）：把预测句原样保留、只打乱它与参考的配对，得到"流畅但内容无关"
的分数。**任何指标都要和它比**——ROUGE-L 20 看着体面，地板就有 14。
地板计算要跑多次 ROUGE，训练时逐 epoch 的 dev 评测不做（with_floor=False），
只在最终 evaluate 时做。
"""
import collections
import random

import sacrebleu
from sacrebleu.metrics import BLEU
from rouge_chinese import Rouge


def _char_tokens(s):
    """按字切分并用空格连接，给 rouge-chinese 用。"""
    return " ".join(list(s.strip())) or "空"


def corpus_bleu_n(hyps, refs, n):
    """BLEU-n（max_ngram_order=n），sacrebleu 中文分词器，0-100。"""
    if not hyps:
        return float("nan")
    return BLEU(tokenize="zh", max_ngram_order=n).corpus_score(hyps, [refs]).score


def corpus_bleu4(hyps, refs):
    if not hyps:
        return float("nan")
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score


def corpus_chrf(hyps, refs):
    """chrF（字符级 n-gram F 值，默认 char_order=6，word_order=0），0-100。"""
    if not hyps:
        return float("nan")
    return sacrebleu.corpus_chrf(hyps, [refs]).score


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


def _core(hyps, refs):
    d = {"n": len(hyps), "bleu4": corpus_bleu4(hyps, refs)}
    d.update(corpus_rouge(hyps, refs))
    return d


def _extras(hyps, refs):
    return {"bleu1": corpus_bleu_n(hyps, refs, 1),
            "bleu2": corpus_bleu_n(hyps, refs, 2),
            "bleu3": corpus_bleu_n(hyps, refs, 3),
            "chrf": corpus_chrf(hyps, refs)}


def floor_scores(hyps, refs, n_shuffle=5, seed=0):
    """打乱配对的地板：预测不变，只重排与参考的对应关系，取多次均值。"""
    if not hyps:
        return {}
    rng = random.Random(seed)
    acc = collections.defaultdict(float)
    for _ in range(n_shuffle):
        sh = hyps[:]
        rng.shuffle(sh)
        d = _core(sh, refs)
        d.update(_extras(sh, refs))
        for k, v in d.items():
            if k != "n":
                acc[k] += v
    return {k: v / n_shuffle for k, v in acc.items()}


def evaluate(hyps, refs, translators=None, with_floor=False):
    """总体指标 + 按手语者分层（D-009 派生要求：不能只给一个总分）。

    返回键：n, bleu4, bleu1..3, chrf, rouge-1/2/l,
           per_translator（各含 n/bleu4/rouge-*），
           floor（仅 with_floor=True 时，同名指标的打乱配对地板）。
    旧键全部保留，新增键只增不改，训练脚本不受影响。
    """
    res = _core(hyps, refs)
    res.update(_extras(hyps, refs))

    if translators is not None:
        by = collections.defaultdict(lambda: ([], []))
        for h, r, t in zip(hyps, refs, translators):
            by[t][0].append(h)
            by[t][1].append(r)
        res["per_translator"] = {t: _core(h, r) for t, (h, r) in sorted(by.items())}

    if with_floor:
        res["floor"] = floor_scores(hyps, refs)
    return res


def format_report(res, title=""):
    lines = []
    if title:
        lines.append(title)
    lines.append("  n={}  BLEU-4 {:.2f}  BLEU-1/2/3 {:.2f}/{:.2f}/{:.2f}  chrF {:.2f}"
                 .format(res["n"], res["bleu4"], res.get("bleu1", float("nan")),
                         res.get("bleu2", float("nan")), res.get("bleu3", float("nan")),
                         res.get("chrf", float("nan"))))
    lines.append("  ROUGE-1 {:.2f}  ROUGE-2 {:.2f}  ROUGE-L {:.2f}".format(
        res["rouge-1"], res["rouge-2"], res["rouge-l"]))
    fl = res.get("floor")
    if fl:
        lines.append("  地板(打乱配对): BLEU-4 {:.2f}  chrF {:.2f}  ROUGE-L {:.2f}"
                     "  -> 高出地板 BLEU-4 {:+.2f} / chrF {:+.2f} / ROUGE-L {:+.2f}".format(
                         fl["bleu4"], fl["chrf"], fl["rouge-l"],
                         res["bleu4"] - fl["bleu4"], res["chrf"] - fl["chrf"],
                         res["rouge-l"] - fl["rouge-l"]))
    per = res.get("per_translator")
    if per:
        lines.append("  按手语者分层:")
        lines.append("    {:<4}{:>5}{:>10}{:>10}".format("T", "n", "BLEU-4", "ROUGE-L"))
        for t, row in per.items():
            lines.append("    {:<4}{:>5}{:>10.2f}{:>10.2f}".format(
                t, row["n"], row["bleu4"], row["rouge-l"]))
    return "\n".join(lines)
