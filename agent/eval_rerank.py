"""agent 的两行评测（D-027 §五）：n-best 重排 + 门控。

输入：analyze_confidence.py 产出的 analysis_confidence.json（每句的参考句、beam4 n-best 与分数）。
对每句调用一次裁判（Judge.decide），把它选的候选与首选、oracle 比：
  重排行：corpus BLEU-4  首选 vs 裁判选 vs oracle（上限）
  门控行：裁判 ask=true 的句子有多差、ask=false 的有多好；与规则版 τ=0.7 的划分对比
结果落盘可断点续跑（jsonl 按句号缓存），重跑不再花钱。

用法（在仓库根目录，需要对应后端的 API key 环境变量）：
  PYTHONPATH=. python -m agent.eval_rerank --analysis runs/E003_enc1e-4_s3456/analysis_confidence.json --judge anthropic
  可选：--limit 50（先跑 50 条看看）、--workers 4、--context none（test 集没有对话上下文，默认就是 none）
产物：<analysis 同目录>/rerank_<judge>.jsonl（逐句）、rerank_<judge>.json / .md（汇总）
"""
import argparse
import json
import math
import os
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import sacrebleu

from agent.judge import RuleJudge, make_judge

PRICE = {"anthropic": (5.0, 25.0), "deepseek": (0.30, 1.20), "rule": (0.0, 0.0)}   # $/1M 输入、输出（D-027 补充；DeepSeek [待核]）


def corpus_bleu(h, r):
    return sacrebleu.corpus_bleu(h, [r], tokenize="zh").score if h else float("nan")


def sent_chrf(h, r):
    return sacrebleu.sentence_chrf(h, [r]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--judge", default=None, help="anthropic | deepseek | rule（默认按环境变量）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.7, help="规则版门控阈值，作对照")
    ap.add_argument("--retry-errors", action="store_true", help="只重跑缓存里 API 出错回退规则的句子（如余额不足后充值）")
    a = ap.parse_args()

    data = json.load(open(a.analysis, encoding="utf-8"))
    rows = data["per_sentence"][:a.limit] if a.limit else data["per_sentence"]
    judge = make_judge(a.judge)
    out_dir = os.path.dirname(a.analysis)
    cache_p = os.path.join(out_dir, "rerank_{}.jsonl".format(judge.name))
    done = {}
    if os.path.exists(cache_p):
        for line in open(cache_p, encoding="utf-8"):
            d = json.loads(line); done[d["number"]] = d
        if a.retry_errors:
            bad = [k for k, v in done.items() if v["decision"].get("error")]
            for k in bad:
                del done[k]
            with open(cache_p, "w", encoding="utf-8") as f:       # 重写缓存，去掉出错项
                for v in done.values():
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
            print("重跑出错回退的 {} 句".format(len(bad)), flush=True)
    todo = [r for r in rows if r["number"] not in done]
    print("裁判 {} | 共 {} 句，已缓存 {}，待跑 {}".format(judge.name, len(rows), len(rows) - len(todo), len(todo)), flush=True)

    def work(r):
        cands = [(t, s) for t, s in r["nbest"]]
        d = judge.decide(cands, context=())
        return {"number": r["number"], "decision": d.to_dict()}

    t0 = time.time()
    with open(cache_p, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            res = fu.result(); done[res["number"]] = res
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0 or i == len(todo):
                errs = sum(1 for x in done.values() if x["decision"].get("error"))
                print("  {}/{}  {:.0f}s  出错回退 {}".format(i, len(todo), time.time() - t0, errs), flush=True)

    # ---------------- 汇总
    refs, top, chosen, oracle = [], [], [], []
    ask_j, conf, schrf_top, schrf_chosen, changed, improved, worsened = [], [], [], [], 0, 0, 0
    usage_in = usage_out = 0; errors = 0; ms = []
    for r in rows:
        d = done[r["number"]]["decision"]
        texts = [t for t, _ in r["nbest"]]
        c = texts[d["choice"] - 1]
        refs.append(r["ref"]); top.append(r["top"]); chosen.append(c)
        sc = [sent_chrf(t, r["ref"]) for t in texts]; oracle.append(texts[int(max(range(len(sc)), key=lambda k: sc[k]))])
        ask_j.append(bool(d["ask"])); conf.append(r["conf"])
        st_, sc_ = sent_chrf(r["top"], r["ref"]), sent_chrf(c, r["ref"])
        schrf_top.append(st_); schrf_chosen.append(sc_)
        if c != r["top"]:
            changed += 1; improved += sc_ > st_ + 1e-9; worsened += sc_ < st_ - 1e-9
        if d.get("usage"):
            usage_in += d["usage"].get("input", 0); usage_out += d["usage"].get("output", 0)
        errors += bool(d.get("error")); ms.append(d.get("ms", 0))

    n = len(rows)
    b_top, b_chosen, b_oracle = corpus_bleu(top, refs), corpus_bleu(chosen, refs), corpus_bleu(oracle, refs)
    def subset(mask, hyps):
        idx = [i for i in range(n) if mask[i]]
        return {"n": len(idx), "bleu4": corpus_bleu([hyps[i] for i in idx], [refs[i] for i in idx]) if idx else float("nan")}
    ask_rule = [x < a.tau for x in conf]
    gating = {
        "judge": {"ask_rate": sum(ask_j) / n, "kept": subset([not x for x in ask_j], chosen), "asked": subset(ask_j, chosen)},
        "rule_tau": {"tau": a.tau, "ask_rate": sum(ask_rule) / n, "kept": subset([not x for x in ask_rule], top), "asked": subset(ask_rule, top)},
        "agreement": sum(1 for x, y in zip(ask_j, ask_rule) if x == y) / n,
    }
    pin, pout = PRICE.get(judge.name, (0, 0))
    cost = usage_in / 1e6 * pin + usage_out / 1e6 * pout
    summary = {
        "judge": judge.name, "n": n, "analysis": a.analysis,
        "rerank": {"bleu4_top1": b_top, "bleu4_judge": b_chosen, "bleu4_oracle": b_oracle,
                   "delta_judge": b_chosen - b_top, "headroom": b_oracle - b_top,
                   "changed": changed, "improved": improved, "worsened": worsened,
                   "mean_sent_chrf_top1": st.mean(schrf_top), "mean_sent_chrf_judge": st.mean(schrf_chosen)},
        "gating": gating,
        "cost": {"input_tokens": usage_in, "output_tokens": usage_out, "usd": round(cost, 3), "errors_fallback": errors,
                 "mean_ms": int(st.mean(ms)) if ms else 0},
    }
    json.dump(summary, open(os.path.join(out_dir, "rerank_{}.json".format(judge.name)), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = ["# agent 评测：{}（{} 句，无对话上下文）".format(judge.name, n), "",
          "| | BLEU-4 |", "|---|---:|",
          "| 首选（不重排） | {:.2f} |".format(b_top),
          "| **裁判重排** | **{:.2f}**（{:+.2f}） |".format(b_chosen, b_chosen - b_top),
          "| oracle 上限 | {:.2f}（{:+.2f}） |".format(b_oracle, b_oracle - b_top), "",
          "改选 {} 句：改好 {}，改坏 {}，持平 {}（句级 chrF）。".format(changed, improved, worsened, changed - improved - worsened), "",
          "| 门控 | 追问率 | 直接输出 n / BLEU-4 | 追问 n / BLEU-4 |", "|---|---:|---|---|",
          "| 裁判 ask | {:.0%} | {} / {:.2f} | {} / {:.2f} |".format(gating["judge"]["ask_rate"], gating["judge"]["kept"]["n"], gating["judge"]["kept"]["bleu4"], gating["judge"]["asked"]["n"], gating["judge"]["asked"]["bleu4"]),
          "| 规则 τ={:.1f} | {:.0%} | {} / {:.2f} | {} / {:.2f} |".format(a.tau, gating["rule_tau"]["ask_rate"], gating["rule_tau"]["kept"]["n"], gating["rule_tau"]["kept"]["bleu4"], gating["rule_tau"]["asked"]["n"], gating["rule_tau"]["asked"]["bleu4"]),
          "", "裁判与规则的追问判定一致率 {:.0%}。".format(gating["agreement"]),
          "", "成本：输入 {} / 输出 {} token，约 ${:.3f}；平均 {} ms/句；出错回退规则 {} 句。".format(usage_in, usage_out, cost, summary["cost"]["mean_ms"], errors)]
    open(os.path.join(out_dir, "rerank_{}.md".format(judge.name)), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
