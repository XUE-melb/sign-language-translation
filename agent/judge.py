"""D 层 agent 的"裁判"（D-027 §五；依据见 D-028 §四：重排上限 +7.2，置信度校准单调）。

职责只有一个：拿 n-best 候选 + 对话上下文，选一条，并判断是否需要向手语者确认。
硬约束：LLM 只能在候选里选，不能改写、不能合并——这是堵住"把错译润成流畅错译"的地方。

三个后端，一个接口（Judge.decide），运行时用环境变量 SLT_JUDGE 切换：
    anthropic  Claude（官方 SDK，结构化输出 output_config.format）  ← 在澳洲用
    deepseek   DeepSeek（OpenAI 兼容接口，JSON 模式）                ← 回国后用
    rule       置信度门控（不调 API；页面里原来的规则），也是 API 出错时的兜底
agent 的状态、门控、确认流程都在我们的代码里；LLM 只是可插拔的裁判，换厂商不改 agent。
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass

SYSTEM = (
    "你是一个手语翻译系统的复核员。系统把一段中国手语视频翻成中文，给出最多 4 条候选译文和各自的置信度"
    "（0 到 1，越高越自信）。候选来自 beam search：首选通常最可能，但约六成情况下更好的候选不在第一位。\n"
    "你的任务：\n"
    "1. 结合对话上下文（如果有），选出最可能是手语者本意的一条。只能从候选中选，不能改写、不能合并、不能自己造句。\n"
    "2. 判断是否需要向手语者确认（ask）：候选之间意思差别大、或所有候选都不像一句通顺的日常中文、或与上下文明显不搭时 ask=true；"
    "候选彼此只是措辞差异且首选通顺时 ask=false。\n"
    "3. 用一句话说明理由（中文，30 字以内）。\n"
    "常见错误形态：名词被换成形近或音近的词（优惠券→碎片）、动词近义替换（理解→解释）、句尾多出无关成分。"
    "通顺且符合日常语境的优先；不要仅因为置信度高就选它，也不要仅因为长就选它。"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "enum": [1, 2, 3, 4], "description": "选中的候选编号（1 起）"},
        "ask": {"type": "boolean", "description": "是否需要向手语者确认"},
        "reason": {"type": "string", "description": "一句话理由"},
    },
    "required": ["choice", "ask", "reason"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    choice: int            # 1 起的候选编号
    ask: bool              # 是否向手语者确认
    reason: str
    backend: str
    ms: int = 0
    usage: dict | None = None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def user_prompt(candidates, context=()):
    lines = []
    if context:
        lines.append("对话上下文（按时间顺序，最后一句是对方刚说的）：")
        lines += ["- " + c for c in list(context)[-4:]]
        lines.append("")
    lines.append("候选译文（编号、置信度）：")
    for i, (text, score) in enumerate(candidates, 1):
        lines.append("{}. {}  [置信度 {:.2f}]".format(i, text, math.exp(score)))
    lines.append("")
    lines.append("请从中选一条，并判断是否需要确认。")
    return "\n".join(lines)


def _clamp(choice, n):
    try:
        c = int(choice)
    except (TypeError, ValueError):
        return 1
    return min(max(c, 1), n)


class Judge:
    name = "base"

    def decide(self, candidates, context=(), meta=None) -> Decision:
        raise NotImplementedError


class RuleJudge(Judge):
    """置信度门控：首选的 exp(score) 低于 τ 就追问。τ=0.7 来自 D-028 的阈值扫描（覆盖 65%，直接输出部分 24.3）。"""
    name = "rule"

    def __init__(self, tau=0.7):
        self.tau = tau

    def decide(self, candidates, context=(), meta=None):
        conf = math.exp(candidates[0][1])
        ask = conf < self.tau
        return Decision(1, ask, "置信度 {:.2f} {} τ={:.2f}".format(conf, "<" if ask else "≥", self.tau), self.name)


class AnthropicJudge(Judge):
    """Claude 官方 SDK。结构化输出走 output_config.format（JSON schema），保证第一块就是合法 JSON。
    思考默认开（Opus 5 自适应），任务小，effort=low。未接 refusal fallback：候选是日常中文短句，没有拒答面。"""
    name = "anthropic"

    def __init__(self, model="claude-opus-5", effort="low", max_retries=2):
        import anthropic
        self.anthropic = anthropic
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("没有 ANTHROPIC_API_KEY 环境变量；key 只放环境变量，不写进代码或仓库")
        self.client = anthropic.Anthropic(max_retries=max_retries)   # 从 ANTHROPIC_API_KEY 读 key
        self.model = model
        self.effort = effort

    def decide(self, candidates, context=(), meta=None):
        t0 = time.time()
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=4096, system=SYSTEM,
                messages=[{"role": "user", "content": user_prompt(candidates, context)}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": self.effort},
            )
            if r.stop_reason == "refusal":
                d = RuleJudge().decide(candidates, context); d.error = "refusal"; return d
            text = next(b.text for b in r.content if b.type == "text")
            v = json.loads(text)
            usage = {"input": r.usage.input_tokens, "output": r.usage.output_tokens,
                     "cache_read": getattr(r.usage, "cache_read_input_tokens", 0) or 0}
            return Decision(_clamp(v.get("choice"), len(candidates)), bool(v.get("ask")), str(v.get("reason", ""))[:80],
                            self.name, int((time.time() - t0) * 1000), usage)
        except (self.anthropic.APIStatusError, self.anthropic.APIConnectionError, StopIteration, ValueError) as e:
            d = RuleJudge().decide(candidates, context)
            d.error = "{}: {}".format(type(e).__name__, str(e)[:400]); d.ms = int((time.time() - t0) * 1000)
            return d


class OpenAICompatJudge(Judge):
    """DeepSeek 等 OpenAI 兼容接口：JSON 模式 + 温度 0。提示词里必须出现 json 字样（DeepSeek 的要求）。"""
    name = "deepseek"

    def __init__(self, model=None, base_url=None, api_key=None, max_retries=2):
        from openai import OpenAI
        import openai
        self.openai = openai
        self.model = model or os.environ.get("SLT_JUDGE_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                             api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"), max_retries=max_retries)

    def decide(self, candidates, context=(), meta=None):
        t0 = time.time()
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=300,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM + '\n\n只输出一个 JSON 对象：{"choice": 编号, "ask": true/false, "reason": "理由"}'},
                          {"role": "user", "content": user_prompt(candidates, context)}],
            )
            v = json.loads(r.choices[0].message.content)
            usage = {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens} if r.usage else None
            return Decision(_clamp(v.get("choice"), len(candidates)), bool(v.get("ask")), str(v.get("reason", ""))[:80],
                            self.name, int((time.time() - t0) * 1000), usage)
        except (self.openai.APIError, ValueError, KeyError, IndexError) as e:
            d = RuleJudge().decide(candidates, context)
            d.error = "{}: {}".format(type(e).__name__, str(e)[:120]); d.ms = int((time.time() - t0) * 1000)
            return d


def make_judge(name=None, **kw) -> Judge:
    """SLT_JUDGE 未设时：有 ANTHROPIC_API_KEY 用 Claude，否则有 DEEPSEEK_API_KEY 用 DeepSeek，否则规则。"""
    name = (name or os.environ.get("SLT_JUDGE") or
            ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
             "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "rule")).lower()
    if name == "anthropic":
        return AnthropicJudge(**kw)
    if name == "deepseek":
        return OpenAICompatJudge(**kw)
    return RuleJudge(**kw)


if __name__ == "__main__":       # 冒烟：python -m agent.judge [anthropic|deepseek|rule]
    import sys
    j = make_judge(sys.argv[1] if len(sys.argv) > 1 else None)
    cands = [("一年内可以把这块碎片用。", math.log(0.62)), ("一年之内这个优惠券可以用。", math.log(0.55)),
             ("一年内可以把这张卡用。", math.log(0.50)), ("一年内可以用。", math.log(0.41))]
    d = j.decide(cands, context=["请问这个优惠券什么时候过期？"])
    print(j.name, json.dumps(d.to_dict(), ensure_ascii=False))
