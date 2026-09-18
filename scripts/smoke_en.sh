#!/bin/bash
# --lang en 冒烟：英文指标 + How2Sign dev（39 条）当 train/eval 跑 2 轮（原版 mT5，LoRA），看输出带空格、指标可算、best 落盘。
cd /root/autodl-tmp/slt || exit 1
export PYTHONPATH=src
PY=/root/miniconda3/bin/python
rm -f weights/mt5-base/tf_model.h5 weights/mt5-base/flax_model.msgpack
du -sh weights/mt5-base
echo "---- metrics en / zh ----"
$PY - <<'PY'
from slt import metrics as M
h = ["we are going to work on a drill", "i call it painting the wall"]
r = ["we are going to work on an arm drill", "i call it painting the wall."]
M.set_lang("en"); e = M.evaluate(h, r)
print("en: BLEU-4 %.1f  BLEU-1 %.1f  chrF %.1f  R-1 %.1f  R-L %.1f" % (e["bleu4"], e["bleu1"], e["chrf"], e["rouge-1"], e["rouge-l"]))
M.set_lang("zh"); e = M.evaluate(["我喜欢在家看电视。"], ["我喜欢在家看电视。"])
print("zh sanity: BLEU-4 %.1f  R-L %.1f" % (e["bleu4"], e["rouge-l"]))
PY
echo "---- smoke en train: dev 39 clips, 2 epochs, HF mT5 ----"
rm -rf runs/smoke_en
$PY -m slt.train_stage2 --out runs/smoke_en --lang en --hf-mt5 \
    --root /root/autodl-tmp/How2Sign --csv-dir /root/autodl-tmp/How2Sign/csv \
    --unisign-ckpt weights/unisign/csl_daily_pose_only_slt.pth --lora-r 16 --lr 1e-4 \
    --epochs 2 --batch-size 8 --label-smoothing 0.0 --train-split dev --eval-split dev --smoke 2>&1 \
    | grep -E "^epoch|lang=|train .* 条|样例|预测|参考|Error|Traceback|error"
ls runs/smoke_en/best 2>/dev/null
echo "---- test-only path (dev as test? no: real test split 2343 clips is too slow for smoke) ----"
$PY - <<'PY'
import json; m = json.load(open("runs/smoke_en/best/meta.json"))["args"]; print("meta lang/root:", m.get("lang"), m.get("root"))
PY
rm -rf runs/smoke_en
echo "SMOKE END"
