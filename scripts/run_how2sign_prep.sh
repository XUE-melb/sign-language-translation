#!/bin/bash
# How2Sign（ASL → 英文）数据准备链，无人值守（本人 09-18 批准；DECISIONS D-029 待写）。
#  1) 从 HF 镜像 aipieces/How2Sign 下载正面视角句子级片段（train 32 / val 18 / test 12 个 zip，约 35 GB）+ 英文 CSV
#  2) 解压到 How2Sign/video/{train,dev,test}/H2S/*.mp4（val 改名 dev，沿用本项目 dataloader 的划分名）
#  3) 生成 CE-CSL 同列名的 CSV（Number = SENTENCE_NAME，Translator = H2S，"Chinese Sentences" 列放英文）
#  4) 用同一个 RTMPose 提取器（lightweight，D-020）提 133 点 → How2Sign/pose_rtm/{split}/H2S/*.pkl
# 结束打 HOW2SIGN PREP DONE。日志 logs/how2sign_prep.log。数据 CC-BY-NC 4.0，不再分发。
set -u
ROOT=/root/autodl-tmp/How2Sign
SLT=/root/autodl-tmp/slt
LOG=$SLT/logs/how2sign_prep.log
PY=/root/miniconda3/bin/python
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }
mkdir -p "$ROOT/mirror" "$ROOT/video" "$ROOT/csv" "$ROOT/pose_rtm"
source /etc/network_turbo >/dev/null 2>&1
export HF_HUB_DISABLE_XET=1     # Xet CAS 后端在本机 401（09-18 实测），走普通 HTTP 下载

# ---------- 1) 下载（可断点续传，hf_hub 自带校验） ----------
log "下载开始"
cd "$ROOT/mirror"
$PY - 2>&1 <<'PY' | tee -a "$LOG"
import time, os
from huggingface_hub import hf_hub_download, list_repo_files
REPO = "aipieces/How2Sign"
files = list_repo_files(REPO, repo_type="dataset")
want = [f for f in files if f.startswith(("train_rgb_front_clips/", "val_rgb_front_clips/", "test_rgb_front_clips/")) and f.endswith(".zip")]
want += ["how2sign_train.csv", "how2sign_val.csv", "how2sign_test.csv",
         "how2sign_realigned_train.csv", "how2sign_realigned_val.csv", "how2sign_realigned_test.csv"]
t0 = time.time(); tot = 0
for i, f in enumerate(want, 1):
    for attempt in range(5):
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir="."); break
        except Exception as e:
            print("  重试 {} {}: {}".format(attempt + 1, f, str(e)[:120]), flush=True); time.sleep(15)
    else:
        print("!! 放弃 {}".format(f), flush=True); continue
    tot += os.path.getsize(p)
    if i % 5 == 0 or i == len(want):
        el = time.time() - t0
        print("  {}/{}  {:.1f} GB  {:.0f} min  {:.1f} MB/s".format(i, len(want), tot / 1e9, el / 60, tot / 1e6 / max(el, 1)), flush=True)
print("下载完成 {:.1f} GB".format(tot / 1e9))
PY

# ---------- 2) 解压 ----------
for pair in "train:train" "val:dev" "test:test"; do
    src=${pair%%:*}; dst=${pair##*:}
    mkdir -p "$ROOT/video/$dst/H2S"
    n0=$(ls "$ROOT/video/$dst/H2S" 2>/dev/null | wc -l)
    log "解压 $src -> video/$dst/H2S（已有 $n0 个）"
    for z in "$ROOT/mirror/${src}_rgb_front_clips"/*.zip; do
        unzip -q -n -j "$z" -d "$ROOT/video/$dst/H2S" || log "!! 解压失败 $z"      # -n：已有的不重复解压，脚本可重跑
    done
    log "  video/$dst/H2S 现有 $(ls "$ROOT/video/$dst/H2S" | wc -l) 个 mp4"
done

# ---------- 3) CSV（原始对齐版：片段是按原始时间切的，文本用原始版才对得上） ----------
$PY - 2>&1 <<'PY' | tee -a "$LOG"
import csv, os
ROOT = "/root/autodl-tmp/How2Sign"
for src, dst in (("train", "train"), ("val", "dev"), ("test", "test")):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "mirror", "how2sign_%s.csv" % src), encoding="utf-8"), delimiter="\t"))
    have = set(f[:-4] for f in os.listdir(os.path.join(ROOT, "video", dst, "H2S")) if f.endswith(".mp4"))
    out, miss = [], 0
    for r in rows:
        name = r["SENTENCE_NAME"].strip()
        if name not in have:
            miss += 1; continue
        out.append({"Number": name, "Translator": "H2S", "Chinese Sentences": r["SENTENCE"].strip(), "Gloss": "", "Note": r["VIDEO_ID"]})
    with open(os.path.join(ROOT, "csv", dst + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Number", "Translator", "Chinese Sentences", "Gloss", "Note"]); w.writeheader(); w.writerows(out)
    print("csv {}: {} 行（CSV 里有、片段缺失 {}；片段有、CSV 没有 {}）".format(dst, len(out), miss, len(have) - len(out)), flush=True)
PY

# ---------- 4) 提取关键点（同 CE-CSL：lightweight，最高分的人，[w,h] 归一化） ----------
cd "$SLT"
for SPLIT in dev test train; do
    log "RTMPose 提取 $SPLIT 开始"
    bash scripts/run_rtmpose.sh scripts/extract_rtmpose.py --root "$ROOT" --csv-dir "$ROOT/csv" --split "$SPLIT" --workers 3 2>&1 | grep -E "待处理|ETA|完成|!!" | tee -a "$LOG"
    log "RTMPose 提取 $SPLIT 结束：$(ls "$ROOT/pose_rtm/$SPLIT/H2S" 2>/dev/null | wc -l) 个 pkl"
done
du -sh "$ROOT/mirror" "$ROOT/video" "$ROOT/pose_rtm" | tee -a "$LOG"
log "================ HOW2SIGN PREP DONE ================"
