"""B 层服务端（D-027）：持有模型的常驻进程，浏览器 / 端侧客户端只发数据收文字。

启动（本地）：
    set PYTHONPATH=src
    .venv-infer\\Scripts\\python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
环境变量：SLT_RUN（训练产物目录，默认 runs/E003_enc1e-4_s3456；空字符串 = 零样本底座）

接口（三种输入模式，D-027 §四）：
    GET  /                      页面
    GET  /api/info              模型信息
    POST /api/translate         JSON {keypoints:(T,133,2), scores:(T,133), n_best, beams}   ← 关键点流模式的一次性版本
    POST /api/translate_pkl     multipart 文件（CE-CSL/pose_rtm 的 pkl 格式）
    POST /api/translate_video   multipart 视频 → 服务端 rtmlib 提关键点 → 翻译；顺带返回关键点供画骨架   ← 上传模式
    WS   /ws/stream             端侧逐帧发 {type:"frame", kp:[[x,y]*133], sc:[..]}，每 partial_every 帧回一次部分结果，
                                {type:"end"} 回最终 n-best                                   ← 关键点流模式（C 层卖点）

GPU 上一次只跑一个任务（asyncio.Lock）；torch 调用放线程池，不卡事件循环。
"""
import asyncio
import json
import os
import pickle
import tempfile
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from slt.infer import Translator

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
RUN = os.environ.get("SLT_RUN", "runs/E003_enc1e-4_s3456")   # 阶段 3，按 dev 选的 seed（D-026 结论）


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    app.state.tr = Translator(RUN or None)
    app.state.px = None                                   # rtmlib 提取器，第一次上传视频时再建
    app.state.lock = asyncio.Lock()
    app.state.load_seconds = round(time.time() - t0, 1)
    print("模型就绪 {:.1f}s | device {} | {}".format(app.state.load_seconds, app.state.tr.device, app.state.tr.info), flush=True)
    yield


app = FastAPI(title="SLT demo", lifespan=lifespan)


def _nbest_json(nb):
    return [{"text": t, "score": round(float(s), 4)} for t, s in nb]


async def _translate(feats, n_best=4, beams=4):
    async with app.state.lock:
        loop = asyncio.get_running_loop()
        t0 = time.time()
        nb = await loop.run_in_executor(None, lambda: app.state.tr.translate(feats, n_best=n_best, num_beams=max(beams, 1)))
        return nb, round((time.time() - t0) * 1000)


def _extractor():
    if app.state.px is None:
        from slt.pose_extract import PoseExtractor
        app.state.px = PoseExtractor()
        print("rtmlib 就绪 device={} providers={}".format(app.state.px.device, app.state.px.providers), flush=True)
    return app.state.px


# ------------------------------------------------------------------ 页面与信息

@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(STATIC, "index.html")
    if os.path.exists(p):
        return FileResponse(p)
    return HTMLResponse("<h3>SLT demo</h3><p>接口见 /docs</p>")


@app.get("/api/info")
async def info():
    tr = app.state.tr
    return {"run": RUN, "device": tr.device, "load_seconds": app.state.load_seconds,
            "info": {k: str(v) for k, v in tr.info.items()}, "max_frames": tr.max_frames}


# ------------------------------------------------------------------ 测试集片段（骨架回放模式）

REPO = os.path.abspath(os.path.join(HERE, ".."))
POSE_ROOT = os.path.join(REPO, "CE-CSL", "pose_rtm")
CSV_DIR = os.path.join(REPO, "TFNet", "data", "CE-CSL")


def _clip_index():
    """扫本地有的 pkl（服务器上是全量，本地只拷了部分），配上参考句。私下展示用；关键点不是视频，不出仓库。"""
    if getattr(app.state, "clips", None) is not None:
        return app.state.clips
    import csv
    refs = {}
    for split in ("train", "dev", "test"):
        p = os.path.join(CSV_DIR, split + ".csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding="utf-8")):
                refs[r["Number"].strip()] = r["Chinese Sentences"].strip()
    clips = []
    for split in ("test", "dev", "train"):
        d = os.path.join(POSE_ROOT, split)
        if not os.path.isdir(d):
            continue
        for signer in sorted(os.listdir(d)):
            sd = os.path.join(d, signer)
            if not os.path.isdir(sd):
                continue
            for fn in sorted(os.listdir(sd)):
                if fn.endswith(".pkl"):
                    num = fn[:-4]
                    clips.append({"number": num, "split": split, "signer": signer,
                                  "text": refs.get(num, ""), "path": os.path.join(sd, fn)})
    app.state.clips = clips
    return clips


@app.get("/api/clips")
async def list_clips(split: str = "test", signer: str = ""):
    rows = [c for c in _clip_index() if c["split"] == split and (not signer or c["signer"] == signer)]
    signers = sorted({c["signer"] for c in _clip_index() if c["split"] == split})
    return {"split": split, "signers": signers, "n": len(rows),
            "clips": [{k: c[k] for k in ("number", "signer", "text")} for c in rows[:500]]}


@app.get("/api/clip/{number}")
async def get_clip(number: str):
    hit = next((c for c in _clip_index() if c["number"] == number), None)
    if hit is None:
        raise HTTPException(404, "没有这条片段的关键点：{}".format(number))
    with open(hit["path"], "rb") as f:
        d = pickle.load(f)
    K = np.asarray(d["keypoints"], np.float32); S = np.asarray(d["scores"], np.float32)
    return {"number": number, "split": hit["split"], "signer": hit["signer"], "text": hit["text"],
            "frames": int(len(K)), "fps": 30,
            "keypoints": K.round(4).tolist(), "scores": S.round(3).tolist()}


# ------------------------------------------------------------------ 三种输入

@app.post("/api/translate")
async def translate_keypoints(body: dict):
    try:
        K = np.asarray(body["keypoints"], dtype=np.float32)
        S = np.asarray(body["scores"], dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, "需要 keypoints (T,133,2) 与 scores (T,133)：{}".format(e))
    if K.ndim != 3 or K.shape[1:] != (133, 2) or S.shape != K.shape[:2]:
        raise HTTPException(400, "形状不对：keypoints {} scores {}".format(K.shape, S.shape))
    feats = app.state.tr.features_from_keypoints(K, S)
    nb, ms = await _translate(feats, int(body.get("n_best", 4)), int(body.get("beams", 4)))
    return {"nbest": _nbest_json(nb), "frames": int(len(K)), "frames_used": int(len(feats)), "ms": ms}


@app.post("/api/translate_pkl")
async def translate_pkl(file: UploadFile = File(...), n_best: int = Form(4), beams: int = Form(4)):
    d = pickle.loads(await file.read())
    feats = app.state.tr.features_from_keypoints(d["keypoints"], d["scores"])
    nb, ms = await _translate(feats, n_best, beams)
    return {"nbest": _nbest_json(nb), "frames": int(len(d["keypoints"])), "frames_used": int(len(feats)), "ms": ms,
            "keypoints": np.asarray(d["keypoints"], np.float32).round(4).tolist(),
            "scores": np.asarray(d["scores"], np.float32).round(3).tolist()}


@app.post("/api/translate_video")
async def translate_video(file: UploadFile = File(...), n_best: int = Form(4), beams: int = Form(4),
                          max_seconds: float = Form(20.0)):
    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        path = f.name
    try:
        async with app.state.lock:
            loop = asyncio.get_running_loop()
            K, S, meta = await loop.run_in_executor(None, lambda: _extractor().video(path, max_seconds))
    finally:
        os.unlink(path)
    if len(K) == 0:
        raise HTTPException(422, "视频里没有解出任何帧")
    feats = app.state.tr.features_from_keypoints(K, S)
    nb, ms = await _translate(feats, n_best, beams)
    return {"nbest": _nbest_json(nb), "frames": int(len(K)), "frames_used": int(len(feats)), "ms": ms,
            "extract": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in meta.items()},
            "keypoints": K.round(4).tolist(), "scores": S.round(3).tolist()}


# ------------------------------------------------------------------ D 层 agent：裁判

@app.get("/api/agent/info")
async def agent_info():
    return {"judge": _judge().name, "available": ["anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None,
                                                  "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None, "rule"]}


def _judge():
    if getattr(app.state, "judge", None) is None:
        import sys
        if REPO not in sys.path:                    # agent/ 在仓库根目录，服务用 PYTHONPATH=src 启动
            sys.path.insert(0, REPO)
        from agent.judge import make_judge
        app.state.judge = make_judge()
        print("agent 裁判：{}".format(app.state.judge.name), flush=True)
    return app.state.judge


@app.post("/api/agent/decide")
async def agent_decide(body: dict):
    """输入 {nbest:[{text,score}], context:[str]}；输出裁判决定。LLM 只在候选里选（agent/judge.py）。"""
    nb = body.get("nbest") or []
    if not nb:
        raise HTTPException(400, "nbest 为空")
    cands = [(c["text"], float(c["score"])) for c in nb][:4]
    ctx = [str(x) for x in (body.get("context") or [])][-6:]
    loop = asyncio.get_running_loop()
    d = await loop.run_in_executor(None, lambda: _judge().decide(cands, ctx))
    out = d.to_dict(); out["text"] = cands[d.choice - 1][0]
    return out


# ------------------------------------------------------------------ 关键点流

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """端侧逐帧推关键点。协议：
        → {"type":"start", "partial_every": 30, "n_best": 4, "beams": 4}   可选
        → {"type":"frame", "kp": [[x,y]*133], "sc": [133]}                 每帧
        → {"type":"end"}                                                    结束一句
        ← {"type":"partial", "text":..., "frames": n, "ms": ..}            每 partial_every 帧
        ← {"type":"final", "nbest": [...], "frames": n, "ms": ..}
    """
    await ws.accept()
    kps, scs = [], []
    cfg = {"partial_every": 30, "n_best": 4, "beams": 4}
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "start":
                cfg.update({k: int(msg[k]) for k in ("partial_every", "n_best", "beams") if k in msg})
                kps, scs = [], []
                await ws.send_json({"type": "ready", **cfg})
            elif t == "frame":
                kps.append(np.asarray(msg["kp"], np.float32)); scs.append(np.asarray(msg["sc"], np.float32))
                if cfg["partial_every"] and len(kps) % cfg["partial_every"] == 0:
                    feats = app.state.tr.features_from_keypoints(np.stack(kps), np.stack(scs))
                    nb, ms = await _translate(feats, 1, 1)          # 部分结果用 greedy，快
                    await ws.send_json({"type": "partial", "text": nb[0][0], "score": round(float(nb[0][1]), 4),
                                        "frames": len(kps), "ms": ms})
            elif t == "end":
                if not kps:
                    await ws.send_json({"type": "final", "nbest": [], "frames": 0, "ms": 0})
                    continue
                feats = app.state.tr.features_from_keypoints(np.stack(kps), np.stack(scs))
                nb, ms = await _translate(feats, cfg["n_best"], cfg["beams"])
                await ws.send_json({"type": "final", "nbest": _nbest_json(nb), "frames": len(kps), "ms": ms})
                kps, scs = [], []
            elif t == "reset":
                kps, scs = [], []
            else:
                await ws.send_json({"type": "error", "msg": "未知消息类型 {}".format(t)})
    except WebSocketDisconnect:
        return
