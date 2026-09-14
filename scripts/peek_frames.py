#!/usr/bin/env python
"""抽帧并画上骨架，用来肉眼确认「手腕出画」到底是什么情况。"""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

mp_h = mp.solutions.holistic
mp_d = mp.solutions.drawing_utils
mp_s = mp.solutions.drawing_styles


def montage(video, out_png, n=4, width=640):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    picks = [int(total * f) for f in (0.2, 0.4, 0.6, 0.8)][:n]

    hol = mp_h.Holistic(static_image_mode=False, model_complexity=1,
                        refine_face_landmarks=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    tiles, info = [], []
    for i, fi in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            continue
        res = hol.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        vis = bgr.copy()
        if res.pose_landmarks:
            mp_d.draw_landmarks(vis, res.pose_landmarks, mp_h.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_s.get_default_pose_landmarks_style())
            lm = res.pose_landmarks.landmark
            for idx, tag in ((15, "LW"), (16, "RW")):
                x, y = lm[idx].x, lm[idx].y
                inside = 0 <= x <= 1 and 0 <= y <= 1
                info.append("f{} {} ({:.2f},{:.2f}) {}".format(
                    fi, tag, x, y, "in" if inside else "OUT"))
                px, py = int(np.clip(x, 0, 1) * w), int(np.clip(y, 0, 1) * h)
                cv2.circle(vis, (px, py), 18, (0, 0, 255) if not inside else (0, 255, 0), -1)
        for lms in (res.left_hand_landmarks, res.right_hand_landmarks):
            if lms:
                mp_d.draw_landmarks(vis, lms, mp_h.HAND_CONNECTIONS)
        lab = "frame {}  hands L={} R={}".format(
            fi, int(res.left_hand_landmarks is not None),
            int(res.right_hand_landmarks is not None))
        cv2.putText(vis, lab, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        tiles.append(cv2.resize(vis, (width, int(h * width / w))))
    hol.close()
    cap.release()

    if not tiles:
        print("无帧:", video)
        return
    top = np.hstack(tiles[:2])
    grid = np.vstack([top, np.hstack(tiles[2:4])]) if len(tiles) >= 4 else top
    cv2.imwrite(out_png, grid)
    print("{}  {}x{} {}帧 -> {}".format(os.path.basename(video), w, h, total, out_png))
    for s in info:
        print("   ", s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--outdir", default="/root/autodl-tmp/slt/logs/peek")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        montage(v, os.path.join(args.outdir, name + ".png"))
