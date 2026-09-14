import csv, os, collections, sys

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
csv_p = f"TFNet/data/CE-CSL/{split}.csv"
vid_root = f"CE-CSL/video/{split}"

rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
csv_cnt = collections.Counter(r["Translator"].strip() for r in rows)
if not os.path.isdir(vid_root):
    print(f"{vid_root} 不存在"); sys.exit(0)
disk_cnt = {d: len(os.listdir(os.path.join(vid_root, d)))
            for d in sorted(os.listdir(vid_root))
            if os.path.isdir(os.path.join(vid_root, d))}

print(f"[{split}] CSV 行数 {len(rows)} | 磁盘 mp4 总数 {sum(disk_cnt.values())}")
print(f"{'T':<4}{'CSV':>6}{'DISK':>6}  状态")
for t in sorted(set(csv_cnt) | set(disk_cnt)):
    c, d = csv_cnt.get(t, 0), disk_cnt.get(t, 0)
    print(f"{t:<4}{c:>6}{d:>6}  {'OK' if c == d else '<<< 不一致'}")

missing = [r for r in rows
           if not os.path.exists(os.path.join(vid_root, r["Translator"].strip(),
                                              r["Number"].strip() + ".mp4"))]
print(f"\nCSV 有记录但磁盘缺文件: {len(missing)}")
for r in missing[:8]:
    print("   ", r["Number"], r["Translator"])

# 反向：磁盘有文件但 CSV 无记录
known = {(r["Translator"].strip(), r["Number"].strip() + ".mp4") for r in rows}
extra = [(t, f) for t in disk_cnt for f in os.listdir(os.path.join(vid_root, t))
         if (t, f) not in known]
print(f"磁盘有文件但 CSV 无记录: {len(extra)}")
for e in extra[:8]:
    print("   ", e)
