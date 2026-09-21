"""
QA tool for Medetec raw images (fixed thresholds v2).
Run:  python scripts/qa_medetec.py
Moves bad images to _qa_reject/<class>/, borderline to _qa_reject_review/<class>/
Writes data/raw/medetec/qa_report.csv. Nothing is deleted.
"""
from pathlib import Path
import csv, hashlib, shutil
import numpy as np
from PIL import Image

ROOT = Path("data/raw/medetec")
CLASSES = ["diabetic", "pressure", "_venous_inbox"]
MIN_DIM = 150
BLUR_REJECT = 0.0     # disabled: 35mm scans are inherently soft
BLUR_REVIEW = 0.0     # disabled
BRIGHT_LOW, BRIGHT_HIGH = 0, 255   # exposure gate bhi off

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def ahash(p):
    img = Image.open(p).convert("L").resize((8, 8), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def laplacian_var(gray):
    g = gray.astype(np.float32)
    lap = 4*g[1:-1,1:-1] - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]
    return float(lap.var())

def dest_for(verdict, cls):
    base = ROOT / {"reject": "_qa_reject", "review": "_qa_reject_review", "duplicate": "_qa_reject"}[verdict]
    d = base / cls
    d.mkdir(parents=True, exist_ok=True)
    return d

def main():
    md5_seen, ah_seen = {}, {}
    rows = []
    counts = {}
    for cls in CLASSES:
        cdir = ROOT / cls
        if not cdir.exists():
            continue
        k = r = v = d = 0
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            h_ = ah = None
            blur = bright = -1.0
            w = h = 0
            dupof = ""
            try:
                img = Image.open(f)
                w, h = img.size
                gray = np.asarray(img.convert("L"))
                bright = float(gray.mean())
                blur = laplacian_var(gray)
                h_ = md5(f)
                ah = ahash(f)
                verdict, reason = "keep", ""
                if h_ in md5_seen:
                    verdict, reason, dupof = "duplicate", "exact duplicate", md5_seen[h_]
                elif min(w, h) < MIN_DIM:
                    verdict, reason = "reject", f"too small {w}x{h}"
                elif ah is not None and any((ah ^ b).bit_count() <= 4 for b in ah_seen):
                    dupof = next(name for key, name in ah_seen.items() if (ah ^ key).bit_count() <= 4)
                    verdict, reason = "review", "near duplicate"
                elif blur < BLUR_REJECT:
                    verdict, reason = "reject", "extremely blurry"
                elif not (BRIGHT_LOW <= bright <= BRIGHT_HIGH):
                    verdict, reason = "reject", f"exposure {bright:.0f}"
                elif blur < BLUR_REVIEW:
                    verdict, reason = "review", "soft focus borderline"
            except Exception as e:
                verdict, reason = "reject", f"unreadable ({type(e).__name__})"
            if verdict == "keep":
                md5_seen.setdefault(h_, f.name)
                ah_seen.setdefault(ah, f.name)
            rows.append([f.name, cls, verdict, f"{blur:.1f}", f"{bright:.0f}", f"{w}x{h}", (h_ or "")[:12], dupof, reason])
            if verdict == "keep":
                k += 1
            else:
                if verdict == "review":
                    v += 1
                elif verdict == "duplicate":
                    d += 1
                else:
                    r += 1
                shutil.move(str(f), str(dest_for(verdict, cls) / f.name))
        counts[cls] = (k, r, v, d)
    with open(ROOT / "qa_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "class", "verdict", "blur_score", "brightness", "size", "md5", "duplicate_of", "reason"])
        w.writerows(rows)
    print("\nMedetec QA summary (fixed thresholds v2)")
    tk = tr = tv = td = 0
    for cls, (k, r, v, d) in counts.items():
        print(f"  {cls}: keep={k}, reject={r}, review={v}, duplicate={d}")
        tk, tr, tv, td = tk+k, tr+r, tv+v, td+d
    print(f"Total: keep={tk}, reject={tr}, review={tv}, duplicate={td}")
    print("Review folder: _qa_reject_review  |  Report: qa_report.csv")

if __name__ == "__main__":
    main()

