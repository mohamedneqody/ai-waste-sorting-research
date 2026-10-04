# -*- coding: utf-8 -*-
"""
توليد/تحديث eg_waste_manifest.csv من الصور الموجودة فعليًا في data/local_egypt/<class>/
- يملأ آليًا: image_id، image_path، true_label (من اسم الفولدر — حقيقة ملفية)
- يترك فارغة: scene_type و difficulty و notes (حكم بشري — تملؤها الطالبة يدويًا)
- عند وجود بيان سابق: يحافظ على أعمدة الحكم البشري المعبأة ولا يمسحها
لا يخترع أي metadata: ما لم يُملأ يدويًا يبقى فارغًا.
"""
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # رسائل عربية سليمة حتى على CP1252

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "local_egypt"
MANIFEST = BASE / "eg_waste_manifest.csv"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTS = {".jpg", ".jpeg", ".png"}

def scan():
    rows = []
    for c in CLASSES:
        folder = IMG_DIR / c
        if not folder.exists():
            print(f"  تحذير: فولدر الفئة {c} غير موجود — سيغيب من البيان حتى يُنشأ")
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in EXTS:
                rows.append({"image_id": f"{c}/{p.name}",
                             "image_path": str(p.relative_to(BASE)),
                             "true_label": c,
                             "scene_type": "", "difficulty": "", "notes": ""})
    return rows

def main():
    if not IMG_DIR.exists():
        raise SystemExit(f"لا يوجد فولدر الصور: {IMG_DIR}\n"
                         "ضعي الصور أولًا حسب docs/image_collection_protocol.md ثم أعيدي التشغيل.")
    rows = scan()
    if not rows:
        raise SystemExit("لم يُعثر على أي صورة (jpg/jpeg/png) داخل الفولدرات الستة.")

    # دمج مع بيان سابق: الحفاظ على الحكم البشري المعبأ
    old = {}
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                old[r.get("image_id", "")] = r
        print(f"بيان سابق به {len(old)} صف — سيُحافظ على scene_type/difficulty/notes المعبأة.")
    kept = 0
    for r in rows:
        o = old.get(r["image_id"])
        if o:
            for k in ("scene_type", "difficulty", "notes"):
                if o.get(k, "").strip():
                    r[k] = o[k]
                    kept += 1
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "image_path", "true_label",
                                          "scene_type", "difficulty", "notes"])
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["true_label"]] = counts.get(r["true_label"], 0) + 1
    total = len(rows)
    print(f"تم كتابة {MANIFEST} — {total} صورة: {counts}")
    print(f"صفوف حُفظ فيها حكم بشري سابق: {kept}")
    empty = total * 2 - sum(1 for r in rows for k in ("scene_type", "difficulty") if r[k].strip())
    print(f"خانات scene_type/difficulty الفارغة المطلوب ملؤها يدويًا: {empty}")
    print("تذكير: scene_type = single أو crowded — difficulty = normal أو transparent_hard")

if __name__ == "__main__":
    main()
