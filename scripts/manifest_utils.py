# -*- coding: utf-8 -*-
"""
مُحقِّق البيان المشترك — تستخدمه سكربتات التقييم والمقارنة
قاعدة صارمة: التقييم يعتمد على eg_waste_manifest.csv فقط، لا على الفولدرات وحدها،
ويجب التطابق التام بين البيان والفولدرات بلا صورة ناقصة من أي جهة.
"""
import csv
from pathlib import Path

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SCENE_TYPES = {"single", "crowded"}
DIFFICULTIES = {"normal", "transparent_hard"}
EXTS = {".jpg", ".jpeg", ".png"}
REQUIRED_COLS = ["image_id", "image_path", "true_label", "scene_type", "difficulty"]

def scan_folder_images(img_dir: Path, base_dir: Path):
    """{مسار نسبي للمشروع: تسمية الفولدر} لكل صورة موجودة فعليًا"""
    out = {}
    if not img_dir.exists():
        return out
    for c in CLASSES:
        for p in sorted((img_dir / c).rglob("*")):
            if p.suffix.lower() in EXTS:
                out[str(p.relative_to(base_dir)).replace("\\", "/")] = c
    return out

def validate_manifest(manifest_path: Path, img_dir: Path, base_dir: Path):
    """
    يرجع (ok, valid_rows, problems)
    ok = لا مشاكل إطلاقًا + تطابق تام بين عدد صفوف البيان الصالحة وصور الفولدرات
    valid_rows = قائمة صفوف مرتبة: image_id, image_path(نسبي للمشروع), true_label, scene_type, difficulty
    """
    problems = []
    if not manifest_path.exists():
        return False, [], [f"ملف البيان غير موجود: {manifest_path} — يُنشأ بتشغيل scripts/make_manifest.py بعد وضع الصور"]
    with manifest_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing_cols:
            return False, [], [f"أعمدة ناقصة في البيان: {missing_cols}"]
        raw_rows = list(reader)
    if not raw_rows:
        return False, [], ["البيان فارغ (لا صفوف) — شغّل scripts/make_manifest.py بعد وضع الصور في الفولدرات"]

    disk = scan_folder_images(img_dir, base_dir)  # rel_path -> folder label
    if not disk:
        problems.append("لا توجد أي صورة في data/local_egypt/<الفئة>/ رغم وجود البيان — ضع الصور أولًا")

    seen_id, seen_path = set(), set()
    valid_rows = []
    for n, r in enumerate(raw_rows, 2):  # السطر 1 = العناوين
        where = f"سطر {n}"
        rid = (r.get("image_id") or "").strip().replace("\\", "/")
        rpath = (r.get("image_path") or "").strip().replace("\\", "/")
        label = (r.get("true_label") or "").strip()
        scene = (r.get("scene_type") or "").strip()
        diff = (r.get("difficulty") or "").strip()
        if not rid:
            problems.append(f"{where}: image_id فارغ"); continue
        if rid in seen_id:
            problems.append(f"{where}: image_id مكرر ({rid})")
        seen_id.add(rid)
        if not rpath:
            problems.append(f"{where} ({rid}): image_path فارغ"); continue
        if rpath in seen_path:
            problems.append(f"{where} ({rid}): image_path مكرر ({rpath})")
        seen_path.add(rpath)
        if label not in CLASSES:
            problems.append(f"{where} ({rid}): true_label غير صحيح ({label or 'فارغ'}) — المسموح: {CLASSES}"); continue
        if scene not in SCENE_TYPES:
            problems.append(f"{where} ({rid}): scene_type غير صالح ({scene or 'فارغ'}) — المسموح: single/crowded")
        if diff not in DIFFICULTIES:
            problems.append(f"{where} ({rid}): difficulty غير صالح ({diff or 'فارغ'}) — المسموح: normal/transparent_hard")
        p = base_dir / rpath
        if not p.exists():
            problems.append(f"{where} ({rid}): الصورة في البيان غير موجودة على القرص ({rpath})"); continue
        folder_label = rpath.split("/")[-2]
        if folder_label != label:
            problems.append(f"{where} ({rid}): true_label ({label}) لا يطابق اسم فولدر الصورة ({folder_label})"); continue
        if rpath not in disk:
            problems.append(f"{where} ({rid}): المسار في البيان خارج فحص الفولدرات"); continue
        valid_rows.append({"image_id": rid, "image_path": rpath, "true_label": label,
                           "scene_type": scene, "difficulty": diff})

    disk_only = sorted(set(disk) - seen_path)
    if disk_only:
        shown = ", ".join(disk_only[:5]) + (" …" if len(disk_only) > 5 else "")
        problems.append(f"{len(disk_only)} صورة موجودة في الفولدرات وغائبة عن البيان: {shown} — "
                        f"أعد تشغيل scripts/make_manifest.py ثم أكمل الحكم البصري")
    ok = (not problems) and len(valid_rows) == len(disk) and len(valid_rows) == len(raw_rows)
    valid_rows.sort(key=lambda r: r["image_path"])
    return ok, valid_rows, problems
