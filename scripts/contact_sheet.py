# -*- coding: utf-8 -*-
"""
أداة المعاينة البصرية (Contact Sheet) — الخطوة الوسيطة بين وضع الصور والتقييم
الأدوار: الطالبة تصور وتضع الصور في الفولدرات فقط. مهندس المشروع (الوكيل) يملأ scene_type/difficulty
وفق القواعد المعلنة أدناه، ويعرض عليها في تقرير واحد فقط الصور غير الواضحة للحسم.

القواعد المعلنة (وثيقة الحكم البصري — تُطبق على كل صورة):
  scene_type = single   : عنصر مخلف واحد مسيطر وواضح يشغل معظم الكادر (~2/3 فأكثر) بخلفية بسيطة
  scene_type = crowded  : عدة عناصر أو مشهد حقيقي مزدحم (ترابيزة/مكتب/سلة) يتداخل فيه العنصر مع غيره
  difficulty = normal   : العنصر الرئيسي غير شفاف أو لا يشبه بصريًا فئة أخرى قابلة للخلط
  difficulty = transparent_hard : العنصر الرئيسي شفاف/واضح يشبه بصريًا الفئة الأخرى (بلاستيك↔زجاج)
  unclear = yes         : الحكم غير مؤكد — تُعرض الصورة على الطالبة في تقرير واحد (unclear_report.md)

الاستخدام (ينفذه الوكيل، لا الطالبة):
  python scripts/contact_sheet.py          → توليد أوراق المعاينة + قالب القرارات
  python scripts/contact_sheet.py apply    → بعد تعبئة القرارات: تحديث البيان + تقرير غير الواضح
"""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from manifest_utils import CLASSES, SCENE_TYPES, DIFFICULTIES, scan_folder_images

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # رسائل عربية سليمة حتى على CP1252

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "local_egypt"
REVIEW = BASE / "review"  # ملفات المعاينة منفصلة عن results/ (المخصص لمخرجات التقييم الفعلية فقط)
DECISIONS = REVIEW / "contact_sheet_decisions.csv"
MANIFEST = BASE / "eg_waste_manifest.csv"
THUMB_W, THUMB_H, COLS, ROWS_PER_SHEET = 220, 170, 4, 6

def load_decisions():
    """قراءة القرارات السابقة إن وجدت (للحفاظ على العمل المنجز)"""
    if not DECISIONS.exists():
        return {}
    with DECISIONS.open(encoding="utf-8-sig") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}

def generate():
    disk = scan_folder_images(IMG_DIR, BASE)
    if not disk:
        print("لا توجد صور في data/local_egypt/<الفئة>/ — ضع الصور أولًا ثم أعد التشغيل.")
        sys.exit(2)
    REVIEW.mkdir(parents=True, exist_ok=True)
    old = load_decisions()

    # قالب/تحديث قرارات: image_id + true_label من الفولدرات فعليًا، والحفاظ على أي حكم سابق
    items = sorted(disk.items())
    with DECISIONS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "true_label", "scene_type", "difficulty", "unclear"])
        for rel, lbl in items:
            o = old.get(f"{lbl}/{Path(rel).name}")
            prev = [o.get("scene_type", ""), o.get("difficulty", ""), o.get("unclear", "")] if o else ["", "", ""]
            w.writerow([f"{lbl}/{Path(rel).name}", lbl] + prev)

    # أوراق المعاينة: شبكة مصغرات مع التعليق image_id + التسمية
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    per_sheet = COLS * ROWS_PER_SHEET
    n_sheets = (len(items) + per_sheet - 1) // per_sheet
    for s in range(n_sheets):
        chunk = items[s * per_sheet:(s + 1) * per_sheet]
        sheet = Image.new("RGB", (COLS * (THUMB_W + 10) + 10,
                                  ROWS_PER_SHEET * (THUMB_H + 34) + 10), "white")
        d = ImageDraw.Draw(sheet)
        for k, (rel, lbl) in enumerate(chunk):
            col, row = k % COLS, k // COLS
            x0, y0 = 10 + col * (THUMB_W + 10), 10 + row * (THUMB_H + 34)
            try:
                im = Image.open(BASE / rel).convert("RGB")
                im.thumbnail((THUMB_W, THUMB_H))
                sheet.paste(im, (x0 + (THUMB_W - im.width) // 2, y0))
            except Exception as e:
                d.text((x0 + 4, y0 + 4), f"ERROR: {e}"[:40], fill="red", font=font)
            iid = f"{lbl}/{Path(rel).name}"
            d.text((x0 + 2, y0 + THUMB_H + 2), iid, fill="black", font=font)
            o = old.get(iid)
            hint = f"scene={o.get('scene_type','?')} diff={o.get('difficulty','?')}" + \
                   (" [unclear]" if o and o.get("unclear") else "") if o else "scene=? diff=?"
            d.text((x0 + 2, y0 + THUMB_H + 16), hint, fill="#555555", font=font)
        out = REVIEW / f"contact_sheet_{s + 1:02d}.png"
        sheet.save(out, dpi=(110, 110))
        print(f"  {out} ({len(chunk)} صورة)")
    print(f"\nأُنتج {n_sheets} ورقة معاينة و {DECISIONS}")
    print("الخطوة التالية (على الوكيل): فحص الأوراق بصريًا وتعبئة scene_type/difficulty/unclear في ملف القرارات وفق القواعد المعلنة أعلاه.")
    print(f"ثم: python scripts/contact_sheet.py apply")

def apply():
    disk = scan_folder_images(IMG_DIR, BASE)
    if not disk:
        print("لا توجد صور — لا شيء للتطبيق."); sys.exit(2)
    if not MANIFEST.exists():
        print("البيان غير موجود — شغّل scripts/make_manifest.py أولًا."); sys.exit(2)
    old = load_decisions()
    if not old:
        print("ملف القرارات فارغ/غير موجود — ولّد الأوراق أولًا (generate) وعبّئ القرارات."); sys.exit(2)

    problems, decisions = [], {}
    for rel, lbl in sorted(disk.items()):
        iid = f"{lbl}/{Path(rel).name}"
        o = old.get(iid)
        if not o:
            problems.append(f"{iid}: لا يوجد صف قرار — أكمل ملف القرارات"); continue
        scene = (o.get("scene_type") or "").strip()
        diff = (o.get("difficulty") or "").strip()
        unclear = (o.get("unclear") or "").strip().lower()
        if scene not in SCENE_TYPES:
            problems.append(f"{iid}: scene_type غير صالح ({scene or 'فارغ'})"); continue
        if diff not in DIFFICULTIES:
            problems.append(f"{iid}: difficulty غير صالح ({diff or 'فارغ'})"); continue
        if unclear not in ("", "yes"):
            problems.append(f"{iid}: unclear يجب أن يكون فارغًا أو yes (الموجود: {unclear})"); continue
        decisions[iid] = {"scene_type": scene, "difficulty": diff, "unclear": unclear}
    if problems:
        print("=== التطبيق متوقف — مشاكل في ملف القرارات ===")
        for p in problems[:20]:
            print("  -", p)
        if len(problems) > 20:
            print(f"  … و{len(problems) - 20} أخرى")
        sys.exit(2)

    # تحديث البيان بالقرارات (بالدمج على image_id — لا يلمس أعمدة أخرى)
    with MANIFEST.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    updated = 0
    for r in rows:
        iid = (r.get("image_id") or "").strip().replace("\\", "/")
        d = decisions.get(iid)
        if d:
            r["scene_type"] = d["scene_type"]
            r["difficulty"] = d["difficulty"]
            updated += 1
    missing = [iid for iid in decisions if iid not in {(r.get("image_id") or "").strip().replace("\\", "/") for r in rows}]
    if missing:
        print(f"=== توقف: {len(missing)} قرار لصور غير موجودة في البيان — أعد تشغيل make_manifest.py ثم apply ===")
        for m in missing[:5]:
            print("  -", m)
        sys.exit(2)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    unclear = sorted(iid for iid, d in decisions.items() if d["unclear"] == "yes")
    report = ["# تقرير الحالات غير الواضحة — يحتاج حسم الطالبة (تقرير واحد)", "",
              f"التاريخ: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
              f"عدد الصور الكلي: {len(decisions)} — غير الواضح: {len(unclear)}", ""]
    if unclear:
        report.append("| image_id | المسار | الحكم المقترح |")
        report.append("|---|---|---|")
        for iid in unclear:
            scene = decisions[iid]["scene_type"]; diff = decisions[iid]["difficulty"]
            report.append(f"| {iid} | data/local_egypt/{iid} | scene={scene}, difficulty={diff} |")
        report += ["", "الطالبة تحسم كل حالة (تأكيد أو تعديل) — يعدّل الوكيل ملف القرارات ويطبق مرة أخيرة."]
    else:
        report.append("لا توجد حالات غير واضحة — الحكم البصري مكتمل وواثق لكل الصور.")
    (REVIEW / "unclear_report.md").write_text("\n".join(report), encoding="utf-8")

    n_unclear = len(unclear)
    print(f"تم تحديث البيان: {updated} صف. تقرير غير الواضح: review/unclear_report.md ({n_unclear} حالة)")
    if n_unclear:
        print("انتظر حسم الطالبة للحالات غير الواضحة قبل اعتماد البيان للتقييم.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        apply()
    else:
        generate()
