# -*- coding: utf-8 -*-
"""
المهمة 5 (محدثة 2026-10-04) — مقارنة النموذجين على كل صور البيان فقط
قاعدة صارمة: المقارنة لا تُنفذ إلا إذا:
  - البيان صالح ومطابق للفولدرات (validate_manifest)
  - كل صورة في البيان موجودة في predictions_mobilenet.csv و predictions_gemini.csv
  - true_label في الملفين متطابق مع تسمية البيان لكل صورة بلا استثناء
لا تُستبعد صور مخالفة ثم تُكمل المقارنة على جزء من البيانات — أي مخالفة توقف المقارنة كلها.
المخرجات (results/): comparison_table.csv + comparison_table.md + confusion_matrices_comparison.png
+ per_class_f1_comparison.png
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifest_utils import CLASSES, validate_manifest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # رسائل عربية سليمة حتى على CP1252

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "local_egypt"
MANIFEST = BASE / "eg_waste_manifest.csv"
OUT = BASE / "results"

def main():
    needed = ["predictions_mobilenet.csv", "predictions_gemini.csv",
              "summary_mobilenet.json", "summary_gemini.json"]
    missing = [f for f in needed if not (OUT / f).exists() or (OUT / f).stat().st_size == 0]
    if missing:
        print("=== المقارنة غير ممكنة الآن — المخرجات الحقيقية التالية غير موجودة ===")
        for m in missing:
            print("  results/" + m)
        print("شغلي أولًا: scripts/evaluate_mobilenet.py ثم scripts/evaluate_gemini.py بعد اكتمال الصور والبيان.")
        sys.exit(2)

    ok, rows, problems = validate_manifest(MANIFEST, IMG_DIR, BASE)
    if not ok:
        print("=== الحارس: البيان غير صالح — لا مقارنة ===")
        for pr in problems:
            print("  -", pr)
        sys.exit(2)

    def load(name):
        with (OUT / name).open(encoding="utf-8") as f:
            return {r["image_id"]: r for r in csv.DictReader(f)}
    pm, pg = load("predictions_mobilenet.csv"), load("predictions_gemini.csv")

    violations = []
    for r in rows:
        iid, lbl = r["image_id"], r["true_label"]
        if iid not in pm:
            violations.append(f"{iid}: غائبة عن predictions_mobilenet.csv")
        elif pm[iid]["true_label"] != lbl:
            violations.append(f"{iid}: true_label في predictions_mobilenet ({pm[iid]['true_label']}) يخالف البيان ({lbl})")
        if iid not in pg:
            violations.append(f"{iid}: غائبة عن predictions_gemini.csv")
        elif pg[iid]["true_label"] != lbl:
            violations.append(f"{iid}: true_label في predictions_gemini ({pg[iid]['true_label']}) يخالف البيان ({lbl})")
    extra_m = sorted(set(pm) - {r["image_id"] for r in rows})
    extra_g = sorted(set(pg) - {r["image_id"] for r in rows})
    if extra_m:
        violations.append(f"{len(extra_m)} صف زائد في predictions_mobilenet خارج البيان: {extra_m[:5]}")
    if extra_g:
        violations.append(f"{len(extra_g)} صف زائد في predictions_gemini خارج البيان: {extra_g[:5]}")
    if violations:
        print("=== المقارنة مرفوضة: الملفان لا يغطيان البيان كاملًا بتسميات مطابقة — لا مقارنة جزئية ===")
        for v in violations[:20]:
            print("  -", v)
        if len(violations) > 20:
            print(f"  … و{len(violations) - 20} مخالفة أخرى")
        sys.exit(2)

    # تحقق صريح: كل predicted_label في الملفين ينتمي للفئات الست — أي قيمة فارغة أو غير صحيحة توقف المقارنة
    invalid = []
    for fname, pred in (("predictions_mobilenet.csv", pm), ("predictions_gemini.csv", pg)):
        for iid in ids:
            v = pred[iid].get("predicted_label", "")
            if v not in CLASSES:
                invalid.append(f"{fname}: {iid} → predicted_label غير صالح ({v or 'فارغ'})")
    if invalid:
        print("=== المقارنة مرفوضة: قيم predicted_label فارغة أو غير صحيحة — لا مقارنة على بيانات معيبة ===")
        for v in invalid[:20]:
            print("  -", v)
        if len(invalid) > 20:
            print(f"  … و{len(invalid) - 20} أخرى")
        sys.exit(2)

    sm = json.loads((OUT / "summary_mobilenet.json").read_text(encoding="utf-8"))
    sg = json.loads((OUT / "summary_gemini.json").read_text(encoding="utf-8"))
    if sg.get("completeness") != "complete":
        print("=== المقارنة مرفوضة: تجربة Gemini غير مكتملة (summary_gemini.json يقول incomplete) ===")
        print("أكمل تجربة Gemini على كل صور البيان أولًا ثم أعد المقارنة.")
        sys.exit(2)

    ids = [r["image_id"] for r in rows]
    def acc(pred):
        return float(np.mean([pred[i]["true_label"] == pred[i]["predicted_label"] for i in ids]))
    def macro_f1_from_preds(pred):
        f1s = []
        for c in CLASSES:
            tp = sum(1 for i in ids if pred[i]["predicted_label"] == c and pred[i]["true_label"] == c)
            fp = sum(1 for i in ids if pred[i]["predicted_label"] == c and pred[i]["true_label"] != c)
            fn = sum(1 for i in ids if pred[i]["predicted_label"] != c and pred[i]["true_label"] == c)
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return float(np.mean(f1s)), f1s
    am, af1m = acc(pm), macro_f1_from_preds(pm)
    ag, af1g = acc(pg), macro_f1_from_preds(pg)

    lines_md = ["| المقياس | MobileNetV2 (محلي) | Gemini Vision (سحابي) |", "|---|---:|---:|"]
    rows_csv = [["metric", "mobilenet", "gemini"]]
    def add(metric, a, b):
        rows_csv.append([metric, a, b]); lines_md.append(f"| {metric} | {a} | {b} |")
    add("عدد الصور (كل صور البيان)", len(ids), len(ids))
    add("Accuracy", f"{am:.4f}", f"{ag:.4f}")
    add("Macro F1", f"{af1m[0]:.4f}", f"{af1g[0]:.4f}")
    for c, fm, fg in zip(CLASSES, af1m[1], af1g[1]):
        add(f"F1_{c}", f"{fm:.4f}", f"{fg:.4f}")
    tm = sm.get("timing_definition", {})
    add("متوسط الزمن المحلي e2e (ms)", tm.get("e2e_avg_ms", "—"), "غير منطبق")
    lats_g = [float(pg[i]["cloud_latency_ms"]) for i in ids if pg[i].get("cloud_latency_ms") not in (None, "")]
    add("متوسط الزمن السحابي incl. شبكة (ms)", "غير منطبق",
        round(float(np.mean(lats_g)), 1) if lats_g else "—")
    add("التكلفة في التجربة", "تشغيل محلي شبه معدوم", sg.get("cost_note", "—"))
    add("الإنترنت مطلوب", "لا", "نعم")

    (OUT / "comparison_table.csv").open("w", newline="", encoding="utf-8").write(
        "\n".join(",".join(map(str, r)) for r in rows_csv))
    (OUT / "comparison_table.md").write_text(
        "# جدول المقارنة النهائي (كل صور البيان — بلا استثناءات)\n\n" + "\n".join(lines_md) + "\n",
        encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, (name, pred, cmap) in zip(ax, [("MobileNetV2", pm, "Blues"), ("Gemini zero-shot", pg, "Purples")]):
        cm = np.zeros((6, 6), int)
        for i in ids:
            cm[CLASSES.index(pred[i]["true_label"]), CLASSES.index(pred[i]["predicted_label"])] += 1
        a.imshow(cm, cmap=cmap)
        a.set_xticks(range(6)); a.set_xticklabels(CLASSES, rotation=45, ha="right")
        a.set_yticks(range(6)); a.set_yticklabels(CLASSES)
        for x in range(6):
            for y in range(6):
                a.text(y, x, str(cm[x, y]), ha="center", va="center",
                       color="white" if cm[x, y] > cm.max() / 2 else "black")
        a.set_title(f"{name} — Acc {acc(pred):.1%}")
    fig.suptitle(f"EG-Waste-Mini — {len(ids)} صورة (كل صور البيان)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices_comparison.png", dpi=150)

    x = np.arange(6)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(x - 0.2, af1m[1], 0.4, label="MobileNetV2")
    ax2.bar(x + 0.2, af1g[1], 0.4, label="Gemini zero-shot")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES)
    ax2.set_ylabel("F1"); ax2.legend(); ax2.set_title("F1 لكل فئة — EG-Waste-Mini")
    fig2.tight_layout(); fig2.savefig(OUT / "per_class_f1_comparison.png", dpi=150)

    print(f"المقارنة أُنتجت من كل صور البيان ({len(ids)} صورة): MobileNet {am:.4f} مقابل Gemini {ag:.4f}")
    print("المخرجات: comparison_table.csv/md + confusion_matrices_comparison.png + per_class_f1_comparison.png")

if __name__ == "__main__":
    main()
