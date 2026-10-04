# -*- coding: utf-8 -*-
"""
جدول المقارنة النهائي على Web-Waste-Mini: MobileNetV2 (محلي) مقابل Gemini Vision (سحابي)
لا يعمل إلا عند اكتمال التجربتين (summary كلاهما complete) — لا مقارنة جزئية إطلاقًا.
المخرجات (results/webset/): comparison_table.md + comparison_table.csv + per_class_f1_comparison.png
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifest_utils import CLASSES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "webset"

def main():
    need = ["predictions_webset.csv", "summary_webset.json",
            "predictions_gemini_webset.csv", "summary_gemini_webset.json"]
    missing = [f for f in need if not (OUT / f).exists() or (OUT / f).stat().st_size == 0]
    if missing:
        print("المخرجات الناقصة:", ", ".join(missing)); sys.exit(2)
    sm = json.loads((OUT / "summary_webset.json").read_text(encoding="utf-8"))
    sg = json.loads((OUT / "summary_gemini_webset.json").read_text(encoding="utf-8"))
    if sg.get("completeness") != "complete":
        print("تجربة Gemini غير مكتملة — لا مقارنة."); sys.exit(2)

    def load(name):
        with (OUT / name).open(encoding="utf-8") as f:
            return {r["image_id"]: r for r in csv.DictReader(f)}
    pm, pg = load("predictions_webset.csv"), load("predictions_gemini_webset.csv")
    ids = sorted(set(pm) & set(pg))
    if not ids:
        print("لا صور مشتركة."); sys.exit(2)
    bad = [i for i in ids if pm[i]["true_label"] != pg[i]["true_label"]
           or pm[i]["predicted_label"] not in CLASSES or pg[i]["predicted_label"] not in CLASSES]
    if bad:
        print(f"قيم غير صالحة في {len(bad)} صف — لا مقارنة. أمثلة:", bad[:5]); sys.exit(2)

    def stats(pred):
        y_t = [pred[i]["true_label"] for i in ids]
        y_p = [pred[i]["predicted_label"] for i in ids]
        acc = float(np.mean(np.array(y_t) == np.array(y_p)))
        f1s = []
        for c in CLASSES:
            tp = sum(1 for i in ids if pred[i]["true_label"] == c and pred[i]["predicted_label"] == c)
            fp = sum(1 for i in ids if pred[i]["true_label"] != c and pred[i]["predicted_label"] == c)
            fn = sum(1 for i in ids if pred[i]["true_label"] == c and pred[i]["predicted_label"] != c)
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return acc, float(np.mean(f1s)), f1s

    am, fm, f1m = stats(pm)
    ag, fg, f1g = stats(pg)

    lines = ["# جدول المقارنة النهائي — Web-Waste-Mini (كل الصور، بلا استثناءات)", "",
             f"عدد الصور: {len(ids)} | تاريخ: MobileNet {sm['run_date'][:10]} / Gemini {sg['test_date'][:10]}", "",
             "| المقياس | MobileNetV2 (محلي) | Gemini Vision (سحابي) |", "|---|---:|---:|"]
    rows_csv = [["metric", "mobilenet", "gemini"]]
    def add(m, a, b):
        rows_csv.append([m, a, b]); lines.append(f"| {m} | {a} | {b} |")
    add("الدقة الكلية Accuracy", f"{am:.4f} ({am:.1%})", f"{ag:.4f} ({ag:.1%})")
    add("Macro F1", f"{fm:.4f}", f"{fg:.4f}")
    for c, a, b in zip(CLASSES, f1m, f1g):
        add(f"F1_{c}", f"{a:.4f}", f"{b:.4f}")
    add("زمن الاستجابة", f"محلي e2e {sm['timing_definition']['e2e_avg_ms']}ms (بلا شبكة)",
        f"سحابي {sg.get('cloud_latency_avg_ms', '—')}ms (شامل الشبكة)")
    add("التكلفة في التجربة", "تشغيل محلي شبه معدوم", sg.get("cost_note", "—"))
    add("الإنترنت مطلوب", "لا", "نعم")
    add("الخصوصية", "الصورة لا تغادر الجهاز", "الصورة تُرسل لخدمة خارجية")
    add("النموذج/الإصدار", "MobileNetV2 (مدرب محليًا على TrashNet)",
        f"{sg['model_requested']} ({', '.join(sg['model_version_observed'])})")

    (OUT / "comparison_table.csv").open("w", newline="", encoding="utf-8").write(
        "\n".join(",".join(map(str, r)) for r in rows_csv))
    (OUT / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, f1m, 0.4, label=f"MobileNetV2 (Acc {am:.1%})")
    ax.bar(x + 0.2, f1g, 0.4, label=f"Gemini zero-shot (Acc {ag:.1%})")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("F1"); ax.legend(); ax.set_title("MobileNetV2 vs Gemini zero-shot — Web-Waste-Mini")
    fig.tight_layout(); fig.savefig(OUT / "per_class_f1_comparison.png", dpi=150)

    print(f"المقارنة أُنتجت من {len(ids)} صورة: MobileNet {am:.1%} مقابل Gemini {ag:.1%}")
    print("المخرجات: comparison_table.md/csv + per_class_f1_comparison.png")

if __name__ == "__main__":
    main()
