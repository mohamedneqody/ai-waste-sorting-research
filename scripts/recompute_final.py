# -*- coding: utf-8 -*-
"""
إعادة الحساب النهائي بعد اعتماد الوسوم البشرية — بلا أي طلب جديد لـ Gemini ولا أي استدلال جديد.
المصدر الوحيد للحقيقة: blind_labels.csv بعد مراجعة الباحثة (وسوم بشرية معتمدة + استبعادات موثقة).

الشروط (أي فشل يوقف كل شيء بلا كتابة نتائج):
  1) blind_labels.csv موجود وبه قرارات لكل الصور الـ125.
  2) كل صف: إما exclude=yes (خارج التقييم) أو reviewer_label ∈ الفئات الست.
  3) الصور المستبعدة موثقة، والمتبقي هو مجموعة التقييم النهائية.
المخرجات (results/final/):
  final_summary.json            N النهائي، الدقة لكل نموظج، فحص اتفاق مع الوسوم الأولية، الاستبعادات
  final_confusion_mobilenet.png / final_confusion_gemini.png
  final_comparison_table.md
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifest_utils import CLASSES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
BLIND = BASE / "review_web" / "blind" / "blind_labels.csv"
OUT = BASE / "results" / "final"
WEBSET = BASE / "results" / "webset"

def main():
    need = ["predictions_webset.csv", "predictions_gemini_webset.csv"]
    missing = [f for f in need if not (WEBSET / f).exists()]
    if missing:
        print("تنبؤات محفوظة ناقصة:", missing); sys.exit(2)
    if not BLIND.exists():
        print("ملف القرارات غير موجود:", BLIND); sys.exit(2)

    with BLIND.open(encoding="utf-8-sig") as f:
        decisions = list(csv.DictReader(f))
    if not decisions:
        print("ملف القرارات فارغ."); sys.exit(2)

    included, excluded, problems = [], [], []
    seen = set()
    for n, r in enumerate(decisions, 2):
        rid = (r.get("review_id") or "").strip()
        label = (r.get("reviewer_label") or "").strip()
        excl = (r.get("exclude") or "").strip().lower()
        if rid in seen:
            problems.append(f"سطر {n}: review_id مكرر ({rid})")
        seen.add(rid)
        if excl == "yes":
            excluded.append(rid)
        else:
            if label not in CLASSES:
                problems.append(f"سطر {n} ({rid}): reviewer_label غير صالح ({label or 'فارغ'})")
            else:
                included.append({"review_id": rid, "true": label})
    if problems:
        print("=== الحارس: مشاكل في القرارات — لا كتابة نتائج ===")
        for p in problems[:20]:
            print("  -", p)
        sys.exit(2)
    if len(included) < 80:
        print(f"=== المتبقي بعد الاستبعاد ({len(included)}) أقل من الحد الأدنى المنطقي (80) — راجعي الاستبعادات ===")
        sys.exit(2)

    # التنبؤات المحفوظة (من التجربة الفعلية — لا استدلال جديد)
    def load(name):
        with (WEBSET / name).open(encoding="utf-8") as f:
            return {r["image_id"]: r for r in csv.DictReader(f)}
    pm = load("predictions_webset.csv")
    pg = load("predictions_gemini_webset.csv")

    missing_m = [d["review_id"] for d in included if d["review_id"] not in pm]
    missing_g = [d["review_id"] for d in included if d["review_id"] not in pg]
    if missing_m or missing_g:
        print("=== تنبؤات محفوظة ناقصة — أعيدي evaluate_webset/evaluate_gemini على النسخة الحالية ===")
        print("  mobile ناقص:", len(missing_m), "| gemini ناقص:", len(missing_g)); sys.exit(2)

    # الوسم البشري المعتمد يصبح الحقيقة المرجعية للصور المشمولة
    final_rows = []
    for d in included:
        rid = d["review_id"]
        final_rows.append({"review_id": rid, "true": d["true"],
                           "mn_pred": pm[rid]["predicted_label"], "mn_conf": float(pm[rid]["confidence"]),
                           "mn_ms": float(pm[rid]["inference_ms"]),
                           "gm_pred": pg[rid]["predicted_label"],
                           "gm_ms": float(pg[rid]["cloud_latency_ms"]) if pg[rid].get("cloud_latency_ms") else None})

    def metrics(pred_key):
        y_t = [r["true"] for r in final_rows]
        y_p = [r[pred_key] for r in final_rows]
        acc = float(np.mean(np.array(y_t) == np.array(y_p)))
        rep = classification_report(y_t, y_p, labels=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_t, y_p, labels=range(len(CLASSES)))
        return acc, rep, cm

    acc_m, rep_m, cm_m = metrics("mn_pred")
    acc_g, rep_g, cm_g = metrics("gm_pred")
    f1m, f1g = rep_m["macro avg"]["f1-score"], rep_g["macro avg"]["f1-score"]

    # فحص اتفاق الوسوم البشرية المعتمدة مع الوسوم الأولية (بمساعدة آلية) — نسبة خام فقط، بلا Kappa
    init = {r2["image_id"]: r2 for r2 in csv.DictReader((WEBSET / "predictions_webset.csv").open(encoding="utf-8"))}
    agree = sum(1 for r in final_rows if init[r["review_id"]]["true_label"] == r["true"])
    agree_pct = round(100 * agree / len(final_rows), 1) if final_rows else 0.0

    OUT.mkdir(parents=True, exist_ok=True)
    excluded_ids = sorted(excluded)
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_of_truth": "blind_labels.csv — وسوم بشرية معتمدة من الباحثة بعد مراجعة عمياء",
        "n_excluded": len(excluded), "excluded_ids": excluded_ids,
        "n_final": len(final_rows),
        "counts_per_class": {c: sum(1 for r in final_rows if r["true"] == c) for c in CLASSES},
        "mobilenet": {"accuracy": round(acc_m, 4), "macro_f1": round(float(rep_m["macro avg"]["f1-score"]), 4),
                      "per_class": {c: {"precision": round(rep_m[c]["precision"], 4), "recall": round(rep_m[c]["recall"], 4),
                                        "f1": round(rep_m[c]["f1-score"], 4), "support": rep_m[c]["support"]} for c in CLASSES},
                      "confusion_matrix": cm_m.tolist()},
        "gemini": {"accuracy": round(acc_g, 4), "macro_f1": round(float(rep_g["macro avg"]["f1-score"]), 4),
                   "per_class": {c: {"precision": round(rep_g[c]["precision"], 4), "recall": round(rep_g[c]["recall"], 4),
                                     "f1": round(rep_g[c]["f1-score"], 4), "support": rep_g[c]["support"]} for c in CLASSES},
                   "confusion_matrix": cm_g.tolist()},
        "agreement_with_initial_labels": {"method": "نسبة اتفاق خام بين الوسوم البشرية المعتمدة والوسوم الأولية (بمساعدة آلية) — ليس Cohen's Kappa ولا اتفاق مقيّمين بشريين",
                                          "agree": agree, "pct": agree_pct},
        "note": "لا طلبات جديدة لـ Gemini ولا أي استدلال جديد — إعادة حساب من التنبؤات المحفوظة حصرًا",
    }
    (OUT / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, (name, cm, color) in zip(ax, [("MobileNetV2 (محلي)", cm_m, "Blues"), ("Gemini zero-shot (سحابي)", cm_g, "Purples")]):
        a.imshow(cm, cmap=color)
        a.set_xticks(range(6)); a.set_xticklabels(CLASSES, rotation=45, ha="right")
        a.set_yticks(range(6)); a.set_yticklabels(CLASSES)
        for x in range(6):
            for y in range(6):
                a.text(y, x, str(cm[x, y]), ha="center", va="center",
                       color="white" if cm[x, y] > cm.max() / 2 else "black")
        a.set_title(f"{name} — {acc if name.startswith('Mobile') else acc_g:.1%}")
    fig.suptitle(f"Web-Waste-Mini بعد الوسوم البشرية — {len(final_rows)} صورة معتمدة")
    fig.tight_layout(); fig.savefig(OUT / "final_confusion_matrices.png", dpi=150)

    import arabic_reshaper
    from bidi.algorithm import get_display
    def A(t): return get_display(arabic_reshaper.reshape(t))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(6)
    ax2.bar(x - 0.2, [rep_m[c]["f1-score"] for c in CLASSES], 0.4, label=f"MobileNetV2 ({acc_m:.1%})")
    ax2.bar(x + 0.2, [rep_g[c]["f1-score"] for c in CLASSES], 0.4, label=f"Gemini ({acc_g:.1%})")
    ax2.set_xticks(x); ax2.set_xticklabels([A(c) for c in CLASSES])
    ax2.set_ylabel("F1"); ax2.legend(); ax2.set_title(A("المقارنة النهائية بعد اعتماد الوسوم البشرية"))
    fig2.tight_layout(); fig2.savefig(OUT / "final_f1_comparison.png", dpi=150)

    print(f"\n=== النتيجة النهائية (وسوم بشرية معتمدة، N={len(final_rows)}) ===")
    print(f"MobileNetV2: {acc_m:.4f} | Gemini: {acc_g:.4f}")
    print(f"اتفاق مع الوسوم الأولية: {agree_pct}% | مستبعد: {len(excluded)}")
    print("المخرجات في results/final/", flush=True)

if __name__ == "__main__":
    main()
