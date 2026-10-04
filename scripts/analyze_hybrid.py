# -*- coding: utf-8 -*-
"""
تحليل النظام الهجين وعتبات الثقة — من التنبؤات الموجودة فقط (بلا أي طلب جديد)
السياسة المعلنة مسبقًا: إذا كانت ثقة MobileNet >= T تُقبل إجابته محليًا؛ وإلا تُصعَّد الصورة إلى Gemini
وتُستخدم إجابته (نفس تنبؤات Gemini الحقيقية المسجلة في التجربة — لا محاكاة ولا أرقام مصنعة).
يُحسب عند كل عتبة: نسبة التصعيد، دقة النظام الهجين، عدد نداءات Gemini الفعلية.
كذلك تحليل معايرة الثقة: متوسط الثقة للإجابات الصحيحة مقابل الخاطئة، وتفصيلًا حسب المشهد والصعوبة.
المخرجات: results/webset/hybrid_analysis.json + charts/hybrid_policy_curve.png + charts/confidence_calibration.png
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "webset"
CHARTS = BASE / "charts"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def ar(t):
    return get_display(arabic_reshaper.reshape(t))

def load(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}

def main():
    pm = load("predictions_webset.csv")
    pg = load("predictions_gemini_webset.csv")
    ids = sorted(set(pm) & set(pg))
    assert len(ids) == 125, f"التغطية غير كاملة: {len(ids)}"

    recs = []
    for i in ids:
        recs.append({
            "image_id": i,
            "true": pm[i]["true_label"],
            "mn_pred": pm[i]["predicted_label"],
            "mn_conf": float(pm[i]["confidence"]),
            "mn_ok": pm[i]["true_label"] == pm[i]["predicted_label"],
            "gm_pred": pg[i]["predicted_label"],
            "gm_ok": pg[i]["true_label"] == pg[i]["predicted_label"],
            "gm_latency": float(pg[i]["cloud_latency_ms"]) if pg[i].get("cloud_latency_ms") else None,
            "scene": pm[i].get("scene_type", ""),
            "difficulty": pm[i].get("difficulty", ""),
        })
    N = len(recs)
    mn_acc = np.mean([r["mn_ok"] for r in recs])
    gm_acc = np.mean([r["gm_ok"] for r in recs])
    gm_lat = np.mean([r["gm_latency"] for r in recs if r["gm_latency"]])
    print(f"الأساس: MobileNet وحده = {mn_acc:.4f} | Gemini وحده = {gm_acc:.4f} | N={N}")

    # السياسة الهجينة عند كل عتبة
    policy = []
    for T in THRESHOLDS:
        accepted = [r for r in recs if r["mn_conf"] >= T]
        escalated = [r for r in recs if r["mn_conf"] < T]
        correct = sum(1 for r in accepted if r["mn_ok"]) + sum(1 for r in escalated if r["gm_ok"])
        policy.append({
            "threshold": T,
            "accepted_local": len(accepted),
            "escalated_to_gemini": len(escalated),
            "escalation_pct": round(100 * len(escalated) / N, 1),
            "hybrid_accuracy": round(correct / N, 4),
            "gemini_calls": len(escalated),
        })
        print(f"  T={T:.2f}: تصعيد {len(escalated):3d} ({policy[-1]['escalation_pct']:5.1f}%) "
              f"→ دقة النظام الهجين {policy[-1]['hybrid_accuracy']:.4f}")

    best = max(policy, key=lambda p: p["hybrid_accuracy"])
    print(f"أفضل عتبة: T={best['threshold']} → دقة {best['hybrid_accuracy']} "
          f"(تصعيد {best['escalation_pct']}% من الصور)")

    # معايرة الثقة
    conf_ok = np.mean([r["mn_conf"] for r in recs if r["mn_ok"]])
    conf_bad = np.mean([r["mn_conf"] for r in recs if not r["mn_ok"]])
    bins = [(0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    calib = []
    for lo, hi in bins:
        sel = [r for r in recs if lo <= r["mn_conf"] < hi]
        if sel:
            calib.append({"bin": f"{lo:.2f}–{hi:.2f}", "n": len(sel),
                          "accuracy": round(float(np.mean([r["mn_ok"] for r in sel])), 4)})
    # حسب المشهد والصعوبة عند أفضل عتبة
    Tb = best["threshold"]
    def subset_stats(flt):
        sel = [r for r in recs if flt(r)]
        return {"n": len(sel),
                "mn_alone_acc": round(float(np.mean([r["mn_ok"] for r in sel])), 4) if sel else None,
                "hybrid_acc": round(float(np.mean([(r["mn_ok"] if r["mn_conf"] >= Tb else r["gm_ok"]) for r in sel])), 4) if sel else None}
    by_scene = {s: subset_stats(lambda r, s=s: r["scene"] == s) for s in ("single", "crowded")}
    by_diff = {d: subset_stats(lambda r, d=d: r["difficulty"] == d) for d in ("normal", "transparent_hard")}

    result = {
        "method": "سياسة هجينة محسومة مسبقًا: ثقة MobileNet >= T تُقبل محليًا؛ وإلا تصعيد لـ Gemini واستخدام تنبؤه الحقيقي المسجل",
        "data_source": "results/webset/predictions_webset.csv + predictions_gemini_webset.csv (نفس الـ125 صورة، بلا أي طلب جديد)",
        "n_images": N,
        "mobile_alone": {"accuracy": round(mn_acc, 4)},
        "gemini_alone": {"accuracy": round(gm_acc, 4), "avg_latency_ms": round(gm_lat, 1)},
        "policy_thresholds": policy,
        "best_threshold": best,
        "confidence_calibration": {
            "mean_conf_when_correct": round(float(conf_ok), 4),
            "mean_conf_when_wrong": round(float(conf_bad), 4),
            "bins": calib,
        },
        "by_scene_at_best": by_scene,
        "by_difficulty_at_best": by_diff,
        "honesty_note": "لا يمكن استبعاد ألفة صور ويب العامة من تدريب نموذج تجاري — التحفظ نفسه الخاص بمقارنة 5-3 ينطبق هنا",
    }
    (OUT / "hybrid_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # الرسم: منحنى السياسة
    Ts = [p["threshold"] for p in policy]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(Ts, [p["hybrid_accuracy"] for p in policy], "o-", color="#2e7d32",
             label=ar("دقة النظام الهجين"))
    ax1.axhline(mn_acc, color="#1565c0", ls="--", label=ar(f"MobileNet وحده ({mn_acc:.0%})"))
    ax1.axhline(gm_acc, color="#6a1b9a", ls=":", label=ar(f"Gemini وحده ({gm_acc:.0%})"))
    ax1.set_xlabel(ar("عتبة الثقة T")); ax1.set_ylabel(ar("دقة النظام الهجين"))
    ax1.set_ylim(0, 1); ax1.set_title(ar("النظام الهجين: تصعيد الصور غير الواثقة إلى Gemini"))
    ax2 = ax1.twinx()
    ax2.bar(Ts, [p["escalation_pct"] for p in policy], width=0.03, alpha=0.25, color="#ef6c00",
            label=ar("نسبة التصعيد %"))
    ax2.set_ylabel(ar("نسبة التصعيد %")); ax2.set_ylim(0, 100)
    ax1.legend(loc="center left", fontsize=9)
    fig.tight_layout(); fig.savefig(CHARTS / "hybrid_policy_curve.png", dpi=150)

    # الرسم: معايرة الثقة
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    labels = [c["bin"] + f"\n(n={c['n']})" for c in calib]
    vals = [c["accuracy"] for c in calib]
    bars = ax2.bar(labels, vals, color=["#c62828" if v < 0.5 else "#2e7d32" for v in vals])
    ax2.axhline(1/6, color="gray", ls=":", label=ar("تخمين عشوائي (1/6)"))
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.0%}", ha="center")
    ax2.set_ylim(0, 1); ax2.set_ylabel(ar("دقة MobileNet داخل شريحة الثقة"))
    ax2.set_title(ar("معايرة الثقة: هل الثقة العالية تعني صحة؟"))
    ax2.legend()
    fig2.tight_layout(); fig2.savefig(CHARTS / "confidence_calibration.png", dpi=150)
    print("المخرجات: results/webset/hybrid_analysis.json + charts/hybrid_policy_curve.png + charts/confidence_calibration.png")

if __name__ == "__main__":
    main()
