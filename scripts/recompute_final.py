# -*- coding: utf-8 -*-
"""
إعادة الحساب النهائي بعد اعتماد «وسوم مرجعية بشرية» من مراجعة عمياء واحدة تعتمدها الباحثة.
المصدر الوحيد للتنبؤات: results/webset/predictions_webset.csv و predictions_gemini_webset.csv
(محفوظتان من التجربة الفعلية) — صفر طلبات جديدة لـ Gemini وصفر استدلال جديد.

جسر الهوية الإلزامي: R###.jpg ↔ SHA-256 ↔ provenance/web_waste_manifest.csv ↔ image_id (wwm_…)
↔ مفاتيح ملفي التنبؤات — يُرفض التنفيذ إلا بمطابقة 125/125 واحدًا لواحد بلا تكرار.

الحارس الصارم (أي فشل = لا كتابة نتائج إطلاقًا):
  - ملف القرارات 125 صفًا بالضبط، R001–R125 كلها مرة واحدة.
  - exclude يقبل فقط yes أو فارغًا؛ exclude=yes ⇒ reviewer_label فارغ وexclude_reason إلزامي.
  - exclude فارغ ⇒ reviewer_label إحدى الفئات الست.
  - predicted_label في ملفي التنبؤات ضمن الفئات الست.
  - رفض أي فئة نهائية بلا عينات، وتحذير واضح عند العدد القليل.
المخرجات (results/final/): final_predictions.csv (review_id، source_image_id، الوسم البشري،
  الوسم الأولي، قرار الاستبعاد وسببه، تنبؤا النموذجين)، final_summary.json،
  final_comparison_table.md/.csv، final_confusion_mobilenet.png، final_confusion_gemini.png،
  final_f1_chart.png.
التصنيف: إعادة حساب لوسوم مرجعية بشرية على مجموعة الاختبار ذاتها — لا تحقق مستقل ولا سياسة نشر.
"""
import csv
import hashlib
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

REVIEW_IDS = {f"R{i:03d}" for i in range(1, 126)}
FEW_WARN = 10


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_accepted(base: Path):
    """{sha256: {source_image_id, initial_label}} لصفوف accepted — من المانيفست الموثق"""
    p = base / "provenance" / "web_waste_manifest.csv"
    if not p.exists():
        return None, [f"المانيفست غير موجود: {p}"]
    out, problems = {}, []
    with p.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("review_status") or "").strip() != "accepted":
                continue
            sha = (r.get("sha256") or "").strip().lower()
            iid = (r.get("image_id") or "").strip()
            lbl = (r.get("true_label") or "").strip()
            if not sha or not iid or lbl not in CLASSES:
                problems.append(f"صف مانيفست غير مكتمل: sha={bool(sha)} id={iid!r} label={lbl!r}")
                continue
            out[sha] = {"source_image_id": iid, "initial_label": lbl}
    if len(out) != 125:
        problems.append(f"صفوف accepted في المانيفست = {len(out)} ≠ 125")
    return (not problems), out, problems


def build_r_map(review_images: Path, accepted_by_sha: dict):
    """خريطة R### → صف المانيفست عبر SHA-256 — وحدة كاملة 125/125 (بلا تكرار ولا فجوة)"""
    problems = []
    files = sorted(review_images.glob("R*.jpg"))
    if len(files) != 125:
        problems.append(f"عدد صور المراجعة {len(files)} ≠ 125")
    mapping, seen_sha = {}, {}
    for p in files:
        rid = p.stem
        h = sha256_file(p)
        entry = accepted_by_sha.get(h)
        if entry is None:
            problems.append(f"{rid}: لا يطابق أي صف مقبول في المانيفست عبر SHA-256")
            continue
        if h in seen_sha:
            problems.append(f"{rid}: نفس محتوى {seen_sha[h]} — تكرار ملفات")
        seen_sha[h] = rid
        mapping[rid] = {"sha": h, "entry": entry}
    covered = len(set(m["entry"]["source_image_id"] for m in mapping.values()))
    if covered != 125:
        problems.append(f"المطابقة عبر SHA تغطي {covered}/125 — المطلوب واحدًا لواحد كاملًا")
    return (not problems and covered == 125), mapping, problems


def validate_decisions(blind_path: Path):
    """الحارس الصارم على ملف القرارات — يرجع (ok, decisions, problems)"""
    problems = []
    if not blind_path.exists():
        return False, [], [f"ملف القرارات غير موجود: {blind_path}"]
    with blind_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        need = ["review_id", "image_file", "reviewer_label", "exclude", "exclude_reason"]
        if [c for c in need if c not in cols]:
            problems.append(f"أعمدة ناقصة في ملف القرارات: {[c for c in need if c not in cols]}")
            return False, [], problems
        rows = list(reader)
    if len(rows) != 125:
        problems.append(f"عدد صفوف القرارات {len(rows)} ≠ 125 بالضبط")
    ids = [(r.get("review_id") or "").strip() for r in rows]
    if set(x for x in ids if x) != REVIEW_IDS:
        missing = sorted(REVIEW_IDS - set(x for x in ids if x))
        problems.append(f"أرقام مراجعة غائبة أو خارج النطاق: {missing[:10]}")
    dup = sorted({x for x in ids if x and ids.count(x) > 1})
    if dup:
        problems.append(f"review_id مكرر: {dup[:10]}")
    decisions = {}
    for n, r in enumerate(rows, 2):
        rid = (r.get("review_id") or "").strip()
        label = (r.get("reviewer_label") or "").strip()
        excl = (r.get("exclude") or "").strip().lower()
        reason = (r.get("exclude_reason") or "").strip()
        if not rid or rid not in REVIEW_IDS:
            if rid:
                problems.append(f"سطر {n}: review_id خارج النطاق ({rid!r})")
            continue
        if excl not in ("", "yes"):
            problems.append(f"سطر {n} ({rid}): exclude يقبل فقط yes أو فارغًا ({excl!r})")
            continue
        if excl == "yes":
            if label:
                problems.append(f"سطر {n} ({rid}): صورة مستبعدة لكن فيها reviewer_label ({label!r})")
                continue
            if not reason:
                problems.append(f"سطر {n} ({rid}): exclude=yes بدون exclude_reason إلزامي")
                continue
            decisions[rid] = {"label": "", "excluded": True, "reason": reason}
        else:
            if label not in CLASSES:
                problems.append(f"سطر {n} ({rid}): reviewer_label غير صالح ({label or 'فارغ'})")
                continue
            decisions[rid] = {"label": label, "excluded": False, "reason": reason}
    return (not problems and len(decisions) == 125), decisions, problems


def load_saved_preds(base: Path):
    """ملفا التنبؤات المحفوظان + فحص نطاق predicted_label"""
    problems = []
    preds = {"mobilenet": {}, "gemini": {}}
    for key, fname in (("mobilenet", "predictions_webset.csv"), ("gemini", "predictions_gemini_webset.csv")):
        p = base / "results" / "webset" / fname
        if not p.exists():
            problems.append(f"ملف التنبؤات غير موجود: {p}")
            continue
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                iid = (r.get("image_id") or "").strip()
                pl = (r.get("predicted_label") or "").strip()
                if pl not in CLASSES:
                    problems.append(f"{fname}: {iid} → predicted_label خارج الفئات ({pl or 'فارغ'})")
                    continue
                preds[key][iid] = r
    return preds, problems


def run(base: Path):
    """يعيد (True, summary) أو (False, problems) — الكتابة في base/results/final فقط عند النجاح الكامل"""
    problems = []
    ok_m, accepted_by_sha, problems1 = load_manifest_accepted(base)
    problems += problems1
    if not ok_m:
        return False, problems

    review_images = base / "review_web" / "blind" / "review_images"
    if not review_images.exists():
        problems.append(f"مجلد صور المراجعة غير موجود: {review_images}")
        return False, problems
    map_ok, rmap, problems2 = build_r_map(review_images, accepted_by_sha)
    problems += problems2

    dec_ok, decisions, problems3 = validate_decisions(base / "review_web" / "blind" / "blind_labels.csv")
    problems += problems3

    preds, problems4 = load_saved_preds(base)
    problems += problems4
    if problems:
        return False, problems

    included, excluded = [], []
    for rid in sorted(rmap):
        d = decisions[rid]
        entry = rmap[rid]["entry"]
        if d["excluded"]:
            excluded.append({"review_id": rid, "source_image_id": entry["source_image_id"],
                             "reason": d["reason"]})
        else:
            included.append({"review_id": rid, "source_image_id": entry["source_image_id"],
                             "human": d["label"], "initial": entry["initial_label"],
                             "mn": preds["mobilenet"][entry["source_image_id"]],
                             "gm": preds["gemini"][entry["source_image_id"]]})
    counts = {c: sum(1 for r in included if r["human"] == c) for c in CLASSES}
    zero = [c for c, v in counts.items() if v == 0]
    if zero:
        problems.append(f"فئات نهائية بلا أي عينة بعد الاستبعاد: {zero} — مرفوض")
    few = {c: v for c, v in counts.items() if 0 < v < FEW_WARN}
    if problems:
        return False, problems
    if few:
        print(f"⚠️ تحذير: فئات بعدد قليل من العينات: {few}", flush=True)

    y_t = [r["human"] for r in included]
    y_m = [r["mn"]["predicted_label"] for r in included]
    y_g = [r["gm"]["predicted_label"] for r in included]
    acc_m = float(np.mean(np.array(y_t) == np.array(y_m)))
    acc_g = float(np.mean(np.array(y_t) == np.array(y_g)))
    rep_m = classification_report(y_t, y_m, labels=CLASSES, output_dict=True, zero_division=0)
    rep_g = classification_report(y_t, y_g, labels=CLASSES, output_dict=True, zero_division=0)
    cm_m = confusion_matrix(y_t, y_m, labels=CLASSES)
    cm_g = confusion_matrix(y_t, y_g, labels=CLASSES)

    agree = sum(1 for r in included if r["initial"] == r["human"])
    agree_pct = round(100 * agree / len(included), 1)

    out = base / "results" / "final"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "final_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "source_image_id", "human_label", "initial_label",
                    "exclude_decision", "exclude_reason",
                    "mobilenet_prediction", "mobilenet_confidence", "mobilenet_ms",
                    "gemini_prediction", "gemini_ms"])
        for r in included:
            w.writerow([r["review_id"], r["source_image_id"], r["human"], r["initial"],
                        "", "", r["mn"]["predicted_label"], r["mn"]["confidence"],
                        r["mn"]["inference_ms"], r["gm"]["predicted_label"],
                        r["gm"].get("cloud_latency_ms", "")])
        for e in excluded:
            w.writerow([e["review_id"], e["source_image_id"], "", "", "yes", e["reason"], "", "", "", ""])

    for name, cm, acc, cmap in (("mobilenet", cm_m, acc_m, "Blues"), ("gemini", cm_g, acc_g, "Purples")):
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        ax.imshow(cm, cmap=cmap)
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASSES)
        for i2 in range(6):
            for j2 in range(6):
                ax.text(j2, i2, str(cm[i2, j2]), ha="center", va="center",
                        color="white" if cm[i2, j2] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Human reference label")
        ax.set_title(f"{name} — Acc {acc:.1%}")
        fig.tight_layout(); fig.savefig(out / f"final_confusion_{name}.png", dpi=150)

    import numpy as _np
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = _np.arange(6)
    ax2.bar(x - 0.2, [rep_m[c]["f1-score"] for c in CLASSES], 0.4, label=f"MobileNetV2 ({acc_m:.1%})")
    ax2.bar(x + 0.2, [rep_g[c]["f1-score"] for c in CLASSES], 0.4, label=f"Gemini ({acc_g:.1%})")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES)
    ax2.set_ylabel("F1"); ax2.legend(); ax2.set_title("Final F1 per class — human-adopted labels")
    fig2.tight_layout(); fig2.savefig(out / "final_f1_chart.png", dpi=150)

    rows_csv = [["metric", "mobilenet", "gemini"]]
    lines = ["# جدول المقارنة النهائي — وسوم بشرية معتمدة بعد مراجعة عمياء واحدة", "",
             f"الصور المشمولة: {len(included)} | المستبعد: {len(excluded)} | التاريخ: {datetime.now().isoformat(timespec='seconds')}", "",
             "| المقياس | MobileNetV2 (محلي) | Gemini Vision (سحابي) |", "|---|---:|---:|"]
    def add(m, a, b):
        rows_csv.append([m, a, b]); lines.append(f"| {m} | {a} | {b} |")
    add("الدقة الكلية", f"{acc_m:.4f}", f"{acc_g:.4f}")
    add("Macro F1", f"{rep_m['macro avg']['f1-score']:.4f}", f"{rep_g['macro avg']['f1-score']:.4f}")
    for c in CLASSES:
        add(f"F1 — {c}", f"{rep_m[c]['f1-score']:.4f}", f"{rep_g[c]['f1-score']:.4f}")
    add("زمن الاستجابة", "محلي 29.9ms (بلا شبكة)", "سحابي 4530ms (شامل الشبكة)")
    (out / "final_comparison_table.csv").open("w", newline="", encoding="utf-8").write(
        "\n".join(",".join(map(str, r)) for r in rows_csv))
    (out / "final_comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "labels": "وسوم مرجعية بشرية تعتمدها الباحثة بعد مراجعة عمياء واحدة — لا تُوصف بحقيقة أرضية",
        "n_excluded": len(excluded), "excluded_ids": [e["review_id"] for e in excluded],
        "n_final_included": len(included),
        "counts_per_class": counts,
        "few_samples_warning": few or None,
        "mobilenet": {"accuracy": round(acc_m, 4), "macro_f1": round(float(rep_m["macro avg"]["f1-score"]), 4)},
        "gemini": {"accuracy": round(acc_g, 4), "macro_f1": round(float(rep_g["macro avg"]["f1-score"]), 4)},
        "agreement_with_initial_labels": {"pct": agree_pct,
                                          "note": "نسبة اتفاق خام بين الوسوم البشرية المعتمدة والوسوم الأولية (بمساعدة آلية) — ليس Cohen's Kappa ولا اتفاق مقيّمين بشريين"},
        "no_new_calls": "صفر طلبات جديدة لـ Gemini وصفر استدلال جديد — إعادة حساب من التنبؤات المحفوظة حصرًا",
    }
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, summary


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    ok, payload = run(base)
    if ok:
        print("\n=== النتيجة النهائية ===")
        print(json.dumps({k: payload[k] for k in ("n_final_included", "n_excluded", "mobilenet", "gemini",
                                                  "agreement_with_initial_labels")}, ensure_ascii=False, indent=1))
        sys.exit(0)
    print("=== فشل الحارس ===")
    for x in (payload if isinstance(payload, list) else [payload]):
        print(" -", x)
    sys.exit(2)


if __name__ == "__main__":
    main()
