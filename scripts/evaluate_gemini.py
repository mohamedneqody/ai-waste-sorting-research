# -*- coding: utf-8 -*-
"""
المهمة 4 (محدثة 2026-10-04 مراجعة ثالثة) — تقييم Gemini Vision (zero-shot) على صور EG-Waste-Mini عبر البيان
القواعد المنفذة:
  - المفتاح من متغير البيئة GEMINI_API_KEY فقط — لا مفتاح في الكود أو أي ملف.
  - المطالبة ثابتة في gemini_prompt.txt — لا تُعدل ولا تتغير حسب أي إجابة أو نتيجة.
  - قبول الإجابة فقط إذا كانت الكلمة كاملة مطابقة لإحدى الفئات الست بعد strip/lower — لا بحث جزئي داخل النص.
  - **رد الخدمة الناجح اتصاليًا لكنه لا يطابق أي فئة = unparseable نهائي (ok=False): إجابة محتوى لا تُعاد،
    وتجعل التجربة غير مكتملة بلا Accuracy أو Macro F1 أو مصفوفة التباس.**
  - MIME type حسب امتداد الملف الحقيقي (jpg/jpeg → image/jpeg، png → image/png).
  - سياسة إعادة المحاولة: الأخطاء التقنية فقط (شبكة، 429، 5xx) تُعاد تلقائيًا بنفس الطلب والمطالبة نفسها
    حتى MAX_TECH_ATTEMPTS مع تسجيل عدد المحاولات وسبب كل فشل. الطلب الناجح لا يُعاد أبدًا، والإجابة
    غير القابلة للتحليل لا تُعاد، وأخطاء HTTP الدائمة تُسجل فشلًا نهائيًا.
  - قاعدة الاكتمال: المقاييس تُحسب فقط إذا نجحت كل صور البيان — لا مقاييس من مجموعة جزئية.
  - checkpoint سجل أحداث (سطر لكل محاولة) ويخزن model_version في كل حدث نجاح؛ الملخص يجمع الإصدارات
    من كل أحداث النجاح (بما فيها تشغيلات سابقة) حتى لا يختفي إصدار النموذج عند استكمال تشغيل مقطوع.
"""
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifest_utils import CLASSES, scan_folder_images, validate_manifest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # رسائل عربية سليمة حتى على CP1252

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "local_egypt"
MANIFEST = BASE / "eg_waste_manifest.csv"
PROMPT_FILE = BASE / "gemini_prompt.txt"
CKPT = BASE / "results" / "gemini_checkpoint.jsonl"
OUT = BASE / "results"
MIN_TOTAL, MIN_PER_CLASS = 100, 10
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
MAX_TECH_ATTEMPTS = 3  # إجمالي المحاولات لكل صورة للأخطاء التقنية (الطلب والمطالبة ثابتان)
FINAL_KINDS = {"success", "unparseable", "http_permanent"}

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
SLEEP_S = float(os.environ.get("GEMINI_SLEEP", "4"))

def print_setup_steps(reason: str):
    print(f"=== لا يمكن تشغيل Gemini الآن: {reason} ===")
    print("خطوات التشغيل الصحيحة (بدون وضع أي مفتاح في ملفات المشروع):")
    print("  1) مفتاح مجاني من Google AI Studio: https://aistudio.google.com/apikey")
    print("  2) في PowerShell:  setx GEMINI_API_KEY \"المفتاح\"   ثم افتحي طرفية جديدة")
    print("  3) اختياري: setx GEMINI_MODEL \"gemini-2.0-flash\"  و  setx GEMINI_SLEEP \"4\"")
    print("  4) الصور في data/local_egypt/<الفئة>/ + البيان صالح →  python scripts/evaluate_gemini.py")
    print("لن تُكتب أو تُحاكى أي نتيجة Gemini قبل توفر المفتاح والصور والبيان الصالح فعليًا.")

def parse_answer(raw: str):
    """تصنيف إجابة الخدمة: كاملة المطابقة → (الفئة، success)؛ غير المطابقة → ('', unparseable)
    unparseable إجابة محتوى نهائية — لا تُعاد وليست خطأ اتصال."""
    pred = (raw or "").strip().lower()
    if pred in CLASSES:
        return pred, "success"
    return "", "unparseable"

def classify_exception(e):
    """تصنيف الاستثناء: خطأ تقني قابل لإعادة المحاولة أم دائم"""
    if isinstance(e, urllib.error.HTTPError):
        return "http_retryable" if e.code in RETRYABLE_HTTP else "http_permanent"
    return "network"  # URLError / Timeout / أي فشل اتصال

def call_gemini(prompt_text: str, img_path: Path, mime: str):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime, "data": b64}},
            {"text": prompt_text},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    latency = (time.perf_counter() - t0) * 1000
    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
    model_version = data.get("modelVersion", MODEL)
    pred, kind = parse_answer(raw)
    return raw, pred, kind, round(latency, 1), model_version

def build_summary(rows, last_event, test_date, prompt_text):
    """يبني الملخص من الحالات النهائية فقط — المقاييس تُحسب عند الاكتمال الكامل حصريًا (دالة نقية قابلة للاختبار)"""
    outcomes = {"success": [], "unparseable": [], "http_permanent": [], "technical_pending": []}
    for r in rows:
        iid = r["image_id"]
        ev = last_event.get(iid)
        if ev is None:
            outcomes["technical_pending"].append(iid)  # لم تُحاول (توقف سابق)
        elif ev.get("kind") == "success":
            outcomes["success"].append(iid)
        elif ev.get("kind") in ("unparseable", "http_permanent"):
            outcomes[ev["kind"]].append(iid)
        else:
            outcomes["technical_pending"].append(iid)  # استنفاد المحاولات التقنية
    complete = (len(outcomes["success"]) == len(rows)) and not any(
        outcomes[k] for k in ("unparseable", "http_permanent", "technical_pending"))
    # جمع إصدارات النموذج من كل أحداث النجاح (تشمل تشغيلات سابقة) — لا يختفي عند الاستكمال
    versions = sorted({ev["model_version"] for ev in last_event.values()
                       if ev.get("kind") == "success" and ev.get("model_version")}) or ["لم يُستلم أي رد"]
    summary = {
        "test_date": test_date,
        "manifest_file": "eg_waste_manifest.csv",
        "manifest_validated": True,
        "model_requested": MODEL,
        "model_version_observed": versions,
        "prompt_file": "gemini_prompt.txt",
        "prompt_text": prompt_text,
        "n_images_manifest": len(rows),
        "n_success": len(outcomes["success"]),
        "n_unparseable": len(outcomes["unparseable"]),
        "n_http_permanent": len(outcomes["http_permanent"]),
        "n_technical_pending": len(outcomes["technical_pending"]),
        "technical_pending_ids": outcomes["technical_pending"][:20],
        "completeness": "complete" if complete else "incomplete",
        "completeness_note": "المقاييس تُحسب فقط عند نجاح كل صور البيان — لا مقاييس من مجموعة جزئية",
        "retry_policy": f"الأخطاء التقنية (شبكة/429/5xx) تُعاد بنفس الطلب والمطالبة حتى {MAX_TECH_ATTEMPTS} محاولات؛ "
                        "النجاح لا يُعاد بسبب نتيجته؛ الإجابة غير القابلة للتحليل (unparseable) نهائية لا تُعاد؛ "
                        "أخطاء HTTP الدائمة تُسجل فشلًا نهائيًا",
        "accuracy": None, "macro_f1": None, "per_class": {}, "confusion_matrix": [],
        "timing_definition": "زمن الاستجابة السحابي الكلي من بدء الإرسال حتى استلام الرد (يشمل الشبكة) — لا يقارن مباشرة بالزمن المحلي",
        "cost_note": "الحساب المجاني (Google AI Studio) — تكلفة هذه التجربة = 0 جنيه؛ لا يُعمم على استخدام مدفوع",
    }
    if complete:
        label_of = {r["image_id"]: r["true_label"] for r in rows}  # التسمية من البيان — الأحداث لا تخزنها
        valid = [last_event[r["image_id"]] for r in rows]
        y_true = [label_of[r["image_id"]] for r in valid]
        y_pred = [r["predicted_label"] for r in valid]
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)  # labels نصية مطابقة للبيانات
        lats = [r["cloud_latency_ms"] for r in valid if r.get("cloud_latency_ms")]
        summary.update({
            "accuracy": round(acc, 4),
            "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
            "per_class": {c: {"precision": round(rep[c]["precision"], 4), "recall": round(rep[c]["recall"], 4),
                              "f1": round(rep[c]["f1-score"], 4), "support": rep[c]["support"]} for c in CLASSES},
            "confusion_matrix": cm.tolist(), "confusion_matrix_labels": CLASSES,
            "cloud_latency_avg_ms": round(float(np.mean(lats)), 1) if lats else None,
        })
    return summary, outcomes

def main():
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    disk = scan_folder_images(IMG_DIR, BASE)
    counts = {c: 0 for c in CLASSES}
    for lbl in disk.values():
        counts[lbl] += 1
    total = len(disk)
    if total < MIN_TOTAL or any(v < MIN_PER_CLASS for v in counts.values()):
        print(f"=== الحارس: الصور غير مكتملة ({total} صورة موجودة، المطلوب ≥{MIN_TOTAL} و≥{MIN_PER_CLASS} لكل فئة) ===")
        for c in CLASSES:
            print(f"  {c:10s}: {counts[c]}")
        print_setup_steps("الصور غير مكتملة بعد")
        sys.exit(2)
    ok, rows, problems = validate_manifest(MANIFEST, IMG_DIR, BASE)
    if not ok:
        print("=== الحارس: البيان غير صالح — لا تقييم ولا كتابة أي نتيجة ===")
        for pr in problems:
            print("  -", pr)
        print_setup_steps("البيان غير صالح")
        sys.exit(2)
    if not API_KEY:
        print_setup_steps("متغير البيئة GEMINI_API_KEY غير مضبوط")
        sys.exit(2)

    OUT.mkdir(exist_ok=True)
    # استعادة آخر حدث لكل صورة من سجل الأحداث؛ الصور المعلقة تقنيًا تُعاد، والنهائية لا تُعاد
    last_event = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
                last_event[ev["image_id"]] = ev
            except Exception:
                pass
    n_pending_before = sum(1 for ev in last_event.values() if ev.get("kind") not in FINAL_KINDS)
    print(f"Checkpoint: {len(last_event)} صورة لها محاولات سابقة "
          f"({n_pending_before} معلقة تقنيًا ستُعاد). عينات البيان: {len(rows)}", flush=True)

    test_date = datetime.now().isoformat(timespec="seconds")
    with CKPT.open("a", encoding="utf-8") as ck:
        for i, r in enumerate(rows, 1):
            iid = r["image_id"]
            prev = last_event.get(iid)
            if prev and prev.get("kind") in FINAL_KINDS:
                continue  # نهائي (نجاح أو unparseable أو فشل دائم) — لا يُعاد
            attempts = prev.get("attempts", 0) if prev else 0
            p = BASE / r["image_path"]
            mime = MIME.get(p.suffix.lower(), "image/jpeg")
            while True:
                attempts += 1
                try:
                    raw, pred, kind, lat, mv = call_gemini(prompt_text, p, mime)
                    if kind == "success":
                        ev = {"image_id": iid, "attempt": attempts, "kind": "success", "ok": True,
                              "predicted_label": pred, "raw_response": raw[:60],
                              "cloud_latency_ms": lat, "model_version": mv, "error": ""}
                        final = True
                    else:
                        # إجابة محتوى لا تطابق أي فئة — unparseable نهائي: لا تُعاد وليست خطأ اتصال
                        ev = {"image_id": iid, "attempt": attempts, "kind": "unparseable", "ok": False,
                              "predicted_label": "", "raw_response": raw[:60],
                              "cloud_latency_ms": lat, "model_version": mv,
                              "error": "إجابة من الخدمة لا تطابق أي فئة من الفئات الست"}
                        final = True
                except Exception as e:
                    kind = classify_exception(e)
                    ev = {"image_id": iid, "attempt": attempts, "kind": kind, "ok": False,
                          "predicted_label": "", "raw_response": "", "cloud_latency_ms": None,
                          "error": (f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else str(e))[:120]}
                    final = not (kind in ("http_retryable", "network") and attempts < MAX_TECH_ATTEMPTS)
                ck.write(json.dumps(ev, ensure_ascii=False) + "\n"); ck.flush()
                last_event[iid] = ev
                if final:
                    break
                print(f"  {iid}: خطأ تقني ({ev['error']}) — محاولة {attempts}/{MAX_TECH_ATTEMPTS} بعد انتظار…", flush=True)
                time.sleep(SLEEP_S * 2)
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
            time.sleep(SLEEP_S)

    summary, outcomes = build_summary(rows, last_event, test_date, prompt_text)

    with (OUT / "predictions_gemini.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "true_label", "predicted_label", "raw_response",
                    "cloud_latency_ms", "model", "test_date"])
        for r in rows:
            ev = last_event.get(r["image_id"])
            if ev:
                w.writerow([r["image_id"], r["true_label"], ev.get("predicted_label", ""),
                            ev.get("raw_response", "") or ev.get("error", ""),
                            ev.get("cloud_latency_ms", ""), MODEL, test_date])

    (OUT / "summary_gemini.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary["completeness"] == "complete":
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = summary["confusion_matrix"]
        ax.imshow(cm, cmap="Purples")
        ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True (EG-Waste-Mini)")
        ax.set_title(f"Gemini zero-shot — Acc {summary['accuracy']:.1%}")
        fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_gemini.png", dpi=150)
        print(f"\n=== Gemini مكتمل: {summary['n_images_manifest']} صورة | Accuracy={summary['accuracy']} ===", flush=True)
        print("المخرجات: results/predictions_gemini.csv و results/summary_gemini.json و results/confusion_matrix_gemini.png", flush=True)
    else:
        print(f"\n=== تجربة Gemini غير مكتملة: نجح {summary['n_success']} | غير قابل للتحليل {summary['n_unparseable']} | "
              f"HTTP دائم {summary['n_http_permanent']} | معلق تقنيًا {summary['n_technical_pending']} ===", flush=True)
        print("لا تُحسب Accuracy أو Macro F1 أو مصفوفة التباس من مجموعة جزئية — أعيدي التشغيل لإكمال المعلق تقنيًا "
              "(النجاحات وunparseable لا تُعاد). للتصفير الكامل امسحي results/gemini_checkpoint.jsonl.", flush=True)
        print("السجل الخام: results/predictions_gemini.csv — لا مقاييس أُنتجت.", flush=True)

if __name__ == "__main__":
    main()
