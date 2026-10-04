# -*- coding: utf-8 -*-
"""
تقييم Gemini Vision (zero-shot) على مجموعة Web-Waste-Mini — بروتوكول ثابت ومطالبة واحدة
مستقل عن سكربتات EG المعتمدة؛ يعمل على صفوف accepted في provenance/web_waste_manifest.csv.

المتطلبات:
  1) متغير بيئة GEMINI_API_KEY (مجاني من Google AI Studio):
     PowerShell:  setx GEMINI_API_KEY "المفتاح"   ثم طرفية جديدة
  2) لا يوجد أي مفتاح في الكود أو الملفات.

القواعد (نفس المعايير المعتمدة):
  - المطالبة ثابتة في gemini_prompt.txt — لا تعديل ولا إعادة طلب ناجح بسبب نتيجته.
  - قبول الإجابة فقط إذا كانت كلمة كاملة مطابقة لإحدى الفئات الست بعد strip/lower؛
    غير ذلك = unparseable نهائي → التجربة غير مكتملة بلا أي مقاييس.
  - إعادة المحاولة للأخطاء التقنية فقط (شبكة/429/5xx) حتى 3 محاولات بنفس الطلب، مع تسجيل المحاولات والأسباب.
  - model_version يخزن في كل حدث نجاح ويجمع في الملخص من كل الأحداث.
  - كل صورة: SHA-256 متحقق قبل الإرسال — لا تُرسل صورة تغيرت.
المخرجات (results/webset/): predictions_gemini_webset.csv + summary_gemini_webset.json
"""
import base64
import csv
import hashlib
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

from manifest_utils import CLASSES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
WEB_MANIFEST = BASE / "provenance" / "web_waste_manifest.csv"
PROMPT_FILE = BASE / "gemini_prompt.txt"
CKPT = BASE / "results" / "webset" / "gemini_checkpoint.jsonl"
OUT = BASE / "results" / "webset"
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
MAX_TECH_ATTEMPTS = 3
FINAL_KINDS = {"success", "unparseable", "http_permanent"}

API_KEYS = []
def load_api_keys():
    """يجمع المفاتيح من متغيرات البيئة: GEMINI_API_KEY ثم GEMINI_API_KEY_2..N — تدوير تلقائي يرفع السرعة دون تجاوز حدود كل مفتاح"""
    keys = []
    if os.environ.get("GEMINI_API_KEY", "").strip():
        keys.append(os.environ["GEMINI_API_KEY"].strip())
    i = 2
    while os.environ.get(f"GEMINI_API_KEY_{i}", "").strip():
        keys.append(os.environ[f"GEMINI_API_KEY_{i}"].strip())
        i += 1
    out, seen = [], set()
    for k in keys:
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

API_KEYS = load_api_keys()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest").strip()
MAX_TECH_ATTEMPTS = 3 * max(len(API_KEYS), 1)  # لكل مفتاح 3 محاولات تقنية
SLEEP_S = float(os.environ.get("GEMINI_SLEEP", "1.5" if len(API_KEYS) > 1 else "4"))
_key_state = {"idx": 0}

def next_key():
    """تدوير round-robin على المفاتيح — يرجع (المفتاح، رقمه الترتيبي 1..n)"""
    k = API_KEYS[_key_state["idx"] % len(API_KEYS)]
    _key_state["idx"] += 1
    return k, (_key_state["idx"] - 1) % len(API_KEYS) + 1

def print_setup_steps(reason: str):
    print(f"=== لا يمكن تشغيل Gemini الآن: {reason} ===")
    print("1) مفتاح مجاني: https://aistudio.google.com/apikey")
    print("2) PowerShell:  setx GEMINI_API_KEY \"المفتاح\"   (وممكن عدة مفاتيح: GEMINI_API_KEY_2، _3، _4 للتدوير ورفع السرعة)")
    print("3) التشغيل:  python scripts/evaluate_gemini_webset.py")
    print("لا تُكتب أو تُحاكى أي نتيجة قبل توفر المفتاح فعليًا.")

def parse_answer(raw: str):
    pred = (raw or "").strip().lower()
    return (pred, "success") if pred in CLASSES else ("", "unparseable")

def classify_exception(e):
    if isinstance(e, urllib.error.HTTPError):
        return "http_retryable" if e.code in RETRYABLE_HTTP else "http_permanent"
    return "network"

def call_gemini(prompt_text: str, img_path: Path, mime: str, key: str):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime, "data": b64}},
            {"text": prompt_text},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())
    latency = (time.perf_counter() - t0) * 1000
    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
    mv = data.get("modelVersion", MODEL)
    pred, kind = parse_answer(raw)
    return raw, pred, kind, round(latency, 1), mv

def main():
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not WEB_MANIFEST.exists():
        print(f"مانيفست الويب غير موجود: {WEB_MANIFEST}"); sys.exit(2)
    with WEB_MANIFEST.open(encoding="utf-8-sig") as f:
        accepted = [r for r in csv.DictReader(f) if (r.get("review_status") or "").strip() == "accepted"]
    rows = []
    problems = []
    for r in accepted:
        rel = (r.get("image_path") or "").strip().replace("\\", "/")
        label = (r.get("true_label") or "").strip()
        p = BASE / rel
        if label not in CLASSES or not p.exists() or p.parent.name != label:
            problems.append(rel); continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != (r.get("sha256") or "").strip().lower():
            problems.append(f"{rel} (SHA mismatch)"); continue
        rows.append({"image_id": (r.get("image_id") or "").strip(), "image_path": rel,
                     "true_label": label, "mime": MIME.get(p.suffix.lower(), "image/jpeg")})
    if problems or len(rows) < 100:
        print(f"=== الحارس: بيانات غير سليمة ({len(rows)} صالحة) ===")
        for pr in problems[:20]:
            print("  -", pr)
        sys.exit(2)
    if not API_KEYS:
        print_setup_steps("لا يوجد GEMINI_API_KEY في متغيرات البيئة")
        sys.exit(2)
    print(f"العينات: {len(rows)} | الموديل: {MODEL} | المفاتيح المتاحة للتدوير: {len(API_KEYS)} "
          f"| الانتظار {SLEEP_S}s بين الطلبات (~{int(len(rows)*SLEEP_S/60) + 1} دقيقة)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    last_event = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
                last_event[ev["image_id"]] = ev
            except Exception:
                pass
    pend = sum(1 for ev in last_event.values() if ev.get("kind") not in FINAL_KINDS)
    print(f"Checkpoint: {len(last_event)} سابقة ({pend} معلقة تقنيًا ستُعاد).", flush=True)

    test_date = datetime.now().isoformat(timespec="seconds")
    with CKPT.open("a", encoding="utf-8") as ck:
        for i, r in enumerate(rows, 1):
            iid = r["image_id"]
            prev = last_event.get(iid)
            if prev and prev.get("kind") in FINAL_KINDS:
                continue
            attempts = prev.get("attempts", 0) if prev else 0
            p = BASE / r["image_path"]
            while True:
                attempts += 1
                key, kidx = next_key()
                try:
                    raw, pred, kind, lat, mv = call_gemini(prompt_text, p, r["mime"], key)
                    if kind == "success":
                        ev = {"image_id": iid, "attempt": attempts, "kind": "success", "ok": True,
                              "predicted_label": pred, "raw_response": raw[:60],
                              "cloud_latency_ms": lat, "model_version": mv, "key_index": kidx, "error": ""}
                        final = True
                    else:
                        ev = {"image_id": iid, "attempt": attempts, "kind": "unparseable", "ok": False,
                              "predicted_label": "", "raw_response": raw[:60],
                              "cloud_latency_ms": lat, "model_version": mv, "key_index": kidx,
                              "error": "إجابة لا تطابق أي فئة"}
                        final = True
                except Exception as e:
                    kind = classify_exception(e)
                    ev = {"image_id": iid, "attempt": attempts, "kind": kind, "ok": False,
                          "predicted_label": "", "raw_response": "", "cloud_latency_ms": None,
                          "key_index": kidx,
                          "error": (f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else str(e))[:120]}
                    final = not (kind in ("http_retryable", "network") and attempts < MAX_TECH_ATTEMPTS)
                ck.write(json.dumps(ev, ensure_ascii=False) + "\n"); ck.flush()
                last_event[iid] = ev
                if final:
                    break
                # 429: المفتاح التالي جاهز فورًا تقريبًا؛ باقي الأخطاء التقنية: انتظار أطول
                time.sleep(SLEEP_S if ev["kind"] == "http_retryable" else SLEEP_S * 2)
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
            time.sleep(SLEEP_S)

    outcomes = {"success": [], "unparseable": [], "http_permanent": [], "technical_pending": []}
    for r in rows:
        ev = last_event.get(r["image_id"])
        k = ev.get("kind") if ev else None
        if k in ("success", "unparseable", "http_permanent"):
            outcomes[k].append(r["image_id"])
        else:
            outcomes["technical_pending"].append(r["image_id"])
    complete = len(outcomes["success"]) == len(rows) and not any(
        outcomes[k] for k in ("unparseable", "http_permanent", "technical_pending"))
    versions = sorted({ev["model_version"] for ev in last_event.values()
                       if ev.get("kind") == "success" and ev.get("model_version")}) or ["لم يُستلم أي رد"]

    with (OUT / "predictions_gemini_webset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "true_label", "predicted_label", "raw_response",
                    "cloud_latency_ms", "model", "test_date"])
        for r in rows:
            ev = last_event.get(r["image_id"])
            if ev:
                w.writerow([r["image_id"], r["true_label"], ev.get("predicted_label", ""),
                            ev.get("raw_response", "") or ev.get("error", ""),
                            ev.get("cloud_latency_ms", ""), MODEL, test_date])

    summary = {
        "test_date": test_date,
        "dataset": "Web-Waste-Mini (accepted rows, SHA-verified)",
        "model_requested": MODEL,
        "model_version_observed": versions,
        "prompt_file": "gemini_prompt.txt",
        "prompt_text": prompt_text,
        "n_images_manifest": len(rows),
        "n_success": len(outcomes["success"]),
        "n_unparseable": len(outcomes["unparseable"]),
        "n_http_permanent": len(outcomes["http_permanent"]),
        "n_technical_pending": len(outcomes["technical_pending"]),
        "completeness": "complete" if complete else "incomplete",
        "completeness_note": "المقاييس تُحسب فقط عند نجاح كل صور المجموعة — لا مقاييس جزئية",
        "retry_policy": f"تقني (شبكة/429/5xx) يُعاد حتى {MAX_TECH_ATTEMPTS} بنفس الطلب مع تدوير المفاتيح؛ النجاح لا يُعاد؛ unparseable نهائي",
        "n_api_keys_rotated": len(API_KEYS),
        "key_rotation_note": "التدوير round-robin على المفاتيح يوزع الحمل فيبقى داخل حدود كل مفتاح مجاني مع سرعة أعلى — لا يتجاوز حصة أي مفتاح بمفرده",
        "accuracy": None, "macro_f1": None, "per_class": {}, "confusion_matrix": [],
        "timing_definition": "زمن الاستجابة السحابي الكلي شامل الشبكة — لا يقارن مباشرة بالزمن المحلي",
        "cost_note": "الحساب المجاني — تكلفة التجربة = 0 جنيه؛ لا يُعمم على استخدام مدفوع",
    }
    if complete:
        label_of = {r["image_id"]: r["true_label"] for r in rows}
        valid = [last_event[r["image_id"]] for r in rows]
        y_true = [label_of[r["image_id"]] for r in valid]
        y_pred = [r["predicted_label"] for r in valid]
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        lats = [r["cloud_latency_ms"] for r in valid if r.get("cloud_latency_ms")]
        summary.update({
            "accuracy": round(acc, 4),
            "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
            "per_class": {c: {"precision": round(rep[c]["precision"], 4), "recall": round(rep[c]["recall"], 4),
                              "f1": round(rep[c]["f1-score"], 4), "support": rep[c]["support"]} for c in CLASSES},
            "confusion_matrix": cm.tolist(), "confusion_matrix_labels": CLASSES,
            "cloud_latency_avg_ms": round(float(np.mean(lats)), 1) if lats else None,
        })
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Purples")
        ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True (Web-Waste-Mini)")
        ax.set_title(f"Gemini zero-shot — Acc {acc:.1%}")
        fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_gemini_webset.png", dpi=150)

    (OUT / "summary_gemini_webset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if complete:
        print(f"\n=== Gemini مكتمل: {summary['n_images_manifest']} صورة | Accuracy={summary['accuracy']} | Macro F1={summary['macro_f1']} ===", flush=True)
    else:
        print(f"\n=== غير مكتمل: نجح {summary['n_success']} | unparseable {summary['n_unparseable']} | "
              f"HTTP دائم {summary['n_http_permanent']} | معلق {summary['n_technical_pending']} — لا مقاييس ===", flush=True)
    print("المخرجات: results/webset/predictions_gemini_webset.csv و summary_gemini_webset.json", flush=True)

if __name__ == "__main__":
    main()
