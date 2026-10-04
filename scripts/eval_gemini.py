# -*- coding: utf-8 -*-
"""
تقييم Gemini Vision (zero-shot بدون تدريب) على نفس مجموعة المخلفات المصرية المحلية
للمقارنة مع النموذج المحلي المدرب: الدقة + السرعة + التكلفة.

المتطلبات:
  1) متغير بيئة GEMINI_API_KEY (مجاني من Google AI Studio):
     في PowerShell:  setx GEMINI_API_KEY "مفتاحك_هنا"   ثم افتح طرفية جديدة
  2) الصور في D:\waste_research\data\local_egypt\<class>\*.jpg (نفس مدخلات eval_local.py)

التشغيل:  python scripts/eval_gemini.py
- في Checkpoint تلقائي (JSONL) فلو اتقطع ويعد تاني مش هيكرر الصور اللي خلصت.
- المعدل الافتراضي 4 ثوانٍ/صورة احترامًا لحدود الحساب المجاني (~150 صورة ≈ 12 دقيقة).
"""
import base64
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"D:\waste_research")
LOCAL = BASE / "data" / "local_egypt"
CKPT = BASE / "data" / "gemini_results.jsonl"
OUT = BASE / "models" / "metrics_gemini.json"
CHARTS = BASE / "charts"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("لا يوجد GEMINI_API_KEY في البيئة. اعملي مفتاح مجاني من Google AI Studio ثم: "
                     "setx GEMINI_API_KEY \"...\" وافتحي طرفية جديدة.")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
SLEEP_S = float(os.environ.get("GEMINI_SLEEP", "4"))

PROMPT = ("Classify this photo of a single waste item into exactly one of these categories: "
          "cardboard, glass, metal, paper, plastic, trash. "
          "Answer with ONLY the category word, nothing else.")

samples = []
for ci, c in enumerate(CLASSES):
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in (LOCAL / c).glob(ext):
            samples.append((p, ci))
print(f"العينات: {len(samples)} | الموديل: {MODEL} | التوقف كل {SLEEP_S}s", flush=True)
if not samples:
    raise SystemExit("مفيش صور. شغلي eval_local.py الأول لتتأكدي من المجموعة.")

# استعادة المحاولات السابقة (checkpoint)
done = {}
if CKPT.exists():
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            if r.get("ok"):
                done[r["file"]] = r
        except Exception:
            pass
print(f"مكتمل سابقًا (checkpoint): {len(done)}", flush=True)

def call_gemini(img_path):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    dt = (time.perf_counter() - t0) * 1000
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
    return text, dt

latencies, errors = [], 0
with CKPT.open("a", encoding="utf-8") as ck:
    for i, (p, ci) in enumerate(samples, 1):
        if str(p) in done:
            continue
        try:
            text, dt = call_gemini(p)
            pred = next((c for c in CLASSES if c in text), None)
            rec = {"file": str(p), "true": CLASSES[ci], "raw": text[:40],
                   "pred": pred, "latency_ms": round(dt, 1), "ok": pred is not None}
        except urllib.error.HTTPError as e:
            rec = {"file": str(p), "true": CLASSES[ci], "raw": f"HTTP {e.code}", "pred": None, "ok": False}
            errors += 1
        ck.write(json.dumps(rec, ensure_ascii=False) + "\n"); ck.flush()
        done[str(p)] = rec
        if i % 10 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}", flush=True)
        time.sleep(SLEEP_S)

valid = [r for r in done.values() if r["ok"]]
y_true = [CLASSES.index(r["true"]) for r in valid]
y_pred = [CLASSES.index(r["pred"]) if r["pred"] in CLASSES else -1 for r in valid]
acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
lats = [r["latency_ms"] for r in valid if r.get("latency_ms")]

metrics = {
    "model": f"Google {MODEL} (zero-shot, بدون تدريب)",
    "n_evaluated": len(valid), "n_failed": len(done) - len(valid) + errors,
    "accuracy": round(acc, 4),
    "per_class": {k: {"precision": round(v["precision"], 4), "recall": round(v["recall"], 4),
                      "f1": round(v["f1-score"], 4), "support": v["support"]}
                  for k, v in report.items() if k in CLASSES},
    "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
    "latency_api_ms_avg": round(float(np.mean(lats)), 1) if lats else None,
    "cost_note": "حساب مجاني (Google AI Studio) — تكلفة 0 جنيه في هذه التجربة",
    "confusion_matrix": cm.tolist(), "classes": CLASSES,
}
OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Purples")
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True (Egyptian waste)")
ax.set_title(f"Gemini zero-shot — Acc {acc:.1%}")
fig.tight_layout(); fig.savefig(CHARTS / "gemini_confusion_matrix.png", dpi=150)

print(f"\n=== Gemini zero-shot على المخلفات المصرية: {acc:.1%} ({len(valid)} صورة) ===", flush=True)
loc = BASE / "models" / "metrics_local.json"
if loc.exists():
    la = json.loads(loc.read_text(encoding="utf-8"))["accuracy_local"]
    print(f"مقارنة سريعة: المحلي المدرب {la:.1%} مقابل Gemini {acc:.1%}", flush=True)
print("تم الحفظ:", OUT, flush=True)
