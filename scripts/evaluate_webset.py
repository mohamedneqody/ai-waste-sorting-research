# -*- coding: utf-8 -*-
"""
المحور الثاني للتجربة — تقييم MobileNetV2 المدرّب (النموذج المعتمد دون تغيير) على Web-Waste-Mini
صور ويب عامة مرخصة (Wikimedia Commons) — اختبار تعميم خارجي موسوم بصدق، منفصل تمامًا عن خط EG-Waste-Mini.

الضمانات (نفس معايير الخط المعتمد):
  - لا يعدّل أي سكربت معتمد ولا النموذج ولا قاعدة SQLite — يكتب في results/webset/ فقط.
  - بوابات صارمة قبل أي حساب: كل صورة مقبولة موجودة + SHA-256 مطابق للمانيفست + التسمية تطابق الفولدر
    + scene_type/difficulty صحيحان + قابلية قراءة كل صورة (preflight كامل — لا تقييم جزئي).
  - labels=CLASSES نصية، معالجة استدلال eval_tf فقط بلا Augmentation، مزامنة CUDA قبل/بعد القياس،
    model.eval() + torch.no_grad()، إحماء على أول صورة فقط دون استهلاكها.
المخرجات (results/webset/): predictions_webset.csv / classification_report_webset.csv /
  confusion_matrix_webset.png / summary_webset.json (+ تفصيل الدقة حسب scene_type و difficulty)
"""
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifest_utils import CLASSES, SCENE_TYPES, DIFFICULTIES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
WEB_MANIFEST = BASE / "provenance" / "web_waste_manifest.csv"
MODEL_PATH = BASE / "models" / "mobilenetv2_trashnet_best.pt"
OUT = BASE / "results" / "webset"
MIN_TOTAL, MIN_PER_CLASS, WARMUP = 100, 10, 3
SEED = 42  # للتوثيق؛ التقييم بلا عشوائية (ترتيب المسارات الأبجدي)

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_webset():
    """تحميل صفوف accepted والتحقق الكامل — يرجع (rows, counts, None) أو (None, None, رسالة فشل)"""
    if not WEB_MANIFEST.exists():
        return None, None, f"مانيفست الويب غير موجود: {WEB_MANIFEST}"
    with WEB_MANIFEST.open(encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))
    accepted = [r for r in raw_rows if (r.get("review_status") or "").strip() == "accepted"]
    counts = {c: 0 for c in CLASSES}
    problems, rows = [], []
    for r in accepted:
        rel = (r.get("image_path") or "").strip().replace("\\", "/")
        label = (r.get("true_label") or "").strip()
        scene = (r.get("scene_type") or "").strip()
        diff = (r.get("difficulty") or "").strip()
        sha = (r.get("sha256") or "").strip().lower()
        iid = (r.get("image_id") or "").strip()
        if label not in CLASSES:
            problems.append(f"{iid}: تسمية غير صحيحة ({label or 'فارغ'})"); continue
        counts[label] += 1
        p = BASE / rel
        if not p.exists():
            problems.append(f"{iid}: الملف غير موجود ({rel})"); continue
        if p.parent.name != label:
            problems.append(f"{iid}: التسمية ({label}) لا تطابق الفولدر ({p.parent.name})"); continue
        if scene not in SCENE_TYPES:
            problems.append(f"{iid}: scene_type غير صالح ({scene or 'فارغ'})"); continue
        if diff not in DIFFICULTIES:
            problems.append(f"{iid}: difficulty غير صالح ({diff or 'فارغ'})"); continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            problems.append(f"{iid}: SHA-256 لا يطابق المانيفست — الملف تغير"); continue
        rows.append({"image_id": iid, "image_path": rel, "true_label": label,
                     "scene_type": scene, "difficulty": diff})
    total = len(rows)
    if problems:
        return None, None, "مشاكل في التحقق:\n  - " + "\n  - ".join(problems[:20])
    if total < MIN_TOTAL or any(v < MIN_PER_CLASS for v in counts.values()):
        return None, None, (f"المجموعة غير كافية: {total} صورة (المطلوب ≥{MIN_TOTAL} و≥{MIN_PER_CLASS} لكل فئة) — "
                            f"العدّاد: {counts}")
    return rows, counts, None

def preflight_readable(rows):
    bad = []
    for r in rows:
        try:
            with Image.open(BASE / r["image_path"]) as im:
                im.verify()
        except Exception as e:
            bad.append((r["image_id"], str(e)[:100]))
    return bad

def main():
    rows, counts, err = load_webset()
    if rows is None:
        print(f"=== الحارس: {err} ===")
        print("لم يُنفَّذ أي حساب ولم تُكتب أي نتيجة.")
        sys.exit(2)
    bad = preflight_readable(rows)
    if bad:
        print("=== الحارس: صور غير قابلة للقراءة — التوقف الكامل (لا تقييم جزئي) ===")
        for iid, e in bad:
            print("  -", iid, "→", e)
        sys.exit(2)
    print(f"الحارس نجح: {len(rows)} صورة مقبولة ومتحقق منها (SHA+تسمية+قراءة). تحميل النموذج المعتمد…", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    model = model.to(device).eval()
    device_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"العينات: {len(rows)} | الجهاز: {device_label}", flush=True)

    rows.sort(key=lambda r: r["image_path"])
    recs, lat_e2e, lat_gpu_fwd, fatal = [], [], [], []
    with torch.no_grad():
        warm = tf(Image.open(BASE / rows[0]["image_path"]).convert("RGB")).unsqueeze(0).to(device)
        for _ in range(WARMUP):
            model(warm)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for r in rows:
            p = BASE / r["image_path"]
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                t1 = time.perf_counter()
                out = model(img)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                probs = torch.softmax(out[0], dim=0)
                conf, idx = probs.max(0)
                recs.append({"image_id": r["image_id"], "true_label": r["true_label"],
                             "predicted_label": CLASSES[idx.item()], "confidence": round(float(conf), 4),
                             "inference_ms": round((t2 - t0) * 1000, 1), "device": device_label,
                             "scene_type": r["scene_type"], "difficulty": r["difficulty"]})
                lat_e2e.append((t2 - t0) * 1000)
                lat_gpu_fwd.append((t2 - t1) * 1000)
            except Exception as e:
                fatal.append({"image_id": r["image_id"], "error": str(e)[:120]})
                print(f"  فشل أثناء الاستدلال ({r['image_id']}): {e} — إيقاف كامل قبل كتابة أي نتيجة", flush=True)
                break
    if fatal:
        print("=== التقييم متوقف — لا يُنشأ أي ملف نتائج ===")
        for f in fatal:
            print("  -", f["image_id"], "→", f["error"])
        sys.exit(2)

    model_cpu = model.to("cpu").eval()
    lat_cpu = []
    with torch.no_grad():
        for r in rows:
            try:
                img = tf(Image.open(BASE / r["image_path"]).convert("RGB")).unsqueeze(0)
                t1 = time.perf_counter()
                model_cpu(img)
                lat_cpu.append((time.perf_counter() - t1) * 1000)
            except Exception:
                pass

    y_true = [r["true_label"] for r in recs]
    y_pred = [r["predicted_label"] for r in recs]
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    macro_f1 = float(rep["macro avg"]["f1-score"])

    def acc_of(preds, flt):
        sel = [(p["true_label"], p["predicted_label"]) for p in preds if flt(p)]
        return round(float(np.mean([a == b for a, b in sel])), 4) if sel else None
    by_scene = {s: acc_of(recs, lambda p, s=s: p["scene_type"] == s) for s in sorted({p["scene_type"] for p in recs})}
    by_diff = {d: acc_of(recs, lambda p, d=d: p["difficulty"] == d) for d in sorted({p["difficulty"] for p in recs})}
    scene_counts = {s: sum(1 for p in recs if p["scene_type"] == s) for s in sorted({p["scene_type"] for p in recs})}
    diff_counts = {d: sum(1 for p in recs if p["difficulty"] == d) for d in sorted({p["difficulty"] for p in recs})}

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "predictions_webset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "true_label", "predicted_label", "confidence",
                                          "inference_ms", "device", "scene_type", "difficulty"])
        w.writeheader(); w.writerows(recs)
    with (OUT / "classification_report_webset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c in CLASSES:
            v = rep[c]
            w.writerow([c, round(v["precision"], 4), round(v["recall"], 4), round(v["f1-score"], 4), v["support"]])
        w.writerow(["accuracy", "", "", round(acc, 4), len(recs)])
        for k in ("macro avg", "weighted avg"):
            v = rep[k]
            w.writerow([k, round(v["precision"], 4), round(v["recall"], 4), round(v["f1-score"], 4), v["support"]])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (Web-Waste-Mini)")
    ax.set_title(f"Web images (public, licensed) — Acc {acc:.1%}")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_webset.png", dpi=150)

    summary = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Web-Waste-Mini — 125 صورة ويب عامة مرخصة (Wikimedia Commons)، مقبولة بمراجعة موثقة",
        "role": "المحور الثاني (اختبار تعميم خارجي على صور ويب) — منفصل عن الاختبار الميداني المحلي EG-Waste-Mini",
        "manifest_source": "provenance/web_waste_manifest.csv (صفوف accepted فقط، SHA-256 متحقق)",
        "n_images": len(recs),
        "counts_per_class": counts,
        "scene_type_counts": scene_counts, "difficulty_counts": diff_counts,
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "accuracy_by_scene_type": by_scene, "accuracy_by_difficulty": by_diff,
        "per_class": {c: {"precision": round(rep[c]["precision"], 4), "recall": round(rep[c]["recall"], 4),
                          "f1": round(rep[c]["f1-score"], 4), "support": rep[c]["support"]} for c in CLASSES},
        "confusion_matrix": cm.tolist(), "confusion_matrix_labels": CLASSES,
        "device": device_label,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "timing_definition": {
            "csv_inference_ms": "end-to-end محلي لكل صورة (قراءة+معالجة+forward) — لا شبكة",
            "gpu_sync_note": "torch.cuda.synchronize() قبل وبعد كل قياس + بعد الإحماء",
            "warmup_note": f"{WARMUP} تمريرات على أول صورة فقط (غير مقيسة ولا مستهلكة)",
            "gpu_forward_only_avg_ms": round(float(np.mean(lat_gpu_fwd)), 1) if lat_gpu_fwd else None,
            "cpu_forward_only_avg_ms": round(float(np.mean(lat_cpu)), 1) if lat_cpu else None,
            "e2e_avg_ms": round(float(np.mean(lat_e2e)), 1) if lat_e2e else None},
        "model_source": "models/mobilenetv2_trashnet_best.pt (النموذج المعتمد، لم يُعَد تدريبه)",
        "internal_trashnet_reference": "Accuracy=89.21%, Macro F1=0.879 (دقة داخلية منفصلة — لا تخلط)",
        "integrity_note": "لا تقييم جزئي: preflight كامل قبل الاستدلال وأي فشل يوقف كل شيء قبل الكتابة",
    }
    (OUT / "summary_webset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== النتيجة الفعلية على Web-Waste-Mini: Accuracy={acc:.4f} | Macro F1={macro_f1:.4f} ({len(recs)} صورة) ===", flush=True)
    print(f"الدقة حسب نوع المشهد: {by_scene}", flush=True)
    print(f"الدقة حسب الصعوبة: {by_diff}", flush=True)
    print("المخرجات في results/webset/:", flush=True)
    for n in ("predictions_webset.csv", "classification_report_webset.csv",
              "confusion_matrix_webset.png", "summary_webset.json"):
        print("  " + n, flush=True)

if __name__ == "__main__":
    main()
