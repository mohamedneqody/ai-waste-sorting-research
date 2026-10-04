# -*- coding: utf-8 -*-
"""
المهمة 3 (محدثة 2026-10-04) — تقييم MobileNetV2 المدرّب على EG-Waste-Mini عبر البيان المعتمد
لا يبدأ التقييم إلا إذا:
  1) صور الفولدرات كافية (≥100 إجمالًا و≥10 لكل فئة)
  2) eg_waste_manifest.csv موجود وغير فارغ
  3) كل صورة لها صف واحد فقط في البيان، true_label صحيح ويطابق اسم فولدر الصورة
  4) scene_type و difficulty معبآن بقيم صحيحة (single/crowded — normal/transparent_hard)
  5) تطابق تام بين البيان والفولدرات — لا صورة في بيان غير موجودة ولا صورة على القرص غائبة عن البيان
التقييم يعتمد على صور البيان فقط (لا الفولدرات وحدها).
المخرجات في results/ عند النجاح: predictions_mobilenet.csv / classification_report_mobilenet.csv /
confusion_matrix_mobilenet.png / summary_mobilenet.json
"""
import csv
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

from manifest_utils import CLASSES, scan_folder_images, validate_manifest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # رسائل عربية سليمة حتى على CP1252

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "local_egypt"
MANIFEST = BASE / "eg_waste_manifest.csv"
MODEL_PATH = BASE / "models" / "mobilenetv2_trashnet_best.pt"
OUT = BASE / "results"
MIN_TOTAL, MIN_PER_CLASS, WARMUP = 100, 10, 3
SEED = 42  # ثابتة للتوثيق؛ التقييم بلا عشوائية (ترتيب المسار الأبجدي)

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def guards():
    disk = scan_folder_images(IMG_DIR, BASE)
    counts = {c: 0 for c in CLASSES}
    for lbl in disk.values():
        counts[lbl] += 1
    total = len(disk)
    if total < MIN_TOTAL or any(v < MIN_PER_CLASS for v in counts.values()):
        print("=== الحارس: المجموعة غير مكتملة — لم يُنفَّذ أي حساب ولم تُكتب أي نتيجة ===", flush=True)
        print(f"الموجود فعليًا ({total} صورة من {MIN_TOTAL} كحد أدنى):", flush=True)
        for c in CLASSES:
            need = max(0, MIN_PER_CLASS - counts[c])
            print(f"  {c:10s}: {counts[c]:4d}  {'(يكمل ' + str(need) + ' على الأقل)' if need else ''}", flush=True)
        print("المرجع: docs/image_collection_protocol.md — ثم python scripts/make_manifest.py", flush=True)
        return None
    ok, rows, problems = validate_manifest(MANIFEST, IMG_DIR, BASE)
    if not ok:
        print("=== الحارس: البيان غير صالح — لا تقييم ولا كتابة أي نتيجة ===", flush=True)
        for pr in problems:
            print("  -", pr, flush=True)
        print("البيان هو أساس التقييم: صلح المشاكل أعلاه (أو أعد توليد البيان ثم أكمل الحكم البصري) وأعد التشغيل.", flush=True)
        return None
    return rows

def preflight_readable(rows):
    """فحص قابلية قراءة كل صورة قبل بدء الاستدلال — أي صورة تالفة توقف التقييم كله (لا تقييم جزئي)"""
    bad = []
    for r in rows:
        try:
            with Image.open(BASE / r["image_path"]) as im:
                im.verify()
        except Exception as e:
            bad.append((r["image_id"], str(e)[:100]))
    return bad

def main():
    rows = guards()
    if rows is None:
        sys.exit(2)
    bad = preflight_readable(rows)
    if bad:
        print("=== الحارس: صور غير قابلة للقراءة — التقييم متوقف بالكامل ولا تُكتب أي نتيجة ===", flush=True)
        for iid, err in bad:
            print("  -", iid, "→", err, flush=True)
        print("استبعد/استبدل الصور التالفة (مع إبقاء البيان مطابقًا) ثم أعد التشغيل.", flush=True)
        sys.exit(2)
    print(f"الحارس نجح: {len(rows)} صورة في بيان صالح ومقروء بالكامل. تحميل النموذج الحالي (بدون تغيير)...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    model = model.to(device).eval()
    device_label = (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

    rows = sorted(rows, key=lambda r: r["image_path"])
    print(f"العينات: {len(rows)} | الجهاز: {device_label}", flush=True)

    recs, lat_e2e, lat_gpu_fwd, fatal = [], [], [], []
    # إحماء غير مقيس: 3 تمريرات على أول صورة فقط — لا تُستهلك من التقييم (كل الصور تُقيَّم)
    with torch.no_grad():
        warm_img = tf(Image.open(BASE / rows[0]["image_path"]).convert("RGB")).unsqueeze(0).to(device)
        for _ in range(WARMUP):
            model(warm_img)
    if device.type == "cuda":
        torch.cuda.synchronize()  # تنظيف طابور GPU قبل بدء القياس
    with torch.no_grad():
        for r in rows:
            p = BASE / r["image_path"]
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()  # مزامنة قبل بدء القياس — لا يتسرب وقت عمليات غير متزامنة سابقة
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
                             "predicted_label": CLASSES[idx.item()],
                             "confidence": round(float(conf), 4),
                             "inference_ms": round((t2 - t0) * 1000, 1),
                             "device": device_label,
                             "scene_type": r["scene_type"], "difficulty": r["difficulty"]})
                lat_e2e.append((t2 - t0) * 1000)
                lat_gpu_fwd.append((t2 - t1) * 1000)
            except Exception as e:
                fatal.append({"image_id": r["image_id"], "error": str(e)[:120]})
                print(f"  فشل غير متوقع أثناء الاستدلال ({r['image_id']}): {e} — إيقاف كامل قبل كتابة أي نتيجة", flush=True)
                break

    # قياس forward-only على CPU لكل الصور (سيناريو التشغيل بدون GPU)
    model_cpu = model.to("cpu").eval()
    lat_cpu_fwd = []
    with torch.no_grad():
        for r in rows:
            try:
                img = tf(Image.open(BASE / r["image_path"]).convert("RGB")).unsqueeze(0)
                t1 = time.perf_counter()
                model_cpu(img)
                lat_cpu_fwd.append((time.perf_counter() - t1) * 1000)
            except Exception:
                pass

    # أي فشل أثناء الاستدلال يمنع كتابة أي نتيجة — لا تقييم جزئي
    if fatal:
        print("=== التقييم متوقف: فشل أثناء الاستدلال — لا يُنشأ results ولا أي نتيجة ===", flush=True)
        for f in fatal:
            print("  -", f["image_id"], "→", f["error"], flush=True)
        sys.exit(2)

    y_true = [r["true_label"] for r in recs]
    y_pred = [r["predicted_label"] for r in recs]
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    # الإصلاح: labels نصية مطابقة لطبيعة y_true/y_pred — labels عددية كانت تنتج مصفوفة أصفار بصمت
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    macro_f1 = float(rep["macro avg"]["f1-score"])

    OUT.mkdir(exist_ok=True)
    with (OUT / "predictions_mobilenet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "true_label", "predicted_label", "confidence",
                                          "inference_ms", "device", "scene_type", "difficulty"])
        w.writeheader(); w.writerows(recs)

    with (OUT / "classification_report_mobilenet.csv").open("w", newline="", encoding="utf-8") as f:
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
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (EG-Waste-Mini)")
    ax.set_title(f"MobileNetV2 on EG-Waste-Mini — Acc {acc:.1%}")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_mobilenet.png", dpi=150)

    scene_counts = {s: sum(1 for r in recs if r["scene_type"] == s) for s in sorted({r["scene_type"] for r in recs})}
    diff_counts = {d: sum(1 for r in recs if r["difficulty"] == d) for d in sorted({r["difficulty"] for r in recs})}
    summary = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "manifest_file": "eg_waste_manifest.csv",
        "manifest_validated": True,
        "preflight_note": "تم فحص قابلية قراءة كل صورة قبل الاستدلال — أي صورة تالفة توقف التقييم كله (لا تقييم جزئي)",
        "n_images": len(recs),
        "counts_per_class": counts,
        "scene_type_counts": scene_counts, "difficulty_counts": diff_counts,
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_class": {c: {"precision": round(rep[c]["precision"], 4), "recall": round(rep[c]["recall"], 4),
                          "f1": round(rep[c]["f1-score"], 4), "support": rep[c]["support"]} for c in CLASSES},
        "confusion_matrix": cm.tolist(), "confusion_matrix_labels": CLASSES,
        "device": device_label,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "seed_note": "التقييم بلا عشوائية (ترتيب مسار البيان الأبجدي)؛ البذرة 42 استُخدمت في تدريب TrashNet فقط",
        "timing_definition": {
            "csv_inference_ms": "end-to-end محلي لكل صورة = قراءة + معالجة مسبقة + forward (بدون كتابة ملفات)؛ لا يشمل أي شبكة",
            "gpu_sync_note": "torch.cuda.synchronize() قبل وبعد كل قياس GPU + بعد الإحماء — حتى لا يظهر الزمن أقل من الحقيقي بسبب التنفيذ غير المتزامن",
            "inference_mode_note": "model.eval() مفعلة وكل حلقات الاستدلال داخل torch.no_grad()",
            "warmup_note": f"{WARMUP} تمريرات إحماء على أول صورة فقط (غير مقيسة ولا تُستهلك من التقييم — كل الصور تُقيَّم)",
            "gpu_forward_only_avg_ms": round(float(np.mean(lat_gpu_fwd)), 1) if lat_gpu_fwd else None,
            "cpu_forward_only_avg_ms": round(float(np.mean(lat_cpu_fwd)), 1) if lat_cpu_fwd else None,
            "e2e_avg_ms": round(float(np.mean(lat_e2e)), 1) if lat_e2e else None,
            "cloud_note": "هذه أزمنة محلية فقط — لا تقارن مباشرة بزمن استجابة خدمة سحابية"},
        "model_source": "models/mobilenetv2_trashnet_best.pt (النموذج الحالي، لم يُعَد تدريبه)",
        "internal_trashnet_reference": "Accuracy=89.21%, Macro F1=0.879 (دقة داخلية منفصلة تمامًا عن هذا الاختبار الخارجي)",
    }
    (OUT / "summary_mobilenet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== النتيجة الفعلية على EG-Waste-Mini: Accuracy={acc:.4f} | Macro F1={macro_f1:.4f} "
          f"({len(recs)} صورة) ===", flush=True)
    print("المخرجات:", flush=True)
    for name in ("predictions_mobilenet.csv", "classification_report_mobilenet.csv",
                 "confusion_matrix_mobilenet.png", "summary_mobilenet.json"):
        print("  results/" + name, flush=True)

if __name__ == "__main__":
    main()
