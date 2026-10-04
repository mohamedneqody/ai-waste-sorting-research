# -*- coding: utf-8 -*-
"""
تقييم النموذج المدرّب (MobileNetV2 على TrashNet) على مجموعة المخلفات المصرية المحلية EG-Waste-Mini
المخرجات: الدقة + تقرير لكل فئة + confusion matrix + سرعة الاستدلال (ms/صورة) على GPU وCPU
يُشغَّل بعد تسليم الصور في D:\waste_research\data\local_egypt\<class>\*.jpg
"""
import json
import time
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

BASE = Path(r"D:\waste_research")
LOCAL = BASE / "data" / "local_egypt"
MODEL_PATH = BASE / "models" / "mobilenetv2_trashnet_best.pt"
OUT = BASE / "models" / "metrics_local.json"
CHARTS = BASE / "charts"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# فحص المجموعة أولًا
counts = {c: len(list((LOCAL / c).glob("*.jpg"))) + len(list((LOCAL / c).glob("*.jpeg"))) +
          len(list((LOCAL / c).glob("*.png"))) for c in CLASSES} if LOCAL.exists() else {}
total = sum(counts.values())
print("المجموعة المحلية:", counts, "| الإجمالي:", total, flush=True)
if total < 100:
    raise SystemExit(f"المجموعة غير مكتملة ({total} صورة < 100). أكملوا الجمع حسب البروتوكول أولًا.")

model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

samples = []
for ci, c in enumerate(CLASSES):
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in (LOCAL / c).glob(ext):
            samples.append((p, ci))
print(f"عدد العينات: {len(samples)}", flush=True)

y_true, y_pred = [], []
latencies_gpu = []
with torch.no_grad():
    for p, ci in samples:
        try:
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        except Exception as e:
            print(f"تعذر فتح {p.name}: {e}", flush=True)
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_gpu.append((time.perf_counter() - t0) * 1000)
        y_pred.append(out.argmax(1).item())
        y_true.append(ci)

acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
print(f"\n=== الدقة على المخلفات المصرية المحلية: {acc:.1%} (على {len(y_true)} صورة) ===", flush=True)
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0), flush=True)

# سرعة الاستدلال على CPU أيضًا (سيناريو التشغيل بدون GPU — مهم للقرار التطبيقي)
model_cpu = model.to("cpu").eval()
latencies_cpu = []
with torch.no_grad():
    for p, ci in samples[:50]:
        img = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        t0 = time.perf_counter()
        model_cpu(img)
        latencies_cpu.append((time.perf_counter() - t0) * 1000)

metrics = {
    "dataset": "EG-Waste-Mini — صور مصرية حقيقية من البيئة الجامعية",
    "counts_per_class": counts,
    "n_evaluated": len(y_true),
    "accuracy_local": round(acc, 4),
    "per_class": {k: {"precision": round(v["precision"], 4), "recall": round(v["recall"], 4),
                      "f1": round(v["f1-score"], 4), "support": v["support"]}
                  for k, v in report.items() if k in CLASSES},
    "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
    "latency_gpu_ms_avg": round(float(np.mean(latencies_gpu)), 1) if latencies_gpu else None,
    "latency_cpu_ms_avg": round(float(np.mean(latencies_cpu)), 1),
    "confusion_matrix": cm.tolist(), "classes": CLASSES,
    "note": "مقارنة أساسية مقابل دقة النموذج على اختبار TrashNet: 89.21%",
}
OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Oranges")
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True (Egyptian waste)")
ax.set_title(f"Local Egyptian Waste — Acc {acc:.1%}")
fig.tight_layout(); fig.savefig(CHARTS / "local_confusion_matrix.png", dpi=150)
print("تم الحفظ:", OUT, "و charts/local_confusion_matrix.png", flush=True)
