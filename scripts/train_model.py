# -*- coding: utf-8 -*-
"""
تدريب نموذج فرز المخلفات — MobileNetV2 + Transfer Learning على TrashNet
المخرجات: النموذج المدرب + دقة الاختبار + confusion matrix + منحنى التدريب + مقاييس JSON
"""
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE = Path(r"D:\waste_research")
DATA = BASE / "data" / "dataset-resized"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)
(CHARTS := BASE / "charts").mkdir(exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[1/5] Device: {DEVICE}", flush=True)

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds = torchvision.datasets.ImageFolder(str(DATA), transform=eval_tf)
assert full_ds.classes == CLASSES, f"unexpected classes: {full_ds.classes}"
targets = np.array(full_ds.targets)
idx = np.arange(len(targets))

idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=targets, random_state=SEED)
targets_tmp = targets[idx_tmp]
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=targets_tmp, random_state=SEED)
print(f"[2/5] Split: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}", flush=True)

train_ds = torchvision.datasets.ImageFolder(str(DATA), transform=train_tf)
train_sub = Subset(train_ds, idx_train)
val_sub = Subset(full_ds, idx_val)
test_sub = Subset(full_ds, idx_test)

BATCH = 32
train_dl = DataLoader(train_sub, batch_size=BATCH, shuffle=True, num_workers=0)
val_dl = DataLoader(val_sub, batch_size=BATCH, shuffle=False, num_workers=0)
test_dl = DataLoader(test_sub, batch_size=BATCH, shuffle=False, num_workers=0)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model = model.to(DEVICE)

EPOCHS = 12
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
print(f"[3/5] Training {EPOCHS} epochs...", flush=True)
t0 = time.time()
best_val = 0.0
for ep in range(1, EPOCHS + 1):
    model.train()
    tl, tc, tn = 0.0, 0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(xb); loss = crit(out, yb)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tl += loss.item() * yb.size(0)
        tc += (out.argmax(1) == yb).sum().item(); tn += yb.size(0)
    train_loss, train_acc = tl / tn, tc / tn

    model.eval()
    vl, vc, vn = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            vl += crit(out, yb).item() * yb.size(0)
            vc += (out.argmax(1) == yb).sum().item(); vn += yb.size(0)
    val_loss, val_acc = vl / vn, vc / vn
    sched.step()

    hist["train_loss"].append(train_loss); hist["train_acc"].append(train_acc)
    hist["val_loss"].append(val_loss); hist["val_acc"].append(val_acc)
    flag = ""
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), MODELS / "mobilenetv2_trashnet_best.pt")
        flag = "  <- best saved"
    print(f"    epoch {ep:02d}/{EPOCHS}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}{flag}", flush=True)

train_time = time.time() - t0
print(f"[4/5] Training done in {train_time:.1f}s. Loading best weights...", flush=True)
model.load_state_dict(torch.load(MODELS / "mobilenetv2_trashnet_best.pt", weights_only=True))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        out = model(xb.to(DEVICE))
        y_pred += out.argmax(1).cpu().tolist(); y_true += yb.tolist()

acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
print(f"[5/5] TEST ACCURACY: {acc:.4f}", flush=True)
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0), flush=True)

metrics = {
    "model": "MobileNetV2 (transfer learning, ImageNet-pretrained)",
    "dataset": "TrashNet — 2527 images, 6 classes (cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137)",
    "split": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test), "seed": SEED},
    "epochs": EPOCHS, "batch": BATCH, "optimizer": "Adam lr=1e-4 cosine",
    "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
    "train_time_sec": round(train_time, 1),
    "test_accuracy": round(acc, 4),
    "per_class": {k: {"precision": round(v["precision"], 4), "recall": round(v["recall"], 4),
                      "f1": round(v["f1-score"], 4), "support": v["support"]}
                  for k, v in report.items() if k in CLASSES},
    "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
    "confusion_matrix": cm.tolist(),
    "classes": CLASSES,
}
(MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, EPOCHS + 1), hist["train_loss"], label="train loss")
ax[0].plot(range(1, EPOCHS + 1), hist["val_loss"], label="val loss")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
ax[1].plot(range(1, EPOCHS + 1), hist["train_acc"], label="train acc")
ax[1].plot(range(1, EPOCHS + 1), hist["val_acc"], label="val acc")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].set_title("Accuracy")
fig.tight_layout(); fig.savefig(CHARTS / "training_curve.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(7, 6))
im = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks(range(len(CLASSES))); ax2.set_xticklabels(CLASSES, rotation=45, ha="right")
ax2.set_yticks(range(len(CLASSES))); ax2.set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("True"); ax2.set_title(f"Confusion Matrix — Test Acc {acc:.1%}")
fig2.tight_layout(); fig2.savefig(CHARTS / "confusion_matrix.png", dpi=150)
print("DONE — metrics.json + charts saved", flush=True)
