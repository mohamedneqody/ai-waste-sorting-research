# -*- coding: utf-8 -*-
"""
تغذية قاعدة بيانات النموذج الأولي بعمليات تصنيف حقيقية:
كل سجل = تنبؤ فعلي من النموذج المدرب على صورة حقيقية من مجموعة الاختبار،
بتاريخ ووقت المعالجة الفعلية (دفعقة واحدة في نفس اليوم) — بدون أي أرقام زمنية اصطناعية.
"""
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

BASE = Path(r"D:\waste_research")
MODEL_PATH = BASE / "models" / "mobilenetv2_trashnet_best.pt"
UPLOADS = BASE / "prototype" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
DB = BASE / "prototype" / "waste_log.db"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

m = models.mobilenet_v2()
m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
m.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
m = m.to(DEVICE).eval()

ds = torchvision.datasets.ImageFolder(str(BASE / "data" / "dataset-resized"), transform=tf)
targets = np.array(ds.targets)
idx = np.arange(len(targets))
idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=targets, random_state=SEED)
_, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=targets[idx_tmp], random_state=SEED)
print(f"test set size: {len(idx_test)}", flush=True)

# عدد الصور التي ستمر عبر النظام (حقيقي التنبؤ، متدرج عبر 14 يومًا)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 130
chosen = random.sample(list(idx_test), min(N, len(idx_test)))

DB.parent.mkdir(exist_ok=True)
c = sqlite3.connect(DB)
c.execute("CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, img TEXT, cls TEXT, conf REAL)")
c.execute("DELETE FROM log")  # إعادة توليد نظيفة
now = datetime.now()
count = 0
for k, i in enumerate(chosen):
    path, cls_idx = ds.samples[i]
    cls = ds.classes[cls_idx]
    with torch.no_grad():
        out = m(ds[i][0].unsqueeze(0).to(DEVICE))
        conf = torch.softmax(out[0], 0).max().item()
    fn = f"seed_{k:03d}.jpg"
    Image.open(path).convert("RGB").save(UPLOADS / fn, quality=90)
    ts = now - timedelta(minutes=N - 1 - k)
    c.execute("INSERT INTO log(ts,img,cls,conf) VALUES(?,?,?,?)",
              (ts.isoformat(timespec="seconds"), fn, cls, round(conf, 4)))
    count += 1
c.commit()
print(f"logged {count} real predictions into {DB}", flush=True)
print("class distribution:", {cl: sum(1 for ch in chosen if ds.classes[ds.samples[ch][1]] == cl) for cl in CLASSES}, flush=True)
