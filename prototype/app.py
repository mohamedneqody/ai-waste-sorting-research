# -*- coding: utf-8 -*-
"""
النموذج الأولي — نظام ذكي لفرز المخلفات
Upload → AI Classification → Recommended Bin → Dashboard
"""
import io
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for

BASE = Path(__file__).parent
MODEL_PATH = Path(r"D:\waste_research\models\mobilenetv2_trashnet_best.pt")
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
DB = BASE / "waste_log.db"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
AR_NAME = {"cardboard": "كرتون", "glass": "زجاج", "metal": "معدن",
           "paper": "ورق", "plastic": "بلاستيك", "trash": "مخلفات عامة"}
BIN = {  # لون الحاوية + توجيهها
    "cardboard": ("أزرق", "ورق وكرتون", "#1565c0", True),
    "paper": ("أزرق", "ورق وكرتون", "#1565c0", True),
    "plastic": ("أصفر", "بلاستيك", "#f9a825", True),
    "glass": ("أخضر", "زجاج", "#2e7d32", True),
    "metal": ("رمادي", "معادن", "#546e7a", True),
    "trash": ("أسود", "مخلفات عامة (غير قابل للتدوير)", "#212121", False),
}

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    m = models.mobilenet_v2()
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    m.eval()
    return m

MODEL = load_model()

_conn = None
def db():
    # اتصال واحد مشترك — فتح اتصال جديد لكل استدعاء كان يجعل INSERT غير ملتزم (يُفقد بصمت)
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS log("
                      "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, img TEXT, cls TEXT, conf REAL)")
    return _conn

def predict(img: Image.Image):
    with torch.no_grad():
        out = MODEL(tf(img.convert("RGB")).unsqueeze(0))
        probs = torch.softmax(out[0], dim=0)
        conf, i = probs.max(0)
        return CLASSES[i], float(conf), [float(p) for p in probs]

app = Flask(__name__)

PAGE = """<!doctype html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
<title>نظام ذكي لفرز المخلفات</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;font-family:'Segoe UI',Tahoma,Arial,sans-serif}
body{margin:0;background:#f4f6f8;color:#1f2937}
header{background:#1b5e20;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.85;font-size:13px}
nav{background:#2e7d32;padding:10px 28px}nav a{color:#fff;text-decoration:none;margin-left:22px;font-size:14px}
.wrap{max-width:960px;margin:24px auto;padding:0 16px}
.card{background:#fff;border-radius:12px;padding:22px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:18px}
.btn{background:#2e7d32;color:#fff;border:none;padding:10px 22px;border-radius:8px;font-size:15px;cursor:pointer}
.kpis{display:flex;gap:14px;flex-wrap:wrap}
.kpi{flex:1;min-width:160px;background:#fff;border-radius:12px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.kpi .n{font-size:26px;font-weight:700;color:#1b5e20}.kpi .t{font-size:13px;color:#6b7280}
.bar{height:18px;border-radius:9px;color:#fff;font-size:11px;line-height:18px;padding:0 8px;margin:4px 0}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:8px 10px;border-bottom:1px solid #eee;text-align:right}
.badge{padding:2px 10px;border-radius:10px;color:#fff;font-size:12px}
.bin{display:inline-block;width:56px;height:64px;border-radius:6px 6px 14px 14px;border:2px solid #333;position:relative;vertical-align:middle;margin-left:10px}
.bin:after{content:'';position:absolute;top:-10px;left:8px;right:8px;height:8px;background:inherit;border-radius:4px}
.ok{color:#2e7d32;font-weight:700}.no{color:#c62828;font-weight:700}
canvas{max-width:100%}
</style></head><body>
<header><h1>♻️ النظام الذكي لفرز المخلفات</h1>
<p>توظيف الذكاء الاصطناعي وتحليل البيانات لتحسين إعادة التدوير في البيئة الجامعية</p></header>
<nav><a href="/">تصنيف مخلف</a><a href="/dashboard">لوحة التحليلات</a></nav>
<div class="wrap">{{inner|safe}}</div></body></html>"""

@app.route("/")
def home():
    inner = """
<div class="card"><h3>تصوير / رفع مخلف للتصنيف</h3>
<p>ارفع صورة المخلف، وسيقوم نموذج الذكاء الاصطناعي (MobileNetV2) بتحديد نوعه والحاوية المناسبة له.</p>
<form action="/predict" method="post" enctype="multipart/form-data">
<input type="file" name="img" accept="image/*" required style="margin:12px 0">
<br><button class="btn" type="submit">تصنيف المخلف</button></form></div>
<div class="card"><h3>الفئات التي يتعرف عليها النظام</h3>
<table><tr><th>النوع</th><th>الحاوية</th><th>قابل للتدوير</th></tr>
{% for c, (b, bt, col, rec) in bins.items() %}
<tr><td>{{names[c]}}</td><td><span class="bin" style="background:{{col}}"></span> الحاوية {{b}} — {{bt}}</td>
<td>{% if rec %}<span class="ok">نعم</span>{% else %}<span class="no">لا</span>{% endif %}</td></tr>{% endfor %}</table></div>"""
    return render_template_string(PAGE, inner=render_template_string(inner, bins=BIN, names=AR_NAME))

@app.route("/predict", methods=["POST"])
def predict_route():
    f = request.files["img"]
    img = Image.open(io.BytesIO(f.read()))
    cls, conf, probs = predict(img)
    bname, btxt, col, rec = BIN[cls]
    fn = f"{uuid.uuid4().hex[:10]}.jpg"
    img.convert("RGB").save(UPLOADS / fn, quality=90)
    cur = db().execute("INSERT INTO log(ts,img,cls,conf) VALUES(?,?,?,?)",
                       (datetime.now().isoformat(timespec="seconds"), fn, cls, round(conf, 4)))
    db().commit()
    return redirect(url_for("result", log_id=cur.lastrowid))

@app.route("/result/<int:log_id>")
def result(log_id):
    row = db().execute("SELECT ts,img,cls,conf FROM log WHERE id=?", (log_id,)).fetchone()
    if not row:
        return "not found", 404
    ts, fn, cls, conf = row
    bname, btxt, col, rec = BIN[cls]
    inner = f"""
<div class="card"><h3>نتيجة التصنيف</h3>
<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center">
<img src="/uploads/{fn}" style="max-width:240px;border-radius:10px;border:1px solid #ddd">
<div><div style="font-size:30px;font-weight:700">{AR_NAME[cls]}</div>
<div style="margin:6px 0">ثقة النموذج: <b>{conf:.1%}</b></div>
<div><span class="bin" style="background:{col}"></span>
<b>الحاوية {bname}</b> — {btxt}</div>
<div style="margin-top:10px">{'<span class=ok>✔ قابل لإعادة التدوير</span>' if rec else '<span class=no>✖ غير قابل لإعادة التدوير</span>'}</div>
<div style="color:#6b7280;font-size:12px;margin-top:8px">رقم العملية: {log_id} — <span dir="ltr">{ts.replace("T", " ")}</span></div>
</div></div></div>
<a class="btn" href="/">تصنيف مخلف آخر</a> <a class="btn" style="background:#1565c0" href="/dashboard">لوحة التحليلات</a>"""
    return render_template_string(PAGE, inner=inner)

@app.route("/uploads/<fn>")
def up(fn): return send_from_directory(UPLOADS, fn)

@app.route("/dashboard")
def dashboard():
    rows = db().execute("SELECT ts, cls, conf FROM log ORDER BY ts").fetchall()
    total = len(rows)
    per = {k: 0 for k in CLASSES}
    rec_n = 0
    confs = []
    for _, cls, conf in rows:
        per[cls] += 1
        if BIN[cls][3]: rec_n += 1
        confs.append(conf)
    rec_pct = rec_n / total if total else 0
    avg_conf = (sum(confs) / len(confs)) if confs else 0
    active = sum(1 for k in CLASSES if per[k] > 0)
    mx = max(per.values()) if total else 1
    inner = f"""
<div class="card" style="background:#fff8e1;border:1px solid #e0c36a;padding:10px 14px"><p style="margin:0;color:#7a5c00;font-size:13px">عرض تشغيلي توضيحي لإحصاءات معالجة صور اختبار، وليست قياسًا لمخلفات ميدانية أو لمعدل إعادة تدوير.</p></div>
<div class="kpis">
<div class="kpi"><div class="n">{total}</div><div class="t">إجمالي صور الاختبار المعالجة</div></div>
<div class="kpi"><div class="n">{rec_pct:.0%}</div><div class="t">نسبة صور الاختبار المصنفة كقابلة لإعادة التدوير</div></div>
<div class="kpi"><div class="n">{avg_conf:.0%}</div><div class="t">متوسط ثقة النموذج</div></div>
<div class="kpi"><div class="n">{active}</div><div class="t">فئات تم رصدها</div></div>
</div>
<div class="card"><h3>توزيع المخلفات حسب النوع</h3>
{''.join(f"<div style='display:flex;align-items:center;gap:10px'><div style='width:90px'>{AR_NAME[c]}</div><div class='bar' style='background:{BIN[c][2]};width:{(per[c]/mx)*70:.0f}%'>{per[c]}</div></div>" for c in CLASSES)}
</div>
<div class="card"><h3>آخر عمليات التصنيف</h3><table><tr><th>التاريخ</th><th>النوع</th><th>الحاوية</th><th>الثقة</th></tr>
{''.join(f"<tr><td><span dir='ltr'>{ts.replace('T', ' ')}</span></td><td>{AR_NAME[cls]}</td><td><span class='badge' style='background:{BIN[cls][2]}'>{BIN[cls][0]}</span></td><td>{conf:.0%}</td></tr>" for ts, cls, conf in rows[-8:][::-1])}</table></div>"""
    return render_template_string(PAGE, inner=inner)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
