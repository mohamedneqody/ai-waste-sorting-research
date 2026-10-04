# فرز المخلفات بالذكاء الاصطناعي — دراسة تجريبية موثقة قابلة لإعادة التشغيل

**العنوان:** توظيف الذكاء الاصطناعي وتحليل البيانات في فرز المخلفات: دراسة تجريبية لتقييم تعميم نماذج التعلم العميق على صور عامة ومقارنتها بنموذج رؤية تجاري

**هذا المستودع هو السجل الكامل القابل لإعادة التشغيل لكل رقم في البحث** — لا يوجد أي رقم غير مولد من تجربة فعلية بسكربت موجود هنا.

**⏰ التسليم:** الأربعاء 14/10/2026 — الإدارة العامة لرعاية الطلاب (أصل + صورتين + استمارة مختومة + البطاقة والكارنيه)

## النتائج الفعلية (من ملفات التجارب، ليست تقديرات)

| الاختبار | الدقة | Macro F1 | المصدر |
|---|---:|---:|---|
| داخلي على TrashNet (380 صورة اختبار) | **89.21%** | 0.879 | `models/metrics.json` |
| خارجي على Web-Waste-Mini (125 صورة ويب مرخصة) | **32.00%** | 0.2826 | `results/webset/summary_webset.json` |
| Gemini Vision Zero-shot على الصور ذاتها | **88.80%** | 0.8876 | `results/webset/summary_gemini_webset.json` |

**القراءة:** النموذج الخفيف المدرّب داخلًا ينهار خارج توزيع تدريبه (89.21% → 32.00%)، بينما يحقق Gemini على الصور ذاتها 88.80% — مع تحفظ أن صور ويب عامة قد تكون مألوفة لنموذج تجاري مدرَّب على بيانات إنترنت واسعة (حدود الدراسة 7-4).

## بنية المستودع

| المسار | المحتوى | الحالة |
|---|---|---|
| `بحث_فرز_المخلفات_بالذكاء_الاصطناعي.docx` / `.pdf` | البحث النهائي المنسق (32 صفحة، بملحق المستودع) | ✅ نهائي |
| `مسودة_بحث_فرز_المخلفات_بالذكاء_الاصطناعي.md` | مصدر البحث القابل للتحرير (الفصول + الملاحق أ–د) | ✅ |
| `scripts/train_model.py` | تدريب MobileNetV2 على TrashNet → `models/metrics.json` + `charts/` | ✅ |
| `models/mobilenetv2_trashnet_best.pt` | النموذج المدرب (نتائج الجدول أعلاه ناتجه) | ✅ |
| `data/web_waste_mini/` | مجموعة الاختبار الخارجية: 125 صورة ويب مرخصة في 6 فئات | ✅ |
| `provenance/` | سجل الإثبات: مانيفست 410 صور (قبول/رفض بالسبب)، المنهجية، إسناد التراخيص | ✅ |
| `scripts/evaluate_webset.py` + `evaluate_gemini_webset.py` + `compare_webset.py` | التقييم الخارجي والمقارنة → `results/webset/` | ✅ |
| `prototype/app.py` | النموذج الأولي (Flask + SQLite) | ✅ |
| `docs/` + `PROJECT_AUDIT.md` | منهجية الجمع، سجل التدقيق والتغييرات | ✅ |
| `scripts/eval_local.py` / `eval_gemini.py` / `evaluate_mobilenet.py` / `make_manifest.py` / `contact_sheet.py` | أدوات مسار ميداني محلي (لم يُستخدم في الدراسة المنشورة — محفوظة كأدوات أعمال مستقبلية) | مرجعية |

> **تنويه النموذج الأولي:** تطبيق Flask + لوحة إحصاءات = **تشغيل توضيحي** لمعالجة صور اختبار (131 عملية في يوم واحد) — وليس قياسًا لمخلفات ميدانية أو معدل إعادة تدوير أو استخدامًا طلابيًا.

## إعادة إنتاج النتائج

### 1) بيانات TrashNet (للتدريب الداخلي)

المستودع لا يتضمن نسخًا من TrashNet (رخصة غير معلنة) — تُنزَّل من مصدرها ثم توضع محليًا:

- المصدر الأكاديمي المرجعي: [github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- المرآة المستخدمة فعليًا في هذه الدراسة (نفس المحتوى، ~41 ميجا):
  `https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip`
- بعد فك الضغط تكون الفئات الست في `data/dataset-resized/{cardboard,glass,metal,paper,plastic,trash}/` ثم:

```bash
python scripts/train_model.py
```

### 2) التقييم الخارجي (Web-Waste-Mini)

المجموعة جاهزة في `data/web_waste_mini/` (125 صورة، مانيفست إثبات في `provenance/`):

```bash
python scripts/evaluate_webset.py
```

### 3) مقارنة Gemini (اختيارية — يتطلب مفتاحًا مجانيًا)

مفتاح من [Google AI Studio](https://aistudio.google.com/apikey) يُضبط في متغير البيئة `GEMINI_API_KEY` (لا يُكتب في أي ملف)، ثم:

```bash
python scripts/evaluate_gemini_webset.py
python scripts/compare_webset.py
```

### 4) النموذج الأولي (تشغيل توضيحي)

```bash
python prototype/app.py   # ثم فتح http://127.0.0.1:5050
```

## الروابط الأساسية

| المصدر | الرابط |
|---|---|
| مجموعة الاختبار الخارجية (المصدر) | [Wikimedia Commons](https://commons.wikimedia.org/) — إسناد كل صورة في `provenance/WEB_IMAGE_ATTRIBUTION.md` |
| بيانات التدريب | [TrashNet — Stanford CS229 (Yang & Thung 2016)](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf) |
| النموذج الأساس | [MobileNetV2 — Sandler et al., CVPR 2018](https://doi.org/10.1109/CVPR.2018.00474) |
| نموذج المقارنة | [Gemini API — Google AI Studio](https://ai.google.dev/gemini-api/docs) |
| الإطار القانوني المصري | [قانون تنظيم إدارة المخلفات 202/2020](https://www.eeaa.gov.eg/Laws/56/index) |
| سياق عالمي | [World Bank — What a Waste 2.0](https://doi.org/10.1596/978-1-4648-1329-0) |

## التراخيص

- **صور Web-Waste-Mini:** كل صورة بترخيصها المعلن (CC0 / Public Domain / CC BY / CC BY-SA) — الإسناد الكامل في `provenance/WEB_IMAGE_ATTRIBUTION.md`.
- **TrashNet:** تُستخدم لأغراض بحثية مع الإحالة لمصدرها الأكاديمي؛ لا تُرفع نسخها في هذا المستودع.
- **نص البحث والمستندات:** جميع الحقوق محفوظة للطالبة الباحثة.
- **الأمان:** لا يوجد أي مفتاح API داخل المستودع أو تاريخه — تُضبط عبر متغيرات البيئة فقط.
