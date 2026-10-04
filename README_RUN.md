# README_RUN — دليل إعادة إنتاج نتائج الدراسة المنشورة

> **سير العمل النهائي المعتمد:** التجربة المنشورة تستخدم مجموعة **Web-Waste-Mini** (125 صورة ويب عامة مرخصة من Wikimedia Commons، إثبات كامل في `provenance/`) — لا تستخدم أي تصوير ميداني.
> أدوات المسار الميداني المحلي (`evaluate_mobilenet.py` وغيره على `data/local_egypt/`) محفوظة كأدوات أعمال مستقبلية وليست جزءًا من الدراسة المنشورة.

## 0) البيئة (المستخدمة في الدراسة — مرجع)

Python 3.14 + torch/torchvision (CUDA) + scikit-learn + matplotlib + Pillow + Flask — على NVIDIA T1200 Laptop GPU.

## 1) بيانات TrashNet (للتدريب الداخلي المرجعي)

- المصدر الأكاديمي: [github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- المرآة المستخدمة فعليًا في هذه الدراسة: [HuggingFace — garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip) (~41 ميجا)
- التنزيل والوضع: فك الضغط بحيث تكون الفئات في `data/dataset-resized/{cardboard,glass,metal,paper,plastic,trash}/`
- ثم: `python scripts/train_model.py` → يعيد إنتاج `models/metrics.json` (Accuracy = 0.8921، Macro F1 = 0.879) و`charts/`

> هذه الدقة **الداخلية** المرجعية — تُعرض منفصلة تمامًا عن نتائج الاختبار الخارجي.

## 2) مجموعة الاختبار الخارجية Web-Waste-Mini

- جاهزة في `data/web_waste_mini/` (125 صورة، 6 فئات) بمانيفست إثبات كامل في `provenance/` (المصدر، الرخصة، المؤلف، SHA-256 لكل صورة).
- بيان التحقق: `eg_waste_manifest.csv` في جذر المشروع.

## 3) تقييم MobileNetV2 على Web-Waste-Mini (RQ1)

```bash
python scripts/evaluate_webset.py
```

- الحواجز: تحقق SHA-256 لكل صورة + قابلية قراءة كاملة قبل الاستدلال — أي خلل يوقف كل شيء بلا كتابة نتائج.
- المخرجات في `results/webset/`: `predictions_webset.csv`، `classification_report_webset.csv`، `confusion_matrix_webset.png`، `summary_webset.json` (الدقة الفعلية 0.3200 وMacro F1 0.2826، وتفصيل حسب نوع المشهد والصعوبة، وتعريف الزمن المحلي).

## 4) تقييم Gemini Vision على الصور ذاتها (RQ2)

مفتاح مجاني من [Google AI Studio](https://aistudio.google.com/apikey) يُضبط في متغير البيئة فقط:

```powershell
setx GEMINI_API_KEY "المفتاح"    # ثم فتح طرفية جديدة
```

```bash
python scripts/evaluate_gemini_webset.py
```

- المطالبة ثابتة في `gemini_prompt.txt` (لا تُعدل)، محاولة واحدة لكل صورة، checkpoint يستكمل التشغيل المقطوع.
- **قاعدة الاكتمال:** أي صورة فاشلة أو إجابة غير قابلة للتحليل = التجربة غير مكتملة ولا تُحسب مقاييس جزئية.
- نتيجة الدراسة المنشورة: 125/125 نجاح — Accuracy = 0.8880، Macro F1 = 0.8876، بـ `gemini-flash-lite-latest` (الإصدار المستجيب: gemini-3.5-flash-lite)، بتاريخ 2026-10-04.

## 5) المقارنة النهائية

```bash
python scripts/compare_webset.py
```

يرفض المقارنة إلا بتطابق كامل مع البيان، ويخرج `comparison_table.md/csv` + `per_class_f1_comparison.png` — أرقام الفصل الخامس جاهزة منها مباشرة.

## 6) النموذج الأولي (تشغيل توضيحي)

```bash
python prototype/app.py   # http://127.0.0.1:5050
```

- **تشغيل توضيحي لمعالجة صور اختبار** (131 عملية في يوم واحد بسجل SQLite) — وليس قياسًا لمخلفات ميدانية أو معدل إعادة تدوير أو استخدامًا طلابيًا؛ والتنويه مثبت داخل الواجهة نفسها.

## مسار ميداني مستقبلي (خارج الدراسة المنشورة)

أدوات مسار التصوير الميداني المحلي محفوظة وجاهزة للتوسعة المستقبلية: `make_manifest.py` + `contact_sheet.py` + `evaluate_mobilenet.py` على `data/local_egypt/` — مع نفس حواجز السلامة (لا تقييم جزئي، لا نتائج بلا صور موسومة).

## أخطاء شائعة

| الرسالة | الحل |
|---|---|
| "بيانات غير سليمة" في أي سكربت | الرسالة تحدد الصف/الملف بالضبط — صححه وأعد التشغيل |
| "لا يوجد GEMINI_API_KEY" | `setx GEMINI_API_KEY "..."` ثم طرفية جديدة |
| HTTP 429 متكرر | انتظار أطول: `setx GEMINI_SLEEP "8"` — أو الاكتفاء بالموديلات ذات الحصة الأوسع |
| نتيجة Gemini "غير مكتملة" | إعادة التشغيل (checkpoint يكمل) — لن تُحسب مقاييس جزئية أبدًا |
