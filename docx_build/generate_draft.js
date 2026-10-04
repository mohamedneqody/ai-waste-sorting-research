// -*- coding: utf-8 -*-
// محول المسودة العربية → DOCX أكاديمي RTL (غلاف + ملخص روماني + فهرس + متن عربي)
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, PageNumber, NumberFormat, SectionType,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableOfContents, PageBreak,
} = require("docx");
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size");

const BASE = "D:\\waste_research";
const DRAFT = path.join(BASE, "مسودة_بحث_فرز_المخلفات_بالذكاء_الاصطناعي.md");
const OUT = path.join(BASE, "Waste_AI_Research.docx");

// الأشكال: رقم الشكل → مسار الصورة
const FIGURES = {
  "3-1": "review_web/contact_sheet_plastic.png",
  "4-1": "screenshots/home_page.png",
  "4-2": "screenshots/result_page.png",
  "4-3": "screenshots/dashboard.png",
  "5-1": "charts/training_curve.png",
  "5-2": "charts/confusion_matrix.png",
  "5-3": "results/webset/confusion_matrix_webset.png",
  "5-4": "results/webset/confusion_matrix_gemini_webset.png",
  "5-5": "results/webset/per_class_f1_comparison.png",
};

const F_BODY = 28;   // 14pt Traditional Arabic
const F_CAP = 22;    // 11pt captions
const F_H1 = 34, F_H2 = 30, F_H3 = 28;
const FONT = { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "Traditional Arabic", eastAsia: "Times New Roman" };
const NB = { style: BorderStyle.NONE };

function ar(text, opts = {}) {
  // تشغيلة عربية RTL مع خط Complex Script مطابق
  return new TextRun({
    text,
    rightToLeft: true,
    font: FONT,
    size: opts.size || F_BODY,
    sizeComplexScript: opts.size || F_BODY,
    bold: !!opts.bold,
    boldComplexScript: !!opts.bold,
    italics: !!opts.italics,
    color: opts.color || "000000",
  });
}

// تحليل **غامق** و`كود` داخل سطر إلى تشغيلات
function inlineRuns(text, opts = {}) {
  const runs = [];
  const parts = String(text).split(/\*\*(.+?)\*\*/g);
  parts.forEach((part, i) => {
    if (!part) return;
    const bold = i % 2 === 1;
    const clean = part.replace(/`([^`]*)`/g, "$1");
    runs.push(ar(clean, { ...opts, bold: bold || opts.bold }));
  });
  return runs.length ? runs : [ar(" ", opts)];
}

function bodyPara(text, opts = {}) {
  return new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 480 },
    spacing: { line: 330, after: 60 },
    children: inlineRuns(text, opts),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    bidirectional: true,
    alignment: AlignmentType.CENTER,
    spacing: { before: 380, after: 220, line: 380, lineRule: "atLeast" },
    children: [ar(text, { size: F_H1, bold: true })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { before: 280, after: 160, line: 360, lineRule: "atLeast" },
    children: [ar(text, { size: F_H2, bold: true })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { before: 200, after: 120, line: 340, lineRule: "atLeast" },
    children: [ar(text, { size: F_H3, bold: true })],
  });
}

function figure(num, title) {
  const rel = FIGURES[num];
  if (!rel) return [bodyPara(`[شكل ${num} غير موجود]`)];
  const abs = path.join(BASE, rel);
  const buf = fs.readFileSync(abs);
  const dim = sizeOf.imageSize ? sizeOf.imageSize(buf) : sizeOf(buf);
  const maxW = 530, maxH = 440;
  let w = maxW, h = Math.round(maxW * dim.height / dim.width);
  if (h > maxH) { h = maxH; w = Math.round(maxH * dim.width / dim.height); }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      keepNext: true,
      children: [new ImageRun({ data: buf, transformation: { width: w, height: h }, type: "png" })],
    }),
    new Paragraph({
      bidirectional: true,
      alignment: AlignmentType.CENTER,
      spacing: { after: 200, line: 300 },
      children: [ar(`شكل (${num}): ${title}`, { size: F_CAP, bold: true, color: "333333" })],
    }),
  ];
}

function threeLineTable(headerCells, dataRows) {
  const mk = (text, isHeader) => new TableCell({
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "F5F7FA" } : undefined,
    children: [new Paragraph({
      bidirectional: true,
      alignment: AlignmentType.CENTER,
      spacing: { line: 300 },
      children: inlineRuns(text, { size: 24, bold: isHeader }),
    })],
  });
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: "000000" },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000" },
      left: NB, right: NB, insideHorizontal: NB, insideVertical: NB,
    },
    rows: [
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: headerCells.map(t => {
          return new TableCell({
            margins: { top: 60, bottom: 60, left: 100, right: 100 },
            shading: { type: ShadingType.CLEAR, fill: "F5F7FA" },
            borders: { bottom: { style: BorderStyle.SINGLE, size: 3, color: "000000" }, top: NB, left: NB, right: NB },
            children: [new Paragraph({
              bidirectional: true, alignment: AlignmentType.CENTER, spacing: { line: 300 },
              children: inlineRuns(t, { size: 24, bold: true }),
            })],
          });
        }),
      }),
      ...dataRows.map(cells => new TableRow({
        cantSplit: true,
        children: cells.map(t => new TableCell({
          margins: { top: 60, bottom: 60, left: 100, right: 100 },
          children: [new Paragraph({
            bidirectional: true, alignment: AlignmentType.CENTER, spacing: { line: 300 },
            children: inlineRuns(t, { size: 24 }),
          })],
        })),
      })),
    ],
  });
}

function codeBlock(lines) {
  return new Table({
    visuallyRightToLeft: false,
    width: { size: 80, type: WidthType.PERCENTAGE },
    alignment: AlignmentType.CENTER,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "BBBBBB" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "BBBBBB" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "BBBBBB" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "BBBBBB" },
      insideHorizontal: NB, insideVertical: NB,
    },
    rows: [new TableRow({
      cantSplit: true,
      children: [new TableCell({
        margins: { top: 120, bottom: 120, left: 200, right: 200 },
        shading: { type: ShadingType.CLEAR, fill: "FAFAFA" },
        children: lines.map(l => new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { line: 280 },
          children: [new TextRun({ text: l, size: 20, font: { ascii: "Consolas", hAnsi: "Consolas", cs: "Consolas" }, color: "333333" })],
        })),
      })],
    })],
  });
}

// ============ تحليل المسودة ============
const md = fs.readFileSync(DRAFT, "utf-8");
const lines = md.split(/\r?\n/);
const startIdx = lines.findIndex(l => l.startsWith("# الملخص"));
const bodyLines = lines.slice(startIdx);

const elements = [];
const tocEntries = [];
let abstractStart = -1, bodyStart = -1;
let i = 0;
while (i < bodyLines.length) {
  let line = bodyLines[i].trimEnd();
  if (/^#\s/.test(line)) {
    const txt = line.replace(/^#\s*/, "");
    if (/^الملخص/.test(txt)) abstractStart = elements.length;
    if (/^الفصل الأول/.test(txt) && bodyStart === -1) bodyStart = elements.length;
    if (bodyStart !== -1 && !/^الملخص/.test(txt)) tocEntries.push({ level: 1, text: txt });
    elements.push(h1(txt)); i++; continue;
  }
  if (/^##\s/.test(line)) {
    const txt = line.replace(/^##\s*/, "");
    if (bodyStart !== -1) tocEntries.push({ level: 2, text: txt });
    elements.push(h2(txt)); i++; continue;
  }
  if (/^###\s/.test(line)) { elements.push(h3(line.replace(/^###\s*/, ""))); i++; continue; }
  if (/^\|/.test(line)) {
    const tbl = [];
    while (i < bodyLines.length && /^\|/.test(bodyLines[i].trim())) { tbl.push(bodyLines[i].trim()); i++; }
    const rows = tbl.filter(r => !/^\|[\s:|-]+\|$/.test(r))
      .map(r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
    if (rows.length >= 2) elements.push(threeLineTable(rows[0], rows.slice(1)), new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  }
  if (/^```/.test(line)) {
    const buf = [];
    i++;
    while (i < bodyLines.length && !/^```/.test(bodyLines[i])) { buf.push(bodyLines[i]); i++; }
    i++;
    elements.push(codeBlock(buf), new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  }
  const fig = line.match(/^\*\*شكل \((\d-\d)\):\s*([^*]+)\*\*/);
  if (fig) { elements.push(...figure(fig[1], fig[2].trim())); i++; continue; }
  if (/^>\s?/.test(line)) { elements.push(bodyPara(line.replace(/^>\s?/, ""), { noIndent: true, italics: true, color: "333333" })); i++; continue; }
  if (/^---\s*$/.test(line) || line.trim() === "") { i++; continue; }
  if (/^(-|\d+\.)\s/.test(line.trim())) { elements.push(bodyPara(line.trim().replace(/^- /, "• "), { noIndent: true })); i++; continue; }
  elements.push(bodyPara(line));
  i++;
}

// ============ الغلاف ============
function coverPara(text, size, bold, before, after) {
  return new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.CENTER,
    spacing: { before, after, line: Math.ceil((size / 2) * 23), lineRule: "atLeast" },
    children: [ar(text, { size, bold })],
  });
}
const infoRows = [
  ["إعداد الطالبة", "[الاسم الثلاثي]"],
  ["الكلية", "[اسم الكلية]"],
  ["الفرقة / القسم", "[يُستكمل]"],
  ["إشراف", "[يُستكمل إن وجد]"],
];
const coverInfoTable = new Table({
  visuallyRightToLeft: true,
  width: { size: 62, type: WidthType.PERCENTAGE },
  alignment: AlignmentType.CENTER,
  borders: { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB },
  rows: infoRows.map(([label, value]) => new TableRow({
    cantSplit: true,
    children: [
      new TableCell({
        width: { size: 32, type: WidthType.PERCENTAGE },
        borders: { top: NB, left: NB, right: NB, bottom: NB },
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.RIGHT, spacing: { line: 340, lineRule: "atLeast" }, children: [ar(label + ":", { size: 28, bold: true })] })],
      }),
      new TableCell({
        width: { size: 68, type: WidthType.PERCENTAGE },
        borders: { top: NB, left: NB, right: NB, bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" } },
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { line: 340, lineRule: "atLeast" }, children: [ar(value, { size: 28 })] })],
      }),
    ],
  })),
});
const coverChildren = [
  coverPara("[اسم الجامعة]", 40, true, 900, 200),
  coverPara("[اسم الكلية]", 30, false, 0, 500),
  coverPara("بحث علمي مقدم للمسابقة البحثية الجامعية", 26, false, 400, 100),
  coverPara("موضوع: تدوير المخلفات وإعادة استخدامها", 26, true, 0, 700),
  coverPara("توظيف الذكاء الاصطناعي وتحليل البيانات في فرز المخلفات:", 34, true, 600, 120),
  coverPara("دراسة تجريبية لتقييم تعميم نماذج التعلم العميق على صور عامة ومقارنتها بنموذج رؤية تجاري", 28, true, 0, 1100),
  coverInfoTable,
  coverPara("أكتوبر 2026", 28, false, 1100, 0),
];

// ============ الرأس والتذييل ============
const DOC_TITLE = "فرز المخلفات بالذكاء الاصطناعي — دراسة تجريبية";
function buildHeader() {
  return new Header({ children: [
    new Paragraph({
      bidirectional: true, alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "000000" } },
      children: [ar(DOC_TITLE, { size: 18, color: "333333" })],
    }),
  ] });
}
function buildFooter() {
  return new Footer({ children: [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: "- ", size: 21, font: FONT }),
        new TextRun({ children: [PageNumber.CURRENT], size: 21, font: FONT }),
        new TextRun({ text: " -", size: 21, font: FONT }),
      ],
    }),
  ] });
}

const PAGE = {
  size: { width: 11906, height: 16838 },
  margin: { top: 1440, bottom: 1440, left: 1701, right: 1417, header: 850, footer: 992 },
};

let TOC_PAGES = {};
try { TOC_PAGES = JSON.parse(fs.readFileSync(path.join(__dirname, "toc_pages.json"), "utf-8")); } catch (e) {}

const tocSection = [
  new Paragraph({
    bidirectional: true, alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 300, line: 400, lineRule: "atLeast" },
    children: [ar("قائمة المحتويات", { size: F_H1, bold: true })],
  }),
  ...tocEntries.map(e => new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { line: 330, after: e.level === 1 ? 60 : 30 },
    indent: e.level === 2 ? { right: 400 } : undefined,
    children: [ar(`${e.text}  ..........  ${TOC_PAGES[e.text] !== undefined ? TOC_PAGES[e.text] : "…"}`,
      { size: e.level === 1 ? 28 : 26, bold: e.level === 1 })],
  })),
];

const abstractChildren = elements.slice(abstractStart, bodyStart);
const bodyChildren = elements.slice(bodyStart);

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: F_BODY, color: "000000" },
        paragraph: { spacing: { line: 330 } },
      },
      heading1: { run: { font: FONT, size: F_H1, bold: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 480, after: 300, line: 400 } } },
      heading2: { run: { font: FONT, size: F_H2, bold: true, color: "000000" }, paragraph: { spacing: { before: 360, after: 200, line: 380 } } },
      heading3: { run: { font: FONT, size: F_H3, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 140, line: 360 } } },
    },
  },
  features: { updateFields: true },
  sections: [
    { // 1) الغلاف — بلا ترقيم
      properties: { page: PAGE, titlePage: true },
      children: coverChildren,
    },
    { // 2) الملخص — روماني من i
      properties: { type: SectionType.NEXT_PAGE, page: { ...PAGE, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } } },
      headers: { default: buildHeader() },
      footers: { default: buildFooter() },
      children: abstractChildren,
    },
    { // 3) قائمة المحتويات — روماني متواصل
      properties: { type: SectionType.NEXT_PAGE, page: { ...PAGE, pageNumbers: { formatType: NumberFormat.UPPER_ROMAN } } },
      headers: { default: buildHeader() },
      footers: { default: buildFooter() },
      children: tocSection,
    },
    { // 4) المتن — عربي من 1
      properties: { type: SectionType.NEXT_PAGE, page: { ...PAGE, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      headers: { default: buildHeader() },
      footers: { default: buildFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("DOCX written:", OUT, "| elements:", elements.length, "| abstract:", abstractChildren.length, "| body:", bodyChildren.length);
});
