# -*- coding: utf-8 -*-
"""استخراج رقم الصفحة الظاهر لكل عنوان عبر UNO — يكتب toc_pages.json لتمرير التوليد الثاني"""
import subprocess
import time
import json
import os

SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
DOC_URL = "file:///D:/waste_research/Waste_AI_Research.docx"
OUT_JSON = r"D:\waste_research\docx_build\toc_pages.json"

import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def main():
    listener = subprocess.Popen([
        SOFFICE, "--headless", "--norestore", "--nologo", "--nodefault",
        "--accept=socket,host=127.0.0.1,port=2003;urp;StarOffice.ServiceManager",
    ])
    try:
        localContext = uno.getComponentContext()
        resolver = localContext.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", localContext)
        ctx = None
        for _ in range(30):
            try:
                ctx = resolver.resolve(
                    "uno:socket,host=127.0.0.1,port=2003;urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(1)
        if ctx is None:
            raise RuntimeError("فشل الاتصال")
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(DOC_URL, "_blank", 0, (prop("Hidden", True),))

        controller = doc.getCurrentController()
        vc = controller.getViewCursor()
        enum = doc.Text.createEnumeration()
        headings = []  # (text, style, physical_page)
        while enum.hasMoreElements():
            par = enum.nextElement()
            try:
                if not par.supportsService("com.sun.star.text.Paragraph"):
                    continue
            except Exception:
                continue
            style = par.ParaStyleName
            if style in ("Heading 1", "Heading 2"):
                txt = par.getString().strip()
                vc.gotoRange(par.getStart(), False)
                headings.append({"text": txt, "level": 1 if style == "Heading 1" else 2,
                                 "page": vc.getPage()})
        # إزاحة الترقيم: أول عنوان متن (الفصل الأول) = صفحة 1 ظاهرة
        body_first = next(h for h in headings if h["text"].startswith("الفصل الأول"))
        offset = body_first["page"] - 1
        mapping = {h["text"]: h["page"] - offset for h in headings}
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=1)
        print(f"عناوين: {len(headings)} | إزاحة: {offset} | عينة:",
              dict(list(mapping.items())[:4]))
        doc.close(False)
    finally:
        try:
            desktop.terminate()
        except Exception:
            pass
        time.sleep(2)
        listener.terminate()


if __name__ == "__main__":
    main()
