# -*- coding: utf-8 -*-
"""تحديث فهرس المحتويات داخل DOCX عبر UNO ثم تصدير PDF — يعمل ببايثون LibreOffice المدمج"""
import subprocess
import sys
import time
import os

SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
DOC_URL = "file:///D:/waste_research/Waste_AI_Research.docx"
PDF_URL = "file:///D:/waste_research/Waste_AI_Research.pdf"

import uno  # متاح فقط في بايثون LibreOffice
from com.sun.star.beans import PropertyValue


def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def main():
    # 1) تشغيل المستمع
    listener = subprocess.Popen([
        SOFFICE, "--headless", "--norestore", "--nologo", "--nodefault",
        "--accept=socket,host=127.0.0.1,port=2002;urp;StarOffice.ServiceManager",
    ])
    try:
        # 2) الاتصال مع إعادة محاولة
        localContext = uno.getComponentContext()
        resolver = localContext.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", localContext)
        ctx = None
        for _ in range(30):
            try:
                ctx = resolver.resolve(
                    "uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(1)
        if ctx is None:
            raise RuntimeError("فشل الاتصال بـ LibreOffice")
        smgr = ctx.ServiceManager
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

        # 3) فتح المستند
        doc = desktop.loadComponentFromURL(DOC_URL, "_blank", 0, (prop("Hidden", True),))

        # 4) تحديث الحقول والفهارس (مرتان لاستقرار أرقام الصفحات)
        for round_no in (1, 2):
            doc.refresh()
            try:
                doc.getTextFields().refresh()
            except Exception:
                pass
            idxs = doc.getDocumentIndexes()
            for i in range(idxs.getCount()):
                idxs.getByIndex(i).update()
            time.sleep(1)

        # 5) تصدير PDF
        doc.storeToURL(PDF_URL, (prop("FilterName", "writer_pdf_Export"),))
        doc.close(False)
        print("PDF updated with TOC:", PDF_URL)
    finally:
        try:
            desktop.terminate()
        except Exception:
            pass
        time.sleep(2)
        listener.terminate()


if __name__ == "__main__":
    main()
