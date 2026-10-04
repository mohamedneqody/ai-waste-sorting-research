# -*- coding: utf-8 -*-
"""الأشكال التوضيحية: سلسلة الموضوع + مخطط المفاضلة — بعربية مضمونة الاتجاه"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import arabic_reshaper
from bidi.algorithm import get_display

BASE = Path(__file__).resolve().parent.parent
CHARTS = BASE / "charts"
plt.rcParams["font.family"] = ["Arial", "Tahoma", "DejaVu Sans"]

def ar(t):
    return get_display(arabic_reshaper.reshape(t))

def chain_figure():
    steps = [
        ("صورة المخلف", "#1565c0"),
        ("تصنيف بالذكاء\nالاصطناعي", "#2e7d32"),
        ("الحاوية\nالمناسبة", "#f9a825"),
        ("تقليل اختلاط\nالمواد", "#6a4caa"),
        ("دعم إعادة\nالتدوير", "#1b5e20"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    x = 10.6
    for i, (label, color) in enumerate(steps):
        box = FancyBboxPatch((x - 1.05, 0.9), 2.1, 1.3,
                             boxstyle="round,pad=0.08", fc=color, ec="#333333", lw=1.2)
        ax.add_patch(box)
        ax.text(x, 1.55, ar(label), ha="center", va="center",
                fontsize=12, color="white", weight="bold")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x - 1.15, 1.55), (x - 1.75, 1.55),
                                         arrowstyle="-|>", mutation_scale=22,
                                         color="#444444", lw=2))
        x -= 2.4
    ax.text(6, 2.7, ar("سلسلة القيمة: من الصورة إلى دعم إعادة التدوير"),
            ha="center", fontsize=14, weight="bold", color="#1b5e20")
    fig.tight_layout()
    fig.savefig(CHARTS / "topic_chain.png", dpi=160, bbox_inches="tight")

def tradeoff_figure():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    models = [ar("MobileNetV2\n(محلي)"), ar("Gemini Vision\n(سحابي)")]
    acc = [0.32, 0.888]
    lat = [29.9, 4530.3]
    bars = ax[0].bar(models, acc, color=["#1565c0", "#6a1b9a"], width=0.5)
    for b, v in zip(bars, acc):
        ax[0].text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}", ha="center", weight="bold")
    ax[0].set_ylim(0, 1); ax[0].set_ylabel(ar("الدقة على Web-Waste-Mini"))
    ax[0].set_title(ar("الدقة (125 صورة)"))
    bars2 = ax[1].bar(models, lat, color=["#1565c0", "#6a1b9a"], width=0.5)
    ax[1].set_yscale("log")
    for b, v in zip(bars2, lat):
        ax[1].text(b.get_x() + b.get_width()/2, v * 1.15, f"{v:,.0f}ms", ha="center", weight="bold")
    ax[1].set_ylim(10, 20000); ax[1].set_ylabel(ar("زمن الاستجابة (ms — مقياس لوغاريتمي)"))
    ax[1].set_title(ar("الزمن: محلي بلا شبكة مقابل سحابي شامل الشبكة"))
    ax[1].text(0, 60, ar("خصوصية كاملة\nبلا إنترنت"), ha="center", fontsize=9, color="#1565c0")
    ax[1].text(1, 900, ar("يتطلب اتصالًا\nوخصوصية أقل"), ha="center", fontsize=9, color="#6a1b9a")
    fig.suptitle(ar("المفاضلة الفعلية: الدقة مقابل الزمن والخصوصية"), fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS / "tradeoff.png", dpi=160, bbox_inches="tight")

if __name__ == "__main__":
    chain_figure()
    tradeoff_figure()
    print("charts/topic_chain.png + charts/tradeoff.png")
