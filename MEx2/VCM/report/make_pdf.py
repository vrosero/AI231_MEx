#!/usr/bin/env python3
"""Build VCM_Report.pdf from the report markdown sections (reportlab platypus)."""
import os, re, html
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, Table, TableStyle, PageBreak,
                                KeepTogether, HRFlowable)
from PIL import Image as PILImage

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(BASE, "VCM_Report.pdf")
FIG = os.path.join(BASE, "figures")

INK = HexColor("#1a2332"); ACC = HexColor("#2563eb"); GRAY = HexColor("#6b7280")
LIGHT = HexColor("#eef2ff"); LINE = HexColor("#d1d5db")

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=32,
                            textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=12.5, leading=18,
                               textColor=GRAY, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=19, leading=24,
                         textColor=INK, spaceBefore=18, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14, leading=18,
                         textColor=ACC, spaceBefore=13, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                         textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14.5,
                           textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=10, leading=14,
                             textColor=INK, leftIndent=14, bulletIndent=4, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=12,
                           textColor=INK, backColor=HexColor("#f3f4f6"),
                           borderPadding=6, leftIndent=6, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.8,
                              leading=12, textColor=GRAY, alignment=TA_CENTER, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.8, leading=11.5,
                           textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.8, leading=11.5,
                            textColor=HexColor("#ffffff")),
    "toc": ParagraphStyle("toc", fontName="Helvetica", fontSize=11, leading=20, textColor=INK),
}

def inline(t):
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    # italic only when the *...* does not touch a word char (avoids
    # BRIGHTNESS_20/60/100 being read as <i>60</i> inside inline code)
    t = re.sub(r"(?<![\w/`])\*([^*`\w][^*]*?)\*(?![\w/`])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8.6'>\1</font>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    return t

def scaled_image(path, max_w=170*mm, max_h=105*mm):
    with PILImage.open(path) as im:
        w, h = im_size = im.size
    r = min(max_w / w, max_h / h)
    return Image(path, width=w*r, height=h*r)

def make_table(rows, widths=None):
    header = [Paragraph(inline(c), S["cellh"]) for c in rows[0]]
    body = [[Paragraph(inline(c), S["cell"]) for c in r] for r in rows[1:]]
    t = Table([header] + body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACC),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t

def col_widths(rows, total=170*mm):
    ncol = len(rows[0])
    if ncol == 2: return [total*0.42, total*0.58]
    if ncol == 3: return [total*0.30, total*0.46, total*0.12]
    if ncol == 4: return [total*0.10, total*0.42, total*0.30, total*0.18]
    w = total / ncol
    return [w] * ncol

def parse_md(path, story):
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    fname = os.path.basename(path)
    if fname != "README.md":
        story.append(Paragraph(fname.replace(".md", "").replace("_", " ").upper(),
                               ParagraphStyle("sec", parent=S["subtitle"], textColor=ACC,
                                              fontSize=10.5, spaceAfter=2)))
        story.append(HRFlowable(width="100%", thickness=1.2, color=ACC, spaceAfter=10))
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1; continue
        if ln.startswith("```"):
            j = i + 1; buf = []
            while j < n and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            story.append(Paragraph("<br/>".join(html.escape(b) for b in buf) or " ", S["code"]))
            i = j + 1; continue
        if ln.startswith("|"):
            j = i; rows = []
            while j < n and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                j += 1
            if rows:
                story.append(make_table(rows, col_widths(rows)))
                story.append(Spacer(1, 8))
            i = j; continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)", ln.strip())
        if m:
            img = os.path.join(BASE, m.group(1))
            if os.path.exists(img):
                story.append(Spacer(1, 4))
                story.append(scaled_image(img))
                story.append(Spacer(1, 4))
            i += 1; continue
        if ln.startswith("#### "):
            story.append(Paragraph(inline(ln[5:]), S["h3"])); i += 1; continue
        if ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), S["h3"])); i += 1; continue
        if ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), S["h2"])); i += 1; continue
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), S["h1"])); i += 1; continue
        if ln.startswith("> "):
            story.append(Paragraph("<i>" + inline(ln[2:]) + "</i>",
                        ParagraphStyle("q", parent=S["body"], textColor=GRAY,
                                       leftIndent=10, borderPadding=4)))
            i += 1; continue
        if re.match(r"^[-*] ", ln):
            story.append(Paragraph("• " + inline(ln[2:]), S["bullet"])); i += 1; continue
        if re.match(r"^\d+\. ", ln):
            num, rest = ln.split(". ", 1)
            story.append(Paragraph(f"{num}. " + inline(rest), S["bullet"])); i += 1; continue
        if ln.strip() == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=8))
            i += 1; continue
        # paragraph (merge soft-wrapped lines)
        buf = [ln]; i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#|\||```|!\[|[-*] |\d+\. |>|---)", lines[i]):
            buf.append(lines[i]); i += 1
        story.append(Paragraph(inline(" ".join(buf)), S["body"]))

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(18*mm, 12*mm, "VCM — Tiny Voice Command Model")
    canvas.drawRightString(A4[0]-18*mm, 12*mm, f"page {doc.page}")
    canvas.setStrokeColor(LINE); canvas.setLineWidth(0.5)
    canvas.line(18*mm, 15*mm, A4[0]-18*mm, 15*mm)
    canvas.restoreState()

def build():
    doc = BaseDocTemplate(OUT_PDF, pagesize=A4,
                          leftMargin=20*mm, rightMargin=20*mm,
                          topMargin=18*mm, bottomMargin=20*mm,
                          title="VCM — Tiny Voice Command Model",
                          author="Elveena Grace Rosero")
    doc.addPageTemplates([PageTemplate(id="main", frames=[
        Frame(20*mm, 20*mm, A4[0]-40*mm, A4[1]-38*mm, id="f")], onPage=on_page)])
    story = []
    # cover
    story.append(Spacer(1, 46))
    story.append(Paragraph("VCM — Tiny Voice Command Model", S["title"]))
    story.append(Paragraph("A standalone, on-device voice assistant for smart devices:<br/>"
                           "\"Hello Jarvis\" wake gate · CRNN command classifier · real actions on a Raspberry Pi 4",
                           S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.4, color=ACC, spaceAfter=14))
    cover_rows = [
        ["Course deliverable", "AI 231 MEx 2 — real-world VCM demo (RPi4/5)"],
        ["Device", "Raspberry Pi 4 (4 GB) · ReSpeaker 2-Mics HAT · USB speaker (only two devices)"],
        ["Wake word", "\"Hello Jarvis\" — 249 k-param CNN, AUC 0.9975, live 9/10 fired · 0 false fires"],
        ["Command model", "CRNN · 0.367 M params · 62.78 M FLOPs · ~1.6 MB ONNX"],
        ["Test accuracy", "91.6 % intent / 87.4 % fine (n = 6,305, speaker-disjoint)"],
        ["Holdout accuracy", "88.6 % intent (n = 463, unseen speakers)"],
        ["Standalone", "no cloud models · no LLM · no ASR — pure VCM, commands 1–10"],
    ]
    story.append(make_table(cover_rows, [52*mm, 118*mm]))
    story.append(Spacer(1, 16))
    story.append(Paragraph("Report contents", S["h2"]))
    toc = [
        "01 · Overview — problem, constraints, architecture",
        "02 · Dataset — HF ai231-me2-voice-commands + supplementals (36,604 clips)",
        "03 · Model — CRNN architecture and training",
        "04 · Results — the deployed CRNN",
        "05 · Efficiency — why this model is tiny",
        "06 · Wake word — \"Hello Jarvis\" from scratch",
        "07 · Runtime & demo — the 8 items on real hardware",
        "08 · How to set up on the Raspberry Pi — detailed guide",
        "09 · Verification — hardware tests and honest limits",
    ]
    for t in toc:
        story.append(Paragraph(t, S["toc"]))
    story.append(PageBreak())
    for f in ["README.md", "01_overview.md", "02_dataset.md", "03_model.md", "04_results.md",
              "05_efficiency.md", "06_wakeword.md", "07_runtime_demo.md", "08_rpi_setup.md",
              "09_verification.md"]:
        parse_md(os.path.join(BASE, f), story)
        if f != "09_verification.md":
            story.append(PageBreak())
    doc.build(story)
    print("wrote", OUT_PDF)

if __name__ == "__main__":
    build()