#!/usr/bin/env python3
"""Build vcm_deck.pptx — 2-page deck.

Template layout (course deck template):
  p.1  title banner + 4 quadrants (Model / Dataset / Training on MPS /
       Validation on RPi 4) + architecture diagram strip
  p.2  "To Be Submitted" checklist (6 items) + links block
       (GitHub repo / Dataset location / Training hardware / Model weights)

All numbers are the verified ones from report/ sections 02-09,
model/run_outputs/eval/*.json and rpi/perf_results/*.json.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "vcm_deck.pptx")

INK = RGBColor(0x1A, 0x23, 0x32)
ACC = RGBColor(0x25, 0x63, 0xEB)
ACC2 = RGBColor(0x7C, 0x3A, 0xED)
GOOD = RGBColor(0x05, 0x96, 0x69)
WARN = RGBColor(0xD9, 0x77, 0x06)
GRAY = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xEE, 0xF2, 0xFF)
PALE = RGBColor(0xF8, 0xFA, 0xFC)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def tx(s, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
       font="Calibri", anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = align
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font
        if line_spacing != 1.0:
            p.line_spacing = line_spacing
    return box


def table(s, x, y, w, rows, col_w=None, size=12, row_h=0.42, header=True):
    nr, nc = len(rows), len(rows[0])
    shape = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.margin_top = Emu(18288)
            cell.margin_bottom = Emu(18288)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.name = "Calibri"
                if r == 0 and header:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                else:
                    p.font.color.rgb = INK
            if r == 0 and header:
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACC
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = PALE
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
    return shape


def panel(s, x, y, w, h, title, color):
    """A titled quadrant panel; returns (inner_x, inner_y, inner_w)."""
    card = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = color
    card.line.width = Pt(1.25)
    bar = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.34))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    tx(s, x + 0.12, y + 0.03, w - 0.24, 0.3, title, size=13, bold=True, color=WHITE)
    return x + 0.15, y + 0.42, w - 0.3


# ================================================================== p.1
s = slide()
bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.18))
bg.fill.solid()
bg.fill.fore_color.rgb = INK
bg.line.fill.background()
tx(s, 0.55, 0.12, 12.2, 0.55, "VCM — Voice Controlled Smart Device", size=28, bold=True, color=WHITE)
tx(s, 0.55, 0.66, 12.2, 0.45,
   "Always-on keyword + intent recognition at the edge · no cloud round-trip",
   size=14, color=RGBColor(0x9C, 0xB3, 0xF0))
tx(s, 0.55, 1.24, 12.2, 0.35,
   "15 October 2026 · \u27e8fill: author / team\u27e9 · Model · Dataset · Training on MPS · Validation on RPi 4",
   size=12, color=GRAY)

# ---- quadrant geometry: 2x2 grid, each 6.0 x 2.42
QY1, QY2, QH = 1.72, 4.24, 2.42
QX1, QX2, QW = 0.55, 6.78, 6.0

# --- Model (top-left)
ix, iy, iw = panel(s, QX1, QY1, QW, QH, "Model", ACC)
table(s, ix, iy, iw, [
    ["Item", "Value"],
    ["Parameters / weights", "0.367 M · ~1.6 MB ONNX (CRNN)"],
    ["Topology", "depthwise-separable CNN trunk → BiGRU(128), 37 fine / 20 intent dual head"],
    ["Compute", "62.78 M FLOPs / inference"],
    ["+ wake gate", "0.249 M-param CNN, 1.0 MB (\u201cHello Jarvis\u201d)"],
    ["Runtime", "onnxruntime (CPUExecutionProvider) · 2 thr"],
], col_w=[2.1, 3.6], size=11, row_h=0.33)

# --- Dataset (top-right)
ix, iy, iw = panel(s, QX2, QY1, QW, QH, "Dataset", ACC2)
table(s, ix, iy, iw, [
    ["Item", "Value"],
    ["Source", "HF airimonda/ai231-me2-voice-commands (+4 corpora)"],
    ["Hours / utts", "17.3 h / 29,836 train · 6,305 test · 463 holdout"],
    ["Speakers", "664 (speaker-disjoint splits)"],
    ["Labels", "20 intents · 37 fine (slot) labels · OOS class"],
    ["Splits", "train / val / test disjoint · 6 sources merged"],
], col_w=[1.5, 4.2], size=11, row_h=0.33)

# --- Training on Apple MPS (local Mac) (bottom-left)
ix, iy, iw = panel(s, QX1, QY2, QW, QH, "Training on Apple MPS (local Mac)", GOOD)
table(s, ix, iy, iw, [
    ["Item", "Value"],
    ["Hardware", "Apple M-series · torch MPS backend · device=mps"],
    ["Objective", "CE(fine) + 0.5·CE(intent) · dual-head CRNN"],
    ["Optimiser", "AdamW wd 1e-4 · lr 1e-3 cosine · batch 128"],
    ["Steps / loss", "60 epochs · seed 42 · ~2.5 h wall-clock (MPS)"],
    ["Input", "16 kHz \u2192 log-mel 40\u00d796 · z-scored · energy-norm"],
], col_w=[1.7, 4.0], size=11, row_h=0.33)

# --- Validation on the Raspberry Pi 4 (bottom-right)
ix, iy, iw = panel(s, QX2, QY2, QW, QH, "Validation on the Raspberry Pi 4", WARN)
table(s, ix, iy, iw, [
    ["Item", "Value"],
    ["Keyword / intent acc", "100 % wake (5,000 clips) · 91.6 % intent (CRNN, test split)"],
    ["False-accept rate", "0.13 % wake FAR · 26.1 % FAR_oos @ 0.25"],
    ["Latency p95 / RTF", "9.36 ms classify · 4.85 ms wake (measured on-Pi) · RTF 0.006"],
    ["Live sessions", "Oct 15: 247 wakes · 94 % recognized · 63 % acted ok · Oct 16 retrained gate: 9/10 fired · 0 false fires"],
    ["Runtime", "onnxruntime · 2 threads (swept 1/2/4)"],
], col_w=[2.1, 3.6], size=11, row_h=0.33)

# --- architecture strip (bottom of p.1)
tx(s, 0.55, 6.78, 12.2, 0.3, "Architecture", size=12, bold=True, color=INK)
arch = s.shapes.add_shape(1, Inches(0.55), Inches(7.08), Inches(12.2), Inches(0.34))
arch.fill.solid()
arch.fill.fore_color.rgb = LIGHT
arch.line.color.rgb = ACC
arch.line.width = Pt(1)
tx(s, 0.7, 7.12, 11.9, 0.28,
   "Mic 16 kHz  \u2500\u2500\u25b6  wake CNN (0.249 M)  \u2500\u2500\u25b6  log-mel 40\u00d796  \u2500\u2500\u25b6  CRNN 0.367 M (depthwise-separable CNN trunk \u2192 BiGRU)  \u2500\u2500\u25b6  intent head + slot head  \u2500\u2500\u25b6  dispatcher  \u2500\u2500\u25b6  actuator (LEDs / music / timer / TTS)",
   size=11.5, color=INK, align=PP_ALIGN.CENTER)

# ================================================================== p.2
s = slide()
tx(s, 0.55, 0.35, 12.2, 0.6, "To Be Submitted", size=28, bold=True, color=INK)
tx(s, 0.55, 0.95, 12.2, 0.4, "What the reviewer will look for", size=14, color=GRAY)

table(s, 0.55, 1.45, 12.2, [
    ["#", "Item", "Status"],
    ["1", "Repo public, one-command reproduction", "\u2705 setup.sh + run.sh — verified on the Pi, 2026-10-03"],
    ["2", "Dataset licensed and citable (DOI)", "\u2705 HF dataset \u2192 citable DOI"],
    ["3", "Training logs + final checkpoint committed", "\u2705 train_crnn.jsonl · crnn.pt · onnx/"],
    ["4", "Pi 4 latency reproduced by the posted script", "\u2705 perf.py bench/wake · p95 9.36 ms VCM · 4.85 ms wake (100+100)"],
    ["5", "Held-out test set, unseen speakers", "\u2705 test 6,305 + holdout 463, speaker-disjoint"],
    ["6", "Single deployed model disclosed", "\u2705 CRNN 0.367 M VCM + jarvis wake CNN \u2014 no other model trained or presented"],
], col_w=[0.5, 4.6, 7.1], size=11, row_h=0.44)

tx(s, 0.55, 4.75, 12.2, 0.4, "Links & artefacts", size=16, bold=True, color=INK)
table(s, 0.55, 5.2, 12.2, [
    ["Artefact", "Where"],
    ["GitHub repository", "\u27e8fill: org/repo\u27e9 — public, MIT"],
    ["Dataset location", "huggingface.co/datasets/airimonda/ai231-me2-voice-commands"],
    ["Training hardware", "Apple M-series MPS (local Mac) · 60 epochs · seed 42"],
    ["Model weights", "crnn.pt + crnn.onnx (~1.6 MB) · jarvis_cnn.onnx (1.0 MB)"],
], col_w=[2.6, 9.6], size=11, row_h=0.42)

tx(s, 0.55, 7.02, 12.2, 0.4,
   "Demo: say \u201cHello Jarvis\u201d \u2192 command \u2192 action · spoken answers in the Jarvis voice (Piper en_GB-alan)",
   size=13, color=GRAY, align=PP_ALIGN.CENTER)

prs.save(OUT)
print("saved:", OUT)