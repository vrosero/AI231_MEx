#!/usr/bin/env python3
"""Build VCM_Slides.pptx — 16:9 deck mirroring the report (python-pptx)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
OUT = os.path.join(BASE, "VCM_Slides.pptx")

INK = RGBColor(0x1A, 0x23, 0x32)
ACC = RGBColor(0x25, 0x63, 0xEB)
ACC2 = RGBColor(0x7C, 0x3A, 0xED)
GOOD = RGBColor(0x05, 0x96, 0x69)
WARN = RGBColor(0xD9, 0x77, 0x06)
BAD = RGBColor(0xDC, 0x26, 0x26)
GRAY = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xEE, 0xF2, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide():
    return prs.slides.add_slide(BLANK)

def tx(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
       font="Calibri", anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
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

def bullets(slide, x, y, w, h, items, size=15, color=INK, gap=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        lvl = 0
        if isinstance(it, tuple):
            it, lvl = it
        p.text = ("• " if lvl == 0 else "– ") + it
        p.font.size = Pt(size - lvl * 2)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(gap)
        p.level = lvl
    return box

def header(slide, title, sub=None, color=ACC):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.16))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    tx(slide, 0.55, 0.32, 12.2, 0.8, title, size=28, bold=True, color=INK)
    if sub:
        tx(slide, 0.55, 1.02, 12.2, 0.5, sub, size=14, color=GRAY)

def pic(slide, name, x, y, w=None, h=None):
    from PIL import Image
    p = os.path.join(FIG, name)
    iw, ih = Image.open(p).size
    if w and not h:
        h = w * ih / iw
    if h and not w:
        w = h * iw / ih
    slide.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h

def stat_cards(slide, cards, y=1.7, h=1.5, x0=0.55, w=3.0, gap=0.18):
    x = x0
    for label, value, color in cards:
        card = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
        card.fill.solid(); card.fill.fore_color.rgb = LIGHT
        card.line.color.rgb = color; card.line.width = Pt(1.5)
        card.shadow.inherit = False
        tf = card.text_frame; tf.word_wrap = True
        tf.margin_top = Emu(45720); tf.margin_left = Emu(68580)
        p = tf.paragraphs[0]; p.text = value
        p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = color
        p2 = tf.add_paragraph(); p2.text = label
        p2.font.size = Pt(11.5); p2.font.color.rgb = INK
        x += w + gap

def table(slide, x, y, w, rows, col_w=None, size=12, header=True, row_h=0.34):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.margin_top = Emu(27432); cell.margin_bottom = Emu(27432)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.name = "Calibri"
                if r == 0 and header:
                    p.font.bold = True; p.font.color.rgb = WHITE
                else:
                    p.font.color.rgb = INK
            if r == 0 and header:
                cell.fill.solid(); cell.fill.fore_color.rgb = ACC
            elif r % 2 == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
    return shape

# ---------------------------------------------------------------- 1 title
s = slide()
bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg.fill.solid(); bg.fill.fore_color.rgb = INK; bg.line.fill.background()
tx(s, 0.9, 1.7, 11.5, 1.4, "VCM — Tiny Voice Command Model", size=44, bold=True, color=WHITE)
tx(s, 0.9, 3.0, 11.5, 0.8,
   "\"Hello Jarvis\" wake gate  ·  CRNN command classifier  ·  real actions on a Raspberry Pi 4",
   size=19, color=RGBColor(0x9C, 0xB3, 0xF0))
tx(s, 0.9, 4.1, 11.5, 1.6,
   "AI 231 · MEx 2 — real-world VCM demo\nRaspberry Pi 4 (4 GB) · ReSpeaker 2-Mics HAT · USB speaker (only two devices)\n"
   "Standalone: no cloud models · no LLM · no ASR — pure VCM, commands 1–10",
   size=15, color=RGBColor(0xCB, 0xD5, 0xE1), line_spacing=1.35)

# ------------------------------------------------------------- 2 what it is
s = slide()
header(s, "What this project is", "A tiny, standalone voice assistant — everything on-device")
pic(s, "fig_pipeline.png", 0.55, 1.55, w=12.2)
bullets(s, 0.55, 5.6, 12.2, 1.7, [
    "Listens continuously for \"Hello Jarvis\" (249 k-param CNN gate) — LEDs blink cyan on wake",
    "Classifies the command with a 0.367 M-param CRNN → 37 fine labels / 20 intents — no transcription",
    "Executes real actions: ReSpeaker LEDs (the lights), mp3 music, per-slot timer chimes, weather, time — plus spoken Piper-TTS answers (the \"Jarvis\" voice)",
    "LEDs: blink = listening · steady green = executing · off = done / not recognized",
], size=14.5)

# ------------------------------------------------------------- 3 constraints
s = slide()
header(s, "Meeting the professor's constraints", "tiny · real-time on RPi · standalone · no LLM · no ASR")
stat_cards(s, [
    ("parameters (command model)", "0.367 M", GOOD),
    ("FLOPs / inference", "62.78 M", ACC),
    ("ONNX size", "~1.6 MB", ACC2),
    ("Pi 4 latency (measured)", "9.36 ms", GOOD),
], y=1.6, w=3.0, h=1.6)
table(s, 0.55, 3.6, 12.2, [
    ["Constraint", "How it is met"],
    ["Standalone / no cloud models", "Wake CNN + CRNN run locally via onnxruntime; network only for one action (keyless Open-Meteo weather) — never a model call"],
    ["No LLM", "Fixed 37-label classifier; spoken answers are template sentences, not generated text"],
    ["No full ASR", "Audio → log-mel → intent/fine label directly; no audio→text stage"],
    ["Real-time on RPi 4/5", "62.78 M FLOPs → 9.36 ms p95 measured on a Pi 4 CPU — under 1% of the 1 s frame budget"],
    ["Genuinely tiny", "0.367 M params (~1.6 MB ONNX) + 0.249 M-param wake gate; <120 MB RSS idle"],
], col_w=[3.6, 8.6], size=12.5, row_h=0.52)

# ------------------------------------------------------------- 4 dataset
s = slide()
header(s, "Dataset — 36,604 clips · 37 fine / 20 intents", "HF ai231-me2-voice-commands + supplementals · speaker-disjoint")
pic(s, "fig_dataset.png", 0.55, 1.55, w=12.2)
tx(s, 0.55, 6.6, 12.2, 0.8,
   "Train 29,836 / Test 6,305 / Holdout 463 (21.0 h) — speaker-disjoint. Supplemental merge (synth by voice_split + fil50 persona bank + "
   "1,250 synthetic negatives) levels the skew: COLOR 3,014 → PAUSE 743 (4.1×). Numerals excluded.",
   size=13, color=GRAY)

# ------------------------------------------------------------- 5 model
s = slide()
header(s, "Model — PiCommandCRNN (deployed)", "depthwise-separable CNN trunk → BiGRU, dual head: 37 fine + 20 intent")
pic(s, "fig_crnn.png", 0.55, 1.5, w=12.2)
bullets(s, 0.55, 5.9, 12.2, 1.4, [
    "Loss = CE(fine) + 0.5·CE(intent) — fine head carries the slot, intent head the action group",
    "Recipe: AdamW wd 1e-4 · lr 1e-3 cosine · batch 128 · label smoothing 0.05 · dropout 0.35 · seed 42 · 60 epochs on Apple MPS",
    "The trunk pools 40×96 → 5×12 before the BiGRU: temporal modelling costs 12 steps × 192 dims — 62.78 M FLOPs",
], size=14)

# ------------------------------------------------------------- 8 domain adaptation
s = slide()
header(s, "Results \u2014 deployed CRNN (test n = 6,305)", "one model, speaker-disjoint evaluation", color=GOOD)
pic(s, "fig_results.png", 0.55, 1.5, w=8.6)
table(s, 9.4, 1.7, 3.4, [
    ["Metric", "Value"],
    ["Intent acc", "91.6 %"],
    ["Intent mF1", "90.3 %"],
    ["Fine acc", "87.4 %"],
    ["Holdout intent", "88.6 %"],
], col_w=[2.2, 1.2], size=12.5, row_h=0.42)
bullets(s, 9.4, 4.4, 3.4, 2.6, [
    "speaker-disjoint test + holdout \u2014 no clip seen twice",
    "OOS recall 0.739 \u00b7 FAR_oos 0.261 @ min_conf 0.25",
    "real-voice OOS remains the honest weak point",
], size=12.5)

# ------------------------------------------------------------- 9 efficiency
s = slide()
header(s, "Efficiency — why it is tiny", "62.78 M FLOPs · real-time headroom")
pic(s, "fig_efficiency.png", 0.55, 1.5, w=12.2)
bullets(s, 0.55, 5.9, 12.2, 1.5, [
    "Total resident model footprint (command + wake): ~2.6 MB weights; whole assistant < 120 MB RSS idle",
    "Always-on cost ≈ 0: RMS pre-gate discards silence; wake CNN scores only while the voiced latch is open",
    "ONNX Runtime CPU is the only ML dependency on the Pi — no PyTorch, no librosa",
], size=14)

# ------------------------------------------------------------- 10 wakeword
s = slide()
header(s, "Wake word — \"Hello Jarvis\" from scratch", "Kaggle jarvis-wake-word-dataset + 122 real recordings + 694 real-room negatives · 5,816 clips", color=ACC2)
pic(s, "fig_wakeword.png", 0.55, 1.5, w=12.2)
tx(s, 0.55, 6.5, 12.2, 0.9,
   "Deployed wake model (1200 synthetic + 122 real positives, 694 real-room negatives, source-disjoint split): AUC 0.9975 · "
   "real-room negative max 0.065 vs threshold 0.99 · live on the Pi: retrained weights 9/10 spoken attempts fired · 0 false fires (Oct 16, evidence/logs_live_20261016.log); "
   "original weights 10/10 real clips fired (scores 0.994–0.999) · 0 false fires on 200 noise frames (Oct 15).",
   size=13, color=GRAY)

# ------------------------------------------------------------- 11 runtime demo
s = slide()
header(s, "Runtime & demo — the 8 build items, verified on hardware", "LED language + all 37 labels dispatched")
pic(s, "fig_leds.png", 0.55, 1.45, w=12.2)
pic(s, "fig_coverage.png", 2.6, 4.42, h=2.85)

# ------------------------------------------------------------- 12 setup
s = slide()
header(s, "How to set up on the RPi", "one-time install (left) → per-demo-day routine (right)", color=GOOD)
pic(s, "fig_setup.png", 0.55, 1.45, w=12.2)
tx(s, 0.55, 6.85, 12.2, 0.6,
   "Full command-by-command guide: report section 08_rpi_setup.md — Wi-Fi switching, SPI/LEDs, config, troubleshooting.",
   size=12.5, color=GRAY)

# ------------------------------------------------------------- 13 verification
s = slide()
header(s, "Verification on hardware — and honest limits", "Raspberry Pi 4 · ReSpeaker 2-Mics HAT · USB speaker", color=WARN)
table(s, 0.55, 1.6, 12.2, [
    ["Component", "Test", "Result"],
    ["Mic", "ReSpeaker capture (hw rejects 12 kHz → 16 k + decimate)", "✅ verified"],
    ["Wake gate", "10 real clips streamed through the gate", "✅ 9/10 fire · 0 false fires on 200 noise frames"],
    ["LEDs", "blink → steady → off cycle", "✅ APA102 GPIO bitbang"],
    ["Lights", "ReSpeaker LEDs on/off/brightness/colour (no bulb)", "✅ dispatcher wired"],
    ["Music", "play/stop/next (ffplay fallback)", "✅ audible"],
    ["TTS answers", "Piper en_GB-alan: TIME/WEATHER real; ALARM/TEMP/REMINDERS simulated", "✅ spoken end-to-end"],
    ["Full loop", "mic → wake → classify → dispatch → LEDs → spoken answer", "✅ end-to-end"],
], col_w=[2.0, 6.6, 3.6], size=12.5, row_h=0.42)
bullets(s, 0.55, 5.6, 12.2, 1.6, [
    "Honest limit: real-voice OOS clips still mis-accepted (synthetic negatives taught kinds, not the domain); min_conf 0.25 fails safe",
    "Wake model trained on TTS data → live-mic threshold tuning recommended before the demo",
], size=13.5)

# ------------------------------------------------------------- 14 closing
s = slide()
bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg.fill.solid(); bg.fill.fore_color.rgb = INK; bg.line.fill.background()
tx(s, 0.9, 1.0, 11.5, 1.0, "Deliverable checklist", size=32, bold=True, color=WHITE)
table(s, 0.9, 2.1, 11.5, [
    ["Requirement", "Status"],
    ["Build a dataset to train VCMs (collective)", "✅ HF dataset + supplementals · 36,604 clips / 21.0 h"],
    ["Build and train a VCM (individual)", "✅ CRNN 0.367 M (the deployed model)"],
    ["Design a benchmark for validating VCMs", "✅ speaker-disjoint splits · intent + fine + macro-F1 + per-intent recall"],
    ["Validate the performance of your VCM", "✅ CRNN test 91.6 % intent · holdout 88.6 % · OOS/FAR analysis"],
    ["Build a real-world demo on RPi4/5", "✅ full loop verified on hardware (two devices only)"],
    ["VCM must be tiny, real-time on RPi4/5", "✅ 0.367 M params · 62.78 M FLOPs · 9.36 ms p95 measured on Pi 4"],
    ["Standalone — no cloud models", "✅ all inference on-device"],
    ["No LLM, no ASR — pure VCM, commands 1–10", "✅ 37-label classifier + dispatcher + TTS answers"],
], col_w=[6.4, 5.0], size=13, row_h=0.52)
tx(s, 0.9, 6.9, 11.5, 0.5, "Run it:  bash run.sh   ·   spoken answers in the Jarvis voice (Piper en_GB-alan)",
   size=15, color=RGBColor(0x9C, 0xB3, 0xF0))

prs.save(OUT)
print("wrote", OUT, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))