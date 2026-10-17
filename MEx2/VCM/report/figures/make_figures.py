#!/usr/bin/env python3
"""Generate all figures for the VCM report (real data, no ASCII art).

CRNN primary model, HF dataset + supplementals, Mac MPS training
training, ReSpeaker-LED lights, TTS-simulated tasks, Open-Meteo weather,
per-slot timer chimes.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
MODEL = os.path.join(HERE, "..", "..", "model")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

INK = "#1a2332"; ACC = "#2563eb"; ACC2 = "#7c3aed"; GOOD = "#059669"
WARN = "#d97706"; BAD = "#dc2626"; GRAY = "#6b7280"; LIGHT = "#e5e7eb"

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)

# ---------------------------------------------------------------- 1. pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")

    stages = [
        ("MIC", "ReSpeaker 2-Mic HAT\n12 kHz stream", "#0ea5e9"),
        ("WAKE GATE", "RMS pre-gate +\n\"Hello Jarvis\" CNN\n(249 k params)", "#2563eb"),
        ("FEATURES", "log-mel 40\u00d796\nz-scored, numpy", "#7c3aed"),
        ("CRNN VCM", "0.367 M params\n37 fine + 20 intent\n62.8 M FLOPs", "#059669"),
        ("DISPATCHER", "plain Python\nmin-conf 0.25", "#d97706"),
        ("ACTIONS", "ReSpeaker LEDs \u00b7 mpv music\nOpen-Meteo \u00b7 time \u00b7 piper TTS\ntimer chimes", "#dc2626"),
    ]
    w, gap, x = 13.5, 2.9, 1.0
    for i, (title, sub, c) in enumerate(stages):
        y = 5.6 if i % 2 == 0 else 4.4
        box = FancyBboxPatch((x, y - 1.55), w, 3.1,
                             boxstyle="round,pad=0.25,rounding_size=0.45",
                             fc=c, ec="none", alpha=0.14)
        ax.add_patch(box)
        edge = FancyBboxPatch((x, y - 1.55), w, 3.1,
                              boxstyle="round,pad=0.25,rounding_size=0.45",
                              fc="none", ec=c, lw=1.8)
        ax.add_patch(edge)
        ax.text(x + w / 2, y + 0.85, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=c)
        ax.text(x + w / 2, y - 0.35, sub, ha="center", va="center", fontsize=8.6,
                color=INK, linespacing=1.45)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.4, y), xytext=(x + w + 0.4, y),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color=GRAY))
        x += w + gap

    ax.text(50, 1.15, "always-on  \u2192  event-driven  \u2192  ~10 ms classify  \u2192  real action",
            ha="center", fontsize=10.5, color=GRAY, style="italic")
    ax.text(50, 9.0, "VCM Voice Command Model \u2014 on-device pipeline (no ASR, no LLM, no cloud)",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)
    save(fig, "fig_pipeline.png")

# ------------------------------------------------------- 2. dataset composition
def fig_dataset():
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.3))

    # (a) sources in the merged TRAIN split (verified from the HF parquets)
    ax = axs[0]
    src = [("supplemental_fil", 14120, ACC2), ("group_synthetic", 12288, ACC),
           ("SLURP", 1010, "#0ea5e9"), ("synthetic_negatives", 1000, WARN),
           ("real_voice", 620, GOOD), ("SNIPS", 331, BAD),
           ("FluentSpeechCommands", 256, "#a855f7"), ("other (6)", 211, GRAY)]
    names = [s[0] for s in src][::-1]
    vals = [s[1] for s in src][::-1]
    cols = [s[2] for s in src][::-1]
    ax.barh(names, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 300, i, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_title("Merged train split by source\n(29,836 clips \u00b7 17.3 h)")
    ax.set_xlabel("clips")
    ax.set_xlim(0, 16500)

    # (b) split
    ax = axs[1]
    splits = [("Train", 29836, ACC), ("Test", 6305, GOOD), ("Holdout", 463, WARN)]
    wedges, _, autot = ax.pie([s[1] for s in splits], labels=None, colors=[s[2] for s in splits],
           autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", lw=2),
           textprops=dict(fontsize=10, color="white", fontweight="bold"))
    ax.legend([f"{n}  {v:,}" for n, v, _ in splits], loc="center left",
              bbox_to_anchor=(0.86, 0.5), fontsize=9.5, frameon=False)
    ax.text(0, 0, "36,604", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.set_title("Train / test / holdout\n(speaker-disjoint)")

    # (c) class balance (merged train, per intent)
    ax = axs[2]
    labels = ["COLOR\n3,014", "ALARM\n2,717", "TEMPERATURE\n2,835", "TIMER\n2,609",
              "median\n~1,050", "PAUSE\n743"]
    vals = [3014, 2717, 2835, 2609, 1050, 743]
    cols = [ACC, ACC, ACC2, ACC2, GRAY, BAD]
    bars = ax.bar(range(6), vals, color=cols, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("train clips (log)")
    ax.set_title("Per-intent balance (merged train)\n4.1\u00d7 skew \u2014 supplementals level it")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}",
                ha="center", fontsize=8, color=INK)
    fig.suptitle("VCM dataset \u2014 HF ai231-me2-voice-commands + supplementals (numerals excluded)",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "fig_dataset.png")

# ------------------------------------------------------------- 3. CRNN arch
def fig_crnn():
    fig, ax = plt.subplots(figsize=(12.8, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")

    blocks = [
        ("INPUT", "log-mel\n1\u00d740\u00d796", "#0ea5e9"),
        ("Conv 3\u00d73\n\u00d7 48", "BN + ReLU\n40\u00d796", "#2563eb"),
        ("MaxPool\n2\u00d72", "", "#2563eb"),
        ("DSConv\n\u00d7 2", "depthwise+pointwise\n20\u00d748", "#7c3aed"),
        ("MaxPool\n2\u00d72", "", "#7c3aed"),
        ("DSConv\n\u00d7 1", "10\u00d724", "#a855f7"),
        ("MaxPool\n2\u00d72", "5\u00d712", "#a855f7"),
        ("squeeze\nfreq", "mean over mel\n\u2192 [B, 12, 192]", "#0ea5e9"),
        ("BiGRU\n128", "2 directions\n\u00d7 12 frames", "#059669"),
        ("mean pool\n+ FC 256", "+ dropout 0.35", "#059669"),
        ("2 HEADS", "fine 37\nintent 20", "#d97706"),
    ]
    w, gap, x = 7.6, 1.25, 1.0
    for i, (title, sub, c) in enumerate(blocks):
        box = FancyBboxPatch((x, 3.0), w, 4.0,
                             boxstyle="round,pad=0.22,rounding_size=0.4",
                             fc=c, ec="none", alpha=0.13)
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((x, 3.0), w, 4.0,
                     boxstyle="round,pad=0.22,rounding_size=0.4",
                     fc="none", ec=c, lw=1.7))
        ax.text(x + w / 2, 5.75, title, ha="center", va="center", fontsize=8.8,
                fontweight="bold", color=c)
        if sub:
            ax.text(x + w / 2, 4.15, sub, ha="center", va="center", fontsize=7.2,
                    color=INK, linespacing=1.4)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, 5.0), xytext=(x + w + 0.3, 5.0),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GRAY))
        x += w + gap
    ax.text(50, 1.3, "0.367 M parameters \u00b7 62.8 M FLOPs \u00b7 1.7 ms Mac CPU \u00b7 "
                     "dual-head loss = CE(fine) + 0.5\u00b7CE(intent)",
            ha="center", fontsize=10.5, color=GRAY, style="italic")
    ax.text(50, 8.6, "PiCommandCRNN \u2014 conv front-end + bidirectional GRU (the deployed VCM)",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)
    save(fig, "fig_crnn.png")

# --------------------------------------------- 4. results (test + holdout bars)
def _load_eval(split, tag):
    p = os.path.join(MODEL, "run_outputs", "eval", f"eval_{split}_{tag}.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None

def fig_results():
    crnn = _load_eval("test", "crnn")
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.2))
    metrics = ["Intent\naccuracy", "Intent\nmacro-F1", "Fine\naccuracy", "Fine\nmacro-F1"]

    ax = axs[0]
    if crnn is not None:
        cv = [crnn["intent_accuracy"] * 100, crnn["intent_macro_f1"] * 100,
              crnn["fine_accuracy"] * 100, crnn["fine_macro_f1"] * 100]
    else:
        cv = [0] * 4
    x = np.arange(4); w = 0.5
    b1 = ax.bar(x, cv, w, label="CRNN 0.367 M (deployed)", color=GOOD)
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{b.get_height():.1f}", ha="center", fontsize=8.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9.5)
    ax.set_ylim(0, 104); ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Test split (n = 6,305, speaker-disjoint)")
    ax.set_ylabel("%")

    # per-intent recall (weakest 5 + strongest 3)
    ax = axs[1]
    if crnn:
        pir = crnn["per_intent_recall"]
        intents = sorted(pir, key=lambda k: pir[k])
        show = intents[:5] + intents[-3:]
        crv = [pir[k] for k in show]
    else:
        show = ["OUT_OF_SCOPE", "VOLUME_UP", "LIGHT_OFF", "PLAY_MUSIC",
                "NEXT", "TEMPERATURE", "TIMER", "CREATE_REMINDER"]
        crv = [0] * 8
    y = np.arange(len(show))
    ax.barh(y, crv, 0.5, label="CRNN", color=GOOD)
    ax.set_yticks(y); ax.set_yticklabels(show, fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("test recall")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Per-intent recall \u2014 weakest 5 and strongest 3")
    fig.suptitle("Deployed CRNN on the speaker-disjoint test split",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "fig_results.png")

# ----------------------------------------------------------- 5. efficiency
def fig_efficiency():
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.0))

    ax = axs[0]
    models = ["CRNN\n(command)", "CNN\n(wake)"]
    params = [0.367, 0.249]
    bars = ax.bar(models, params, color=[GOOD, ACC], width=0.5)
    for b, v in zip(bars, params):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f} M",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("parameters (M)"); ax.set_ylim(0, 0.5)
    ax.set_title("Parameters")

    ax = axs[1]
    flops = [62.8, 18.0]
    bars = ax.bar(models, flops, color=[GOOD, ACC], width=0.5)
    for b, v in zip(bars, flops):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f} M",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("FLOPs / forward"); ax.set_ylim(0, 80)
    ax.set_title("Compute\nboth far under the 1 s frame budget")

    ax = axs[2]
    lat_mac = [0.57, 0.57]
    lat_pi = [9.36, 4.85]
    x = np.arange(2); w = 0.38
    b1 = ax.bar(x - w / 2, lat_mac, w, label="Mac CPU (measured)", color=ACC)
    b2 = ax.bar(x + w / 2, lat_pi, w, label="Raspberry Pi 4 (measured)", color=WARN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{b.get_height():.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("inference latency p95 (ms)"); ax.set_ylim(0, 11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("Latency (p95, prof's 100+100 protocol)\nreal-time with headroom")
    fig.suptitle("Model efficiency \u2014 the deployed pair (command CRNN + wake CNN)",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "fig_efficiency.png")

# --------------------------------------------------------- 6. wake word
def fig_wakeword():
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.2))

    # (a) dataset composition
    ax = axs[0]
    sets = ["positive\n(synthetic, 1200)", "positive_real_1\n(first takes, 62)",
            "positive_real_2\n(re-takes, 60)", "negative\n(3000)", "noise\n(800)",
            "negative_real\n(room, 694)"]
    vals = [1200, 62, 60, 3000, 800, 694]
    bars = ax.bar(sets, vals, color=[ACC, GOOD, GOOD, WARN, GRAY, WARN], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}",
                ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_ylabel("clips"); ax.set_yscale("log"); ax.set_ylim(10, 6000)
    ax.set_title("Kaggle Jarvis corpus\n+ 122 real + 694 room negatives)")

    # (b) score distributions of the deployed CNN (held-out test slice)
    ax = axs[1]
    ax.barh(["room neg (86)", "noise (80)", "negative (300)", "real pos (36)", "syn pos (120)"],
            [0.003, 0.006, 0.002, 0.901, 1.000], color=[WARN, GRAY, WARN, GOOD, ACC], height=0.55)
    ax.set_xlim(0, 1.15); ax.set_xlabel("mean model score")
    for i, v in enumerate([0.003, 0.006, 0.002, 0.901, 1.000]):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
    ax.axvline(0.75, color=BAD, lw=1.6, ls="--")
    ax.text(0.76, 4.05, "deployed threshold 0.75", fontsize=8.6, color=BAD)
    ax.set_title("WakeCNN score separation\n(AUC 0.9975 on held-out)")

    # (c) live gate test on the Pi
    ax = axs[2]
    sets = ["positive\n(10 real)", "noise frames\n(200)"]
    fired = [10, 0]
    bars = ax.bar(sets, fired, color=[GOOD, GRAY], width=0.5)
    for b, v, t in zip(bars, fired, ["10/10", "0/200"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, t, ha="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 12); ax.set_ylabel("clips that woke the gate")
    ax.set_title("Live streamed test\nthrough the real Pi gate loop")
    fig.suptitle("Wake word \u2014 \"Hello Jarvis\" CNN, deployed (249 k params)",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "fig_wakeword.png")

# --------------------------------------------------------- 7. LED state machine
def fig_leds():
    fig, ax = plt.subplots(figsize=(11.8, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")

    states = [
        ("IDLE", "all LEDs off\nlistening for\n\"Hello Jarvis\"", "#9ca3af"),
        ("LISTENING", "3\u00d7 cyan LEDs\nBLINK @ 2.5 Hz\nwake word heard", "#0ea5e9"),
        ("EXECUTING", "3\u00d7 green LEDs\nSTEADY ON\ncommand recognized", GOOD),
        ("DONE / REJECT", "LEDs OFF\ndone \u2014 or command\nnot recognized", "#d97706"),
    ]
    w, gap, x = 20.0, 6.0, 2.0
    for i, (name, sub, c) in enumerate(states):
        box = FancyBboxPatch((x, 3.2), w, 4.2,
                             boxstyle="round,pad=0.3,rounding_size=0.5",
                             fc=c, ec="none", alpha=0.13)
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((x, 3.2), w, 4.2,
                     boxstyle="round,pad=0.3,rounding_size=0.5",
                     fc="none", ec=c, lw=2.0))
        # draw the 3 LEDs
        for k in range(3):
            cx = x + w / 2 + (k - 1) * 2.6
            if name == "IDLE":
                fc = "#e5e7eb"; ec = "#9ca3af"
            elif name == "LISTENING":
                fc = c if k % 2 == 0 else "#e5e7eb"; ec = c
            elif name == "EXECUTING":
                fc = c; ec = c
            else:
                fc = "#e5e7eb"; ec = "#d97706"
            ax.add_patch(Circle((cx, 6.2), 0.85, fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, 4.6, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color=c)
        ax.text(x + w / 2, 3.85, sub, ha="center", va="center", fontsize=8.2,
                color=INK, linespacing=1.4)
        if i < len(states) - 1:
            ax.annotate("", xy=(x + w + gap - 0.5, 5.3), xytext=(x + w + 0.5, 5.3),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color=GRAY))
        x += w + gap
    ax.text(50, 1.2, "the 3 APA102 LEDs are BOTH the status display AND the \"lights\" "
                     "(LIGHT_ON/OFF, COLOR_*, BRIGHTNESS_* drive them directly)",
            ha="center", fontsize=10, color=GRAY, style="italic")
    ax.text(50, 9.0, "ReSpeaker LED state machine (runtime/leds.py)",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)
    save(fig, "fig_leds.png")

# --------------------------------------------------------- 8. setup flow
def fig_setup():
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    steps = [
        ("1", "Flash SD card", "RPi Imager \u00b7 Raspberry Pi OS 64-bit \u00b7 SSH on \u00b7 Wi-Fi baked in", "#0ea5e9"),
        ("2", "First boot + SSH", "ssh <user>@<ip> \u00b7 find IP: arp -a or the router", "#0ea5e9"),
        ("3", "Copy the rpi/ folder", "scp -r rpi/ \u2192 ~/vcm on the Pi (or git clone)", "#0ea5e9"),
        ("4", "bash setup.sh", "apt packages \u00b7 .venv \u00b7 requirements \u00b7 piper voice \u00b7 asks Wi-Fi once", "#2563eb"),
        ("5", "bash run.sh", "live loop: wake \u2192 LEDs \u2192 CRNN \u2192 action \u2192 spoken answer", "#059669"),
        ("6", "Optional benchmarks", "python3 perf.py bench / wake \u2014 deck numbers", "#d97706"),
    ]
    col_x = [4, 54]
    cw = 42
    for i, (n, title, sub, c) in enumerate(steps):
        col = i // 3
        row = i % 3
        x = col_x; y = 88 - row * 26
        ax.add_patch(FancyBboxPatch((x[col], y - 20), cw, 22,
                     boxstyle="round,pad=0.4,rounding_size=1.2",
                     fc=c, ec="none", alpha=0.12))
        ax.add_patch(FancyBboxPatch((x[col], y - 20), cw, 22,
                     boxstyle="round,pad=0.4,rounding_size=1.2",
                     fc="none", ec=c, lw=1.8))
        ax.add_patch(Circle((x[col] + 4.5, y - 4.5), 2.6, fc=c, ec="none"))
        ax.text(x[col] + 4.5, y - 4.5, n, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")
        ax.text(x[col] + 9, y - 3.4, title, ha="left", va="center", fontsize=11.5,
                fontweight="bold", color=INK)
        ax.text(x[col] + 2.5, y - 12.5, sub, ha="left", va="center", fontsize=8.8,
                color=GRAY, wrap=True)
    ax.text(50, 95, "VCM \u2014 Raspberry Pi setup: 6 steps, 2 commands",
            ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(50, 4, "No API keys \u00b7 no accounts \u00b7 no smart-home hardware \u2014 Wi-Fi is the only configuration",
            ha="center", fontsize=10.5, color=GRAY, style="italic")
    save(fig, "fig_setup.png")

# --------------------------------------------------------- 9. action coverage
def fig_coverage():
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    groups = [
        ("LIGHT_ON / LIGHT_OFF", "ReSpeaker APA102 LEDs on/off", WARN),
        ("BRIGHTNESS 20/60/100", "ReSpeaker LED brightness", WARN),
        ("COLOR red/blue/green", "ReSpeaker LED colour", WARN),
        ("PLAY_MUSIC / PAUSE / STOP / NEXT", "mpv JSON-IPC \u00b7 5 mp3 playlist \u00b7 vol \u00b110%", GOOD),
        ("VOLUME_UP / VOLUME_DOWN", "mpv volume \u00b110% per command", GOOD),
        ("TIMER 10s/30s/1m", "per-slot chime MP3 (audio/timer/)", GOOD),
        ("WEATHER", "Open-Meteo (keyless) \u00b7 spoken", ACC),
        ("TIME", "RPi system clock \u00b7 spoken", ACC),
        ("ALARM 6AM/8AM/9PM", "piper TTS confirmation (simulated)", ACC2),
        ("TEMPERATURE 18/22/26", "piper TTS confirmation (simulated)", ACC2),
        ("CREATE_REMINDER (3) / LIST_REMINDERS", "piper TTS + in-memory store", ACC2),
        ("CALL / MESSAGE", "piper TTS confirmation (simulated)", ACC2),
    ]
    for i, (label, backend, c) in enumerate(groups):
        y = len(groups) - i
        ax.add_patch(FancyBboxPatch((0.0, y - 0.32), 0.42, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=c, ec="none", alpha=0.15))
        ax.add_patch(FancyBboxPatch((0.0, y - 0.32), 0.42, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="none", ec=c, lw=1.4))
        ax.text(0.21, y, label, ha="center", va="center", fontsize=9.0,
                fontweight="bold", color=INK)
        ax.add_patch(FancyBboxPatch((0.46, y - 0.32), 0.54, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="#f3f4f6", ec="#d1d5db", lw=1.0))
        ax.text(0.73, y, backend, ha="center", va="center", fontsize=8.8, color=GRAY)
    ax.set_xlim(0, 1); ax.set_ylim(0.3, len(groups) + 0.9); ax.axis("off")
    ax.text(0.5, len(groups) + 0.62, "All 37 fine labels \u2192 real actions (two devices only)",
            ha="center", fontsize=13.5, fontweight="bold", color=INK, transform=ax.transData)
    save(fig, "fig_coverage.png")

# ------------------------------------------------- 10. training curve (real log)
def fig_training_curve():
    log = os.path.join(MODEL, "run_outputs", "logs", "train_crnn.jsonl")
    recs = [json.loads(l) for l in open(log) if l.strip()]
    ep = [r["epoch"] for r in recs]
    vi = [r["val_intent_acc"] * 100 for r in recs]
    vf = [r["val_fine_acc"] * 100 for r in recs]
    tl = [r["train_loss"] for r in recs]
    best_i = int(np.argmax(vi))

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(ep, vi, color=ACC, lw=2.2, marker="o", ms=2.5, label="val intent acc")
    ax.plot(ep, vf, color=ACC2, lw=2.2, marker="s", ms=2.5, label="val fine acc")
    ax.axvline(ep[best_i], color=GOOD, lw=1.4, ls="--")
    ax.text(ep[best_i] + 0.6, 12, f"best: {vi[best_i]:.1f} % @ epoch {ep[best_i]}",
            fontsize=9.5, color=GOOD)
    ax2 = ax.twinx()
    ax2.plot(ep, tl, color=WARN, lw=1.6, ls=":", label="train loss (right)")
    ax2.set_ylabel("train loss", color=WARN)
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=WARN)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, max(max(vi), max(vf)) * 1.2); ax.set_xlim(1, len(ep))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, frameon=False, loc="center right")
    ax.set_title("Training dynamics \u2014 CRNN, Mac MPS, 60 epochs, batch 128, lr 1e-3, seed 42\n"
                 "(real per-epoch log: run_outputs/logs/train_crnn.jsonl)")
    save(fig, "fig_training_curve.png")

if __name__ == "__main__":
    fig_pipeline(); fig_dataset(); fig_crnn(); fig_results()
    fig_efficiency(); fig_wakeword(); fig_leds(); fig_setup(); fig_coverage()
    fig_training_curve()
    print("ALL FIGURES DONE")