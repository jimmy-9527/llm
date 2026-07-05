#!/usr/bin/env python3
"""Plot eval/accuracy vs. training step for the SFT dataset-size sweep."""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

RUN_DIR = "runs/sft_experiment"
OUT = "runs/sft_experiment/accuracy_vs_step.png"

# Ordered dataset sizes -> sequential blue ramp (light = less data, dark = more).
# dy = vertical label nudge (points) to declutter endpoints that coincide at step 2000.
SERIES = [
    ("128", "128", "#86b6ef", -1),
    ("256", "256", "#5598e7", 0),
    ("512", "512", "#2a78d6", 1),
    ("1024", "1024", "#1c5cab", 9),
    ("full", "full", "#104281", 0),
]

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e2"


def load(size):
    xs, ys = [], []
    path = os.path.join(RUN_DIR, f"samples{size}_all", "log.jsonl")
    for line in open(path):
        j = json.loads(line)
        if j.get("type") == "eval_metrics":
            xs.append(j["step"])
            ys.append(j["metrics"]["eval/accuracy"])
    return xs, ys


fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for key, label, color, dy in SERIES:
    xs, ys = load(key)
    ax.plot(xs, ys, "-", color=color, lw=2, marker="o", ms=5,
            mec="#fcfcfb", mew=1.0, zorder=3)
    # direct end-label, nudged vertically to avoid endpoint collisions
    ax.annotate(label, (xs[-1], ys[-1]), color=color, fontsize=9,
                xytext=(6, dy), textcoords="offset points",
                va="center", ha="left", fontweight="bold")

# Zero-shot baseline reference.
ax.axhline(0.03, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.annotate("zero-shot baseline (0.03)", (200, 0.03), color=MUTED, fontsize=8,
            va="bottom", ha="left")

ax.set_title("SFT dataset-size sweep — eval accuracy vs. step",
             color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
ax.set_xlabel("Training step", color=MUTED, fontsize=10)
ax.set_ylabel("eval/accuracy (MATH, n=500)", color=MUTED, fontsize=10)

ax.set_ylim(0.0, 0.54)
ax.set_xlim(0, 4300)
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
ax.tick_params(colors=MUTED, labelsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(GRID)

# legend naming the encoding
ax.plot([], [], " ", label="# unique training examples")
ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=MUTED,
          handlelength=0, handletextpad=0)

fig.tight_layout()
fig.savefig(OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"wrote {OUT}")
