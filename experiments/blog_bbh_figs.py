"""Generate BBH figures for the jjd.io blog post — solarized palette.

Per-task data: swollm/results/v13/bbh_full.json (baseline 30.22%, swollm 87.84%).
Field/placement data: Epoch AI BBH leaderboard (standardized 3-shot CoT), pasted 2026-06-27.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- solarized ---------------------------------------------------------------
S = dict(base03="#002b36", base02="#073642", base01="#586e75", base00="#657b83",
         base0="#839496", base1="#93a1a1", base2="#eee8d5", base3="#fdf6e3",
         yellow="#b58900", orange="#cb4b16", red="#dc322f", magenta="#d33682",
         violet="#6c71c4", blue="#268bd2", cyan="#2aa198", green="#859900")
FACE = S["base3"]
INK, LAB, TICK = S["base02"], S["base01"], S["base00"]
C_PROVED, C_RECOG, C_WALL = S["cyan"], S["violet"], S["base1"]
C_BASE, C_LINE, C_PARTIAL = S["base02"], S["orange"], S["yellow"]
GA = 0.62  # gained-segment alpha

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": FACE, "figure.facecolor": FACE, "savefig.facecolor": FACE,
    "axes.edgecolor": S["base1"], "axes.labelcolor": LAB,
    "text.color": INK, "xtick.color": TICK, "ytick.color": INK,
    "figure.dpi": 150,
})

# (task, baseline%, swollm%, mechanism)
DATA = [
    ("boolean_expressions", 75.7, 100.0, "proved"), ("disambiguation_qa", 31.2, 100.0, "recognized"),
    ("dyck_languages", 10.1, 100.0, "proved"), ("geometric_shapes", 8.9, 100.0, "proved"),
    ("logical_deduction_five", 19.0, 100.0, "proved"), ("logical_deduction_seven", 14.6, 100.0, "proved"),
    ("logical_deduction_three", 32.0, 100.0, "proved"), ("multistep_arithmetic_two", 0.4, 100.0, "proved"),
    ("navigate", 51.0, 100.0, "proved"), ("object_counting", 6.5, 100.0, "proved"),
    ("reasoning_about_colored_objects", 17.0, 100.0, "proved"), ("snarks", 46.3, 100.0, "recognized"),
    ("temporal_sequences", 28.3, 100.0, "recognized"), ("tracking_shuffled_five", 19.8, 100.0, "proved"),
    ("tracking_shuffled_seven", 14.2, 100.0, "proved"), ("tracking_shuffled_three", 31.6, 100.0, "proved"),
    ("web_of_lies", 49.0, 100.0, "proved"), ("word_sorting", 37.7, 100.0, "proved"),
    ("formal_fallacies", 54.3, 99.6, "proved"), ("penguins_in_a_table", 23.1, 99.3, "proved"),
    ("ruin_names", 28.3, 99.2, "recognized"), ("hyperbaton", 48.6, 96.0, "recognized"),
    ("causal_judgement", 58.7, 58.7, "wall"), ("sports_understanding", 54.7, 54.7, "wall"),
    ("date_understanding", 19.0, 28.3, "proved"), ("movie_recommendation", 22.3, 80.0, "recognized"),
    ("salient_translation_error", 13.8, 42.4, "recognized"),
]
# native-dispatch aggregate (includes the movie/salient recognition probes)
BASE_AGG, SWOLLM_AGG, HONEST_CV = 30.22, 92.5, 89.5
MECH = {"proved": C_PROVED, "recognized": C_RECOG, "wall": C_WALL}
lab = lambda t: t.replace("_", " ")
IMG = "/Users/jdonaldson/Projects/jjd.io/posts/images/"

# ======================================================= FIG 1 — per-task leaderboard
rows = sorted(DATA, key=lambda r: (r[2], r[1]))
fig, ax = plt.subplots(figsize=(9.2, 9.6))
for y, (t, b, s, m) in enumerate(rows):
    ax.barh(y, b, color=C_BASE, height=0.66, zorder=3)
    if s > b:
        ax.barh(y, s - b, left=b, color=MECH[m], alpha=GA, height=0.66, zorder=3,
                edgecolor=MECH[m], linewidth=0.4)
    ax.text(s + 1.2, y, f"{s:.0f}", va="center", ha="left", fontsize=7.5, color=LAB)
ax.axvline(BASE_AGG, color=C_LINE, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.text(BASE_AGG + 0.6, len(rows) - 0.2, f"vanilla avg {BASE_AGG:.0f}%",
        color=C_LINE, fontsize=8, va="top")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([lab(t) for t, *_ in rows], fontsize=8)
ax.set_xlim(0, 108)
ax.set_xlabel("accuracy (%)", fontsize=9.5)
ax.set_title("BBH, 27 tasks — SmolLM2-1.7B with Turnstyle (“swollm”)",
             fontsize=13, weight="bold", pad=30, loc="left", color=INK)
ax.text(0, 1.013, "Dark = what the bare model already got (3-shot).  "
        "Color = what the neurosymbolic layer added.",
        transform=ax.transAxes, fontsize=8.5, color=LAB)
ax.legend(handles=[
    Patch(facecolor=C_BASE, label="bare SmolLM2 (3-shot)"),
    Patch(facecolor=C_PROVED, alpha=GA, label="proved — exact symbolic solver"),
    Patch(facecolor=C_RECOG, alpha=GA, label="recognized — hidden-state probe"),
    Patch(facecolor=C_WALL, label="wall — no gain (knowledge-bound)"),
], loc="lower right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.02))
ax.margins(y=0.005)
fig.tight_layout(); fig.savefig(IMG + "bbh_leaderboard.png", bbox_inches="tight")
print("wrote bbh_leaderboard.png")

# ======================================================= FIG 2 — the climb
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.5),
                                gridspec_kw={"width_ratios": [1.15, 1]})
for i, (l, v, c) in enumerate([("vanilla\nSmolLM2", BASE_AGG, C_BASE),
                               ("swollm\n(in-sample)", SWOLLM_AGG, C_PROVED),
                               ("swollm\n(honest CV)", HONEST_CV, C_RECOG)]):
    axA.bar(i, v, color=c, alpha=1.0 if i == 0 else 0.85, width=0.62)
    axA.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, weight="bold", color=INK)
axA.set_xticks(range(3)); axA.set_xticklabels(["vanilla\nSmolLM2", "swollm\n(in-sample)", "swollm\n(honest CV)"], fontsize=8.5)
axA.set_ylim(0, 100); axA.set_ylabel("BBH aggregate (%)", fontsize=9)
axA.set_title("30% → ~90%, same 1.7B model", fontsize=11, weight="bold", loc="left", color=INK)
axA.annotate("", xy=(1, SWOLLM_AGG - 3), xytext=(0, BASE_AGG + 3),
             arrowprops=dict(arrowstyle="->", color=C_LINE, lw=1.5))
axA.text(0.5, (BASE_AGG + SWOLLM_AGG) / 2 + 6, "+62pp", color=C_LINE, fontsize=9, ha="center", weight="bold")
n_wall = sum(1 for r in DATA if r[3] == "wall")
n_solved = sum(1 for r in DATA if r[3] != "wall" and r[2] >= 96.0)
n_partial = len(DATA) - n_wall - n_solved
for i, (l, n, c) in enumerate([("solved ≥96%", n_solved, C_PROVED),
                               ("partial", n_partial, C_PARTIAL),
                               ("wall (knowledge)", n_wall, C_WALL)]):
    axB.bar(i, n, color=c, width=0.62, alpha=0.85)
    axB.text(i, n + 0.3, str(n), ha="center", fontsize=11, weight="bold", color=INK)
axB.set_xticks(range(3)); axB.set_xticklabels(["solved ≥96%", "partial", "wall (knowledge)"], fontsize=8.5)
axB.set_ylim(0, 24); axB.set_ylabel("number of tasks", fontsize=9)
axB.set_title("where the 27 tasks land", fontsize=11, weight="bold", loc="left", color=INK)
fig2.tight_layout(); fig2.savefig(IMG + "bbh_climb.png", bbox_inches="tight")
print(f"wrote bbh_climb.png (solved={n_solved} partial={n_partial} wall={n_wall})")

# ======================================================= FIG 3 — field placement
# Epoch AI BBH leaderboard (3-shot CoT). (name, score, params_label, bucket)
#   bucket: 'swollm' | 'weight' (<=2B, swollm's class) | 'other'
FIELD = [
    ("swollm · SmolLM2-1.7B + Turnstyle", 89.5, "1.7B", "swollm"),
    ("Gemini 1.5 Pro", 89.2, "", "other"),
    ("DeepSeek-V3", 87.5, "671B MoE", "other"),
    ("Llama 3.1-405B", 82.9, "405B", "other"),
    ("phi-3-medium", 81.4, "14B", "other"),
    ("Qwen2.5-72B", 79.8, "72B", "other"),
    ("phi-3-small", 79.1, "7.4B", "other"),
    ("GPT-4 (Jun 2023)", 75.1, "", "other"),
    ("phi-3-mini", 71.7, "3.8B", "other"),
    ("GPT-3.5 Turbo", 61.6, "", "other"),
    ("Phi-2", 59.4, "2.7B", "other"),
    ("Mistral 7B", 56.1, "7B", "other"),
    ("Gemma 2B", 35.2, "2B", "weight"),
    ("bare SmolLM2-1.7B (no Turnstyle)", 30.2, "1.7B", "weight"),
    ("Qwen-1.8B", 28.2, "1.8B", "weight"),
]
COLOR_OF = {"swollm": S["cyan"], "weight": S["orange"], "other": S["base1"]}
rows3 = sorted(FIELD, key=lambda r: r[1])
fig3, ax3 = plt.subplots(figsize=(9.4, 7.2))
swollm_y = None
for y, (nm, sc, pp, bk) in enumerate(rows3):
    ax3.barh(y, sc, color=COLOR_OF[bk], height=0.68,
             alpha=0.95 if bk != "other" else 0.8, zorder=3,
             edgecolor=S["base02"] if bk == "swollm" else "none",
             linewidth=1.3 if bk == "swollm" else 0)
    tail = f"{sc:.1f}"
    if bk == "swollm":
        tail += "  ★"; swollm_y = y
        ax3.plot(SWOLLM_AGG, y, marker="|", ms=16, mew=2.4, color=S["violet"], zorder=5)
    ax3.text(sc + 0.7, y, tail, va="center", ha="left", fontsize=8,
             color=INK, weight="bold" if bk == "swollm" else "normal")
ylabels = [f"{nm}  ·{pp}" if pp else nm for (nm, sc, pp, bk) in rows3]
ax3.set_yticks(range(len(rows3))); ax3.set_yticklabels(ylabels, fontsize=8.2)
for t, (_, _, _, bk) in zip(ax3.get_yticklabels(), rows3):
    if bk == "swollm": t.set_color(S["cyan"]); t.set_fontweight("bold")
    elif bk == "weight": t.set_color(S["orange"])
ax3.plot([], [], color=S["violet"], marker="|", ms=10, mew=2.4, ls="none",
         label="swollm in-sample upper bound (92.5%)")
ax3.set_xlim(0, 100); ax3.set_xlabel("BBH accuracy (%) — Epoch AI, 3-shot CoT", fontsize=9.5)
ax3.set_title("Where “swollm” lands on the BBH field", fontsize=13, weight="bold",
              pad=60, loc="left", color=INK)
ax3.text(0, 1.078,
         "Orange = SmolLM2's own ~2B weight class (incl. the bare model).",
         transform=ax3.transAxes, fontsize=8.5, color=LAB)
ax3.text(0, 1.040,
         "⚠ Not apples-to-apples: the others are general 3-shot-CoT reasoners; swollm\n"
         "is task-routed (zero-shot, held-out probes) and abstains on 2 wall tasks.",
         transform=ax3.transAxes, fontsize=8.2, color=S["red"], weight="bold",
         va="top", linespacing=1.5)
ax3.legend(handles=[
    Patch(facecolor=S["cyan"], label="swollm (1.7B + neurosymbolic layer)"),
    Patch(facecolor=S["orange"], label="≈2B weight class (general models)"),
    Patch(facecolor=S["base1"], label="larger / frontier general models"),
], loc="lower right", fontsize=8, frameon=False)
fig3.tight_layout(); fig3.savefig(IMG + "bbh_field.png", bbox_inches="tight")
print("wrote bbh_field.png")

# ======================================================= FIG 4 — OG card
figog, axog = plt.subplots(figsize=(6.4, 3.35)); axog.axis("off")
axog.text(0.5, 0.74, "⊢ Turnstyle", ha="center", fontsize=34, weight="bold", color=S["base02"], transform=axog.transAxes)
axog.text(0.5, 0.52, "a 1.7B model that stops guessing", ha="center", fontsize=15, color=S["base01"], transform=axog.transAxes)
axog.text(0.5, 0.26, "BBH:  30%  →  ~90%", ha="center", fontsize=22, weight="bold", color=S["cyan"], transform=axog.transAxes)
axog.text(0.5, 0.09, "prove it, recognize it, or admit the wall", ha="center", fontsize=11, color=S["base00"], style="italic", transform=axog.transAxes)
figog.savefig(IMG + "bbh_og.png", bbox_inches="tight", dpi=150)
print("wrote bbh_og.png")

# ======================================================= FIG 5 — size vs BBH scatter
import numpy as np
# (model, params_B, BBH, bucket) — params known models only (size axis needs a number)
SCATTER = [
    ("swollm", 1.7, 89.5, "swollm"),
    ("DeepSeek-V3", 671, 87.5, "other"),
    ("Llama 3.1-405B", 405, 82.9, "other"),
    ("phi-3-medium", 14, 81.4, "other"),
    ("Qwen2.5-72B", 72, 79.8, "other"),
    ("phi-3-small", 7.4, 79.1, "other"),
    ("phi-3-mini", 3.8, 71.7, "other"),
    ("Mistral 7B", 7, 56.1, "other"),
    ("Phi-2", 2.7, 59.4, "other"),
    ("Gemma 2B", 2.0, 35.2, "weight"),
    ("SmolLM2-1.7B (bare)", 1.7, 30.2, "weight"),
    ("Qwen-1.8B", 1.8, 28.2, "weight"),
]
fig5, ax5 = plt.subplots(figsize=(8.8, 6.0))
gx = np.array([np.log10(p) for _, p, _, b in SCATTER if b == "other"])
gy = np.array([s for _, _, s, b in SCATTER if b == "other"])
mfit, cfit = np.polyfit(gx, gy, 1)
xs = np.linspace(np.log10(1.4), np.log10(800), 50)
ax5.plot(10**xs, mfit * xs + cfit, color=S["base1"], lw=1.5, ls="--", zorder=1,
         label="general-model trend")
for nm, p, s, bk in SCATTER:
    if bk == "swollm":
        continue
    col = S["orange"] if bk == "weight" else S["base00"]
    ax5.scatter(p, s, s=72, color=col, edgecolor=S["base02"], linewidth=0.6, zorder=3, alpha=0.9)
    ax5.annotate(nm, (p, s), xytext=(7, -2), textcoords="offset points",
                 fontsize=7.5, color=S["base01"], va="center")
# the same-size vertical leap, bare SmolLM2 -> swollm
ax5.annotate("", xy=(1.7, 86.5), xytext=(1.7, 33.0),
             arrowprops=dict(arrowstyle="->", color=S["orange"], lw=1.7, ls=(0, (3, 2))))
ax5.text(1.92, 48, "+59pp\nsame weights", color=S["orange"], fontsize=8.5,
         ha="left", va="center", weight="bold")
# swollm — stands alone
ax5.scatter(1.7, 89.5, marker="*", s=640, color=S["cyan"], edgecolor=S["base02"],
            linewidth=1.4, zorder=6)
ax5.annotate("swollm  ·  1.7B  ·  89.5%", (1.7, 89.5), xytext=(16, 2),
             textcoords="offset points", fontsize=10.5, weight="bold", color=S["cyan"], va="center")
ax5.set_xscale("log")
ax5.set_xlim(1.2, 1300); ax5.set_ylim(20, 97)
ax5.set_xticks([2, 5, 10, 30, 100, 400, 1000])
ax5.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax5.set_xlabel("parameters (billions, log scale)", fontsize=10)
ax5.set_ylabel("BBH accuracy (%)", fontsize=10)
ax5.set_title("Model size vs BBH — swollm breaks the trend",
              fontsize=12.5, weight="bold", pad=14, loc="left", color=INK)
ax5.text(0, 1.015, "Orange = SmolLM2's ~2B weight class. A 1.7B model + Turnstyle sits a full "
         "head above models 200× its size.", transform=ax5.transAxes, fontsize=8.5, color=LAB)
ax5.grid(True, which="major", color=S["base2"], lw=0.6)
ax5.legend(loc="lower right", fontsize=8.5, frameon=False)
fig5.tight_layout(); fig5.savefig(IMG + "bbh_scatter.png", bbox_inches="tight")
print("wrote bbh_scatter.png")
