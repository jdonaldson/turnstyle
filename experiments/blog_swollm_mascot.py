"""Overlay the 'swollm' wordmark on the swole-doge image."""
import os
from PIL import Image, ImageDraw, ImageFont

SRC = "/Users/jdonaldson/Desktop/anyone-elses-shiba-super-swole-or-just-mine-v0-n9j407ld6d4f1.png.webp"
OUT = "/Users/jdonaldson/Projects/jjd.io/posts/images/swollm_mascot.png"

img = Image.open(SRC).convert("RGBA")
W, H = img.size
print("source size", W, H)

# solarized
CYAN = (42, 161, 152, 255)     # #2aa198
INK  = (7, 54, 66, 255)        # #073642 (outline)

# pick a heavy font
CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
]
font_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
size = int(W * 0.19)
if font_path:
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        font = ImageFont.truetype(font_path, size, index=0)
    print("font:", font_path)
else:
    from matplotlib import font_manager
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), size)
    print("font: matplotlib DejaVu Sans Bold")

draw = ImageDraw.Draw(img)
text = "swollm"
bb = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, size // 16))
tw, th = bb[2] - bb[0], bb[3] - bb[1]
x = (W - tw) / 2 - bb[0]
y = H - th - bb[1] - int(H * 0.015)   # sit near the bottom edge
draw.text((x, y), text, font=font, fill=CYAN,
          stroke_width=max(2, size // 16), stroke_fill=INK)

img.save(OUT)
print("wrote", OUT, "->", img.size)

# ---- wide social OG card (1200x628): doge on the right, wordmark on the left ----
def load_font(paths, sz):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), sz)

CREAM = (253, 246, 227, 255)   # base3
OW, OH = 1200, 628
card = Image.new("RGBA", (OW, OH), CREAM)

# raw doge (no baked text), scaled to fit height
doge = Image.open(SRC).convert("RGBA")
dh = int(OH * 0.94)
dw = int(doge.width * dh / doge.height)
doge = doge.resize((dw, dh), Image.LANCZOS)
card.alpha_composite(doge, (OW - dw - 30, (OH - dh) // 2))

d = ImageDraw.Draw(card)
impact = load_font(CANDIDATES, 150)
arial  = load_font(["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"], 52)
arial_sm = load_font(["/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc"], 38)
d.text((64, 150), "swollm", font=impact, fill=CYAN, stroke_width=5, stroke_fill=INK)
d.text((68, 320), "a 1.7B model that stops guessing", font=arial_sm, fill=(88, 110, 117, 255))
d.text((68, 380), "BBH:  30%  →  ~90%", font=arial, fill=INK)
OG = "/Users/jdonaldson/Projects/jjd.io/posts/images/swollm_og.png"
card.convert("RGB").save(OG)
print("wrote", OG, "->", card.size)
