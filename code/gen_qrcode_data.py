#!/usr/bin/env python3
"""Generate QR code pointing to the experiment data webpage."""
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

# The URL that the QR code will point to
URL = "https://github.com/nono0529/cis-research-fair-microgrid"

qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_M,  # M = ~15% correction (good for posters)
    box_size=20,
    border=2,
)
qr.add_data(URL)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

w, h = img.size
label_h = 55

# Create a taller canvas with a white label area at the bottom
labeled = Image.new("RGB", (w, h + label_h), "white")
labeled.paste(img, (0, 0))

draw = ImageDraw.Draw(labeled)
try:
    font_title = ImageFont.truetype("arialbd.ttf", 20)  # Arial Bold
except OSError:
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font_title = ImageFont.load_default()

try:
    font_sub = ImageFont.truetype("arial.ttf", 14)
except OSError:
    font_sub = ImageFont.load_default()

# Label text
draw.text((w / 2, h + 8), "Scan for Experiment Data", fill="black", font=font_title, anchor="mt")
draw.text((w / 2, h + 32), "PV-Battery Microgrid Optimization", fill="#666666", font=font_sub, anchor="mt")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig_qr_code_data.png")
labeled.save(out, dpi=(300, 300))
print(f"Saved: {out}")
print(f"URL encoded: {URL}")
print(f"QR version: {qr.version}, modules: {qr.modules_count}x{qr.modules_count}")
print(f"Recommended print size on A0 poster: ~35-40mm square")
