#!/usr/bin/env python3
"""QR code that reveals the full reference list when scanned."""
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

refs = """REFERENCES

[1] Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. Proc. ICNN'95, 4, 1942-1948. IEEE.

[2] Stackhouse, P. W. et al. (2018). POWER (Prediction of Worldwide Energy Resources). NASA Langley. https://power.larc.nasa.gov/

[3] Upadhyay, S. & Sharma, M. P. (2014). A review on configurations, control and sizing methodologies of hybrid energy systems. Renewable and Sustainable Energy Reviews, 38, 47-63.

[4] Maleki, A. & Askarzadeh, A. (2014). Optimal sizing of a PV-wind-diesel system with battery storage. Solar Energy, 99, 272-282.

[5] Fadaee, M. & Radzi, M. A. M. (2012). Multi-objective optimization of a stand-alone hybrid renewable energy system. Renewable and Sustainable Energy Reviews, 16(5), 3364-3369.

[6] Kaabeche, A., Belhamel, M. & Ibtiouen, R. (2011). Sizing optimization of grid-independent hybrid PV/wind system. Energy, 36(2), 1214-1222.

[7] Mandelli, S. et al. (2016). Effect of load profile uncertainty on the optimum sizing of off-grid PV systems. Sustainable Energy Technologies and Assessments, 18, 34-47.

[8] Zhou, N. et al. (2018). Household electricity consumption profiles in China. Energy and Buildings, 172, 112-124.

[9] Bhattacharyya, S. C. & Palit, D. (2016). Mini-grids for rural electrification of developing countries. Energy for Sustainable Development, 31, 1-13.

[10] IRENA (2024). Renewable Power Generation Costs in 2023. Abu Dhabi: IRENA.

[11] Lazard (2024). Levelized Cost of Energy Analysis - Version 17.0.

[12] IEA (2024). World Energy Outlook 2024. Paris: IEA.

CIS Research Fair 2026
Nono (Yuanjing Zhou) · CQUPT
github.com/nono0529/cis-research-fair-microgrid"""

qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=16,
    border=2,
)
qr.add_data(refs)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

w, h = img.size
label_h = 45
labeled = Image.new("RGB", (w, h + label_h), "white")
labeled.paste(img, (0, 0))

draw = ImageDraw.Draw(labeled)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()
draw.text((w / 2, h + 10), "Scan for References", fill="black", font=font, anchor="mt")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig_qr_code.png")
labeled.save(out, dpi=(300, 300))
print(f"Saved: {out}")
print(f"QR version: {qr.version}, modules: {qr.modules_count}x{qr.modules_count}")
print(f"Chars: {len(refs)}")
print(f"At 35mm print: ~{qr.modules_count/(35/25.4):.1f} modules/mm")
