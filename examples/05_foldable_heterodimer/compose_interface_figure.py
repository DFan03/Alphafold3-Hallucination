"""Add titles and a legend to panel_step0.png / panel_step83.png (from make_interface_figure.py) -> interface_step0_vs_step83.png.
Needs Pillow + matplotlib (for the DejaVu font)."""
import json, os
import matplotlib
from PIL import Image, ImageDraw, ImageFont

here = os.path.dirname(os.path.abspath(__file__))
st = json.load(open(os.path.join(here, "panel_stats.json")))
font = lambda sz, bold=False: ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)
ink, mute = "#1f2328", "#59636e"
P = [Image.open(os.path.join(here, f"panel_{k}.png")).convert("RGB") for k in ("step0", "step83")]
w, h = P[0].size
top, bot = 250, 150
W, H = 2 * w, top + h + bot
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
d.text((W // 2, 40), "Interface of the GB1 (A) + protein L (B) heterodimer: step 0 vs step 83", font=font(54, True), fill=ink, anchor="mt")
heads = [("Step 0 (start)", "ipTM 0.08", "s0"), ("Step 83 (best, last accepted)", "ipTM 0.74", "s83")]
for i, (t, m, k) in enumerate(heads):
    x = i * w + w // 2
    d.text((x, 115), f"{t}   |   {m}", font=font(46, True), fill=ink, anchor="mt")
    s = st[k]
    d.text((x, 172), f"{s['iface_A'] + s['iface_B']} interface residues (A {s['iface_A']} + B {s['iface_B']})   |   {s['polar']} polar contacts", font=font(34), fill=mute, anchor="mt")
    img.paste(P[i], (i * w, top))
d.line([(w, top + 40), (w, top + h - 40)], fill="#d8dee4", width=3)
y = top + h + 40
x = 150
for c, txt in (("#1f77b4", "chain A = GB1"), ("#ff7f0e", "chain B = protein L B1"), ("#2ca02c", "polar contacts (N/O <= 3.5 A)")):
    d.rounded_rectangle([x, y, x + 46, y + 30], 6, fill=c)
    d.text((x + 62, y - 2), txt, font=font(36), fill=ink)
    x += 62 + int(d.textlength(txt, font=font(36))) + 110
d.text((150, y + 62), "sticks = residues within 4.5 A of the other chain; both structures aligned on chain A, same camera", font=font(32), fill=mute)
img.save(os.path.join(here, "interface_step0_vs_step83.png"), optimize=True)
print("wrote", img.size)
