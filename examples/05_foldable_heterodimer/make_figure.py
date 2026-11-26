#!/usr/bin/env python3
"""Draw iptm_loss_trajectory.svg from steps_summary.tsv (stdlib only): ipTM (top) and loss (bottom) over the 100 MCMC steps."""
import csv, os
here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "steps_summary.tsv")), delimiter="\t"))
W, H, L, R_, T = 760, 640, 64, 24, 70
PH, GAP = 190, 60
ink, mute, grid, blue, gray = "#1f2328", "#59636e", "#d8dee4", "#1f6feb", "#8c959f"
def X(s): return L + (W - L - R_) * s / 100.0
def panel(y0, key_c, key_cur, lo, hi, ticks, title, note, ann):
    Y = lambda v: y0 + PH * (1 - (v - lo) / (hi - lo))
    o = [f'<text x="{L}" y="{y0-14}" font-size="14" font-weight="600" fill="{ink}">{title}</text>',
         f'<text x="{W-R_}" y="{y0-14}" font-size="12" text-anchor="end" fill="{mute}">{note}</text>']
    for t in ticks:
        o.append(f'<line x1="{L}" x2="{W-R_}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{Y(t)+4:.1f}" font-size="11" text-anchor="end" fill="{mute}">{t:g}</text>')
    for s in range(0, 101, 20):
        o.append(f'<text x="{X(s):.1f}" y="{y0+PH+16}" font-size="11" text-anchor="middle" fill="{mute}">{s}</text>')
    for r in rows:
        o.append(f'<circle cx="{X(int(r["step"])):.1f}" cy="{Y(float(r[key_c])):.1f}" r="2.4" fill="{gray}" fill-opacity="0.75"/>')
    pts, prev = [], None
    for r in rows:
        x, v = X(int(r["step"])), float(r[key_cur])
        if prev is not None: pts.append(f"{x:.1f},{prev:.1f}")
        pts.append(f"{x:.1f},{Y(v):.1f}"); prev = Y(v)
    o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{blue}" stroke-width="2" stroke-linejoin="round"/>')
    for r in rows:
        if r["accepted"] == "True":
            o.append(f'<circle cx="{X(int(r["step"])):.1f}" cy="{Y(float(r[key_cur])):.1f}" r="4" fill="{blue}" stroke="#ffffff" stroke-width="2"/>')
    for step, dx, dy, anchor, text in ann:
        r = rows[step]; o.append(f'<text x="{X(step)+dx:.1f}" y="{Y(float(r[key_cur]))+dy:.1f}" font-size="11.5" text-anchor="{anchor}" fill="{ink}" stroke="#ffffff" stroke-width="3.5" paint-order="stroke">{text}</text>')
    o.append(f'<text x="{(L+W-R_)/2:.0f}" y="{y0+PH+36}" font-size="12" text-anchor="middle" fill="{mute}">MCMC step</text>')
    return "\n".join(o)
a = {k: rows[i] for i, k in ((0, "s0"), (80, "s80"), (83, "s83"))}
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
       f'<title>af3-hall foldable GB1 + protein L heterodimer: ipTM and loss over 100 MCMC steps</title>',
       f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="8" fill="#ffffff" stroke="{grid}"/>',
       f'<text x="{L}" y="30" font-size="16" font-weight="600" fill="{ink}">ipTM rises from 0.08 to 0.74 in 100 steps (best at step 83)</text>',
       f'<circle cx="{L+6}" cy="48" r="2.4" fill="{gray}" fill-opacity="0.75"/><text x="{L+16}" y="52" font-size="12" fill="{mute}">proposed candidate each step</text>',
       f'<line x1="{L+210}" x2="{L+234}" y1="48" y2="48" stroke="{blue}" stroke-width="2"/><circle cx="{L+222}" cy="48" r="4" fill="{blue}" stroke="#fff" stroke-width="2"/><text x="{L+242}" y="52" font-size="12" fill="{mute}">current chain (dots = accepted steps)</text>',
       panel(T + 20, "iptm_cand", "iptm_current", 0, 1, [0, 0.25, 0.5, 0.75, 1], "ipTM (not in the loss)", "higher = better interface",
             [(0, 8, -12, "start", "step 0: 0.08"), (80, -10, -14, "end", "step 80: 0.48 (A:A20K)"), (83, -10, 4, "end", "step 83: 0.74 (B:V49G)")]),
       panel(T + 20 + PH + GAP + 20, "loss_candidate", "loss_current", 0, 0.7, [0, 0.2, 0.4, 0.6], "dual loss = 0.5 * ((1 - pLDDT) + (1 - pTM))", "lower = better",
             [(0, 8, 18, "start", "step 0: 0.354"), (83, 10, 18, "start", "step 83: 0.150")]),
       "</svg>"]
open(os.path.join(here, "iptm_loss_trajectory.svg"), "w").write("\n".join(svg))
print("wrote svg")
