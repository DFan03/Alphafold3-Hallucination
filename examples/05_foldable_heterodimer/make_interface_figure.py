"""Render the A/B interface at step 0 and step 83 with PyMOL (ray traced), both aligned on chain A and drawn with the SAME camera.
Run from this folder:   pymol -cq make_interface_figure.py        -> panel_step0.png, panel_step83.png
Then add the captions:  python compose_interface_figure.py        -> interface_step0_vs_step83.png
Chain A = GB1 (blue), chain B = protein L B1 (orange); sticks = residues within 4.5 A of the other chain; green dashes = polar contacts
(N/O pairs <= 3.5 A). Use open-source PyMOL (conda-forge `pymol-open-source`): the licensed app stamps an evaluation watermark when unlicensed."""
import json, os
from pymol import cmd

here = os.getcwd()
FILES = {"s0": "step_0000_acc_AB_pos.cif", "s83": "step_0083_acc_AB_pos.cif"}
BLUE, ORANGE, LBLUE, LORANGE = "0x1f77b4", "0xff7f0e", "0xaecde6", "0xffcf9f"

cmd.reinitialize()
for k, f in FILES.items():
    cmd.load(os.path.join(here, f), k)
cmd.align("s83 and chain A and name CA", "s0 and chain A and name CA", cycles=0)  # same frame for GB1
cmd.bg_color("white"); cmd.hide("everything")
stats = {}
for k in FILES:
    cmd.show("cartoon", k)
    cmd.color(LBLUE, f"{k} and chain A"); cmd.color(LORANGE, f"{k} and chain B")
    cmd.select(f"{k}_iA", f"byres ({k} and chain A within 4.5 of ({k} and chain B))")
    cmd.select(f"{k}_iB", f"byres ({k} and chain B within 4.5 of ({k} and chain A))")
    for c, sel in ((BLUE, f"{k}_iA"), (ORANGE, f"{k}_iB")):
        cmd.show("sticks", f"{sel} and not hydrogens"); cmd.color(c, f"{sel} and elem C")
    cmd.color("red", f"({k}_iA or {k}_iB) and elem O"); cmd.color("0x1a1aa6", f"({k}_iA or {k}_iB) and elem N")
    pairs = cmd.find_pairs(f"{k} and chain A and elem N+O", f"{k} and chain B and elem N+O", cutoff=3.5)
    cmd.distance(f"{k}_polar", f"{k} and chain A and elem N+O", f"{k} and chain B and elem N+O", 3.5, mode=0)
    cmd.hide("labels", f"{k}_polar"); cmd.color("0x2ca02c", f"{k}_polar")
    stats[k] = dict(iface_A=cmd.count_atoms(f"{k}_iA and name CA"), iface_B=cmd.count_atoms(f"{k}_iB and name CA"), polar=len(pairs))
cmd.set("stick_radius", 0.17); cmd.set("dash_gap", 0.3); cmd.set("dash_radius", 0.09)
cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_transparency", 0.1); cmd.set("cartoon_side_chain_helper", 1)
cmd.set("antialias", 2); cmd.set("ray_trace_mode", 0); cmd.set("ray_shadow", 0); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1)
cmd.set("ray_opaque_background", 1); cmd.set("specular", 0.15)
cmd.orient("s0 or s83"); cmd.zoom("s0 or s83", buffer=-3)
for k in FILES:
    other = [o for o in FILES if o != k][0]
    cmd.enable(k); cmd.disable(other)
    cmd.enable(f"{k}_polar"); cmd.disable(f"{other}_polar")
    cmd.png(os.path.join(here, f"panel_{k.replace('s', 'step')}.png"), width=1400, height=1150, dpi=200, ray=1)
json.dump(stats, open(os.path.join(here, "panel_stats.json"), "w"))
print("stats", stats)
