#!/usr/bin/env python3
"""tmol 0.2.0 FastRelax worker for af3_hall `--fastrelax`. Runs in an env with torch + tmol 0.2.0 + the RFD4-Proteina `rfproteina` package (tmol_ddg_core), NOT in the AF3/JAX env.

Persistent mode (used by af3_hall.relax.TmolRelaxer): reads one JSON request per stdin line, writes one JSON reply per line:
  request  {"in_cif": "...", "out_cif": "...", "repeats": 1, "cartesian": true, "scope": "all"|"shell", "shell": 8.0,
            "focus_chains": ["B"]}
  reply    {"ok": true, "E_before": .., "E_after": .., "ca_rmsd": .., "relax_sec": .., "build_sec": .., "n_res": ..}
           or {"ok": false, "error": "..."}  (the worker stays alive after an error)
One-shot mode for tests:  relax_tmol_worker.py --once IN.cif OUT.cif [--repeats N] [--scope all|shell]

Method = tmol FastRelax (Cartesian, beta2016) from rfproteina.metrics.physics.tmol_ddg_core, same calls as
the tmol examples of RFD4-Proteina. Only heavy atoms present in the input are written back
(coordinates of the original atom array are replaced; atom order/names/chain ids are untouched), so the output CIF has the same
atoms as the AF3 CIF. Energies are tmol block-pair sums: use only as before/after deltas, not as absolute kcal/mol.
Rosetta FastRelax is not used: tmol runs on the GPU in-process and needs no Rosetta install.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np


def _load_array(path: str):
    import biotite.structure as bs
    import biotite.structure.io.pdbx as pdbx

    cf = pdbx.CIFFile.read(path)
    arr = pdbx.get_structure(cf, model=1, include_bonds=True)
    if isinstance(arr, bs.AtomArrayStack):
        arr = arr[0]
    return arr[~np.isnan(arr.coord).any(axis=1)]


def _ca_rmsd(a0, a1) -> float:
    ca0 = a0[(a0.atom_name == "CA") & (~a0.hetero)].coord
    ca1 = a1[(a1.atom_name == "CA") & (~a1.hetero)].coord
    if len(ca0) < 3 or len(ca0) != len(ca1):
        return float("nan")
    p0, p1 = ca0.mean(0), ca1.mean(0)
    u, _, vt = np.linalg.svd((ca1 - p1).T @ (ca0 - p0))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return float(np.sqrt((((ca1 - p1) @ (u @ d @ vt) - (ca0 - p0)) ** 2).sum(1).mean()))


class Relaxer:
    def __init__(self, device: str = "cuda:0") -> None:
        import torch

        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self._ctx_cache: dict[tuple, tuple] = {}

    def _context(self, prep, hetero_sig: tuple):
        from rfproteina.metrics.physics.tmol_ddg_core import build_context_from_atom_array, create_score_function

        if hetero_sig not in self._ctx_cache:
            ctx = build_context_from_atom_array(prep, self.dev, prepare_ligands=bool(hetero_sig), strict_ligands=bool(hetero_sig))
            self._ctx_cache[hetero_sig] = (ctx, create_score_function(self.dev, param_db=ctx.parameter_database))
        return self._ctx_cache[hetero_sig]

    def _energy(self, sfxn, pose) -> float:
        with self.torch.no_grad():
            bp = sfxn.render_block_pair_scoring_module(pose)(pose.coords, sum_terms=False)[:, 0]
        return float(bp.sum().item())

    def relax(self, req: dict) -> dict:
        import biotite.structure as bs
        import biotite.structure.io.pdbx as pdbx
        from rfproteina.metrics.physics.tmol_ddg_core import (
            pose_stack_from_atom_array, prepare_tmol_atom_array, relax_pose_stack, relax_shell)

        torch = self.torch
        t0 = time.time()
        arr = _load_array(req["in_cif"])
        prep = prepare_tmol_atom_array(arr)
        hetero_sig = tuple(sorted({str(r) for r in np.unique(arr.res_name[arr.hetero])}))
        ctx, sfxn = self._context(prep, hetero_sig)
        pose = pose_stack_from_atom_array(prep, self.dev, context=ctx, no_optH=False)
        starts = bs.get_residue_starts(arr, add_exclusive_stop=True)
        blocks = (pose.block_type_ind[0] >= 0).nonzero().flatten().tolist()
        if len(blocks) != len(starts) - 1:
            raise ValueError(f"tmol blocks {len(blocks)} != residues {len(starts) - 1} (tmol dropped residues)")
        e0 = self._energy(sfxn, pose)
        t_build = time.time() - t0

        t1 = time.time()
        repeats = int(req.get("repeats", 1))
        with torch.inference_mode(False), torch.enable_grad():
            if req.get("scope", "all") == "shell":
                focus = set(req.get("focus_chains") or [])
                shell = float(req.get("shell", 8.0))
                other = [i for i in range(len(starts) - 1) if str(arr.chain_id[starts[i]]) not in focus]
                foc = [i for i in range(len(starts) - 1) if str(arr.chain_id[starts[i]]) in focus]
                if not foc or not other:
                    raise ValueError("scope=shell needs focus_chains and at least one other chain")
                res_xyz = [arr.coord[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
                oth_xyz = np.concatenate([res_xyz[i] for i in other])
                mask = torch.zeros((1, pose.max_n_blocks), dtype=torch.bool, device=self.dev)
                for i in foc:
                    d = np.sqrt(((res_xyz[i][:, None] - oth_xyz[None]) ** 2).sum(-1)).min()
                    if d <= shell:
                        mask[0, blocks[i]] = True
                pose_rel = relax_shell(pose, sfxn, mask, num_repeats=repeats)
            else:
                pose_rel = relax_pose_stack(pose, sfxn, cartesian=bool(req.get("cartesian", True)), num_repeats=repeats)
        e1 = self._energy(sfxn, pose_rel)
        t_relax = time.time() - t1

        out = arr.copy()
        xyz = pose_rel.coords[0].detach().cpu().numpy()
        pbt = pose_rel.packed_block_types
        n_set = 0
        for bi, (s0, s1) in enumerate(zip(starts[:-1], starts[1:])):
            bti = int(pose_rel.block_type_ind[0, bi])
            if bti < 0:
                continue
            a2i = pbt.active_block_types[bti].atom_to_idx
            off = int(pose_rel.block_coord_offset[0, bi])
            for k in range(s0, s1):
                nm = str(out.atom_name[k]).strip()
                if nm in a2i:
                    out.coord[k] = xyz[off + a2i[nm]]
                    n_set += 1
        if n_set < 0.95 * len(out):
            raise ValueError(f"only {n_set}/{len(out)} atoms mapped back from tmol")
        cf = pdbx.CIFFile()
        pdbx.set_structure(cf, out, include_bonds=True)
        cf.write(req["out_cif"])
        return {"ok": True, "E_before": e0, "E_after": e1, "ca_rmsd": _ca_rmsd(arr, out), "relax_sec": round(t_relax, 3),
                "build_sec": round(t_build, 3), "n_res": len(starts) - 1, "n_atoms": int(len(out)), "n_mapped": n_set}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--once", nargs=2, metavar=("IN_CIF", "OUT_CIF"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--scope", default="all", choices=["all", "shell"])
    ap.add_argument("--shell", type=float, default=8.0)
    ap.add_argument("--focus-chains", default="B")
    a = ap.parse_args()
    rx = Relaxer(a.device)
    if a.once:
        r = rx.relax({"in_cif": a.once[0], "out_cif": a.once[1], "repeats": a.repeats, "scope": a.scope, "shell": a.shell,
                      "focus_chains": [c for c in a.focus_chains.split(",") if c]})
        print(json.dumps(r))
        return 0
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # library prints must not corrupt the JSON protocol
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = rx.relax(json.loads(line))
        except Exception as exc:  # keep the worker alive; the client decides raise/skip
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
        real_stdout.write(json.dumps(reply) + "\n")
        real_stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
