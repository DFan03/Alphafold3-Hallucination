"""Structure containers, pure-Python mmCIF/PDB parsing, interface masking, H-bond, Rg, aspect ratio, steric clash, and core RMSD utilities."""
from __future__ import annotations

import dataclasses
import math
import shlex
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})
POLAR_ELEMENTS: frozenset[str] = frozenset({"N", "O"})
STANDARD_AA3: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
})
SUGAR_CCDS: frozenset[str] = frozenset({"NAG", "BMA", "MAN", "FUC", "GAL"})
PYRANOSE_RING_ATOMS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "O5")
GLYCOSIDIC_ACCEPTOR_ATOMS: frozenset[str] = frozenset({"O2", "O3", "O4", "O6"})
AROMATIC_RING_ATOMS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}
EXPECTED_SUGAR_STEREO: dict[str, tuple[str, int]] = {
    "NAG": ("4C1", +1),
    "BMA": ("4C1", +1),
    "GAL": ("4C1", +1),
    "MAN": ("4C1", -1),
    "FUC": ("1C4", +1),
}


@dataclasses.dataclass
class ChainAtoms:
    """Heavy-atom container for a single protein, glycan, or ligand chain."""

    chain_id: str
    resnames: list[str]
    resids: list[int]
    atom_names: list[str]
    elements: list[str]
    xyz: np.ndarray  # shape [n_atoms, 3]
    atom_res: np.ndarray  # shape [n_atoms], 0-based index into resnames/resids

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape((-1, 3))
        self.atom_res = np.asarray(self.atom_res, dtype=np.int64).reshape((-1,))
        self._res_atom_indices: list[np.ndarray] = [
            np.where(self.atom_res == i)[0] for i in range(len(self.resnames))
        ]

    @property
    def n_res(self) -> int:
        return len(self.resnames)

    def _atom_coord(self, res_idx: int, atom_name: str) -> np.ndarray | None:
        for idx in self._res_atom_indices[res_idx]:
            if self.atom_names[idx] == atom_name:
                return self.xyz[idx]
        return None

    def ca_coords(self) -> np.ndarray:
        """Return [n_res, 3] CA coordinates (fallback to first heavy atom if CA missing)."""
        coords: list[np.ndarray] = []
        for i in range(self.n_res):
            pt = self._atom_coord(i, "CA")
            if pt is None and len(self._res_atom_indices[i]) > 0:
                pt = self.xyz[self._res_atom_indices[i][0]]
            if pt is not None:
                coords.append(pt)
        return np.asarray(coords, dtype=np.float64).reshape((-1, 3))

    def cb_or_ca_coords(self) -> np.ndarray:
        """Return [n_res, 3] representative token coordinates (CB, or CA for GLY/missing CB)."""
        coords: list[np.ndarray] = []
        for i in range(self.n_res):
            pt = self._atom_coord(i, "CB")
            if pt is None:
                pt = self._atom_coord(i, "CA")
            if pt is None and len(self._res_atom_indices[i]) > 0:
                pt = self.xyz[self._res_atom_indices[i][0]]
            if pt is not None:
                coords.append(pt)
        return np.asarray(coords, dtype=np.float64).reshape((-1, 3))


@dataclasses.dataclass
class ComplexCoords:
    """Multi-chain heavy-atom representation supporting monomers, oligomers, ligands, and glycans."""

    target: ChainAtoms
    binder: ChainAtoms
    glycan_chains: dict[str, ChainAtoms]
    all_chains: dict[str, ChainAtoms] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.all_chains:
            chains: dict[str, ChainAtoms] = {}
            if self.target.n_res > 0:
                chains[self.target.chain_id] = self.target
            if self.binder.n_res > 0:
                chains[self.binder.chain_id] = self.binder
            chains.update(self.glycan_chains)
            self.all_chains = chains

    @property
    def glycan_xyz(self) -> np.ndarray:
        if not self.glycan_chains:
            return np.zeros((0, 3), dtype=np.float64)
        pts = [c.xyz for c in self.glycan_chains.values() if len(c.xyz) > 0]
        return np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)

    @property
    def glycan_polar_xyz(self) -> np.ndarray:
        pts: list[np.ndarray] = []
        for c in self.glycan_chains.values():
            mask = [i for i, el in enumerate(c.elements) if el in POLAR_ELEMENTS]
            if mask:
                pts.append(c.xyz[mask])
        return np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)


def _split_cif_line(line: str) -> list[str]:
    if "'" not in line and '"' not in line:
        return line.split()
    return shlex.split(line)


def _infer_element(atom_name: str, type_symbol: str) -> str:
    sym = type_symbol.strip().upper()
    if sym and sym not in (".", "?"):
        return sym
    cleaned = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    return cleaned[0] if cleaned else "C"


def _build_chain_atoms(
    chain_id: str,
    records: list[tuple[int, str, str, str, tuple[float, float, float]]],
) -> ChainAtoms:
    resnames: list[str] = []
    resids: list[int] = []
    atom_names: list[str] = []
    elements: list[str] = []
    xyz: list[tuple[float, float, float]] = []
    atom_res: list[int] = []
    seen_res: dict[tuple[int, str], int] = {}

    for rid, rname, aname, el, coord in records:
        key = (rid, rname)
        if key not in seen_res:
            seen_res[key] = len(resnames)
            resnames.append(rname)
            resids.append(rid)
        r_idx = seen_res[key]
        atom_names.append(aname)
        elements.append(el)
        xyz.append(coord)
        atom_res.append(r_idx)

    return ChainAtoms(
        chain_id=chain_id,
        resnames=resnames,
        resids=resids,
        atom_names=atom_names,
        elements=elements,
        xyz=np.asarray(xyz, dtype=np.float64).reshape((-1, 3)),
        atom_res=np.asarray(atom_res, dtype=np.int64),
    )


def _parse_mmcif_records(text: str) -> dict[str, list[tuple[int, str, str, str, tuple[float, float, float]]]]:
    lines = text.splitlines()
    n = len(lines)
    i = 0
    by_chain: dict[str, list[tuple[int, str, str, str, tuple[float, float, float]]]] = {}

    while i < n:
        ln = lines[i].strip()
        if ln == "loop_":
            i += 1
            headers: list[str] = []
            while i < n and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1
            if headers and all(h.startswith("_atom_site.") for h in headers):
                col = {h.split(".", 1)[1]: idx for idx, h in enumerate(headers)}
                idx_group = col.get("group_PDB")
                idx_atom = col.get("label_atom_id", col.get("auth_atom_id"))
                idx_comp = col.get("label_comp_id", col.get("auth_comp_id"))
                idx_asym = col.get("auth_asym_id", col.get("label_asym_id"))
                idx_seq = col.get("auth_seq_id", col.get("label_seq_id"))
                idx_x = col.get("Cartn_x")
                idx_y = col.get("Cartn_y")
                idx_z = col.get("Cartn_z")
                idx_sym = col.get("type_symbol")
                idx_alt = col.get("label_alt_id")
                idx_model = col.get("pdbx_PDB_model_num")
                first_model: str | None = None

                while i < n:
                    raw = lines[i].strip()
                    if not raw or raw.startswith("#"):
                        i += 1
                        continue
                    if raw == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                        break
                    toks = _split_cif_line(raw)
                    i += 1
                    if len(toks) < len(headers):
                        continue
                    if idx_model is not None:
                        mval = toks[idx_model]
                        if first_model is None:
                            first_model = mval
                        elif mval != first_model:
                            continue
                    if idx_alt is not None and toks[idx_alt] not in (".", "?", "A"):
                        continue
                    grp = toks[idx_group] if idx_group is not None else "ATOM"
                    if grp not in ("ATOM", "HETATM"):
                        continue
                    aname = toks[idx_atom]
                    rname = toks[idx_comp]
                    chain = toks[idx_asym]
                    el = _infer_element(aname, toks[idx_sym] if idx_sym is not None else "")
                    if el in ("H", "D"):
                        continue
                    seq_tok = toks[idx_seq] if idx_seq is not None else "1"
                    rid = int(seq_tok) if seq_tok not in (".", "?") else 1
                    coord = (float(toks[idx_x]), float(toks[idx_y]), float(toks[idx_z]))
                    by_chain.setdefault(chain, []).append((rid, rname, aname, el, coord))
                continue
        i += 1
    return by_chain


def _parse_pdb_records(text: str) -> dict[str, list[tuple[int, str, str, str, tuple[float, float, float]]]]:
    by_chain: dict[str, list[tuple[int, str, str, str, tuple[float, float, float]]]] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        alt = line[16:17].strip()
        if alt not in ("", "A"):
            continue
        aname = line[12:16].strip()
        rname = line[17:20].strip()
        chain = line[21:22].strip() or "A"
        rid_str = line[22:26].strip()
        rid = int(rid_str) if rid_str else 1
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        el = _infer_element(aname, line[76:78] if len(line) >= 78 else "")
        if el in ("H", "D"):
            continue
        by_chain.setdefault(chain, []).append((rid, rname, aname, el, (x, y, z)))
    return by_chain


def load_structure(
    path: str | Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    glycan_chain_ids: Sequence[str] | None = None,
) -> ComplexCoords:
    """Load mmCIF or PDB structure (heavy atoms only) into ComplexCoords."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".cif", ".mmcif") or "_atom_site." in text:
        by_chain = _parse_mmcif_records(text)
    else:
        by_chain = _parse_pdb_records(text)

    def _empty_chain(cid: str) -> ChainAtoms:
        return ChainAtoms(cid, [], [], [], [], np.zeros((0, 3)), np.zeros((0,), dtype=np.int64))

    all_chains = {cid: _build_chain_atoms(cid, recs) for cid, recs in by_chain.items()}
    tgt = all_chains.get(target_chain, _empty_chain(target_chain))
    bnd = all_chains.get(binder_chain, _empty_chain(binder_chain))
    if glycan_chain_ids is not None:
        g_set = set(glycan_chain_ids)
        glycan_chains = {cid: c_atoms for cid, c_atoms in all_chains.items() if cid in g_set}
    else:
        glycan_chains = {
            cid: c_atoms
            for cid, c_atoms in all_chains.items()
            if cid not in (target_chain, binder_chain)
            and any(rn in SUGAR_CCDS for rn in c_atoms.resnames)
        }
    return ComplexCoords(target=tgt, binder=bnd, glycan_chains=glycan_chains, all_chains=all_chains)


def cif_to_pdb(cif_path: str | Path, pdb_path: str | Path) -> None:
    """Convert mmCIF to PDB (protein + hetero chains) for ProteinMPNN/LigandMPNN or visualization."""
    try:
        import gemmi

        st = gemmi.read_structure(str(cif_path))
        st.remove_hydrogens()
        st.write_pdb(str(pdb_path))
        return
    except ImportError:
        pass

    cx = load_structure(cif_path)
    lines: list[str] = []
    serial = 1
    for cid in sorted(cx.all_chains):
        chain = cx.all_chains[cid]
        is_het = (cid in cx.glycan_chains) or (
            chain.n_res > 0 and not all(rn in STANDARD_AA3 for rn in chain.resnames)
        )
        rec_kw = "HETATM" if is_het else "ATOM  "
        for idx in range(len(chain.atom_names)):
            r_idx = int(chain.atom_res[idx])
            rname = chain.resnames[r_idx][:3].rjust(3)
            rid = chain.resids[r_idx]
            aname = chain.atom_names[idx]
            aname_fmt = f" {aname:<3s}" if len(aname) < 4 else aname[:4]
            el = chain.elements[idx][:2].rjust(2)
            x, y, z = chain.xyz[idx]
            lines.append(
                f"{rec_kw}{serial:5d} {aname_fmt} {rname} {chain.chain_id:1s}{rid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el}"
            )
            serial += 1
    lines.append("END")
    Path(pdb_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_interface_mask(
    cx: ComplexCoords,
    interface_dist: float = 6.0,
    explicit_positions: Sequence[int] | None = None,
    mutable_chain: str = "B",
    partner_chains: Sequence[str] | None = None,
) -> list[int]:
    """Return sorted 0-based mutable chain indices within `interface_dist` (A) of partner/glycan/ligand chains."""
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    n_b = m_chain.n_res
    if explicit_positions:
        pos = sorted({int(p) - 1 for p in explicit_positions})
        bad = [p + 1 for p in pos if not (0 <= p < n_b)]
        if bad:
            raise ValueError(f"explicit_positions out of range 1..{n_b}: {bad}")
        return pos

    if n_b == 0 or len(m_chain.xyz) == 0:
        return []

    if partner_chains is not None:
        ref_pts_list = [
            cx.all_chains[cid].xyz
            for cid in partner_chains
            if cid in cx.all_chains and len(cx.all_chains[cid].xyz) > 0
        ]
    else:
        ref_pts_list = [
            c.xyz for cid, c in cx.all_chains.items() if cid != mutable_chain and len(c.xyz) > 0
        ]
    if not ref_pts_list:
        return []
    ref_pts = np.vstack(ref_pts_list)

    d_mat = np.linalg.norm(m_chain.xyz[:, None, :] - ref_pts[None, :, :], axis=-1)
    atom_min_d = d_mat.min(axis=1)
    mutable: list[int] = []
    for r_idx in range(n_b):
        atom_idx = m_chain._res_atom_indices[r_idx]
        if len(atom_idx) > 0 and float(atom_min_d[atom_idx].min()) <= float(interface_dist):
            mutable.append(r_idx)
    return mutable


def compute_core_positions(seq_len: int, mutable_positions: Sequence[int]) -> list[int]:
    """Return sorted 0-based positions belonging to the frozen fold core."""
    mut_set = set(int(p) for p in mutable_positions)
    return [i for i in range(seq_len) if i not in mut_set]


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, mobile_centroid, target_centroid) aligning mobile onto target via Kabsch SVD."""
    mc = mobile.mean(axis=0)
    tc = target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    det_sign = float(np.sign(np.linalg.det(u @ vt)))
    if det_sign == 0.0:
        det_sign = 1.0
    d_mat = np.diag([1.0, 1.0, det_sign])
    rot = u @ d_mat @ vt
    return rot, mc, tc


def compute_core_ca_rmsd(
    pred_cx: ComplexCoords,
    ref_cx: ComplexCoords,
    core_positions: Sequence[int] | None = None,
    mutable_chain: str = "B",
    align_chain: str = "A",
) -> float:
    """Superpose `align_chain` CA onto reference (or mutable core if single-chain) and return core CA RMSD (A)."""
    p_bnd_chain = pred_cx.all_chains.get(mutable_chain, pred_cx.binder)
    r_bnd_chain = ref_cx.all_chains.get(mutable_chain, ref_cx.binder)
    p_bnd = p_bnd_chain.ca_coords()
    r_bnd = r_bnd_chain.ca_coords()

    if len(p_bnd) == 0 or len(p_bnd) != len(r_bnd):
        return float("nan")

    if core_positions:
        idx = [int(i) for i in core_positions if 0 <= int(i) < len(p_bnd)]
    else:
        idx = list(range(len(p_bnd)))
    if not idx:
        idx = list(range(len(p_bnd)))

    p_tgt_chain = pred_cx.all_chains.get(align_chain, pred_cx.target)
    r_tgt_chain = ref_cx.all_chains.get(align_chain, ref_cx.target)
    p_tgt = p_tgt_chain.ca_coords()
    r_tgt = r_tgt_chain.ca_coords()

    if align_chain != mutable_chain and len(p_tgt) > 0 and len(p_tgt) == len(r_tgt):
        rot, mc, tc = kabsch_align(p_tgt, r_tgt)
        aligned_bnd = (p_bnd[idx] - mc) @ rot + tc
    else:
        rot, mc, tc = kabsch_align(p_bnd[idx], r_bnd[idx])
        aligned_bnd = (p_bnd[idx] - mc) @ rot + tc

    sq = np.sum((aligned_bnd - r_bnd[idx]) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq)))


def count_glycan_hbonds(
    cx: ComplexCoords,
    cutoff: float = 3.5,
    mutable_chain: str = "B",
    glycan_chains: Sequence[str] | None = None,
) -> int:
    """Count polar H-bond contacts (mutable chain N/O to glycan N/O heavy atoms <= cutoff A)."""
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    if glycan_chains is not None:
        g_pts = [
            cx.all_chains[cid].xyz[[i for i, el in enumerate(cx.all_chains[cid].elements) if el in POLAR_ELEMENTS]]
            for cid in glycan_chains
            if cid in cx.all_chains
        ]
        g_pts = [p for p in g_pts if len(p) > 0]
        g_pol = np.vstack(g_pts) if g_pts else np.zeros((0, 3), dtype=np.float64)
    else:
        g_pol = cx.glycan_polar_xyz

    if len(g_pol) == 0 or len(m_chain.xyz) == 0:
        return 0
    b_pol_idx = [i for i, el in enumerate(m_chain.elements) if el in POLAR_ELEMENTS]
    if not b_pol_idx:
        return 0
    b_pol = m_chain.xyz[b_pol_idx]
    d_mat = np.linalg.norm(b_pol[:, None, :] - g_pol[None, :, :], axis=-1)
    return int(np.sum(d_mat <= float(cutoff)))


def count_glycan_contacts(
    cx: ComplexCoords,
    cutoff: float = 4.5,
    mutable_chain: str = "B",
) -> int:
    """Count mutable chain heavy atoms within `cutoff` (A) of any glycan heavy atom."""
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    g_xyz = cx.glycan_xyz
    if len(g_xyz) == 0 or len(m_chain.xyz) == 0:
        return 0
    d_mat = np.linalg.norm(m_chain.xyz[:, None, :] - g_xyz[None, :, :], axis=-1)
    return int(np.sum(d_mat.min(axis=1) <= float(cutoff)))


def count_nonstem_sugars_hit(
    cx: ComplexCoords,
    cutoff: float = 4.5,
    mutable_chain: str = "B",
    target_chain: str = "A",
    anchor_res_1based: int | None = None,
) -> int:
    """Count contacted non-chitobiose-stem sugar chains within `cutoff` (A) of `mutable_chain`."""
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    if len(m_chain.xyz) == 0 or not cx.glycan_chains:
        return 0

    nag_cids = sorted(
        cid
        for cid, ch in cx.glycan_chains.items()
        if len(ch.xyz) > 0 and any(rn == "NAG" for rn in ch.resnames)
    )
    stem_cids: set[str] = set()
    if nag_cids:
        t_chain = cx.all_chains.get(target_chain, cx.target)
        nd2_pt: np.ndarray | None = None
        if t_chain.n_res > 0:
            if anchor_res_1based is not None:
                for r_idx, rid in enumerate(t_chain.resids):
                    if rid == int(anchor_res_1based):
                        nd2_pt = t_chain._atom_coord(r_idx, "ND2")
                        if nd2_pt is None:
                            nd2_pt = t_chain._atom_coord(r_idx, "CA")
                        break
            if nd2_pt is None:
                for r_idx, rn in enumerate(t_chain.resnames):
                    if rn == "ASN":
                        nd2_pt = t_chain._atom_coord(r_idx, "ND2")
                        if nd2_pt is not None:
                            break

        def _c1_or_cent(ch: ChainAtoms) -> np.ndarray:
            for r_idx in range(ch.n_res):
                c1 = ch._atom_coord(r_idx, "C1")
                if c1 is not None:
                    return c1
            return ch.xyz.mean(axis=0)

        if nd2_pt is not None:
            r1 = min(nag_cids, key=lambda c: float(np.linalg.norm(_c1_or_cent(cx.glycan_chains[c]) - nd2_pt)))
        else:
            r1 = nag_cids[0]
        stem_cids.add(r1)
        rem_nags = [c for c in nag_cids if c != r1]
        if rem_nags:
            r1_ch = cx.glycan_chains[r1]
            r1_ref = r1_ch._atom_coord(0, "O4") if r1_ch.n_res > 0 else None
            if r1_ref is None:
                r1_ref = r1_ch.xyz.mean(axis=0)
            r2 = min(rem_nags, key=lambda c: float(np.linalg.norm(_c1_or_cent(cx.glycan_chains[c]) - r1_ref)))
            stem_cids.add(r2)

    hit = 0
    cut = float(cutoff)
    for cid, g_chain in cx.glycan_chains.items():
        if cid in stem_cids or len(g_chain.xyz) == 0:
            continue
        d_mat = np.linalg.norm(m_chain.xyz[:, None, :] - g_chain.xyz[None, :, :], axis=-1)
        if float(d_mat.min()) <= cut:
            hit += 1
    return hit


def _ring_centroid_and_normal(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cent = pts.mean(axis=0)
    centered = pts - cent
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    nrm = float(np.linalg.norm(normal))
    if nrm > 1e-12:
        normal = normal / nrm
    return cent, normal


def count_ch_pi_contacts(
    cx: ComplexCoords,
    dist_cutoff: float = 5.0,
    tilt_deg: float = 35.0,
    mutable_chain: str = "B",
) -> int:
    """Count PHE/TYR/TRP residues in `mutable_chain` forming CH-pi stacking with a sugar pyranose ring."""
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    if m_chain.n_res == 0 or not cx.glycan_chains:
        return 0

    sugar_rings: list[tuple[np.ndarray, np.ndarray]] = []
    for g_chain in cx.glycan_chains.values():
        if len(g_chain.xyz) < 3:
            continue
        ring_idx = [i for i, an in enumerate(g_chain.atom_names) if an in PYRANOSE_RING_ATOMS]
        pts = g_chain.xyz[ring_idx] if len(ring_idx) >= 4 else g_chain.xyz
        if len(pts) >= 3:
            sugar_rings.append(_ring_centroid_and_normal(pts))

    if not sugar_rings:
        return 0

    cos_cut = math.cos(math.radians(float(tilt_deg)))
    d_cut = float(dist_cutoff)
    contacts = 0
    for r_idx, rname in enumerate(m_chain.resnames):
        ring_names = AROMATIC_RING_ATOMS.get(rname)
        if ring_names is None:
            continue
        pts_list: list[np.ndarray] = []
        for an in ring_names:
            pt = m_chain._atom_coord(r_idx, an)
            if pt is not None:
                pts_list.append(pt)
        if len(pts_list) < 4:
            continue
        acent, anorm = _ring_centroid_and_normal(np.asarray(pts_list, dtype=np.float64))
        if any(
            float(np.linalg.norm(acent - scent)) <= d_cut
            and abs(float(np.dot(anorm, snorm))) >= cos_cut
            for scent, snorm in sugar_rings
        ):
            contacts += 1
    return contacts


def count_target_protein_contacts(
    cx: ComplexCoords,
    cutoff: float = 4.5,
    target_chain: str = "A",
    mutable_chain: str = "B",
) -> int:
    """Count distinct residues in `target_chain` with >= 1 heavy atom within `cutoff` (A) of `mutable_chain`."""
    t_chain = cx.all_chains.get(target_chain, cx.target)
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    if t_chain.n_res == 0 or len(t_chain.xyz) == 0 or len(m_chain.xyz) == 0:
        return 0
    d_mat = np.linalg.norm(t_chain.xyz[:, None, :] - m_chain.xyz[None, :, :], axis=-1)
    atom_hit = d_mat.min(axis=1) <= float(cutoff)
    cnt = 0
    for r_idx in range(t_chain.n_res):
        a_idx = t_chain._res_atom_indices[r_idx]
        if len(a_idx) > 0 and bool(atom_hit[a_idx].any()):
            cnt += 1
    return cnt


def _cremer_pople_pucker(ring_o5_to_c5: np.ndarray) -> tuple[float, float]:
    """Compute Cremer-Pople (Q, theta_deg) for ordered 6-ring atoms (O5, C1, C2, C3, C4, C5)."""
    if ring_o5_to_c5.shape != (6, 3):
        return 0.0, float("nan")
    rel = ring_o5_to_c5 - ring_o5_to_c5.mean(axis=0)
    j = np.arange(6, dtype=np.float64)
    ang = 2.0 * math.pi * j / 6.0
    rp = (rel * np.sin(ang)[:, None]).sum(axis=0)
    rpp = (rel * np.cos(ang)[:, None]).sum(axis=0)
    n_vec = np.cross(rp, rpp)
    n_len = float(np.linalg.norm(n_vec))
    if n_len < 1e-9:
        return 0.0, float("nan")
    z = rel @ (n_vec / n_len)
    ang2 = 4.0 * math.pi * j / 6.0
    q2_cos = math.sqrt(1.0 / 3.0) * float(np.sum(z * np.cos(ang2)))
    q2_sin = -math.sqrt(1.0 / 3.0) * float(np.sum(z * np.sin(ang2)))
    q2 = math.hypot(q2_cos, q2_sin)
    alt = np.where((np.arange(6) % 2) == 0, 1.0, -1.0)
    q3 = math.sqrt(1.0 / 6.0) * float(np.sum(alt * z))
    return float(math.hypot(q2, q3)), float(math.degrees(math.atan2(q2, q3)))


def compute_glyco_stereo_ok(
    cx: ComplexCoords,
    anchor_res_1based: int | None = None,
    expected_n_sugars: int | None = None,
    target_chain: str = "A",
) -> int:
    """Validate N-glycan attachment, no O1 leak, glycosidic bonds, Cremer-Pople pucker, and C1 anomer.

    Returns 1 if valid (or bare without expected glycan), 0 on missing atoms or stereochemical violation.
    """
    if not cx.glycan_chains:
        return 1 if expected_n_sugars in (None, 0) else 0
    if expected_n_sugars is not None and len(cx.glycan_chains) != int(expected_n_sugars):
        return 0

    sugars: dict[str, tuple[str, dict[str, np.ndarray]]] = {}
    for cid, g_chain in cx.glycan_chains.items():
        if g_chain.n_res == 0:
            return 0
        ccd = g_chain.resnames[0]
        amap: dict[str, np.ndarray] = {}
        for idx, aname in enumerate(g_chain.atom_names):
            amap[aname] = g_chain.xyz[idx]
        if "O1" in amap:
            return 0
        sugars[cid] = (ccd, amap)

    t_chain = cx.all_chains.get(target_chain, cx.target)
    if t_chain.n_res == 0:
        return 0

    cg_xyz: np.ndarray | None = None
    nd2_xyz: np.ndarray | None = None
    if anchor_res_1based is not None:
        for r_idx, rid in enumerate(t_chain.resids):
            if rid == int(anchor_res_1based) and t_chain.resnames[r_idx] == "ASN":
                cg_xyz = t_chain._atom_coord(r_idx, "CG")
                nd2_xyz = t_chain._atom_coord(r_idx, "ND2")
                break
    else:
        best_asn_d = float("inf")
        for r_idx, rn in enumerate(t_chain.resnames):
            if rn != "ASN":
                continue
            cg = t_chain._atom_coord(r_idx, "CG")
            nd2 = t_chain._atom_coord(r_idx, "ND2")
            if cg is None or nd2 is None:
                continue
            for _, amap in sugars.values():
                if "C1" in amap:
                    d = float(np.linalg.norm(nd2 - amap["C1"]))
                    if d < best_asn_d:
                        best_asn_d = d
                        cg_xyz, nd2_xyz = cg, nd2

    if cg_xyz is None or nd2_xyz is None:
        return 0

    r1_cid: str | None = None
    r1_nd2_c1 = float("inf")
    for cid, (_, amap) in sugars.items():
        if "C1" in amap:
            d = float(np.linalg.norm(nd2_xyz - amap["C1"]))
            if d < r1_nd2_c1:
                r1_nd2_c1 = d
                r1_cid = cid
    if r1_cid is None:
        return 0

    r1_c1 = sugars[r1_cid][1]["C1"]
    v1 = cg_xyz - nd2_xyz
    v2 = r1_c1 - nd2_xyz
    denom_a = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom_a < 1e-9:
        return 0
    angle_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / denom_a)))))
    if not (110.0 <= angle_deg <= 130.0):
        return 0

    bond_lengths: list[float] = [r1_nd2_c1]
    partner_xyz: dict[str, np.ndarray] = {r1_cid: nd2_xyz}
    for cid, (_, amap) in sugars.items():
        if cid == r1_cid:
            continue
        if "C1" not in amap:
            return 0
        c1_xyz = amap["C1"]
        best_d = float("inf")
        best_o: np.ndarray | None = None
        for o_cid, (_, o_amap) in sugars.items():
            if o_cid == cid:
                continue
            for oname in GLYCOSIDIC_ACCEPTOR_ATOMS:
                if oname in o_amap:
                    d = float(np.linalg.norm(c1_xyz - o_amap[oname]))
                    if d < best_d:
                        best_d = d
                        best_o = o_amap[oname]
        if best_o is None:
            return 0
        bond_lengths.append(best_d)
        partner_xyz[cid] = best_o

    if not bond_lengths or any(d < 1.35 or d > 1.55 for d in bond_lengths):
        return 0

    cp_order = ("O5", "C1", "C2", "C3", "C4", "C5")
    for cid, (ccd, amap) in sugars.items():
        if any(an not in amap for an in cp_order):
            return 0
        ring_pts = np.asarray([amap[an] for an in PYRANOSE_RING_ATOMS], dtype=np.float64)
        centered = ring_pts - ring_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        plane_rmsd = float(np.sqrt(np.mean((centered @ vt[-1]) ** 2)))
        if plane_rmsd < 0.15:
            return 0

        q_val, theta_deg = _cremer_pople_pucker(np.asarray([amap[an] for an in cp_order], dtype=np.float64))
        exp_chair, exp_sign = EXPECTED_SUGAR_STEREO.get(ccd, ("4C1", +1))
        if math.isnan(theta_deg) or q_val < 0.35:
            return 0
        if exp_chair == "4C1" and theta_deg > 45.0:
            return 0
        if exp_chair != "4C1" and theta_deg < 135.0:
            return 0

        x_xyz = partner_xyz.get(cid)
        if x_xyz is None:
            return 0
        vo5 = amap["O5"] - amap["C1"]
        vc2 = amap["C2"] - amap["C1"]
        vx = x_xyz - amap["C1"]
        denom_v = float(np.linalg.norm(vo5) * np.linalg.norm(vc2) * np.linalg.norm(vx))
        if denom_v < 1e-9:
            return 0
        ascore = float(np.dot(np.cross(vo5, vc2), vx) / denom_v)
        if exp_sign > 0 and ascore < 0.15:
            return 0
        if exp_sign < 0 and ascore > -0.15:
            return 0

    return 1


def compute_rg_ratio(chain: ChainAtoms) -> float:
    """Radius of gyration relative to compact globule baseline 2.2 * N^0.38."""
    ca = chain.ca_coords()
    if len(ca) == 0:
        return 1.0
    rg = float(np.sqrt(np.mean(np.sum((ca - ca.mean(axis=0)) ** 2, axis=1))))
    return rg / (2.2 * (len(ca) ** 0.38))


def compute_aspect_ratio(chain: ChainAtoms) -> float:
    """CA principal-axis aspect ratio sqrt(lambda_max / lambda_mid) >= 1.0."""
    ca = chain.ca_coords()
    if len(ca) < 4:
        return 1.0
    centered = ca - ca.mean(axis=0)
    cov = (centered.T @ centered) / max(1, len(ca))
    evals = np.linalg.eigvalsh(cov)
    evals = np.maximum(evals, 1e-6)
    return float(math.sqrt(float(evals[-1]) / float(evals[1])))


def compute_contact_score(
    cx: ComplexCoords,
    mutable_chain: str = "B",
    cutoff: float = 14.0,
    k: int = 10,
) -> float:
    """ColabDesign-style CB-CB interface contact loss in [0, 1] (0 = tight contact, 1 = no contact)."""
    m_chain = cx.all_chains.get(mutable_chain, cx.binder)
    m_cb = m_chain.cb_or_ca_coords()
    if len(m_cb) == 0:
        return 1.0
    other_cbs = [
        c.cb_or_ca_coords()
        for cid, c in cx.all_chains.items()
        if cid != mutable_chain and c.n_res > 0
    ]
    if not other_cbs:
        return 0.0
    o_cb = np.vstack(other_cbs)
    d = np.linalg.norm(m_cb[:, None, :] - o_cb[None, :, :], axis=-1).min(axis=1)
    return float(np.mean(np.minimum(np.sort(d)[:k], cutoff)) / cutoff)


def compute_plddt(
    confidences: Mapping[str, Any],
    chain_id: str | Sequence[str] | None = None,
) -> float:
    """Compute mean atom pLDDT normalized to [0, 1] across `chain_id`(s) if provided, else full complex."""
    plddts = confidences.get("atom_plddts")
    if not plddts:
        return 0.0
    arr = np.asarray(plddts, dtype=np.float64)
    if chain_id is not None and confidences.get("atom_chain_ids"):
        wanted = {chain_id} if isinstance(chain_id, str) else set(chain_id)
        mask = np.array([c in wanted for c in confidences["atom_chain_ids"]], dtype=bool)
        if mask.any():
            arr = arr[mask]
    mean_val = float(np.mean(arr))
    return mean_val / 100.0 if mean_val > 1.0 else mean_val


def extract_token_coords(
    cx: ComplexCoords,
    token_chain_ids: Sequence[str],
    protein_chains: Sequence[str] = ("A", "B"),
) -> np.ndarray | None:
    """Build [N_tok, 3] representative token coordinates aligned to AF3 `token_chain_ids`.

    Protein chains use CB (or CA for GLY); ligand/glycan chains use each heavy atom coordinate in order.
    """
    if not token_chain_ids:
        return None
    prot_set = set(protein_chains)
    per_chain: dict[str, np.ndarray] = {}
    for cid, c_atoms in cx.all_chains.items():
        if cid in prot_set:
            per_chain[cid] = c_atoms.cb_or_ca_coords()
        else:
            per_chain[cid] = c_atoms.xyz

    cursors: dict[str, int] = {}
    out: list[np.ndarray] = []
    for cid in token_chain_ids:
        arr = per_chain.get(cid)
        pos = cursors.get(cid, 0)
        if arr is None or pos >= len(arr):
            return None
        out.append(arr[pos])
        cursors[cid] = pos + 1
    return np.asarray(out, dtype=np.float64)


def _parse_residue_range(spec: Sequence[int] | str) -> tuple[int, int]:
    if isinstance(spec, str):
        cleaned = spec.replace("-", ",")
        parts = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
    else:
        parts = [int(x) for x in spec]
    if len(parts) != 2:
        raise ValueError(f"Residue range must have 2 elements (start, end), got {spec!r}")
    lo, hi = parts[0], parts[1]
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


@dataclasses.dataclass(frozen=True)
class StericClashRef:
    """Pre-parsed context reference structure split into aligned receptor CA and steric obstacle atoms."""

    ref_ca: np.ndarray  # [n_ref, 3] CA atoms used to align predicted receptor onto context reference
    obstacle_heavy_xyz: np.ndarray  # [n_hv, 3] steric obstacle heavy atoms
    obstacle_ca_xyz: np.ndarray  # [n_ca, 3] steric obstacle CA atoms
    ref_residues: tuple[int, int] | None


def load_steric_clash_ref(
    ref_cif: str | Path,
    ref_residues: Sequence[int] | str | None = None,
    chain_id: str = "A",
) -> StericClashRef:
    """Load context reference CIF/PDB once, splitting aligned receptor CA vs flanking/context obstacle atoms.

    If `ref_residues` is `(start, end)` (1-based inclusive), residues in `[start, end]` on `chain_id`
    form the alignment anchor and all atoms outside `[start, end]` (plus any other non-binder chains)
    form the steric obstacle. If `ref_residues` is None, all CA atoms on `chain_id` form the anchor
    and all other chains (excluding `"B"`) form the steric obstacle.
    """
    cx = load_structure(ref_cif, target_chain=chain_id, binder_chain="B")
    ref_chain = cx.all_chains.get(chain_id, cx.target)
    if ref_chain.n_res == 0:
        raise ValueError(f"Chain {chain_id!r} not found or empty in steric clash reference {ref_cif}")

    parsed_range = _parse_residue_range(ref_residues) if ref_residues is not None else None
    ref_ca_list: list[np.ndarray] = []
    obs_ca_list: list[np.ndarray] = []
    obs_hv_parts: list[np.ndarray] = []

    if parsed_range is not None:
        lo, hi = parsed_range
        obs_hv_indices: list[int] = []
        for r_idx, rid in enumerate(ref_chain.resids):
            atom_idx = ref_chain._res_atom_indices[r_idx]
            ca_pt = ref_chain._atom_coord(r_idx, "CA")
            if lo <= rid <= hi:
                if ca_pt is not None:
                    ref_ca_list.append(ca_pt)
            else:
                obs_hv_indices.extend(int(i) for i in atom_idx)
                if ca_pt is not None:
                    obs_ca_list.append(ca_pt)
        if obs_hv_indices:
            obs_hv_parts.append(ref_chain.xyz[obs_hv_indices])
    else:
        for r_idx in range(ref_chain.n_res):
            ca_pt = ref_chain._atom_coord(r_idx, "CA")
            if ca_pt is not None:
                ref_ca_list.append(ca_pt)

    for cid, ch in cx.all_chains.items():
        if cid in (chain_id, "B") or ch.n_res == 0:
            continue
        if len(ch.xyz) > 0:
            obs_hv_parts.append(ch.xyz)
        ca_extra = ch.ca_coords()
        if len(ca_extra) > 0:
            obs_ca_list.extend(ca_extra)

    ref_ca = np.asarray(ref_ca_list, dtype=np.float64).reshape((-1, 3))
    obs_ca = np.asarray(obs_ca_list, dtype=np.float64).reshape((-1, 3))
    obs_hv = np.vstack(obs_hv_parts) if obs_hv_parts else np.zeros((0, 3), dtype=np.float64)
    return StericClashRef(
        ref_ca=ref_ca,
        obstacle_heavy_xyz=obs_hv,
        obstacle_ca_xyz=obs_ca,
        ref_residues=parsed_range,
    )


def compute_steric_clashes(
    pred_cx: ComplexCoords,
    steric_ref: StericClashRef,
    target_chain: str = "A",
    binder_chain: str = "B",
    hv_cutoff: float = 2.5,
    ca_cutoff: float = 3.8,
    pred_residues: Sequence[int] | str | None = None,
) -> dict[str, int]:
    """Rigidly align predicted target CA onto `steric_ref.ref_ca` and count binder-vs-obstacle clashes."""
    t_chain = pred_cx.all_chains.get(target_chain, pred_cx.target)
    b_chain = pred_cx.all_chains.get(binder_chain, pred_cx.binder)
    if (
        t_chain.n_res == 0
        or b_chain.n_res == 0
        or len(steric_ref.ref_ca) == 0
        or len(steric_ref.obstacle_heavy_xyz) == 0
    ):
        return {
            "steric_hv25": 0,
            "steric_ca38": 0,
            "steric_hv_buf": 0,
            "steric_ca_buf": 0,
            "intact_clash_ok": 1,
        }

    if pred_residues is not None:
        lo, hi = _parse_residue_range(pred_residues)
        sub_ca = [
            pt
            for r_idx, rid in enumerate(t_chain.resids)
            if lo <= rid <= hi and (pt := t_chain._atom_coord(r_idx, "CA")) is not None
        ]
        if len(sub_ca) != len(steric_ref.ref_ca):
            return {
                "steric_hv25": 0,
                "steric_ca38": 0,
                "steric_hv_buf": 0,
                "steric_ca_buf": 0,
                "intact_clash_ok": 1,
            }
        p_ca = np.asarray(sub_ca, dtype=np.float64)
    else:
        p_ca = t_chain.ca_coords()
        if len(p_ca) != len(steric_ref.ref_ca) and steric_ref.ref_residues is not None:
            lo, hi = steric_ref.ref_residues
            sub_ca = [
                pt
                for r_idx, rid in enumerate(t_chain.resids)
                if lo <= rid <= hi and (pt := t_chain._atom_coord(r_idx, "CA")) is not None
            ]
            if len(sub_ca) == len(steric_ref.ref_ca):
                p_ca = np.asarray(sub_ca, dtype=np.float64)
            else:
                return {
                    "steric_hv25": 0,
                    "steric_ca38": 0,
                    "steric_hv_buf": 0,
                    "steric_ca_buf": 0,
                    "intact_clash_ok": 1,
                }
        elif len(p_ca) != len(steric_ref.ref_ca):
            return {
                "steric_hv25": 0,
                "steric_ca38": 0,
                "steric_hv_buf": 0,
                "steric_ca_buf": 0,
                "intact_clash_ok": 1,
            }

    rot, mc, tc = kabsch_align(p_ca, steric_ref.ref_ca)
    b_hv_aligned = (b_chain.xyz - mc) @ rot + tc
    d2_hv = np.sum(
        (b_hv_aligned[:, None, :] - steric_ref.obstacle_heavy_xyz[None, :, :]) ** 2,
        axis=-1,
    )
    steric_hv25 = int(np.sum(d2_hv < 2.5 ** 2))
    steric_hv_buf = int(np.sum(d2_hv < float(hv_cutoff) ** 2))

    b_ca = b_chain.ca_coords()
    if len(b_ca) > 0 and len(steric_ref.obstacle_ca_xyz) > 0:
        b_ca_aligned = (b_ca - mc) @ rot + tc
        d2_ca = np.sum(
            (b_ca_aligned[:, None, :] - steric_ref.obstacle_ca_xyz[None, :, :]) ** 2,
            axis=-1,
        )
        steric_ca38 = int(np.sum(d2_ca < 3.8 ** 2))
        steric_ca_buf = int(np.sum(d2_ca < float(ca_cutoff) ** 2))
    else:
        steric_ca38 = 0
        steric_ca_buf = 0

    return {
        "steric_hv25": steric_hv25,
        "steric_ca38": steric_ca38,
        "steric_hv_buf": steric_hv_buf,
        "steric_ca_buf": steric_ca_buf,
        "intact_clash_ok": int(steric_hv25 == 0 and steric_ca38 == 0),
    }
