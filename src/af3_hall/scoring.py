"""Composable @register_loss / @register_multistate_loss registry & Dunbrack/Overath ipSAE_min."""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import AA3_TO_1
from .proposals import cp_count, sequon_count
from .structure import (
    ComplexCoords,
    StericClashRef,
    compute_aspect_ratio,
    compute_contact_score,
    compute_core_ca_rmsd,
    compute_glyco_stereo_ok,
    compute_plddt,
    compute_rg_ratio,
    compute_steric_clashes,
    count_ch_pi_contacts,
    count_glycan_contacts,
    count_glycan_hbonds,
    count_nonstem_sugars_hit,
    count_target_protein_contacts,
    extract_token_coords,
)

_GLYCOFORM_SUGAR_COUNTS: dict[str, int] = {
    "A1G0F": 7,
    "A3G0": 9,
    "A3G0F": 9,
    "A3G3F": 12,
    "A4G2F": 11,
}

PAE_MAX = 31.75  # AF3 PAE scale (0..31.75 A)

LossFn = Callable[[dict[str, Any]], float]
MultistateLossFn = Callable[[dict[str, dict[str, Any]], Mapping[str, Any]], tuple[float, dict[str, Any]]]

LOSS_REGISTRY: dict[str, LossFn] = {}
MULTISTATE_LOSS_REGISTRY: dict[str, MultistateLossFn] = {}


def register_loss(name: str) -> Callable[[LossFn], LossFn]:
    """Decorator registering a per-state loss function (lower = better) by name."""

    def _decorator(fn: LossFn) -> LossFn:
        LOSS_REGISTRY[name] = fn
        return fn

    return _decorator


def register_multistate_loss(name: str) -> Callable[[MultistateLossFn], MultistateLossFn]:
    """Decorator registering a cross-state loss plugin returning (loss, extras_dict)."""

    def _decorator(fn: MultistateLossFn) -> MultistateLossFn:
        MULTISTATE_LOSS_REGISTRY[name] = fn
        return fn

    return _decorator


# ---------------- Exact Dunbrack/Overath ipSAE_min (PAE<10, dist<10, d0>=1.0) ----------------
def d0_protein(n_valid: int) -> float:
    """Dunbrack/Overath d0 formula floored at 1.0 A: max(1.0, 1.24*(max(n,27)-15)^(1/3) - 1.8)."""
    eff = max(int(n_valid), 27)
    return max(1.0, 1.24 * (eff - 15.0) ** (1.0 / 3.0) - 1.8)


def compute_ipsae_directed(
    pae_block: np.ndarray,
    dist_block: np.ndarray | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> float:
    """Directed ipSAE = max_i mean_{j in valid_i} 1 / (1 + (pae[i,j] / d0(n_valid_i))^2)."""
    if pae_block.size == 0 or pae_block.shape[0] == 0 or pae_block.shape[1] == 0:
        return 0.0

    valid_mask = pae_block < float(pae_cutoff)
    if dist_block is not None:
        if dist_block.shape != pae_block.shape:
            raise ValueError(
                f"dist_block shape {dist_block.shape} != pae_block shape {pae_block.shape}"
            )
        valid_mask = valid_mask & (dist_block < float(dist_cutoff))

    best = 0.0
    for i in range(pae_block.shape[0]):
        row_valid = valid_mask[i]
        n_valid = int(row_valid.sum())
        if n_valid == 0:
            continue
        d0 = d0_protein(n_valid)
        vals = pae_block[i, row_valid]
        score = float(np.mean(1.0 / (1.0 + (vals / d0) ** 2)))
        if score > best:
            best = score
    return float(best)


def compute_ipsae_min(
    pae: np.ndarray,
    token_chain_ids: Sequence[str],
    token_coords: np.ndarray | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    target_chain: str = "A",
    binder_chain: str = "B",
    target_chains: Sequence[str] | None = None,
    glycan_chains: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute Dunbrack/Overath ipSAE_min (PAE<10, dist<10, d0>=1.0) for mutable chain vs partner/glycan/ligand."""
    pae_arr = np.asarray(pae, dtype=np.float64)
    n_tok = pae_arr.shape[0]
    if pae_arr.ndim != 2 or pae_arr.shape[0] != pae_arr.shape[1]:
        raise ValueError(f"PAE must be square 2D matrix, got {pae_arr.shape}")
    if len(token_chain_ids) != n_tok:
        raise ValueError(
            f"token_chain_ids length {len(token_chain_ids)} != PAE size {n_tok}"
        )

    dist_mat: np.ndarray | None = None
    if token_coords is not None:
        coords = np.asarray(token_coords, dtype=np.float64)
        if coords.shape != (n_tok, 3):
            raise ValueError(f"token_coords shape {coords.shape} != ({n_tok}, 3)")
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    b_idx = [i for i, c in enumerate(token_chain_ids) if c == binder_chain]
    t_set = set(target_chains) if target_chains is not None else {target_chain}
    t_idx = [i for i, c in enumerate(token_chain_ids) if c in t_set and c != binder_chain]
    if glycan_chains is not None:
        g_set = set(glycan_chains)
        g_idx = [i for i, c in enumerate(token_chain_ids) if c in g_set and c != binder_chain]
    else:
        g_idx = [
            i
            for i, c in enumerate(token_chain_ids)
            if c not in t_set and c != binder_chain
        ]
    rec_idx = t_idx + g_idx

    if not b_idx or not rec_idx:
        return {
            "ipsae_min": 0.0,
            "ipsae_max": 0.0,
            "ipsae_b2r": 0.0,
            "ipsae_r2b": 0.0,
            "ipsae_glycan_min": 0.0,
        }

    d_b2r = dist_mat[np.ix_(b_idx, rec_idx)] if dist_mat is not None else None
    d_r2b = dist_mat[np.ix_(rec_idx, b_idx)] if dist_mat is not None else None
    b2r = compute_ipsae_directed(
        pae_arr[np.ix_(b_idx, rec_idx)], d_b2r, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
    )
    r2b = compute_ipsae_directed(
        pae_arr[np.ix_(rec_idx, b_idx)], d_r2b, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
    )

    if g_idx:
        d_b2g = dist_mat[np.ix_(b_idx, g_idx)] if dist_mat is not None else None
        d_g2b = dist_mat[np.ix_(g_idx, b_idx)] if dist_mat is not None else None
        b2g = compute_ipsae_directed(
            pae_arr[np.ix_(b_idx, g_idx)], d_b2g, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
        )
        g2b = compute_ipsae_directed(
            pae_arr[np.ix_(g_idx, b_idx)], d_g2b, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff
        )
        gly_min = min(b2g, g2b)
    else:
        gly_min = 0.0

    return {
        "ipsae_min": float(min(b2r, r2b)),
        "ipsae_max": float(max(b2r, r2b)),
        "ipsae_b2r": float(b2r),
        "ipsae_r2b": float(r2b),
        "ipsae_glycan_min": float(gly_min),
    }


def extract_binder_iptm(summary: Mapping[str, Any], chain_idx: int = 1) -> float:
    """Extract chain ipTM from AF3 summary_confidences (`chain_iptm[chain_idx]`, fallback `iptm`)."""
    chain_iptm = summary.get("chain_iptm")
    if (
        isinstance(chain_iptm, Sequence)
        and len(chain_iptm) > chain_idx
        and chain_iptm[chain_idx] is not None
    ):
        return float(chain_iptm[chain_idx])
    return float(summary.get("iptm") or 0.0)


def _mean_inter_chain_pae(pae: np.ndarray, token_chain_ids: Sequence[str]) -> float:
    if pae.size == 0 or len(token_chain_ids) != pae.shape[0]:
        return float(np.mean(pae)) if pae.size else PAE_MAX
    chains = np.asarray(token_chain_ids)
    inter = chains[:, None] != chains[None, :]
    if not inter.any():
        return float(np.mean(pae))
    return float(np.mean(pae[inter]))


def _pae_asymmetry(
    pae: np.ndarray,
    token_chain_ids: Sequence[str],
    binder_chain: str,
) -> float:
    if pae.size == 0 or len(token_chain_ids) != pae.shape[0]:
        return 0.0
    b_idx = [i for i, c in enumerate(token_chain_ids) if c == binder_chain]
    r_idx = [i for i, c in enumerate(token_chain_ids) if c != binder_chain]
    if not b_idx or not r_idx:
        return 0.0
    b2r = float(np.mean(pae[np.ix_(b_idx, r_idx)]))
    r2b = float(np.mean(pae[np.ix_(r_idx, b_idx)]))
    return abs(b2r - r2b) / PAE_MAX


# ---------------- Registered AFhall Standard Losses ----------------
@register_loss("plddt")
def _loss_plddt(ctx: dict[str, Any]) -> float:
    return float(1.0 - ctx["plddt"])


@register_loss("ptm")
def _loss_ptm(ctx: dict[str, Any]) -> float:
    return float(1.0 - ctx["ptm"])


@register_loss("iptm")
def _loss_iptm(ctx: dict[str, Any]) -> float:
    return float(1.0 - ctx["iptm"])


@register_loss("pae")
def _loss_pae(ctx: dict[str, Any]) -> float:
    return float(ctx["pae_mean"] / PAE_MAX)


@register_loss("ipae")
def _loss_ipae(ctx: dict[str, Any]) -> float:
    return float(ctx["ipae_mean"] / PAE_MAX)


@register_loss("pae_asym")
def _loss_pae_asym(ctx: dict[str, Any]) -> float:
    return float(ctx.get("pae_asym", 0.0))


@register_loss("dual")
def _loss_dual(ctx: dict[str, Any]) -> float:
    return float(0.5 * ((1.0 - ctx["plddt"]) + (1.0 - ctx["ptm"])))


@register_loss("rmsd")
def _loss_rmsd(ctx: dict[str, Any]) -> float:
    return float(ctx["rmsd_core"])


@register_loss("contact")
def _loss_contact(ctx: dict[str, Any]) -> float:
    return float(ctx["contact_score"])


@register_loss("rg")
def _loss_rg(ctx: dict[str, Any]) -> float:
    return float(max(0.0, ctx["rg_ratio"] - 1.0))


@register_loss("aspect_ratio")
def _loss_aspect_ratio(ctx: dict[str, Any]) -> float:
    return float(max(0.0, ctx.get("aspect_ratio", 1.0) - 1.0))


# ---------------- Registered All-Atom / Glyco / Steric Plugins ----------------
@register_loss("ipsae_min")
def _loss_ipsae_min(ctx: dict[str, Any]) -> float:
    return float(1.0 - ctx["ipsae_min"])


@register_loss("glycan_hbonds")
def _loss_glycan_hbonds(ctx: dict[str, Any]) -> float:
    return float(-ctx["hbonds_glycan"])


@register_loss("steric_clash")
def _loss_steric_clash(ctx: dict[str, Any]) -> float:
    hv = ctx.get("steric_hv_buf", ctx.get("steric_hv25", 0))
    ca = ctx.get("steric_ca_buf", ctx.get("steric_ca38", 0))
    return float(hv + 0.5 * ca)


def compute_objective_J(
    ipsae_min_glyco: float,
    iptm_glyco: float,
    iptm_bare: float,
    ipsae_min_bare: float,
    hbonds_glycan: int | float,
    rmsd_core: float,
    cfg: Mapping[str, Any],
    *,
    steric_hv25: int = 0,
    steric_ca38: int = 0,
    glycan_contacts_45: int | float = 0,
    nonstem_sugars_hit: int | float = 0,
    ch_pi_contacts: int | float = 0,
    binder_plddt: float = 0.0,
    glyco_stereo_ok: int = 1,
    rg_ratio: float = 1.0,
    target_protein_contacts: int | float = 0,
    n_forbidden_aa: int = 0,
) -> float:
    """Compute composite glyco-delta objective J (higher = better) with opt-in additive terms."""
    w1 = float(cfg.get("w1_ipsae_glyco", 1.0))
    w2 = float(cfg.get("w2_delta_iptm", 1.0))
    w3 = float(cfg.get("w3_delta_ipsae", 1.0))
    w4 = float(cfg.get("w4_hbonds_glycan", 0.05))
    w5 = float(cfg.get("w5_rmsd_core", 0.10))
    steric_w = float(cfg.get("steric_clash_weight", 0.0))

    w_gc = float(cfg.get("w_glycan_contacts", 0.0))
    w_ns = float(cfg.get("w_nonstem_sugars", 0.0))
    w_cp = float(cfg.get("w_ch_pi", 0.0))
    w_pl = float(cfg.get("w_binder_plddt", 0.0))
    w_st = float(cfg.get("w_stereo_penalty", 0.0))
    w_rg = float(cfg.get("w_rg_penalty", 0.0))
    w_tc = float(cfg.get("w_excess_target_contacts", 0.0))
    tc_max = int(cfg.get("target_max_contacts", 14))
    w_fa = float(cfg.get("w_forbidden_aa", 0.0))

    delta_iptm = float(iptm_glyco) - float(iptm_bare)
    delta_ipsae = float(ipsae_min_glyco) - float(ipsae_min_bare)
    safe_rmsd = 0.0 if (rmsd_core is None or math.isnan(float(rmsd_core))) else float(rmsd_core)
    steric_pen = steric_w * (float(steric_hv25) + 0.5 * float(steric_ca38))

    j_val = (
        w1 * float(ipsae_min_glyco)
        + w2 * delta_iptm
        + w3 * delta_ipsae
        + w4 * float(hbonds_glycan)
        - w5 * safe_rmsd
        - steric_pen
        + w_gc * min(float(glycan_contacts_45) / 20.0, 1.5)
        + w_ns * min(float(nonstem_sugars_hit) / 3.0, 2.0)
        + w_cp * min(float(ch_pi_contacts), 4.0)
        + w_pl * float(binder_plddt)
        - w_st * float(1 - int(glyco_stereo_ok))
        - w_rg * max(0.0, float(rg_ratio) - 1.0)
        - w_tc * float(max(0.0, float(target_protein_contacts) - float(tc_max)))
        - w_fa * float(n_forbidden_aa)
    )
    return float(j_val)


@register_multistate_loss("glyco_delta")
def _multistate_glyco_delta(
    state_ctx: dict[str, dict[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """2-state (glyco - bare) objective plugin maximizing J (returns loss = -J)."""
    glyco = state_ctx.get("glyco")
    if glyco is None:
        glyco = next(iter(state_ctx.values()))
    bare = state_ctx.get("bare")
    ipsae_g = float(glyco["ipsae_min"])
    iptm_g = float(glyco["iptm"])
    hbonds_g = int(glyco["hbonds_glycan"])
    rmsd_core = float(glyco["rmsd_core"])
    hv25 = int(glyco.get("steric_hv25", 0))
    ca38 = int(glyco.get("steric_ca38", 0))
    hv_buf = int(glyco.get("steric_hv_buf", hv25))
    ca_buf = int(glyco.get("steric_ca_buf", ca38))
    clash_ok = int(glyco.get("intact_clash_ok", int(hv25 == 0 and ca38 == 0)))

    gc_45 = int(glyco["glycan_contacts_45"])
    nonstem_hit = int(glyco.get("nonstem_sugars_hit", 0))
    ch_pi = int(glyco.get("ch_pi_contacts", 0))
    b_plddt = float(glyco.get("binder_plddt", glyco.get("plddt", 0.0)))
    stereo_ok = int(glyco.get("glyco_stereo_ok", 1))
    rg_ratio = float(glyco.get("rg_ratio", 1.0))
    t_contacts = int(glyco.get("target_protein_contacts", 0))
    n_forb = int(glyco.get("n_forbidden_aa", 0))

    ipsae_b = float(bare["ipsae_min"]) if bare is not None else 0.0
    iptm_b = float(bare["iptm"]) if bare is not None else 0.0

    j_val = compute_objective_J(
        ipsae_min_glyco=ipsae_g,
        iptm_glyco=iptm_g,
        iptm_bare=iptm_b,
        ipsae_min_bare=ipsae_b,
        hbonds_glycan=hbonds_g,
        rmsd_core=rmsd_core,
        cfg=cfg,
        steric_hv25=hv_buf,
        steric_ca38=ca_buf,
        glycan_contacts_45=gc_45,
        nonstem_sugars_hit=nonstem_hit,
        ch_pi_contacts=ch_pi,
        binder_plddt=b_plddt,
        glyco_stereo_ok=stereo_ok,
        rg_ratio=rg_ratio,
        target_protein_contacts=t_contacts,
        n_forbidden_aa=n_forb,
    )
    extras = {
        "J": float(j_val),
        "ipsae_min_glyco": ipsae_g,
        "ipsae_glycan_min": float(glyco["ipsae_glycan_min"]),
        "ipsae_min_bare": ipsae_b,
        "delta_ipsae_min": float(ipsae_g - ipsae_b),
        "iptm_glyco": iptm_g,
        "iptm_bare": iptm_b,
        "delta_iptm": float(iptm_g - iptm_b),
        "hbonds_glycan": hbonds_g,
        "glycan_contacts_45": gc_45,
        "nonstem_sugars_hit": nonstem_hit,
        "ch_pi_contacts": ch_pi,
        "binder_plddt": b_plddt,
        "glyco_stereo_ok": stereo_ok,
        "rg_ratio": rg_ratio,
        "target_protein_contacts": t_contacts,
        "n_forbidden_aa": n_forb,
        "rmsd_core": rmsd_core,
        "steric_hv25": hv25,
        "steric_ca38": ca38,
        "steric_hv_buf": hv_buf,
        "steric_ca_buf": ca_buf,
        "intact_clash_ok": clash_ok,
    }
    return float(-j_val), extras


def parse_loss_spec(
    losses: str | Sequence[str] | None,
    weights: str | Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Parse comma-separated or list loss specification into [(loss_name, weight), ...]."""
    if losses is None or losses == "":
        names = ["dual"]
    elif isinstance(losses, str):
        names = [x.strip() for x in losses.split(",") if x.strip()]
    else:
        names = [str(x).strip() for x in losses if str(x).strip()]

    for name in names:
        if name not in LOSS_REGISTRY and name not in MULTISTATE_LOSS_REGISTRY:
            known = sorted(set(LOSS_REGISTRY) | set(MULTISTATE_LOSS_REGISTRY))
            raise ValueError(f"Unknown loss {name!r}; registered: {known}")

    if weights is None or weights == "":
        w_vals = [1.0] * len(names)
    elif isinstance(weights, str):
        w_vals = [float(x) for x in weights.split(",") if x.strip()]
    else:
        w_vals = [float(x) for x in weights]

    if len(w_vals) != len(names):
        raise ValueError(f"loss_weights count ({len(w_vals)}) != losses count ({len(names)})")
    return list(zip(names, w_vals))


def extract_state_context(
    state_spec: Mapping[str, Any],
    pred: Any,
    ref_cx: ComplexCoords | None,
    core_positions: Sequence[int] | None,
    cfg: Mapping[str, Any],
    *,
    steric_ref: StericClashRef | None = None,
) -> dict[str, Any]:
    """Compute all reusable per-state structural and confidence metrics from an AF3Prediction."""
    pae_cut = float(cfg.get("pae_cutoff", 10.0))
    dist_cut = float(cfg.get("dist_cutoff", 10.0))
    hb_cut = float(cfg.get("hbond_cutoff", 3.5))

    chain_specs = state_spec.get("chains") or []
    prot_chains = [
        c["id"] for c in chain_specs if c.get("type", "protein") == "protein"
    ] or ["A"]
    mut_chains = [
        c["id"] for c in chain_specs if c.get("type", "protein") == "protein" and c.get("mutable")
    ]
    fixed_chains = [
        c["id"] for c in chain_specs if c.get("type", "protein") == "protein" and not c.get("mutable")
    ]
    mutable_chain = mut_chains[0] if mut_chains else prot_chains[-1]
    other_prots = [c for c in prot_chains if c != mutable_chain]
    glycan_cids = list(pred.complex_coords.glycan_chains.keys())
    ligand_cids = [
        cid for cid in pred.complex_coords.all_chains
        if cid not in prot_chains and cid not in glycan_cids and cid != mutable_chain
    ]
    partner_chain = (
        fixed_chains[0]
        if fixed_chains
        else (other_prots[0] if other_prots else (ligand_cids[0] if ligand_cids else mutable_chain))
    )

    tok_coords = pred.token_coords
    if tok_coords is None:
        tok_coords = extract_token_coords(
            pred.complex_coords, pred.token_chain_ids, protein_chains=prot_chains
        )

    if other_prots or glycan_cids or ligand_cids:
        ipsae_res = compute_ipsae_min(
            pred.pae,
            pred.token_chain_ids,
            token_coords=tok_coords,
            pae_cutoff=pae_cut,
            dist_cutoff=dist_cut,
            target_chain=partner_chain,
            binder_chain=mutable_chain,
            target_chains=other_prots + ligand_cids,
            glycan_chains=glycan_cids,
        )
    else:
        ipsae_res = {
            "ipsae_min": 0.0,
            "ipsae_max": 0.0,
            "ipsae_b2r": 0.0,
            "ipsae_r2b": 0.0,
            "ipsae_glycan_min": 0.0,
        }

    mut_idx = prot_chains.index(mutable_chain) if mutable_chain in prot_chains else 0
    iptm_val = extract_binder_iptm(pred.summary, chain_idx=mut_idx)
    if len(prot_chains) == 1 and not glycan_cids and not ligand_cids:
        iptm_val = float(pred.summary.get("iptm") or 0.0)

    plddt_chains = [c for c in (mut_chains or prot_chains) if c in pred.complex_coords.all_chains]
    plddt_val = compute_plddt(
        pred.confidences,
        chain_id=plddt_chains if plddt_chains else None,
    )
    ptm_val = float(pred.summary.get("ptm") or 0.0)
    pae_arr = np.asarray(pred.pae, dtype=np.float64)
    pae_mean = float(np.mean(pae_arr)) if pae_arr.size else PAE_MAX
    ipae_mean = _mean_inter_chain_pae(pae_arr, pred.token_chain_ids)
    pae_asym = _pae_asymmetry(pae_arr, pred.token_chain_ids, mutable_chain)

    hbonds_glycan = count_glycan_hbonds(
        pred.complex_coords, cutoff=hb_cut, mutable_chain=mutable_chain
    )
    contacts_glycan = count_glycan_contacts(
        pred.complex_coords, cutoff=4.5, mutable_chain=mutable_chain
    )
    glycan_ents = [c for c in chain_specs if c.get("type") == "glycan"]
    anchor_res_1based: int | None = None
    expected_n_sugars: int | None = None
    if glycan_ents:
        g_ent = glycan_ents[0]
        if g_ent.get("anchor_res_1based") is not None:
            anchor_res_1based = int(g_ent["anchor_res_1based"])
        if g_ent.get("sugar_nodes"):
            expected_n_sugars = len(g_ent["sugar_nodes"])
        elif g_ent.get("glycoform"):
            expected_n_sugars = _GLYCOFORM_SUGAR_COUNTS.get(str(g_ent["glycoform"]))
    elif cfg.get("target_n_pos") is not None:
        anchor_res_1based = int(cfg["target_n_pos"])

    nonstem_sugars_hit = count_nonstem_sugars_hit(
        pred.complex_coords,
        cutoff=4.5,
        mutable_chain=mutable_chain,
        target_chain=partner_chain,
        anchor_res_1based=anchor_res_1based,
    )
    ch_pi_contacts = count_ch_pi_contacts(
        pred.complex_coords,
        dist_cutoff=5.0,
        tilt_deg=35.0,
        mutable_chain=mutable_chain,
    )
    target_protein_contacts = (
        count_target_protein_contacts(
            pred.complex_coords,
            cutoff=4.5,
            target_chain=partner_chain,
            mutable_chain=mutable_chain,
        )
        if partner_chain != mutable_chain
        else 0
    )
    glyco_stereo_ok = compute_glyco_stereo_ok(
        pred.complex_coords,
        anchor_res_1based=anchor_res_1based,
        expected_n_sugars=expected_n_sugars,
        target_chain=partner_chain,
    )
    rmsd_core = (
        compute_core_ca_rmsd(
            pred.complex_coords,
            ref_cx,
            core_positions=core_positions,
            mutable_chain=mutable_chain,
            align_chain=partner_chain,
        )
        if ref_cx is not None
        else 0.0
    )
    if math.isnan(rmsd_core):
        rmsd_core = 0.0

    m_chain_obj = pred.complex_coords.all_chains.get(mutable_chain, pred.complex_coords.binder)
    rg_ratio = compute_rg_ratio(m_chain_obj)
    aspect_ratio = compute_aspect_ratio(m_chain_obj)
    b_seq_obs = "".join(AA3_TO_1.get(rn, "X") for rn in m_chain_obj.resnames)
    n_forbidden_obs = (cp_count(b_seq_obs) + sequon_count(b_seq_obs)) if b_seq_obs else 0
    contact_score = compute_contact_score(pred.complex_coords, mutable_chain=mutable_chain)
    if steric_ref is not None:
        clashes = compute_steric_clashes(
            pred.complex_coords,
            steric_ref,
            target_chain=partner_chain,
            binder_chain=mutable_chain,
            hv_cutoff=float(cfg.get("steric_hv_cutoff", 2.5)),
            ca_cutoff=float(cfg.get("steric_ca_cutoff", 3.8)),
            pred_residues=cfg.get("steric_pred_residues"),
        )
    else:
        clashes = {
            "steric_hv25": 0,
            "steric_ca38": 0,
            "steric_hv_buf": 0,
            "steric_ca_buf": 0,
            "intact_clash_ok": 1,
        }

    return {
        "plddt": float(plddt_val),
        "binder_plddt": float(plddt_val),
        "ptm": float(ptm_val),
        "iptm": float(iptm_val),
        "pae_mean": float(pae_mean),
        "ipae_mean": float(ipae_mean),
        "pae_asym": float(pae_asym),
        "ipsae_min": float(ipsae_res["ipsae_min"]),
        "ipsae_max": float(ipsae_res["ipsae_max"]),
        "ipsae_glycan_min": float(ipsae_res["ipsae_glycan_min"]),
        "hbonds_glycan": int(hbonds_glycan),
        "glycan_contacts_45": int(contacts_glycan),
        "nonstem_sugars_hit": int(nonstem_sugars_hit),
        "ch_pi_contacts": int(ch_pi_contacts),
        "glyco_stereo_ok": int(glyco_stereo_ok),
        "target_protein_contacts": int(target_protein_contacts),
        "n_forbidden_aa": int(n_forbidden_obs),
        "rmsd_core": float(rmsd_core),
        "rg_ratio": float(rg_ratio),
        "aspect_ratio": float(aspect_ratio),
        "contact_score": float(contact_score),
        "ranking_score": float(pred.summary.get("ranking_score") or 0.0),
        "steric_hv25": int(clashes["steric_hv25"]),
        "steric_ca38": int(clashes["steric_ca38"]),
        "steric_hv_buf": int(clashes["steric_hv_buf"]),
        "steric_ca_buf": int(clashes["steric_ca_buf"]),
        "intact_clash_ok": int(clashes["intact_clash_ok"]),
    }


def evaluate_multistate(
    state_specs: Sequence[Mapping[str, Any]],
    preds_by_state: Mapping[str, Any],
    ref_cxs: Mapping[str, ComplexCoords] | None,
    core_positions: Sequence[int] | None,
    cfg: Mapping[str, Any],
    *,
    steric_ref: StericClashRef | None = None,
    protomer_seqs: Mapping[str, str] | None = None,
    precomputed_state_ctx: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate all states and registered losses (per-state + multistate plugins), returning global loss & J."""
    n_forb_override: int | None = None
    if protomer_seqs is not None:
        n_forb_override = sum(cp_count(s) + sequon_count(s) for s in protomer_seqs.values())

    state_ctx: dict[str, dict[str, Any]] = {}
    for spec in state_specs:
        sname = str(spec["name"])
        if precomputed_state_ctx is not None and sname in precomputed_state_ctx:
            ctx = dict(precomputed_state_ctx[sname])
        else:
            ref_cx = ref_cxs.get(sname) if ref_cxs else None
            ctx = extract_state_context(
                spec,
                preds_by_state[sname],
                ref_cx,
                core_positions,
                cfg,
                steric_ref=steric_ref,
            )
        if n_forb_override is not None:
            ctx["n_forbidden_aa"] = int(n_forb_override)
        state_ctx[sname] = ctx

    loss_spec = parse_loss_spec(cfg.get("losses"), cfg.get("loss_weights"))
    ms_terms = [(n, w) for n, w in loss_spec if n in MULTISTATE_LOSS_REGISTRY]
    ps_terms = [(n, w) for n, w in loss_spec if n in LOSS_REGISTRY]

    total_loss = 0.0
    extras: dict[str, Any] = {}

    if ms_terms:
        w_sum = sum(abs(w) for _, w in ms_terms) or 1.0
        for name, w in ms_terms:
            val, term_extras = MULTISTATE_LOSS_REGISTRY[name](state_ctx, cfg)
            total_loss += (w / w_sum) * float(val) if len(ms_terms) > 1 else w * float(val)
            extras.update(term_extras)

    pos_names = [str(s["name"]) for s in state_specs if float(s.get("weight", 1.0)) >= 0.0]
    primary_name = "glyco" if "glyco" in state_ctx else (pos_names[0] if pos_names else str(state_specs[0]["name"]))
    primary_ctx = state_ctx[primary_name]

    if ps_terms:
        ps_w_sum = sum(w for _, w in ps_terms) or 1.0
        st_w_sum = sum(abs(float(s.get("weight", 1.0))) for s in state_specs) or 1.0
        ps_total = 0.0
        for spec in state_specs:
            sname = str(spec["name"])
            sw = float(spec.get("weight", 1.0))
            ctx = state_ctx[sname]
            st_terms = parse_loss_spec(spec["losses"], spec.get("loss_weights")) if spec.get("losses") else ps_terms
            st_w_norm = sum(w for _, w in st_terms) or 1.0
            raw_state_loss = sum(w * LOSS_REGISTRY[n](ctx) for n, w in st_terms) / st_w_norm
            ctx["loss"] = float(raw_state_loss)
            ps_total += (sw / st_w_sum) * raw_state_loss
        if ms_terms:
            total_loss += ps_total / ps_w_sum
        else:
            total_loss = ps_total
            steric_w = float(cfg.get("steric_clash_weight", 0.0))
            if steric_w > 0.0 and "steric_clash" not in dict(ps_terms):
                total_loss += steric_w * (
                    float(primary_ctx.get("steric_hv_buf", primary_ctx["steric_hv25"]))
                    + 0.5 * float(primary_ctx.get("steric_ca_buf", primary_ctx["steric_ca38"]))
                )

    if "steric_hv25" not in extras:
        extras["steric_hv25"] = int(primary_ctx["steric_hv25"])
        extras["steric_ca38"] = int(primary_ctx["steric_ca38"])
        extras["steric_hv_buf"] = int(primary_ctx.get("steric_hv_buf", primary_ctx["steric_hv25"]))
        extras["steric_ca_buf"] = int(primary_ctx.get("steric_ca_buf", primary_ctx["steric_ca38"]))
        extras["intact_clash_ok"] = int(primary_ctx["intact_clash_ok"])
    for m_k in (
        "nonstem_sugars_hit",
        "ch_pi_contacts",
        "binder_plddt",
        "glyco_stereo_ok",
        "rg_ratio",
        "aspect_ratio",
        "target_protein_contacts",
        "n_forbidden_aa",
    ):
        extras.setdefault(m_k, primary_ctx.get(m_k))

    j_val = float(extras["J"]) if "J" in extras else float(-total_loss)
    return {
        "loss": float(total_loss),
        "J": float(j_val),
        "states": state_ctx,
        **extras,
    }
