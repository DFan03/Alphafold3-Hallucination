"""General multistate simulated-annealing MCMC hallucination & interface evolution engine."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import build_states_from_config
from .predictor import AF3Prediction, AF3Predictor, AF3RunnerProtocol, state_glycan_chain_ids
from .proposals import (
    LigandMPNNBiasedSampler,
    MPNNSampler,
    cp_count,
    is_valid_binder_seq,
    mutation_count,
    mutation_string,
    propose_mutant,
    random_sequence,
    sequon_count,
    validate_binder_seq,
)
from .scoring import evaluate_multistate, extract_state_context
from .structure import (
    ComplexCoords,
    StericClashRef,
    compute_core_positions,
    compute_interface_mask,
    load_steric_clash_ref,
    load_structure,
)


class Logger:
    """UTC ISO-8601 logger writing to stdout and a persistent log file."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.fh = p.open("a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")

    def close(self) -> None:
        self.fh.close()


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()


def temperature(
    step: int,
    t_init: float = 0.05,
    half_life: float = 50.0,
    schedule: str = "exp",
    total_steps: int | None = None,
    t_final: float | None = None,
) -> float:
    """Compute annealing temperature at 1-based `step`."""
    sched = str(schedule).lower()
    if sched == "exp":
        t = float(t_init) * (0.5 ** (max(0, step - 1) / max(1e-9, float(half_life))))
        if t_final is not None:
            t = max(float(t_final), t)
        return float(t)
    if sched == "linear":
        n = max(1, int(total_steps or 1))
        tf = 0.0 if t_final is None else float(t_final)
        frac = min(1.0, max(0.0, (step - 1) / max(1, n - 1)))
        return float(t_init + (tf - float(t_init)) * frac)
    if sched == "const":
        return float(t_init)
    raise ValueError(f"Unknown temp_schedule {schedule!r}")


def metropolis_accept(
    loss_new: float,
    loss_cur: float,
    temp: float,
    margin: float,
    rng: np.random.Generator,
) -> tuple[bool, str, float, float]:
    """Metropolis acceptance rule minimizing `loss` with noise margin hurdle."""
    delta = float(loss_new) - float(loss_cur)
    d_eff = delta + float(margin)
    if d_eff < 0.0:
        return True, "improved", 1.0, float("nan")
    p = math.exp(-d_eff / temp) if temp > 0.0 else 0.0
    u = float(rng.random())
    acc = u < p
    return acc, ("metropolis" if acc else "worse"), p, u


def _step_seed(base_seed: int, *parts: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), *parts]).generate_state(1)[0] % (2**31 - 1))


def _discover_protomers(state_specs: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool]:
    protomers: list[str] = []
    has_fixed_partner = False
    for spec in state_specs:
        for ent in spec.get("chains") or []:
            etype = ent.get("type", "protein")
            if etype == "protein" and ent.get("mutable"):
                pkey = str(ent.get("protomer") or ent["id"])
                if pkey not in protomers:
                    protomers.append(pkey)
            else:
                has_fixed_partner = True
    if not protomers:
        raise ValueError("State specifications must contain at least one mutable protein chain")
    return protomers, has_fixed_partner


def _parse_lengths(raw_len: Any, n_protomers: int) -> list[int]:
    if isinstance(raw_len, (list, tuple)):
        vals = [int(x) for x in raw_len]
    elif isinstance(raw_len, str):
        vals = [int(x.strip()) for x in raw_len.split(",") if x.strip()]
    else:
        vals = [int(raw_len)]
    if len(vals) == 1 and n_protomers > 1:
        vals = vals * n_protomers
    if len(vals) != n_protomers:
        raise ValueError(f"binder_length count ({len(vals)}) != protomers ({n_protomers})")
    return vals


def _is_clean_protomers(
    seqs: Mapping[str, str],
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
) -> bool:
    """Return True iff every protomer has 0 Cys, 0 Pro, 0 sequon and satisfies exclusion rules."""
    return all(
        cp_count(s) == 0
        and sequon_count(s) == 0
        and is_valid_binder_seq(s, exclude_aa=exclude_aa, forbid_sequon=forbid_sequon)
        for s in seqs.values()
    )


def _resolve_start_protomers(
    cfg: Mapping[str, Any],
    protomers: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    exclude_aa = str(cfg.get("exclude_aa", "CP"))
    forbid_sequon = bool(cfg.get("forbid_sequon", True))
    allow_init_forbidden = bool(cfg.get("allow_init_forbidden", False))
    out: dict[str, str] = {}

    if cfg.get("binder_seq"):
        seq = str(cfg["binder_seq"]).strip().upper()
        if len(protomers) == 1:
            out[protomers[0]] = seq
        else:
            parts = [s.strip() for s in seq.split(",")]
            if len(parts) != len(protomers):
                raise ValueError("binder_seq must match protomer count")
            out = dict(zip(protomers, parts))
    elif cfg.get("seqs"):
        seq_list = [str(s).strip().upper() for s in cfg["seqs"]]
        if len(seq_list) == 1 and len(protomers) == 1:
            out[protomers[0]] = seq_list[0]
        elif len(seq_list) >= len(protomers):
            out = {p: seq_list[i] for i, p in enumerate(protomers)}
        else:
            raise ValueError("seqs length < mutable protomer count")
    elif cfg.get("seq_file"):
        with Path(cfg["seq_file"]).open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if cfg.get("seq_ids"):
            wanted = set(cfg["seq_ids"])
            rows = [r for r in rows if r.get("design_id") in wanted]
        if not rows:
            raise ValueError("No starting sequences found in seq_file")
        out[protomers[0]] = str(rows[0]["sequence"]).strip().upper()
        for p in protomers[1:]:
            out[p] = out[protomers[0]]
    else:
        lengths = _parse_lengths(cfg.get("binder_length", 80), len(protomers))
        for p, L in zip(protomers, lengths):
            out[p] = random_sequence(L, rng, exclude_aa=exclude_aa, forbid_sequon=forbid_sequon)

    val_ex = "" if allow_init_forbidden else exclude_aa
    val_seq = False if allow_init_forbidden else forbid_sequon
    for p, seq in out.items():
        validate_binder_seq(
            seq, f"start protomer {p}", exclude_aa=val_ex, forbid_sequon=val_seq
        )
    return out


def _compute_mutable_masks_from_coords(
    cfg: Mapping[str, Any],
    state_specs: Sequence[Mapping[str, Any]],
    coords_by_state: Mapping[str, ComplexCoords],
    protomer_seqs: Mapping[str, str],
    has_fixed_partner: bool,
) -> dict[str, list[int]]:
    mode = str(cfg.get("mask_mode", "auto")).lower()
    explicit = cfg.get("explicit_positions")
    if explicit is not None and mode in ("auto", "explicit", "interface"):
        pos = sorted({int(x) - 1 for x in explicit})
        masks: dict[str, list[int]] = {}
        for pkey, seq in protomer_seqs.items():
            bad = [x + 1 for x in pos if not (0 <= x < len(seq))]
            if bad:
                raise ValueError(f"explicit_positions out of range for protomer {pkey}: {bad}")
            masks[pkey] = list(pos)
        return masks

    if mode == "auto":
        mode = "interface" if has_fixed_partner else "all"

    if mode == "all":
        return {pkey: list(range(len(seq))) for pkey, seq in protomer_seqs.items()}

    if mode == "interface":
        dist_cut = float(cfg.get("interface_dist", 6.0))
        masks = {pkey: [] for pkey in protomer_seqs}
        for spec in state_specs:
            sname = str(spec["name"])
            cx = coords_by_state.get(sname)
            if cx is None:
                continue
            for ent in spec.get("chains") or []:
                if ent.get("type", "protein") == "protein" and ent.get("mutable"):
                    pkey = str(ent.get("protomer") or ent["id"])
                    if len(cx.all_chains) > 1:
                        m_idx = compute_interface_mask(
                            cx,
                            interface_dist=dist_cut,
                            mutable_chain=str(ent["id"]),
                        )
                        masks[pkey] = sorted(set(masks[pkey]) | set(m_idx))
        for pkey, seq in protomer_seqs.items():
            if not masks[pkey]:
                masks[pkey] = list(range(len(seq)))
        return masks

    raise ValueError(f"Unknown mask_mode {mode!r}")


def _compute_mutable_masks(
    cfg: Mapping[str, Any],
    state_specs: Sequence[Mapping[str, Any]],
    preds_by_state: Mapping[str, AF3Prediction],
    protomer_seqs: Mapping[str, str],
    has_fixed_partner: bool,
) -> dict[str, list[int]]:
    return _compute_mutable_masks_from_coords(
        cfg,
        state_specs,
        {k: v.complex_coords for k, v in preds_by_state.items()},
        protomer_seqs,
        has_fixed_partner,
    )


def _load_coords_by_state(
    state_specs: Sequence[Mapping[str, Any]],
    cif_paths: Mapping[str, str],
) -> dict[str, ComplexCoords]:
    out: dict[str, ComplexCoords] = {}
    for spec in state_specs:
        sname = str(spec["name"])
        raw_p = cif_paths.get(sname)
        if not raw_p or not Path(raw_p).exists():
            continue
        chain_specs = spec.get("chains") or []
        prot_chains = [c["id"] for c in chain_specs if c.get("type", "protein") == "protein"] or ["A"]
        mut_chains = [c["id"] for c in chain_specs if c.get("type", "protein") == "protein" and c.get("mutable")]
        fixed_chains = [c["id"] for c in chain_specs if c.get("type", "protein") == "protein" and not c.get("mutable")]
        binder_cid = mut_chains[0] if mut_chains else prot_chains[-1]
        other_prots = [c for c in prot_chains if c != binder_cid]
        ligand_cids = [str(c.get("id")) for c in chain_specs if c.get("type") == "ligand" and c.get("id")]
        target_cid = (
            fixed_chains[0]
            if fixed_chains
            else (other_prots[0] if other_prots else (ligand_cids[0] if ligand_cids else binder_cid))
        )
        glycan_cids = state_glycan_chain_ids(spec)
        out[sname] = load_structure(
            raw_p,
            target_chain=target_cid,
            binder_chain=binder_cid,
            glycan_chain_ids=glycan_cids,
        )
    return out


def _copy_state_cifs(
    preds_by_state: Mapping[str, AF3Prediction],
    run_dir: Path,
    prefix: str,
) -> dict[str, str]:
    saved: dict[str, str] = {}
    for sname, pred in preds_by_state.items():
        dst = run_dir / f"{prefix}_{sname}.cif"
        if pred.cif_path.exists():
            shutil.copyfile(pred.cif_path, dst)
        saved[sname] = str(dst)
    return saved


def _seq_signature(protomer_seqs: Mapping[str, str]) -> str:
    if len(protomer_seqs) == 1:
        return next(iter(protomer_seqs.values()))
    return ":".join(f"{k}={protomer_seqs[k]}" for k in sorted(protomer_seqs))


_EXTRA_LOG_KEYS: tuple[str, ...] = (
    "ipsae_min_glyco",
    "ipsae_glycan_min",
    "ipsae_min_bare",
    "delta_ipsae_min",
    "iptm_glyco",
    "iptm_bare",
    "delta_iptm",
    "hbonds_glycan",
    "glycan_contacts_45",
    "nonstem_sugars_hit",
    "ch_pi_contacts",
    "binder_plddt",
    "glyco_stereo_ok",
    "rg_ratio",
    "aspect_ratio",
    "target_protein_contacts",
    "n_forbidden_aa",
    "rmsd_core",
    "steric_hv25",
    "steric_ca38",
    "intact_clash_ok",
)


def run(
    cfg: Mapping[str, Any],
    predictor: AF3Predictor | None = None,
    mpnn: Any | None = None,
    runner: AF3RunnerProtocol | None = None,
) -> Path:
    """Run or cleanly resume a multistate AF3 annealing MCMC trajectory. Returns `run_dir`."""
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    job = str(cfg.get("job") or cfg.get("name") or "af3_hall_run")
    run_dir = Path(
        cfg.get("out_dir")
        or Path(cfg.get("out_root", "./runs"))
        / f"{time.strftime('%Y-%m-%d', time.gmtime())}_af3_hall_{job}"
    )
    work_root = run_dir / "work"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(cfg["log_dir"]) / f"af3_hall_{job}.log"
        if cfg.get("log_dir")
        else (run_dir / "run.log")
    )
    log = Logger(log_path)

    try:
        state_specs = build_states_from_config(cfg)
        protomers, has_fixed_partner = _discover_protomers(state_specs)
        exclude_aa = str(cfg.get("exclude_aa", "CP"))
        forbid_sequon = bool(cfg.get("forbid_sequon", True))
        allow_init_forbidden = bool(cfg.get("allow_init_forbidden", False))
        lazy_neg_interval = int(cfg.get("lazy_neg_interval") or cfg.get("lazy_bare_interval") or 1)
        mask_refresh_interval = int(cfg.get("mask_refresh_interval", 0))
        wf_cap_frac = float(cfg.get("wf_cap_frac", 0.12))
        tmp_root = str(cfg.get("tmp_root") or os.environ.get("AF3_HALL_TMP_DIR", "./tmp"))

        _write_json_atomic(run_dir / "config.json", dict(cfg))
        if predictor is None:
            predictor = AF3Predictor(state_specs, cfg, runner=runner)
        mut_method_cfg = str(cfg.get("mutation_method", "")).lower()
        if mpnn is None and mut_method_cfg == "mpnn":
            mpnn_dir = str(cfg.get("mpnn_dir") or os.environ.get("LIGANDMPNN_DIR", ""))
            if mpnn_dir and Path(mpnn_dir).exists():
                mpnn = MPNNSampler(
                    mpnn_dir,
                    weights=str(cfg.get("mpnn_weights", "proteinmpnn_v_48_020")),
                    temperature=float(cfg.get("mpnn_temp", 0.3)),
                    exclude_aa=exclude_aa,
                )
        elif mpnn is None and mut_method_cfg == "ligand_mpnn_biased":
            mpnn = LigandMPNNBiasedSampler(
                mpnn_dir=cfg.get("mpnn_dir"),
                checkpoint=cfg.get("mpnn_checkpoint"),
                mpnn_python=cfg.get("mpnn_python"),
                temperature=float(cfg.get("mpnn_temp", 0.3)),
                exclude_aa=exclude_aa,
                refresh_interval=int(cfg.get("mpnn_refresh_interval", 5)),
                wf_cap_frac=wf_cap_frac,
                tmp_dir=tmp_root,
            )

        steps_jsonl = run_dir / "steps.jsonl"
        state_json = run_dir / "state.json"
        total_steps = int(cfg.get("steps", 100))

        steric_ref: StericClashRef | None = None
        if cfg.get("steric_clash_ref_cif"):
            steric_ref = load_steric_clash_ref(
                str(cfg["steric_clash_ref_cif"]),
                ref_residues=cfg.get("steric_ref_residues"),
                chain_id=str(cfg.get("target_chain", "A")),
            )

        ref_cxs: dict[str, ComplexCoords] = {}
        best_clean_seqs: dict[str, str] | None = None
        best_clean_eval: dict[str, Any] | None = None
        cur_cifs: dict[str, str] = {}
        if state_json.exists():
            saved = json.loads(state_json.read_text(encoding="utf-8"))
            step0 = int(saved["step_done"]) + 1
            cur_seqs: dict[str, str] = dict(saved["cur_seqs"])
            cur_eval: dict[str, Any] = dict(saved["cur_eval"])
            best_seqs: dict[str, str] = dict(saved["best_seqs"])
            best_eval: dict[str, Any] = dict(saved["best_eval"])
            if saved.get("best_clean_seqs") is not None:
                best_clean_seqs = dict(saved["best_clean_seqs"])
            if saved.get("best_clean_eval") is not None:
                best_clean_eval = dict(saved["best_clean_eval"])
            mutable_masks: dict[str, list[int]] = {
                k: [int(x) for x in v] for k, v in saved["mutable_masks"].items()
            }
            core_positions: list[int] = [int(x) for x in saved["core_positions"]]
            ref_paths: dict[str, str] = dict(saved["ref_cifs"])
            cur_cifs = dict(saved.get("cur_cifs") or {})
            for sname, rpath in ref_paths.items():
                if Path(rpath).exists():
                    ref_cxs[sname] = load_structure(rpath)
            log(f"RESUME job={job} at step {step0}/{total_steps} best_J={best_eval['J']:.4f}")
        else:
            init_rng = np.random.default_rng(_step_seed(seed, 0))
            cur_seqs = _resolve_start_protomers(cfg, protomers, init_rng)
            init_work = work_root / "step_0000"
            init_preds = predictor.predict_all(cur_seqs, seed=_step_seed(seed, 0, 1), work_dir=init_work, tag="step0")

            if cfg.get("ref_cif") and Path(str(cfg["ref_cif"])).exists():
                ref_src = Path(str(cfg["ref_cif"]))
                ref_paths = {}
                for spec in state_specs:
                    sn = str(spec["name"])
                    dst = run_dir / f"ref_{sn}.cif"
                    shutil.copyfile(ref_src, dst)
                    ref_paths[sn] = str(dst)
                    ref_cxs[sn] = load_structure(dst)
            else:
                ref_paths = _copy_state_cifs(init_preds, run_dir, "ref")
                for sn, pth in ref_paths.items():
                    if Path(pth).exists():
                        ref_cxs[sn] = load_structure(pth)

            cur_cifs = _copy_state_cifs(init_preds, run_dir, "cur")
            mutable_masks = _compute_mutable_masks(
                cfg, state_specs, init_preds, cur_seqs, has_fixed_partner
            )
            primary_p = protomers[0]
            core_positions = compute_core_positions(len(cur_seqs[primary_p]), mutable_masks[primary_p])
            cur_eval = evaluate_multistate(
                state_specs,
                init_preds,
                ref_cxs,
                core_positions,
                cfg,
                steric_ref=steric_ref,
                protomer_seqs=cur_seqs,
            )
            best_seqs = dict(cur_seqs)
            best_eval = dict(cur_eval)
            _copy_state_cifs(init_preds, run_dir, "best")
            if _is_clean_protomers(cur_seqs, exclude_aa=exclude_aa, forbid_sequon=forbid_sequon):
                best_clean_seqs = dict(cur_seqs)
                best_clean_eval = dict(cur_eval)
                _copy_state_cifs(init_preds, run_dir, "best_clean")
            shutil.rmtree(init_work, ignore_errors=True)

            seq_sig = _seq_signature(cur_seqs)
            step0_rec = {
                "step": 0,
                "accepted": True,
                "reason": "init",
                "temperature": float(cfg.get("T_init", 0.05)),
                "n_mut": 0,
                "mutations": "none",
                "sequence": seq_sig,
                "protomers": dict(cur_seqs),
                "cys_cnt": sum(s.count("C") for s in cur_seqs.values()),
                "pro_cnt": sum(s.count("P") for s in cur_seqs.values()),
                "sequon_cnt": sum(sequon_count(s) for s in cur_seqs.values()),
                "loss": float(cur_eval["loss"]),
                "loss_cur": float(cur_eval["loss"]),
                "best_loss": float(best_eval["loss"]),
                "J": float(cur_eval["J"]),
                "J_cur": float(cur_eval["J"]),
                "best_J": float(best_eval["J"]),
                "delta_J": 0.0,
                "p_accept": 1.0,
                "neg_evaluated": True,
                "bare_evaluated": True,
                "states": cur_eval["states"],
                "t_step_s": sum(p.wall_s for p in init_preds.values()),
            }
            for extra_k in _EXTRA_LOG_KEYS:
                if extra_k in cur_eval:
                    step0_rec[extra_k] = cur_eval[extra_k]
            _append_jsonl(steps_jsonl, step0_rec)
            _write_json_atomic(
                state_json,
                {
                    "step_done": 0,
                    "cur_seqs": cur_seqs,
                    "cur_eval": cur_eval,
                    "best_seqs": best_seqs,
                    "best_eval": best_eval,
                    "best_clean_seqs": best_clean_seqs,
                    "best_clean_eval": best_clean_eval,
                    "mutable_masks": mutable_masks,
                    "core_positions": core_positions,
                    "ref_cifs": ref_paths,
                    "cur_cifs": cur_cifs,
                },
            )
            log(
                f"INIT job={job} states={[s['name'] for s in state_specs]} "
                f"mutable_lens={ {k: len(v) for k, v in mutable_masks.items()} } "
                f"core_len={len(core_positions)} J={cur_eval['J']:.4f} loss={cur_eval['loss']:.4f}"
            )
            step0 = 1

        primary_p = protomers[0]
        t_run0 = time.time()
        n_prop_cfg = int(cfg.get("proposals_per_step", 1))
        force_map = cfg.get("force_proposals") or {}
        neg_specs = [
            sp for sp in state_specs
            if float(sp.get("weight", 1.0)) < 0.0 or str(sp["name"]) == "bare"
        ]
        neg_names = {str(sp["name"]) for sp in neg_specs}
        pos_specs = [sp for sp in state_specs if str(sp["name"]) not in neg_names]
        can_lazy_neg = (
            lazy_neg_interval > 1
            and len(neg_specs) >= 1
            and len(pos_specs) >= 1
        )

        for step in range(step0, total_steps + 1):
            t_s0 = time.time()
            if mask_refresh_interval > 0 and step % mask_refresh_interval == 0:
                refreshed_cxs = _load_coords_by_state(state_specs, cur_cifs)
                if refreshed_cxs:
                    mutable_masks = _compute_mutable_masks_from_coords(
                        cfg, state_specs, refreshed_cxs, cur_seqs, has_fixed_partner
                    )
                    core_positions = compute_core_positions(
                        len(cur_seqs[primary_p]), mutable_masks[primary_p]
                    )
                    log(
                        f"MASK_REFRESH step={step} mutable_lens="
                        f"{ {k: len(v) for k, v in mutable_masks.items()} }"
                    )

            step_rng = np.random.default_rng(_step_seed(seed, step, 1))
            acc_rng = np.random.default_rng(_step_seed(seed, step, 2))
            T = temperature(
                step,
                t_init=float(cfg.get("T_init", 0.05)),
                half_life=float(cfg.get("half_life", 50.0)),
                schedule=str(cfg.get("temp_schedule", "exp")),
                total_steps=total_steps,
                t_final=cfg.get("T_final"),
            )
            n_mut = mutation_count(step, total_steps, str(cfg.get("mutation_rate", "3-1")), rng=step_rng)

            candidates: list[tuple[dict[str, str], str, str]] = []
            forced = force_map.get(step) or force_map.get(str(step))
            if forced:
                forced_items = [forced] if isinstance(forced, Mapping) else list(forced)
                val_ex = "" if allow_init_forbidden else exclude_aa
                val_seq = False if allow_init_forbidden else forbid_sequon
                for f_item in forced_items:
                    cand_seqs = dict(cur_seqs)
                    if isinstance(f_item, Mapping):
                        cand_seqs.update({k: str(v).strip().upper() for k, v in f_item.items()})
                    else:
                        cand_seqs[primary_p] = str(f_item).strip().upper()
                    for pk, sv in cand_seqs.items():
                        validate_binder_seq(
                            sv, f"forced proposal {pk}", exclude_aa=val_ex, forbid_sequon=val_seq
                        )
                    mut_str = ";".join(
                        f"{pk}:{mutation_string(cur_seqs[pk], cand_seqs[pk])}"
                        if len(protomers) > 1
                        else mutation_string(cur_seqs[pk], cand_seqs[pk])
                        for pk in protomers
                    )
                    candidates.append((cand_seqs, mut_str, "forced"))
            else:
                for k in range(n_prop_cfg):
                    cand_seqs = dict(cur_seqs)
                    target_p = protomers[int(step_rng.integers(0, len(protomers)))]
                    s0_name = str(state_specs[0]["name"])
                    cur_struct = run_dir / f"cur_{s0_name}.cif"
                    best_struct = run_dir / f"best_{s0_name}.cif"
                    ref_struct = cur_struct if cur_struct.exists() else best_struct
                    s0_prots = [
                        c for c in (state_specs[0].get("chains") or [])
                        if c.get("type", "protein") == "protein"
                    ]
                    matching_cids = [
                        str(c["id"]) for c in s0_prots
                        if str(c.get("protomer") or c["id"]) == target_p
                    ]
                    design_cid = matching_cids[0] if matching_cids else (str(s0_prots[-1]["id"]) if s0_prots else "B")
                    fixed_cids = [str(c["id"]) for c in s0_prots if str(c["id"]) != design_cid]
                    prop = propose_mutant(
                        step_rng,
                        cur_seqs[target_p],
                        mutable_masks[target_p],
                        n_mut=n_mut,
                        method=str(cfg.get("mutation_method", "blosum")),
                        blosum_temp=float(cfg.get("blosum_temp", 1.0)),
                        p_nonint=float(cfg.get("p_nonint", 0.0)),
                        exclude_aa=exclude_aa,
                        forbid_sequon=forbid_sequon,
                        mpnn=mpnn,
                        struct_path=ref_struct if ref_struct.exists() else None,
                        tmp_dir=tmp_root,
                        seed=_step_seed(seed, step, 10 + k),
                        design_chain=design_cid,
                        fixed_chains=fixed_cids,
                        allow_cur_forbidden=allow_init_forbidden,
                        wf_cap_frac=wf_cap_frac,
                        step=step,
                    )
                    cand_seqs[target_p] = prop["seq"]
                    mut_str = (
                        f"{target_p}:{prop['muts']}" if len(protomers) > 1 else prop["muts"]
                    )
                    candidates.append((cand_seqs, mut_str, prop["method"]))

            # Score all proposals via persistent AF3 predictor and pick best
            scored_cands: list[tuple[float, dict[str, str], dict[str, Any], dict[str, AF3Prediction], str, bool]] = []
            step_work = work_root / f"step_{step:04d}"
            for k, (cand_seqs, mut_str, _meth) in enumerate(candidates):
                cand_seed = _step_seed(seed, step, 100 + k)
                cand_wdir = step_work / f"p{k}"
                cand_tag = f"s{step}_p{k}"
                neg_evaluated = True
                cur_states = cur_eval.get("states", {})
                if can_lazy_neg and all(nn in cur_states for nn in neg_names):
                    preds: dict[str, AF3Prediction] = {}
                    pos_ctx: dict[str, dict[str, Any]] = {}
                    for sp in pos_specs:
                        sn = str(sp["name"])
                        p_st = predictor.predict_state(
                            sp, cand_seqs, seed=cand_seed, work_dir=cand_wdir, tag=cand_tag
                        )
                        preds[sn] = p_st
                        pos_ctx[sn] = extract_state_context(
                            sp,
                            p_st,
                            ref_cxs.get(sn) if ref_cxs else None,
                            core_positions,
                            cfg,
                            steric_ref=steric_ref,
                        )
                    prov_ctx = dict(pos_ctx)
                    for nn in neg_names:
                        prov_ctx[nn] = dict(cur_states[nn])
                    ev_prov = evaluate_multistate(
                        state_specs,
                        preds,
                        ref_cxs,
                        core_positions,
                        cfg,
                        steric_ref=steric_ref,
                        protomer_seqs=cand_seqs,
                        precomputed_state_ctx=prov_ctx,
                    )
                    pos_improved = (
                        float(ev_prov["J"]) > float(cur_eval["J"])
                        or float(ev_prov["J"]) > float(best_eval["J"])
                    )
                    if not pos_improved:
                        for sp in pos_specs:
                            sn = str(sp["name"])
                            if sn in cur_states:
                                g_new = pos_ctx[sn]
                                g_cur = cur_states[sn]
                                if (
                                    float(g_new["ipsae_min"]) > float(g_cur["ipsae_min"])
                                    or float(g_new["iptm"]) > float(g_cur["iptm"])
                                ):
                                    pos_improved = True
                                    break
                    if (step % lazy_neg_interval == 0) or pos_improved:
                        for n_spec in neg_specs:
                            nsn = str(n_spec["name"])
                            preds[nsn] = predictor.predict_state(
                                n_spec, cand_seqs, seed=cand_seed, work_dir=cand_wdir, tag=cand_tag
                            )
                        ev = evaluate_multistate(
                            state_specs,
                            preds,
                            ref_cxs,
                            core_positions,
                            cfg,
                            steric_ref=steric_ref,
                            protomer_seqs=cand_seqs,
                            precomputed_state_ctx=pos_ctx,
                        )
                        neg_evaluated = True
                    else:
                        ev = ev_prov
                        neg_evaluated = False
                else:
                    preds = predictor.predict_all(
                        cand_seqs,
                        seed=cand_seed,
                        work_dir=cand_wdir,
                        tag=cand_tag,
                    )
                    ev = evaluate_multistate(
                        state_specs,
                        preds,
                        ref_cxs,
                        core_positions,
                        cfg,
                        steric_ref=steric_ref,
                        protomer_seqs=cand_seqs,
                    )
                scored_cands.append((float(ev["loss"]), cand_seqs, ev, preds, mut_str, neg_evaluated))

            scored_cands.sort(key=lambda item: item[0])
            loss_new, cand_seqs, new_eval, new_preds, mut_str, neg_evaluated = scored_cands[0]

            acc, reason, p_acc, _ = metropolis_accept(
                loss_new, float(cur_eval["loss"]), T, float(cfg.get("margin", 0.0)), acc_rng
            )
            delta_j = float(new_eval["J"]) - float(cur_eval["J"])
            loss_cur_before = float(cur_eval["loss"])
            j_cur_before = float(cur_eval["J"])

            if acc:
                val_ex = "" if allow_init_forbidden else exclude_aa
                val_seq = False if allow_init_forbidden else forbid_sequon
                for pk, sv in cand_seqs.items():
                    validate_binder_seq(
                        sv, f"accepted {pk}", exclude_aa=val_ex, forbid_sequon=val_seq
                    )
                cur_seqs = dict(cand_seqs)
                cur_eval = dict(new_eval)
                cur_cifs.update(_copy_state_cifs(new_preds, run_dir, "cur"))
                if float(new_eval["loss"]) < float(best_eval["loss"]):
                    best_seqs = dict(cand_seqs)
                    best_eval = dict(new_eval)
                    _copy_state_cifs(new_preds, run_dir, "best")
                if _is_clean_protomers(cand_seqs, exclude_aa=exclude_aa, forbid_sequon=forbid_sequon):
                    if best_clean_eval is None or float(new_eval["loss"]) < float(best_clean_eval["loss"]):
                        best_clean_seqs = dict(cand_seqs)
                        best_clean_eval = dict(new_eval)
                        _copy_state_cifs(new_preds, run_dir, "best_clean")

            shutil.rmtree(step_work, ignore_errors=True)

            rec: dict[str, Any] = {
                "step": step,
                "accepted": bool(acc),
                "reason": reason,
                "temperature": float(T),
                "n_mut": int(n_mut),
                "mutations": mut_str,
                "sequence": _seq_signature(cur_seqs),
                "proposed_sequence": _seq_signature(cand_seqs),
                "protomers": dict(cur_seqs),
                "cys_cnt": sum(s.count("C") for s in cur_seqs.values()),
                "pro_cnt": sum(s.count("P") for s in cur_seqs.values()),
                "sequon_cnt": sum(sequon_count(s) for s in cur_seqs.values()),
                "loss": float(new_eval["loss"]),
                "loss_cur": float(cur_eval["loss"]),
                "best_loss": float(best_eval["loss"]),
                "J": float(new_eval["J"]),
                "J_cur": float(cur_eval["J"]),
                "best_J": float(best_eval["J"]),
                "delta_J": float(delta_j),
                "p_accept": float(p_acc),
                "neg_evaluated": bool(neg_evaluated),
                "bare_evaluated": bool(neg_evaluated),
                "states": new_eval["states"],
                "t_step_s": time.time() - t_s0,
            }
            for extra_k in _EXTRA_LOG_KEYS:
                if extra_k in new_eval:
                    rec[extra_k] = new_eval[extra_k]

            _append_jsonl(steps_jsonl, rec)
            _write_json_atomic(
                state_json,
                {
                    "step_done": step,
                    "cur_seqs": cur_seqs,
                    "cur_eval": cur_eval,
                    "best_seqs": best_seqs,
                    "best_eval": best_eval,
                    "best_clean_seqs": best_clean_seqs,
                    "best_clean_eval": best_clean_eval,
                    "mutable_masks": mutable_masks,
                    "core_positions": core_positions,
                    "ref_cifs": ref_paths,
                    "cur_cifs": cur_cifs,
                },
            )
            tag = "ACCEPTED" if acc else "REJECTED"
            el = time.time() - t_run0
            pct = 100.0 * step / max(1, total_steps)
            log(
                f"{tag} step={step}/{total_steps} ({pct:.1f}%) T={T:.4f} mut={mut_str} "
                f"J={j_cur_before:.4f}->{new_eval['J']:.4f} (best={best_eval['J']:.4f}) "
                f"loss={loss_cur_before:.4f}->{new_eval['loss']:.4f} p={p_acc:.3f} elapsed={el:.1f}s"
            )

        best_cifs = {
            str(s["name"]): str(run_dir / f"best_{s['name']}.cif") for s in state_specs
        }
        best_clean_cifs = (
            {str(s["name"]): str(run_dir / f"best_clean_{s['name']}.cif") for s in state_specs}
            if best_clean_seqs is not None
            else {}
        )
        best_summary: dict[str, Any] = {
            "job": job,
            "steps_completed": total_steps,
            "best_sequence": _seq_signature(best_seqs),
            "best_protomers": best_seqs,
            "cys_cnt": sum(s.count("C") for s in best_seqs.values()),
            "pro_cnt": sum(s.count("P") for s in best_seqs.values()),
            "sequon_cnt": sum(sequon_count(s) for s in best_seqs.values()),
            "best_J": float(best_eval["J"]),
            "best_loss": float(best_eval["loss"]),
            "best_metrics": best_eval,
            "best_cifs": best_cifs,
            "best_clean_sequence": _seq_signature(best_clean_seqs) if best_clean_seqs is not None else None,
            "best_clean_protomers": best_clean_seqs,
            "best_clean_J": float(best_clean_eval["J"]) if best_clean_eval is not None else None,
            "best_clean_loss": float(best_clean_eval["loss"]) if best_clean_eval is not None else None,
            "best_clean_metrics": best_clean_eval,
            "best_clean_cifs": best_clean_cifs,
        }
        _write_json_atomic(run_dir / "best_summary.json", best_summary)
        if best_clean_seqs is not None and best_clean_eval is not None:
            _write_json_atomic(
                run_dir / "best_clean_summary.json",
                {
                    "job": job,
                    "steps_completed": total_steps,
                    "best_sequence": _seq_signature(best_clean_seqs),
                    "best_protomers": best_clean_seqs,
                    "cys_cnt": 0,
                    "pro_cnt": 0,
                    "sequon_cnt": 0,
                    "best_J": float(best_clean_eval["J"]),
                    "best_loss": float(best_clean_eval["loss"]),
                    "best_metrics": best_clean_eval,
                    "best_cifs": best_clean_cifs,
                },
            )

        with (run_dir / "candidates.tsv").open("w", encoding="utf-8", newline="") as fh:
            cols = [
                "design_id",
                "sequence",
                "cys_cnt",
                "pro_cnt",
                "sequon_cnt",
                "steric_hv25",
                "steric_ca38",
                "intact_clash_ok",
                "best_J",
                "best_loss",
            ]
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
            w.writeheader()
            cand_row_seqs = best_clean_seqs if best_clean_seqs is not None else best_seqs
            cand_row_eval = best_clean_eval if best_clean_eval is not None else best_eval
            w.writerow(
                {
                    "design_id": job,
                    "sequence": _seq_signature(cand_row_seqs),
                    "cys_cnt": sum(s.count("C") for s in cand_row_seqs.values()),
                    "pro_cnt": sum(s.count("P") for s in cand_row_seqs.values()),
                    "sequon_cnt": sum(sequon_count(s) for s in cand_row_seqs.values()),
                    "steric_hv25": int(cand_row_eval.get("steric_hv25", 0)),
                    "steric_ca38": int(cand_row_eval.get("steric_ca38", 0)),
                    "intact_clash_ok": int(cand_row_eval.get("intact_clash_ok", 1)),
                    "best_J": f"{cand_row_eval['J']:.4f}",
                    "best_loss": f"{cand_row_eval['loss']:.4f}",
                }
            )
        if cfg.get("export_candidates_tsv"):
            export_candidates_tsv(run_dir, cfg.get("export_candidates_tsv"), cfg=cfg)
        (run_dir / ".done_af3_hall").touch()
        log(f"DONE best_J={best_eval['J']:.4f} best_loss={best_eval['loss']:.4f} out={run_dir}")
        return run_dir
    finally:
        log.close()


CANDIDATE_EXPORT_COLS: list[str] = [
    "design_id",
    "sequence",
    "design_struct",
    "cys_cnt",
    "pro_cnt",
    "sequon_cnt",
    "steric_hv25",
    "steric_ca38",
    "intact_clash_ok",
    "best_J",
    "best_loss",
]


def export_candidates_tsv(
    run_dir: str | Path,
    out_tsv: str | Path | None = None,
    *,
    top_k: int = 10,
    cfg: Mapping[str, Any] | None = None,
) -> Path:
    """Export top-K distinct hygiene-clean candidates from an `af3-hall` run directory."""
    rdir = Path(run_dir)
    if cfg is None and (rdir / "config.json").exists():
        cfg = json.loads((rdir / "config.json").read_text(encoding="utf-8"))
    cfg = cfg or {}

    job = str(cfg.get("job") or cfg.get("name") or rdir.name)
    clean_cifs = sorted(rdir.glob("best_clean_*.cif"))
    best_cifs = sorted(c for c in rdir.glob("best_*.cif") if not c.name.startswith("best_clean_"))
    primary_cif = clean_cifs[0] if clean_cifs else (best_cifs[0] if best_cifs else None)

    seen: dict[str, dict[str, Any]] = {}

    def _add_clean_cand(seq_str: str, j_val: float, loss_val: float, metrics: Mapping[str, Any]) -> None:
        s = seq_str.strip()
        if not s or cp_count(s) or sequon_count(s):
            return
        if s not in seen or j_val > float(seen[s]["J"]):
            seen[s] = {
                "sequence": s,
                "cys_cnt": 0,
                "pro_cnt": 0,
                "sequon_cnt": 0,
                "steric_hv25": int(metrics.get("steric_hv25", 0)),
                "steric_ca38": int(metrics.get("steric_ca38", 0)),
                "intact_clash_ok": int(metrics.get("intact_clash_ok", 1)),
                "J": float(j_val),
                "loss": float(loss_val),
            }

    if (rdir / "best_summary.json").exists():
        bs = json.loads((rdir / "best_summary.json").read_text(encoding="utf-8"))
        bc_seq = str(bs.get("best_clean_sequence") or "").strip()
        if bc_seq:
            _add_clean_cand(
                bc_seq,
                float(bs.get("best_clean_J", 0.0)),
                float(bs.get("best_clean_loss", 0.0)),
                bs.get("best_clean_metrics") or {},
            )
        b_seq = str(bs.get("best_sequence") or "").strip()
        if b_seq:
            _add_clean_cand(
                b_seq,
                float(bs.get("best_J", 0.0)),
                float(bs.get("best_loss", 0.0)),
                bs.get("best_metrics") or {},
            )

    steps_path = rdir / "steps.jsonl"
    if steps_path.exists():
        for ln in steps_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            seq = str(row.get("proposed_sequence") or row.get("sequence") or "").strip()
            if not seq or cp_count(seq) or sequon_count(seq):
                continue
            j_val = float(row.get("J", 0.0))
            if seq not in seen or j_val > float(seen[seq]["J"]):
                seen[seq] = {
                    "sequence": seq,
                    "cys_cnt": 0,
                    "pro_cnt": 0,
                    "sequon_cnt": 0,
                    "steric_hv25": int(row.get("steric_hv25", 0)),
                    "steric_ca38": int(row.get("steric_ca38", 0)),
                    "intact_clash_ok": int(row.get("intact_clash_ok", 1)),
                    "J": j_val,
                    "loss": float(row.get("loss", -j_val)),
                }

    ranked = sorted(seen.values(), key=lambda r: float(r["J"]), reverse=True)[: max(1, int(top_k))]
    target_path = Path(out_tsv) if out_tsv is not None else (rdir / "top_candidates.tsv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ref_cifs = sorted(rdir.glob("ref_*.cif"))
    fallback_cif = ref_cifs[0] if ref_cifs else primary_cif
    with target_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CANDIDATE_EXPORT_COLS, delimiter="\t")
        w.writeheader()
        for idx, rec in enumerate(ranked):
            did = job if idx == 0 else f"{job}_cand{idx:02d}"
            d_struct = primary_cif if idx == 0 else fallback_cif
            w.writerow(
                {
                    "design_id": did,
                    "sequence": rec["sequence"],
                    "design_struct": str(d_struct) if d_struct is not None else "",
                    "cys_cnt": rec["cys_cnt"],
                    "pro_cnt": rec["pro_cnt"],
                    "sequon_cnt": rec["sequon_cnt"],
                    "steric_hv25": rec["steric_hv25"],
                    "steric_ca38": rec["steric_ca38"],
                    "intact_clash_ok": rec["intact_clash_ok"],
                    "best_J": f"{float(rec['J']):.4f}",
                    "best_loss": f"{float(rec['loss']):.4f}",
                }
            )
    return target_path
