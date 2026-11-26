"""Configuration defaults, portable env-var resolution, YAML/JSON loading, and N-state builders."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

AA3_TO_1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

DEFAULTS: dict[str, Any] = {
    "name": "af3_hall_run",
    "job": None,
    "out_dir": None,
    "out_root": "./runs",
    "log_dir": None,
    "tmp_root": "./tmp",
    "jax_cache_dir": "./tmp/jax_cache_af3",
    "af3_dir": "./alphafold3",
    "model_dir": "./models",
    "db_dir": "./public_databases",
    # General multistate spec (1..N states). When None, built from `oligo` or `target_*`/`ligand_*`.
    "states": None,
    "oligo": None,
    "include_bare": True,
    # Fixed receptor / glycan / ligand convenience config
    "target_seq": None,
    "target_fasta": None,
    "target_pdb": None,
    "target_chain": "A",
    "target_n_pos": None,
    "glycoform": None,
    "target_msa_json": None,
    "ligand_ccd": None,
    "ligand_smiles": None,
    # Optional in-loop steric clash repulsion against a full-length/context reference structure
    "steric_clash_ref_cif": None,
    "steric_clash_weight": 0.0,
    "steric_ref_residues": None,
    "steric_pred_residues": None,
    "steric_hv_cutoff": 2.5,
    "steric_ca_cutoff": 3.8,
    # Optional candidate export path
    "export_candidates_tsv": None,
    # Evolving sequence start & reference pose
    "binder_seq": None,
    "seqs": None,
    "seq_file": None,
    "seq_ids": None,
    "binder_length": 80,
    "ref_cif": None,
    # AF3 inference parameters
    "backend": "in_memory",  # in_memory | subprocess | mock
    "num_recycles": 3,
    "num_diffusion_samples": 1,
    "flash_attention": "triton",
    "gpu_device": 0,
    "buckets": [256, 384, 512, 768, 1024, 1536, 2048],
    # Optional tmol FastRelax of every predicted state BEFORE scoring (CLI --fastrelax); see docs/fastrelax.md
    "fastrelax": False,
    "relax_scope": "all",  # all = whole-complex Cartesian FastRelax | shell = backbone-fixed interface side-chain repack+min
    "relax_repeats": 1,
    "relax_shell": 8.0,  # shell scope: relax residues of the mutable chain(s) within this distance (A) of the other chains
    "relax_cartesian": True,
    "relax_on_fail": "raise",  # raise | skip (keep the raw AF3 structure and flag relax ok=False)
    "relax_python": None,  # interpreter with torch + tmol + rfproteina; default: env AF3_HALL_RELAX_PYTHON, else "python"
    "relax_device": "cuda:0",
    "relax_timeout_s": 900,
    # Per-step structure trajectory in <run>/trajectory/ is ALWAYS written for accepted steps; "all" also keeps rejected candidates.
    "save_trajectory": "accepted",
    # Simulated-annealing MCMC parameters
    "seed": 0,
    "steps": 100,
    "proposals_per_step": 1,
    "T_init": 0.05,
    "half_life": 50.0,
    "temp_schedule": "exp",
    "T_final": None,
    "margin": 0.0,
    "mutation_rate": "3-1",
    "mutation_method": "blosum",
    "blosum_temp": 1.0,
    "wf_cap_frac": 0.12,
    "mpnn_dir": "",
    "mpnn_weights": "proteinmpnn_v_48_020",
    "mpnn_python": sys.executable,
    "mpnn_checkpoint": "",
    "mpnn_refresh_interval": 5,
    "mpnn_temp": 0.3,
    # Mutable-position mask & sequence hygiene
    "mask_mode": "auto",  # auto | interface | all | explicit
    "interface_dist": 6.0,
    "explicit_positions": None,
    "p_nonint": 0.0,
    "mask_refresh_interval": 0,
    "lazy_neg_interval": 1,
    "lazy_bare_interval": None,  # backward-compatible alias for lazy_neg_interval
    "exclude_aa": "CP",
    "forbid_sequon": True,
    "allow_init_forbidden": False,
    # Composable losses (registered via @register_loss / @register_multistate_loss)
    "losses": None,
    "loss_weights": None,
    # Multistate glyco-delta & additive objective weights
    "w1_ipsae_glyco": 1.0,
    "w2_delta_iptm": 1.0,
    "w3_delta_ipsae": 1.0,
    "w4_hbonds_glycan": 0.05,
    "w5_rmsd_core": 0.10,
    "w_glycan_contacts": 0.0,
    "w_nonstem_sugars": 0.0,
    "w_ch_pi": 0.0,
    "w_binder_plddt": 0.0,
    "w_stereo_penalty": 0.0,
    "w_rg_penalty": 0.0,
    "w_excess_target_contacts": 0.0,
    "target_max_contacts": 14,
    "w_forbidden_aa": 0.0,
    "pae_cutoff": 10.0,
    "dist_cutoff": 10.0,
    "hbond_cutoff": 3.5,
    # Deterministic test hook: {step: [full_seq, ...]}
    "force_proposals": None,
}

_ENV_OVERRIDES: dict[str, str] = {
    "af3_dir": "AF3_CODE_DIR",
    "model_dir": "AF3_MODEL_DIR",
    "db_dir": "AF3_DB_DIR",
    "jax_cache_dir": "AF3_JAX_CACHE_DIR",
    "mpnn_dir": "LIGANDMPNN_DIR",
    "mpnn_checkpoint": "LIGANDMPNN_CHECKPOINT",
    "mpnn_python": "LIGANDMPNN_PYTHON",
    "out_root": "AF3_HALL_OUT_ROOT",
    "tmp_root": "AF3_HALL_TMP_DIR",
}


def get_defaults() -> dict[str, Any]:
    """Return a fresh copy of `DEFAULTS` hydrated from portable environment variables."""
    cfg = copy.deepcopy(DEFAULTS)
    for cfg_key, env_name in _ENV_OVERRIDES.items():
        env_val = os.environ.get(env_name)
        if env_val:
            cfg[cfg_key] = env_val
    return cfg


def _strip_yaml_comment(line: str) -> str:
    in_sq = in_dq = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "#" and not in_sq and not in_dq and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _parse_yaml_scalar(val: str) -> Any:
    s = val.strip()
    if s in ("", "null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s.replace("'", '"'))
        except json.JSONDecodeError:
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                return [_parse_yaml_scalar(x) for x in inner.split(",")] if inner else []
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _fallback_yaml_load(text: str) -> Any:
    """Pure-Python subset YAML loader (mappings, nested lists, inline collections, scalars)."""
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = _strip_yaml_comment(raw).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))

    if not lines:
        return {}
    if len(lines) == 1 and ":" not in lines[0][1] and not lines[0][1].startswith("- "):
        return _parse_yaml_scalar(lines[0][1])

    def _parse_block(idx: int, min_indent: int) -> tuple[Any, int]:
        if idx >= len(lines):
            return {}, idx
        if lines[idx][1].startswith("- "):
            arr: list[Any] = []
            list_indent = lines[idx][0]
            while idx < len(lines) and lines[idx][0] == list_indent and lines[idx][1].startswith("- "):
                body = lines[idx][1][2:].strip()
                idx += 1
                if not body:
                    child, idx = _parse_block(idx, list_indent + 1)
                    arr.append(child)
                elif ":" in body and not body.startswith(("{", "[", "'", '"')):
                    k, v = body.split(":", 1)
                    item: dict[str, Any] = {}
                    v_s = v.strip()
                    if v_s:
                        item[k.strip()] = _parse_yaml_scalar(v_s)
                    else:
                        child, idx = _parse_block(idx, list_indent + 1)
                        item[k.strip()] = child
                    while idx < len(lines) and lines[idx][0] > list_indent:
                        sub_k, sub_v = lines[idx][1].split(":", 1)
                        sub_ind = lines[idx][0]
                        idx += 1
                        if sub_v.strip():
                            item[sub_k.strip()] = _parse_yaml_scalar(sub_v)
                        else:
                            child, idx = _parse_block(idx, sub_ind + 1)
                            item[sub_k.strip()] = child
                    arr.append(item)
                else:
                    arr.append(_parse_yaml_scalar(body))
            return arr, idx

        mapping: dict[str, Any] = {}
        map_indent = lines[idx][0]
        while idx < len(lines) and lines[idx][0] >= min_indent:
            ind, content = lines[idx]
            if ind < map_indent or content.startswith("- "):
                break
            if ":" not in content:
                idx += 1
                continue
            k, v = content.split(":", 1)
            key = k.strip().strip("\"'")
            val_s = v.strip()
            idx += 1
            if val_s:
                mapping[key] = _parse_yaml_scalar(val_s)
            elif idx < len(lines) and lines[idx][0] > ind:
                child, idx = _parse_block(idx, ind + 1)
                mapping[key] = child
            else:
                mapping[key] = None
        return mapping, idx

    parsed, _ = _parse_block(0, 0)
    return parsed


def _parse_scalar_or_yaml(text: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return _fallback_yaml_load(text)


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load default config merged with optional YAML/JSON file and `key=value` overrides."""
    cfg = get_defaults()
    explicit_keys: set[str] = set()
    if path is not None:
        raw_text = Path(path).read_text(encoding="utf-8")
        user = _parse_scalar_or_yaml(raw_text) or {}
        if not isinstance(user, Mapping):
            raise ValueError(f"Config file {path} must parse to a mapping")
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(user)
        explicit_keys.update(user.keys())

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in DEFAULTS:
            raise KeyError(f"Unknown config key {k!r}")
        cfg[k] = _parse_scalar_or_yaml(v)
        explicit_keys.add(k)

    # Synchronize lazy_neg_interval and backward-compatible lazy_bare_interval alias
    if "lazy_bare_interval" in explicit_keys and "lazy_neg_interval" not in explicit_keys:
        if cfg["lazy_bare_interval"] is not None:
            cfg["lazy_neg_interval"] = int(cfg["lazy_bare_interval"])
    cfg["lazy_neg_interval"] = int(cfg.get("lazy_neg_interval") or 1)
    cfg["lazy_bare_interval"] = cfg["lazy_neg_interval"]

    if cfg.get("mutation_rate") is not None:
        cfg["mutation_rate"] = str(cfg["mutation_rate"])
    if cfg.get("loss_weights") is not None and not isinstance(cfg["loss_weights"], (list, tuple)):
        cfg["loss_weights"] = str(cfg["loss_weights"])
    if cfg.get("buckets") is not None:
        if isinstance(cfg["buckets"], str):
            cfg["buckets"] = [int(x.strip()) for x in cfg["buckets"].split(",") if x.strip()]
        else:
            cfg["buckets"] = [int(x) for x in cfg["buckets"]]
    return cfg


def _seq_from_fasta(fasta_path: Path) -> str:
    lines = fasta_path.read_text(encoding="utf-8").splitlines()
    return "".join(ln.strip() for ln in lines if ln.strip() and not ln.startswith(">")).upper()


def _seq_from_pdb(pdb_path: Path, chain: str = "A") -> str:
    seq: list[str] = []
    seen_res: set[str] = set()
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA" or line[21].strip() != chain:
            continue
        res_key = line[22:27].strip()
        if res_key in seen_res:
            continue
        seen_res.add(res_key)
        resname = line[17:20].strip()
        if resname in AA3_TO_1:
            seq.append(AA3_TO_1[resname])
    return "".join(seq)


def resolve_target_seq(cfg: Mapping[str, Any]) -> str | None:
    """Resolve uppercase 1-letter fixed receptor sequence (`target_seq`/`target_fasta`/`target_pdb`)."""
    if cfg.get("target_seq"):
        return str(cfg["target_seq"]).strip().upper()
    if cfg.get("target_fasta"):
        seq = _seq_from_fasta(Path(cfg["target_fasta"]))
        if seq:
            return seq
    if cfg.get("target_pdb"):
        p = Path(cfg["target_pdb"])
        if p.suffix.lower() in (".cif", ".mmcif"):
            from .structure import load_structure

            cx = load_structure(p, target_chain=str(cfg.get("target_chain", "A")))
            seq = "".join(AA3_TO_1.get(rn, "X") for rn in cx.target.resnames)
            return seq or None
        seq = _seq_from_pdb(p, chain=str(cfg.get("target_chain", "A")))
        if seq:
            return seq
    return None


def parse_oligo_states(oligo_str: str) -> list[dict[str, Any]]:
    """Convert AFhall `--oligo` string (e.g. 'A+', 'AB+,AA-,BB-') into general StateSpec dicts."""
    states: list[dict[str, Any]] = []
    for raw in oligo_str.split(","):
        tok = raw.strip()
        if not tok:
            continue
        sign = 1.0
        if tok.endswith("+"):
            chain_letters = tok[:-1]
            sign = 1.0
        elif tok.endswith("-"):
            chain_letters = tok[:-1]
            sign = -1.0
        else:
            chain_letters = tok
        if not chain_letters or not chain_letters.isalpha():
            raise ValueError(f"Invalid oligo token {raw!r}")
        chains = [
            {"type": "protein", "id": chr(ord("A") + i), "mutable": True, "protomer": ch.upper()}
            for i, ch in enumerate(chain_letters)
        ]
        states.append({"name": tok, "weight": sign, "chains": chains})
    if not states:
        raise ValueError(f"Empty oligo specification: {oligo_str!r}")
    return states


def build_glyco_delta_states(
    target_seq: str,
    anchor_res_1based: int | None,
    glycoform: str | None,
    *,
    target_msa_json: str | Path | None = None,
    include_bare: bool = True,
) -> list[dict[str, Any]]:
    """Build 1- or 2-state (glyco, bare) StateSpec list from target sequence and optional glycoform."""
    target_entity: dict[str, Any] = {
        "type": "protein",
        "id": "A",
        "mutable": False,
        "sequence": target_seq,
    }
    if target_msa_json is not None:
        target_entity["msa_json"] = str(target_msa_json)

    binder_entity: dict[str, Any] = {
        "type": "protein",
        "id": "B",
        "mutable": True,
        "protomer": "B",
    }

    states: list[dict[str, Any]] = []
    if glycoform is not None:
        if anchor_res_1based is None:
            raise ValueError("target_n_pos must be specified when glycoform is set")
        states.append(
            {
                "name": "glyco",
                "weight": 1.0,
                "chains": [
                    copy.deepcopy(target_entity),
                    copy.deepcopy(binder_entity),
                    {
                        "type": "glycan",
                        "anchor_chain": "A",
                        "anchor_res_1based": int(anchor_res_1based),
                        "glycoform": str(glycoform),
                    },
                ],
            }
        )
    if include_bare or glycoform is None:
        states.append(
            {
                "name": "bare" if glycoform is not None else "complex",
                "weight": -1.0 if glycoform is not None else 1.0,
                "chains": [
                    copy.deepcopy(target_entity),
                    copy.deepcopy(binder_entity),
                ],
            }
        )
    return states


def build_states_from_config(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve `1..N` StateSpec list from explicit `states`, `oligo`, `target_*`, or `ligand_*`."""
    if cfg.get("states"):
        return copy.deepcopy(list(cfg["states"]))
    if cfg.get("oligo"):
        return parse_oligo_states(str(cfg["oligo"]))

    t_seq = resolve_target_seq(cfg)
    if t_seq:
        n_pos = int(cfg["target_n_pos"]) if cfg.get("target_n_pos") is not None else None
        states = build_glyco_delta_states(
            target_seq=t_seq,
            anchor_res_1based=n_pos,
            glycoform=cfg.get("glycoform"),
            target_msa_json=cfg.get("target_msa_json"),
            include_bare=bool(cfg.get("include_bare", True)),
        )
        if cfg.get("ligand_ccd") or cfg.get("ligand_smiles"):
            lig_ent: dict[str, Any] = {"type": "ligand", "id": "L"}
            if cfg.get("ligand_ccd"):
                lig_ent["ccdCodes"] = [str(cfg["ligand_ccd"]).strip().upper()]
            if cfg.get("ligand_smiles"):
                lig_ent["smiles"] = str(cfg["ligand_smiles"])
            for st in states:
                st["chains"].append(copy.deepcopy(lig_ent))
        return states

    if cfg.get("ligand_ccd") or cfg.get("ligand_smiles"):
        lig_ent = {"type": "ligand", "id": "L"}
        if cfg.get("ligand_ccd"):
            lig_ent["ccdCodes"] = [str(cfg["ligand_ccd"]).strip().upper()]
        if cfg.get("ligand_smiles"):
            lig_ent["smiles"] = str(cfg["ligand_smiles"])
        return [
            {
                "name": "ligand_complex",
                "weight": 1.0,
                "chains": [
                    {"type": "protein", "id": "B", "mutable": True, "protomer": "B"},
                    lig_ent,
                ],
            }
        ]

    return [
        {
            "name": "monomer",
            "weight": 1.0,
            "chains": [{"type": "protein", "id": "A", "mutable": True, "protomer": "A"}],
        }
    ]
