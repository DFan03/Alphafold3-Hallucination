"""Mutation proposals with configurable banned AA (0 Cys / 0 Pro) & 0 N-X-S/T sequon guards."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .structure import cif_to_pdb

CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_ALLOWED = "ADEFGHIKLMNQRSTVWY"  # default allowed set (no Cys, no Pro)
FORBIDDEN_AAS: frozenset[str] = frozenset("CP")
SEQUON_LOOKAHEAD_RE = re.compile(r"(?=N[^P][ST])")
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Carbohydrate-binding log-odds bias favoring polar H-bonders, charged anchors, and CH-pi aromatics
_CARB_AA_BIAS: dict[str, float] = {
    "N": 1.2,
    "Q": 1.2,
    "D": 1.1,
    "E": 1.1,
    "S": 1.0,
    "T": 1.0,
    "R": 1.0,
    "K": 0.9,
    "Y": 1.2,
    "W": 1.2,
    "H": 0.5,
    "F": 0.2,
    "A": -1.2,
}

_BG_FREQ: dict[str, float] = {
    "A": 8.25, "C": 1.38, "D": 5.45, "E": 6.75, "F": 3.86,
    "G": 7.07, "H": 2.27, "I": 5.96, "K": 5.84, "L": 9.66,
    "M": 2.42, "N": 4.06, "P": 4.74, "Q": 3.93, "R": 5.53,
    "S": 6.56, "T": 5.34, "V": 6.87, "W": 1.08, "Y": 2.92,
}

# Standard symmetric BLOSUM62 scores over CANONICAL_AAS
_BLOSUM62_ORDER = CANONICAL_AAS
_BLOSUM62_ROWS: dict[str, tuple[int, ...]] = {
    "A": ( 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2),
    "C": ( 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2),
    "D": (-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3),
    "E": (-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2, 0, 0,-1,-2,-3,-2),
    "F": (-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3),
    "G": ( 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3),
    "H": (-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2),
    "I": (-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1),
    "K": (-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2),
    "L": (-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1),
    "M": (-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1),
    "N": (-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2),
    "P": (-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3),
    "Q": (-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1),
    "R": (-1,-3,-2, 0,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2),
    "S": ( 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2),
    "T": ( 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2),
    "V": ( 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1),
    "W": (-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2),
    "Y": (-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7),
}


class ForbiddenSequenceError(AssertionError):
    """Raised when a sequence violates banned amino acids (Cys/Pro) or N-X-S/T sequon rules."""


def cp_count(seq: str) -> int:
    """Count Cys + Pro residues in `seq`."""
    return sum(seq.count(aa) for aa in FORBIDDEN_AAS)


def sequon_count(seq: str) -> int:
    """Count overlapping N-X-S/T (X != P) glycosylation sequons in `seq`."""
    return len(SEQUON_LOOKAHEAD_RE.findall(seq))


def forbidden_positions(
    seq: str,
    *,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
) -> list[int]:
    """Return sorted 0-based indices of residues violating `exclude_aa` or N-X-S/T sequons."""
    s = seq.strip().upper()
    banned = set(str(exclude_aa).upper())
    bad: set[int] = {i for i, ch in enumerate(s) if ch in banned}
    if forbid_sequon:
        for m in SEQUON_LOOKAHEAD_RE.finditer(s):
            st = m.start()
            bad.add(st)
            bad.add(st + 2)
    return sorted(bad)


def allowed_alphabet(exclude_aa: str = "CP") -> str:
    """Return canonical amino acids excluding all characters in `exclude_aa`."""
    banned = set(str(exclude_aa).upper())
    return "".join(a for a in CANONICAL_AAS if a not in banned)


def validate_binder_seq(
    seq: str,
    what: str = "sequence",
    *,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
) -> None:
    """Enforce banned AA (default 0 Cys / 0 Pro) and 0 N-X-S/T sequons on `seq`."""
    s = seq.strip().upper()
    if not s:
        raise ForbiddenSequenceError(f"{what} is empty")
    banned = set(str(exclude_aa).upper())
    bad_banned = [a for a in s if a in banned]
    if bad_banned:
        raise ForbiddenSequenceError(
            f"{what} contains {len(bad_banned)} forbidden residue(s) from {sorted(banned)}: {s}"
        )
    if forbid_sequon:
        n_seq = sequon_count(s)
        if n_seq > 0:
            raise ForbiddenSequenceError(f"{what} contains {n_seq} N-X-S/T sequon(s): {s}")
    bad = set(s) - set(CANONICAL_AAS)
    if bad:
        raise ForbiddenSequenceError(f"{what} contains non-canonical residues {sorted(bad)}")


def is_valid_binder_seq(
    seq: str,
    *,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
) -> bool:
    """Return True iff `seq` passes banned AA and N-X-S/T sequon rules."""
    try:
        validate_binder_seq(seq, exclude_aa=exclude_aa, forbid_sequon=forbid_sequon)
        return True
    except ForbiddenSequenceError:
        return False


def creates_sequon(seq_chars: Sequence[str], pos: int, aa: str) -> bool:
    """Return True if placing `aa` at 0-based `pos` creates an N-X-S/T sequon."""
    n = len(seq_chars)
    lo = max(0, pos - 2)
    hi = min(n, pos + 3)
    window = list(seq_chars[lo:hi])
    window[pos - lo] = aa
    return sequon_count("".join(window)) > 0


def random_sequence(
    length: int,
    rng: np.random.Generator,
    *,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
) -> str:
    """Generate a random sequence of `length` obeying `exclude_aa` and `forbid_sequon`."""
    alpha = allowed_alphabet(exclude_aa)
    probs = np.array([_BG_FREQ[a] for a in alpha], dtype=np.float64)
    probs /= probs.sum()
    buf: list[str] = []
    for pos in range(length):
        cands = [
            a for a in alpha if not (forbid_sequon and creates_sequon(buf + [a], pos, a))
        ]
        if not cands:
            cands = list(alpha)
        p = np.array([_BG_FREQ[a] for a in cands], dtype=np.float64)
        p /= p.sum()
        buf.append(str(rng.choice(cands, p=p)))
    seq = "".join(buf)
    validate_binder_seq(seq, "random sequence", exclude_aa=exclude_aa, forbid_sequon=forbid_sequon)
    return seq


def mutation_count(
    step: int,
    total_steps: int,
    rate: str = "3-1",
    rng: np.random.Generator | None = None,
) -> int:
    """Return number of residues to mutate at `step` (1-based step in 1..total_steps)."""
    r = str(rate).strip()
    if r.startswith("rand:"):
        lo_s, hi_s = r.split(":", 1)[1].split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if rng is None:
            return max(1, (lo + hi) // 2)
        return int(rng.integers(min(lo, hi), max(lo, hi) + 1))
    if "-" in r:
        hi_s, lo_s = r.split("-", 1)
        hi, lo = int(hi_s), int(lo_s)
        if total_steps <= 1:
            return hi
        return max(1, int(round(hi + (lo - hi) * (step - 1) / max(1, total_steps - 1))))
    return max(1, int(r))


def select_positions(
    rng: np.random.Generator,
    n_mut: int,
    mutable_positions: Sequence[int],
    seq_len: int,
    p_nonint: float = 0.0,
) -> list[int]:
    """Sample `n_mut` unique 0-based positions from `mutable_positions`."""
    all_pos = np.arange(seq_len, dtype=np.int64)
    pool = np.array(
        sorted({int(i) for i in mutable_positions if 0 <= int(i) < seq_len}),
        dtype=np.int64,
    )
    if len(pool) == 0:
        pool = all_pos
    rest = np.setdiff1d(all_pos, pool)

    k_target = min(int(n_mut), seq_len)
    chosen: list[int] = []
    for _ in range(k_target):
        use_rest = len(rest) > 0 and float(rng.random()) < float(p_nonint)
        src = rest if use_rest else pool
        src = np.setdiff1d(src, chosen)
        if len(src) == 0:
            src = np.setdiff1d(pool, chosen)
        if len(src) == 0:
            src = np.setdiff1d(all_pos, chosen)
        chosen.append(int(rng.choice(src)))
    return sorted(chosen)


def _allowed_candidates(
    seq_chars: list[str],
    pos: int,
    current_aa: str,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
    allow_cur_forbidden: bool = False,
) -> list[str]:
    alpha = allowed_alphabet(exclude_aa)
    out: list[str] = []
    for aa in alpha:
        if aa == current_aa:
            continue
        if forbid_sequon and creates_sequon(seq_chars, pos, aa):
            continue
        out.append(aa)
    if not out and allow_cur_forbidden:
        n = len(seq_chars)
        lo = max(0, pos - 2)
        hi = min(n, pos + 3)
        cur_w = "".join(seq_chars[lo:hi])
        cur_seq_cnt = sequon_count(cur_w)
        for aa in alpha:
            if aa == current_aa:
                continue
            win = list(seq_chars[lo:hi])
            win[pos - lo] = aa
            if sequon_count("".join(win)) <= cur_seq_cnt:
                out.append(aa)
    return out


def _apply_carb_bias_and_wf_cap(
    logits: np.ndarray,
    cands: Sequence[str],
    seq_chars: Sequence[str],
    wf_cap_frac: float = 0.12,
) -> np.ndarray:
    """Apply carbohydrate-binding AA log-odds bias and cap W/F fraction."""
    biased = np.asarray(logits, dtype=np.float64).copy()
    for idx, aa in enumerate(cands):
        biased[idx] += _CARB_AA_BIAS.get(aa, 0.0)
    probs = np.exp(biased - biased.max())
    wf_frac = (seq_chars.count("W") + seq_chars.count("F")) / max(1, len(seq_chars))
    if wf_frac >= float(wf_cap_frac):
        non_wf_mask = np.array([aa not in ("W", "F") for aa in cands], dtype=bool)
        if non_wf_mask.any():
            probs = np.where(non_wf_mask, probs, 0.0)
    return probs


def sample_residue(
    rng: np.random.Generator,
    seq_chars: list[str],
    pos: int,
    method: str = "blosum",
    blosum_temp: float = 1.0,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
    wf_cap_frac: float = 0.12,
    allow_cur_forbidden: bool = False,
) -> str:
    """Sample a substitution at `pos` respecting `exclude_aa` and `forbid_sequon`."""
    cur = seq_chars[pos]
    cands = _allowed_candidates(
        seq_chars,
        pos,
        cur,
        exclude_aa=exclude_aa,
        forbid_sequon=forbid_sequon,
        allow_cur_forbidden=allow_cur_forbidden,
    )
    if not cands:
        raise ForbiddenSequenceError(
            f"No valid substitution available at position {pos + 1} (exclude_aa={exclude_aa})"
        )

    m = method.lower()
    if m == "uniform":
        probs = np.ones(len(cands), dtype=np.float64)
    elif m == "frequency_adjusted":
        probs = np.array([_BG_FREQ[a] for a in cands], dtype=np.float64)
    elif m in ("blosum", "blosum62"):
        row = _BLOSUM62_ROWS.get(cur)
        if row is None:
            probs = np.array([_BG_FREQ[a] for a in cands], dtype=np.float64)
        else:
            temp = max(0.05, float(blosum_temp))
            scores = np.array([row[_BLOSUM62_ORDER.index(a)] for a in cands], dtype=np.float64)
            logits = (scores - scores.max()) / temp
            probs = np.exp(logits)
    elif m == "carb_blosum":
        row = _BLOSUM62_ROWS.get(cur)
        temp = max(0.05, float(blosum_temp))
        if row is None:
            logits = np.zeros(len(cands), dtype=np.float64)
        else:
            scores = np.array([row[_BLOSUM62_ORDER.index(a)] for a in cands], dtype=np.float64)
            logits = (scores - scores.max()) / temp
        probs = _apply_carb_bias_and_wf_cap(logits, cands, seq_chars, wf_cap_frac=wf_cap_frac)
    else:
        raise ValueError(f"Unknown mutation_method {method!r}")

    total = float(probs.sum())
    if total <= 0.0 or not np.isfinite(total):
        probs = np.ones(len(cands), dtype=np.float64) / len(cands)
    else:
        probs = probs / total
    return str(rng.choice(cands, p=probs))


class MPNNSampler:
    """ProteinMPNN conditional sampler on CPU fixing non-selected positions and omitting banned AAs."""

    def __init__(
        self,
        mpnn_dir: str | Path,
        weights: str = "proteinmpnn_v_48_020",
        temperature: float = 0.3,
        threads: int = 4,
        exclude_aa: str = "CP",
    ) -> None:
        import torch

        torch.set_num_threads(threads)
        mpnn_path = str(mpnn_dir)
        if mpnn_path not in sys.path:
            sys.path.insert(0, mpnn_path)
        import protein_mpnn_utils as U  # type: ignore[import-not-found]

        self.torch = torch
        self.U = U
        self.temp = float(temperature)
        weight_candidates = [
            Path(mpnn_path) / "vanilla_model_weights" / f"{weights}.pt",
            Path(weights),
        ]
        ck_path = next((p for p in weight_candidates if p.exists()), weight_candidates[0])
        ck = torch.load(str(ck_path), map_location="cpu")
        self.model = U.ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=ck["num_edges"],
        )
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        banned = set(str(exclude_aa).upper()) | {"X"}
        self.omit = np.array([a in banned for a in MPNN_ALPHABET], dtype=np.float32)

    def sample(
        self,
        struct_path: str | Path,
        positions: Sequence[int],
        binder_seq: str,
        seed: int,
        tmp_dir: str | Path = "./tmp",
        design_chain: str = "B",
        fixed_chains: Sequence[str] = ("A",),
    ) -> str:
        """Sample residues at 0-based `positions` given complex structure (converting CIF->PDB in `tmp_dir`)."""
        torch, U = self.torch, self.U
        torch.manual_seed(int(seed))
        src = Path(struct_path)
        pdb_path = src
        cleanup_pdb = False
        if src.suffix.lower() in (".cif", ".mmcif"):
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            pdb_path = Path(tmp_dir) / f"mpnn_{os.getpid()}_{seed}.pdb"
            cif_to_pdb(src, pdb_path)
            cleanup_pdb = True
        try:
            prot = U.StructureDatasetPDB(U.parse_PDB(str(pdb_path)), truncate=None, max_length=2000)[0]
            name = prot["name"]
            pos_set = set(int(p) for p in positions)
            fixed = [i + 1 for i in range(len(binder_seq)) if i not in pos_set]
            chain_id_dict = {name: ([design_chain], list(fixed_chains))}
            fixed_dict = {name: {**{fc: [] for fc in fixed_chains}, design_chain: fixed}}
            (
                X, S, mask, _, chain_M, chain_enc, _, _, _, _, chain_M_pos,
                omit_mask, res_idx, _, _, pssm_coef, pssm_bias, pssm_lo, bias_res, _
            ) = U.tied_featurize([prot], "cpu", chain_id_dict, fixed_dict, None, None, None, None, ca_only=False)
            with torch.no_grad():
                out = self.model.sample(
                    X, torch.randn(chain_M.shape), S, chain_M, chain_enc, res_idx,
                    mask=mask, temperature=self.temp, omit_AAs_np=self.omit,
                    bias_AAs_np=np.zeros(len(MPNN_ALPHABET)), chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                    pssm_multi=0.0, pssm_log_odds_flag=False,
                    pssm_log_odds_mask=(pssm_lo > 0.0).float(), pssm_bias_flag=False,
                    bias_by_res=bias_res,
                )
            seq_idx = out["S"][0][chain_M[0] > 0].tolist()
            seq = "".join(MPNN_ALPHABET[i] for i in seq_idx)
            if len(seq) != len(binder_seq):
                raise ValueError(f"MPNN output len {len(seq)} != seq len {len(binder_seq)}")
            return seq
        finally:
            if cleanup_pdb and pdb_path.exists():
                pdb_path.unlink(missing_ok=True)


class LigandMPNNBiasedSampler:
    """Subprocess LigandMPNN logit sampler (preserving sugar/ligand HETATM context) with carb bias & caching.

    Runs `LigandMPNN/run.py --save_stats 1` inside `mpnn_python` and exports pure-NumPy logits.
    Falls back cleanly to `carb_blosum` whenever `LIGANDMPNN_DIR`, checkpoints, or structures are unset/unavailable.
    """

    def __init__(
        self,
        mpnn_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        mpnn_python: str | Path | None = None,
        temperature: float = 0.3,
        exclude_aa: str = "CP",
        refresh_interval: int = 5,
        wf_cap_frac: float = 0.12,
        tmp_dir: str | Path = "./tmp",
    ) -> None:
        raw_dir = mpnn_dir if mpnn_dir is not None else os.environ.get("LIGANDMPNN_DIR", "")
        raw_ckpt = checkpoint if checkpoint is not None else os.environ.get("LIGANDMPNN_CHECKPOINT", "")
        raw_py = mpnn_python if mpnn_python is not None else os.environ.get("LIGANDMPNN_PYTHON", sys.executable)
        self.mpnn_dir = Path(raw_dir) if str(raw_dir).strip() else None
        self.checkpoint = Path(raw_ckpt) if str(raw_ckpt).strip() else None
        self.mpnn_python = Path(str(raw_py))
        self.temperature = float(temperature)
        self.exclude_aa = str(exclude_aa)
        self.refresh_interval = max(1, int(refresh_interval))
        self.wf_cap_frac = float(wf_cap_frac)
        self.tmp_dir = Path(tmp_dir)
        self._cache: dict[tuple[str, int, str], np.ndarray] = {}

    def is_available(self) -> bool:
        if self.mpnn_dir is None or self.checkpoint is None:
            return False
        return (
            self.mpnn_python.exists()
            and (self.mpnn_dir / "run.py").exists()
            and self.checkpoint.exists()
        )

    def get_log_probs(
        self,
        struct_path: str | Path | None,
        binder_len: int,
        step: int,
        seed: int,
        design_chain: str = "B",
        tmp_dir: str | Path | None = None,
    ) -> np.ndarray | None:
        if struct_path is None:
            return None
        src = Path(struct_path)
        if not src.exists() or not self.is_available():
            return None
        assert self.mpnn_dir is not None and self.checkpoint is not None

        bucket = int(step) // self.refresh_interval
        cache_key = (str(src), bucket, str(design_chain))
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.shape == (binder_len, len(MPNN_ALPHABET)):
                return cached

        work_root = Path(tmp_dir) if tmp_dir is not None else self.tmp_dir
        work_dir = work_root / f"lmpnn_{os.getpid()}_{int(seed)}_{bucket}"
        out_folder = work_dir / "mpnn_out"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_folder.mkdir(parents=True, exist_ok=True)
        pdb_path = work_dir / f"{src.stem}.pdb"
        try:
            if src.suffix.lower() in (".cif", ".mmcif"):
                cif_to_pdb(src, pdb_path)
            else:
                shutil.copyfile(src, pdb_path)

            driver = (
                "import json, pathlib, subprocess, sys, numpy as np, torch\n"
                "run_py, pdb_p, out_d, ckpt, ch, omit_aa, temp, seed = sys.argv[1:9]\n"
                "cmd = [sys.executable, run_py, '--model_type', 'ligand_mpnn', "
                "'--checkpoint_ligand_mpnn', ckpt, '--pdb_path', pdb_p, '--out_folder', out_d, "
                "'--chains_to_design', ch, '--omit_AA', omit_aa, "
                "'--ligand_mpnn_use_atom_context', '1', '--save_stats', '1', "
                "'--batch_size', '1', '--number_of_batches', '1', "
                "'--temperature', temp, '--seed', seed]\n"
                "subprocess.run(cmd, check=True)\n"
                "stem = pathlib.Path(pdb_p).stem\n"
                "pt_path = pathlib.Path(out_d) / 'stats' / f'{stem}.pt'\n"
                "d = torch.load(str(pt_path), map_location='cpu', weights_only=False)\n"
                "cm = d['chain_mask'].numpy().astype(bool)\n"
                "lp = d['log_probs'].numpy()[:, cm, :].mean(axis=0)\n"
                "np.savez(pathlib.Path(out_d) / 'logits.npz', log_probs=lp, chain_mask=cm)\n"
                "pathlib.Path(out_d, 'logits.json').write_text("
                "json.dumps({'shape': list(lp.shape), 'alphabet': 'ACDEFGHIKLMNPQRSTVWYX'}), "
                "encoding='utf-8')\n"
            )
            cmd = [
                str(self.mpnn_python),
                "-c",
                driver,
                str(self.mpnn_dir / "run.py"),
                str(pdb_path),
                str(out_folder),
                str(self.checkpoint),
                str(design_chain),
                str(self.exclude_aa or "CP"),
                str(self.temperature),
                str(int(seed)),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            npz_path = out_folder / "logits.npz"
            json_path = out_folder / "logits.json"
            if npz_path.exists():
                with np.load(npz_path) as data:
                    lp = np.asarray(data["log_probs"], dtype=np.float64)
            elif json_path.exists():
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                lp = np.asarray(payload["log_probs"], dtype=np.float64)
            else:
                return None

            if lp.ndim != 2 or lp.shape[0] != binder_len or lp.shape[1] != len(MPNN_ALPHABET):
                return None
            self._cache[cache_key] = lp
            return lp
        except (OSError, subprocess.SubprocessError, KeyError, ValueError):
            return None
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def sample_position(
        self,
        rng: np.random.Generator,
        seq_chars: list[str],
        pos: int,
        *,
        struct_path: str | Path | None,
        step: int = 0,
        seed: int = 0,
        design_chain: str = "B",
        exclude_aa: str | None = None,
        forbid_sequon: bool = True,
        tmp_dir: str | Path | None = None,
        blosum_temp: float = 1.0,
        allow_cur_forbidden: bool = False,
    ) -> tuple[str, str]:
        """Sample one amino acid at 0-based `pos` using LigandMPNN logits + carb bias, or `carb_blosum` fallback."""
        ex_aa = self.exclude_aa if exclude_aa is None else str(exclude_aa)
        lp = self.get_log_probs(
            struct_path,
            len(seq_chars),
            step=step,
            seed=seed,
            design_chain=design_chain,
            tmp_dir=tmp_dir,
        )
        if lp is None:
            aa = sample_residue(
                rng,
                seq_chars,
                pos,
                method="carb_blosum",
                blosum_temp=blosum_temp,
                exclude_aa=ex_aa,
                forbid_sequon=forbid_sequon,
                wf_cap_frac=self.wf_cap_frac,
                allow_cur_forbidden=allow_cur_forbidden,
            )
            return aa, "carb_blosum"

        cur = seq_chars[pos]
        cands = _allowed_candidates(
            seq_chars,
            pos,
            cur,
            exclude_aa=ex_aa,
            forbid_sequon=forbid_sequon,
            allow_cur_forbidden=allow_cur_forbidden,
        )
        if not cands:
            raise ForbiddenSequenceError(
                f"No valid substitution available at position {pos + 1} (exclude_aa={ex_aa})"
            )
        temp = max(0.05, self.temperature)
        logits = np.array(
            [float(lp[pos, MPNN_ALPHABET.index(a)]) / temp for a in cands],
            dtype=np.float64,
        )
        probs = _apply_carb_bias_and_wf_cap(logits, cands, seq_chars, wf_cap_frac=self.wf_cap_frac)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.ones(len(cands), dtype=np.float64) / len(cands)
        else:
            probs = probs / total
        return str(rng.choice(cands, p=probs)), "ligand_mpnn_biased"


def mutation_string(old: str, new: str) -> str:
    """Return 1-based comma-separated mutation string (e.g. 'A12K,S15E')."""
    return ",".join(f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(old, new)) if a != b) or "none"


def propose_mutant(
    rng: np.random.Generator,
    cur_seq: str,
    mutable_positions: Sequence[int],
    n_mut: int,
    method: str = "blosum",
    blosum_temp: float = 1.0,
    p_nonint: float = 0.0,
    exclude_aa: str = "CP",
    forbid_sequon: bool = True,
    mpnn: Any | None = None,
    struct_path: str | Path | None = None,
    tmp_dir: str | Path = "./tmp",
    seed: int = 0,
    design_chain: str = "B",
    fixed_chains: Sequence[str] = ("A",),
    allow_cur_forbidden: bool = False,
    wf_cap_frac: float = 0.12,
    step: int = 0,
) -> dict[str, Any]:
    """Propose a mutant in `mutable_positions` (purging forbidden `C/P`/sequons first if allowed)."""
    if allow_cur_forbidden:
        validate_binder_seq(cur_seq, "current sequence", exclude_aa="", forbid_sequon=False)
        viol = forbidden_positions(cur_seq, exclude_aa=exclude_aa, forbid_sequon=forbid_sequon)
    else:
        validate_binder_seq(
            cur_seq, "current sequence", exclude_aa=exclude_aa, forbid_sequon=forbid_sequon
        )
        viol = []

    k_target = min(max(1, int(n_mut)), len(cur_seq))
    if viol:
        mut_set = set(int(i) for i in mutable_positions if 0 <= int(i) < len(cur_seq))
        viol_mut = [i for i in viol if i in mut_set]
        viol_nonmut = [i for i in viol if i not in mut_set]
        ordered_viol: list[int] = []
        if viol_mut:
            ordered_viol.extend(int(x) for x in rng.permutation(viol_mut))
        if viol_nonmut:
            ordered_viol.extend(int(x) for x in rng.permutation(viol_nonmut))
        chosen_viol = ordered_viol[:k_target]
        rem_n = k_target - len(chosen_viol)
        if rem_n > 0:
            chosen_set = set(chosen_viol)
            rem_mut = [i for i in mutable_positions if int(i) not in chosen_set]
            if rem_mut:
                rem_pos = select_positions(rng, rem_n, rem_mut, len(cur_seq), p_nonint=p_nonint)
            else:
                avail = [i for i in range(len(cur_seq)) if i not in chosen_set]
                rem_pos = select_positions(rng, rem_n, avail, len(cur_seq), p_nonint=0.0)
            pos = sorted((set(chosen_viol) | set(rem_pos)))[:k_target]
        else:
            pos = sorted(chosen_viol)
    else:
        pos = select_positions(rng, k_target, mutable_positions, len(cur_seq), p_nonint=p_nonint)

    used_method = method
    alpha = allowed_alphabet(exclude_aa)

    if method == "mpnn" and mpnn is not None and struct_path is not None:
        sampled = mpnn.sample(
            struct_path, pos, cur_seq, seed=seed, tmp_dir=tmp_dir,
            design_chain=design_chain, fixed_chains=fixed_chains,
        )
        buf = list(cur_seq)
        changed = False
        for i in pos:
            aa = sampled[i]
            if aa != cur_seq[i] and aa in alpha and not (forbid_sequon and creates_sequon(buf, i, aa)):
                buf[i] = aa
                changed = True
        if changed:
            new_seq = "".join(buf)
            if not allow_cur_forbidden or not viol:
                validate_binder_seq(
                    new_seq, "mpnn proposal", exclude_aa=exclude_aa, forbid_sequon=forbid_sequon
                )
            else:
                validate_binder_seq(new_seq, "mpnn proposal", exclude_aa="", forbid_sequon=False)
            return {
                "seq": new_seq,
                "muts": mutation_string(cur_seq, new_seq),
                "method": "mpnn",
                "positions": pos,
            }
        used_method = "blosum"

    if str(used_method).lower() == "ligand_mpnn_biased":
        sampler = mpnn if hasattr(mpnn, "sample_position") else LigandMPNNBiasedSampler(
            exclude_aa=exclude_aa,
            wf_cap_frac=wf_cap_frac,
            tmp_dir=tmp_dir,
        )
        buf = list(cur_seq)
        methods_used: list[str] = []
        for i in pos:
            aa, m_used = sampler.sample_position(
                rng,
                buf,
                i,
                struct_path=struct_path,
                step=step,
                seed=seed,
                design_chain=design_chain,
                exclude_aa=exclude_aa,
                forbid_sequon=forbid_sequon,
                tmp_dir=tmp_dir,
                blosum_temp=blosum_temp,
                allow_cur_forbidden=allow_cur_forbidden,
            )
            buf[i] = aa
            methods_used.append(m_used)
        new_seq = "".join(buf)
        if not allow_cur_forbidden or not viol:
            validate_binder_seq(
                new_seq, "proposed sequence", exclude_aa=exclude_aa, forbid_sequon=forbid_sequon
            )
        else:
            validate_binder_seq(new_seq, "proposed sequence", exclude_aa="", forbid_sequon=False)
        final_method = "ligand_mpnn_biased" if "ligand_mpnn_biased" in methods_used else "carb_blosum"
        return {
            "seq": new_seq,
            "muts": mutation_string(cur_seq, new_seq),
            "method": final_method,
            "positions": pos,
        }

    fallback_method = "blosum" if used_method == "mpnn" else used_method
    buf = list(cur_seq)
    for i in pos:
        buf[i] = sample_residue(
            rng, buf, i, method=fallback_method, blosum_temp=blosum_temp,
            exclude_aa=exclude_aa, forbid_sequon=forbid_sequon,
            wf_cap_frac=wf_cap_frac, allow_cur_forbidden=allow_cur_forbidden,
        )
    new_seq = "".join(buf)
    if not allow_cur_forbidden or not viol:
        validate_binder_seq(
            new_seq, "proposed sequence", exclude_aa=exclude_aa, forbid_sequon=forbid_sequon
        )
    else:
        validate_binder_seq(new_seq, "proposed sequence", exclude_aa="", forbid_sequon=False)
    return {
        "seq": new_seq,
        "muts": mutation_string(cur_seq, new_seq),
        "method": fallback_method,
        "positions": pos,
    }
