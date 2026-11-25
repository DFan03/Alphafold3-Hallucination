"""AlphaFold 3 predictors: persistent hot-JAX runner, subprocess runner, built-in CPU MockAF3Runner, and multistate JSON/MSA builders."""
from __future__ import annotations

import dataclasses
import datetime
import functools
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .proposals import validate_binder_seq
from .scoring import extract_binder_iptm
from .structure import ComplexCoords, extract_token_coords, load_structure


@dataclasses.dataclass(frozen=True)
class SugarNode:
    """Single monosaccharide residue in a covalent N-glycan tree."""

    node_id: str
    ccd: str
    parent_id: str
    parent_atom: str
    child_atom: str = "C1"


_A1G0F_NODES: tuple[SugarNode, ...] = (
    SugarNode("R1", "NAG", "ASN", "ND2", "C1"),
    SugarNode("R2", "NAG", "R1", "O4", "C1"),
    SugarNode("R_FUC", "FUC", "R1", "O6", "C1"),
    SugarNode("R3", "BMA", "R2", "O4", "C1"),
    SugarNode("R4", "MAN", "R3", "O3", "C1"),
    SugarNode("R5", "MAN", "R3", "O6", "C1"),
    SugarNode("R6", "NAG", "R4", "O2", "C1"),
)

_A3G0F_NODES: tuple[SugarNode, ...] = _A1G0F_NODES + (
    SugarNode("R7", "NAG", "R4", "O4", "C1"),
    SugarNode("R8", "NAG", "R5", "O2", "C1"),
)

_A3G0_NODES: tuple[SugarNode, ...] = tuple(n for n in _A3G0F_NODES if n.node_id != "R_FUC")

_A3G3F_NODES: tuple[SugarNode, ...] = _A3G0F_NODES + (
    SugarNode("R10", "GAL", "R6", "O4", "C1"),
    SugarNode("R11", "GAL", "R7", "O4", "C1"),
    SugarNode("R12", "GAL", "R8", "O4", "C1"),
)

_A4G2F_NODES: tuple[SugarNode, ...] = _A3G0F_NODES + (
    SugarNode("R9", "NAG", "R5", "O6", "C1"),
    SugarNode("R10", "GAL", "R6", "O4", "C1"),
    SugarNode("R11", "GAL", "R8", "O4", "C1"),
)

GLYCOFORM_TREES: dict[str, tuple[SugarNode, ...]] = {
    "A1G0F": _A1G0F_NODES,
    "A3G0": _A3G0_NODES,
    "A3G0F": _A3G0F_NODES,
    "A3G3F": _A3G3F_NODES,
    "A4G2F": _A4G2F_NODES,
}


@dataclasses.dataclass
class AF3Prediction:
    """Parsed AlphaFold 3 prediction output for one state."""

    state_name: str
    cif_path: Path
    summary: dict[str, Any]
    confidences: dict[str, Any]
    complex_coords: ComplexCoords
    pae: np.ndarray
    token_chain_ids: list[str]
    token_coords: np.ndarray | None
    iptm_binder: float
    wall_s: float = 0.0


class AF3RunnerProtocol(Protocol):
    """Pluggable runner protocol implemented by AF3HotRunner, AF3SubprocessRunner, and MockAF3Runner."""

    def predict_json(
        self, payload: dict[str, Any], out_dir: Path, job_name: str
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        """Run AF3 on `payload` and return (cif_path, summary_confidences, confidences)."""
        ...


# ---------------- A3M MSA cropping & caching ----------------
def parse_a3m(text: str) -> list[tuple[str, str]]:
    """Parse A3M text into [(header, sequence_row), ...]."""
    recs: list[tuple[str, str]] = []
    head: str | None = None
    buf: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            if head is not None:
                recs.append((head, "".join(buf)))
            head, buf = ln, []
        else:
            buf.append(ln)
    if head is not None:
        recs.append((head, "".join(buf)))
    return recs


def crop_a3m_row(row: str, start: int, end: int) -> str:
    """Crop one A3M row to 0-based query columns [start, end), keeping interior insertions."""
    out: list[str] = []
    col = 0
    for ch in row:
        if ch.islower():
            if start < col < end:
                out.append(ch)
            continue
        if start <= col < end:
            out.append(ch)
        col += 1
    return "".join(out)


def crop_a3m(text: str, start: int, end: int) -> str:
    """Crop A3M text to query columns [start, end), dropping hits empty in window."""
    recs = parse_a3m(text)
    if not recs:
        return ""
    kept: list[str] = []
    for idx, (hdr, row) in enumerate(recs):
        cropped = crop_a3m_row(row, start, end)
        if idx == 0 or cropped.replace("-", "").upper():
            kept.append(f"{hdr}\n{cropped}")
    return "\n".join(kept) + "\n"


def extract_msa_fields(
    msa_json: Mapping[str, Any] | str | Path | None,
    target_seq: str,
) -> tuple[str, str, list[Any]]:
    """Extract (unpairedMsa, pairedMsa, templates) from a pre-cached AF3 JSON, cropping substrings."""
    t_seq = target_seq.strip().upper()
    if msa_json is None:
        return f">target\n{t_seq}\n", "", []

    if isinstance(msa_json, (str, Path)):
        payload = json.loads(Path(msa_json).read_text(encoding="utf-8"))
    else:
        payload = dict(msa_json)

    prot: Mapping[str, Any] = payload
    if "sequences" in payload:
        for entry in payload["sequences"]:
            if isinstance(entry, Mapping) and "protein" in entry:
                prot = entry["protein"]
                break

    raw_unpaired = str(prot.get("unpairedMsa") or "")
    raw_paired = str(prot.get("pairedMsa") or "")
    raw_templates = list(prot.get("templates") or [])
    cached_seq = str(prot.get("sequence") or "").strip().upper()
    if not cached_seq and raw_unpaired:
        recs = parse_a3m(raw_unpaired)
        if recs:
            cached_seq = "".join(ch for ch in recs[0][1] if not ch.islower()).upper()

    if not cached_seq or cached_seq == t_seq:
        return (
            raw_unpaired or f">target\n{t_seq}\n",
            raw_paired,
            raw_templates,
        )

    if t_seq in cached_seq:
        start = cached_seq.index(t_seq)
        end = start + len(t_seq)
        unpaired = crop_a3m(raw_unpaired, start, end) if raw_unpaired else f">target\n{t_seq}\n"
        if raw_unpaired:
            q_recs = parse_a3m(unpaired)
            if not q_recs or q_recs[0][1].upper() != t_seq:
                raise ValueError("Cropped unpairedMsa query row does not match target_seq")
        paired = crop_a3m(raw_paired, start, end) if raw_paired else ""
        return unpaired, paired, []

    raise ValueError(
        f"target_seq (len={len(t_seq)}) is neither exact match nor substring of "
        f"cached MSA sequence (len={len(cached_seq)})"
    )


# ---------------- AF3 Input JSON Builders ----------------
def _next_available_chain_ids(used: set[str], count: int) -> list[str]:
    out: list[str] = []
    for code in range(ord("C"), ord("Z") + 1):
        ch = chr(code)
        if ch not in used:
            out.append(ch)
            used.add(ch)
            if len(out) == count:
                return out
    for code in range(ord("A"), ord("Z") + 1):
        ch = chr(code)
        if ch not in used:
            out.append(ch)
            used.add(ch)
            if len(out) == count:
                return out
    raise ValueError("Exceeded 26 single-letter AF3 chain IDs")


def _nodes_from_glycan_spec(spec: Mapping[str, Any]) -> tuple[SugarNode, ...]:
    if spec.get("sugar_nodes"):
        return tuple(
            SugarNode(
                node_id=str(n["node_id"]),
                ccd=str(n["ccd"]),
                parent_id=str(n["parent_id"]),
                parent_atom=str(n["parent_atom"]),
                child_atom=str(n.get("child_atom", "C1")),
            )
            for n in spec["sugar_nodes"]
        )
    gf = str(spec["glycoform"])
    if gf not in GLYCOFORM_TREES:
        raise ValueError(f"Unknown glycoform {gf!r}; expected one of {sorted(GLYCOFORM_TREES)}")
    return GLYCOFORM_TREES[gf]


def state_glycan_chain_ids(state_spec: Mapping[str, Any]) -> list[str]:
    """Return ordered chain IDs corresponding strictly to `type == 'glycan'` entities in `state_spec`."""
    used: set[str] = set()
    glycan_cids: list[str] = []
    for ent in state_spec.get("chains") or []:
        etype = ent.get("type", "protein")
        if etype == "protein":
            used.add(str(ent["id"]))
        elif etype == "glycan":
            nodes = _nodes_from_glycan_spec(ent)
            glycan_cids.extend(_next_available_chain_ids(used, len(nodes)))
        elif etype == "ligand":
            cid = str(ent.get("id") or _next_available_chain_ids(used, 1)[0])
            used.add(cid)
    return glycan_cids


def build_state_af3_json(
    name: str,
    state_spec: Mapping[str, Any],
    protomer_seqs: Mapping[str, str],
    msa_cache: Mapping[tuple[str, str], tuple[str, str, list[Any]]] | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Build an AF3 JSON dict (dialect='alphafold3', version=1) for an arbitrary StateSpec."""
    sequences: list[dict[str, Any]] = []
    bonded_pairs: list[list[list[Any]]] = []
    used_chains: set[str] = set()

    for ent in state_spec.get("chains") or []:
        etype = ent.get("type", "protein")
        if etype == "protein":
            cid = str(ent["id"])
            used_chains.add(cid)
            if ent.get("mutable"):
                pkey = str(ent.get("protomer") or cid)
                if pkey not in protomer_seqs:
                    if len(protomer_seqs) == 1:
                        pkey = next(iter(protomer_seqs))
                    else:
                        raise KeyError(f"Missing mutable protomer {pkey!r} in {list(protomer_seqs)}")
                seq = str(protomer_seqs[pkey]).strip().upper()
                unpaired, paired, templates = f">{cid}\n{seq}\n", "", []
            else:
                seq = str(ent["sequence"]).strip().upper()
                msa_key = (str(ent.get("msa_json") or ""), seq)
                if msa_cache is not None and msa_key in msa_cache:
                    unpaired, paired, templates = msa_cache[msa_key]
                elif ent.get("msa_a3m"):
                    unpaired = str(ent["msa_a3m"])
                    paired, templates = "", []
                elif ent.get("msa_json"):
                    unpaired, paired, templates = extract_msa_fields(ent["msa_json"], seq)
                else:
                    unpaired, paired, templates = f">{cid}\n{seq}\n", "", []
            sequences.append(
                {
                    "protein": {
                        "id": cid,
                        "sequence": seq,
                        "unpairedMsa": unpaired,
                        "pairedMsa": paired,
                        "templates": templates,
                    }
                }
            )
        elif etype == "glycan":
            nodes = _nodes_from_glycan_spec(ent)
            anchor_chain = str(ent.get("anchor_chain", "A"))
            anchor_res = int(ent["anchor_res_1based"])
            cids = _next_available_chain_ids(used_chains, len(nodes))
            chain_map = {node.node_id: cids[idx] for idx, node in enumerate(nodes)}
            for node in nodes:
                sequences.append(
                    {"ligand": {"id": chain_map[node.node_id], "ccdCodes": [node.ccd]}}
                )
            for node in nodes:
                child_cid = chain_map[node.node_id]
                if node.parent_id == "ASN":
                    p_ref: list[Any] = [anchor_chain, anchor_res, node.parent_atom]
                else:
                    p_ref = [chain_map[node.parent_id], 1, node.parent_atom]
                bonded_pairs.append([p_ref, [child_cid, 1, node.child_atom]])
        elif etype == "ligand":
            cid = str(ent.get("id") or _next_available_chain_ids(used_chains, 1)[0])
            used_chains.add(cid)
            lig: dict[str, Any] = {"id": cid}
            if ent.get("ccdCodes"):
                lig["ccdCodes"] = list(ent["ccdCodes"])
            if ent.get("smiles"):
                lig["smiles"] = str(ent["smiles"])
            sequences.append({"ligand": lig})
        else:
            raise ValueError(f"Unsupported entity type {etype!r}")

    for pair in state_spec.get("bonded_atom_pairs") or []:
        bonded_pairs.append([list(pair[0]), list(pair[1])])

    payload: dict[str, Any] = {
        "name": str(name),
        "modelSeeds": [int(seed)],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": sequences,
    }
    if state_spec.get("user_ccd"):
        payload["userCCD"] = str(state_spec["user_ccd"])
    if bonded_pairs:
        payload["bondedAtomPairs"] = bonded_pairs
    return payload


def build_af3_glyco_json(
    name: str,
    target_seq: str,
    target_n_pos: int,
    glycoform: str | None,
    binder_seq: str,
    target_msa_fields: tuple[str, str, list[Any]] | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Convenience builder for 2-chain (target A + binder B) with optional covalent glycan tree."""
    t_seq = target_seq.strip().upper()
    if not (1 <= target_n_pos <= len(t_seq)):
        raise ValueError(f"target_n_pos={target_n_pos} out of range 1..{len(t_seq)}")
    if glycoform is not None and t_seq[target_n_pos - 1] != "N":
        raise ValueError(
            f"target_seq[{target_n_pos}]={t_seq[target_n_pos - 1]!r}, expected 'N'"
        )
    chains: list[dict[str, Any]] = [
        {"type": "protein", "id": "A", "mutable": False, "sequence": t_seq},
        {"type": "protein", "id": "B", "mutable": True, "protomer": "B"},
    ]
    if glycoform is not None:
        chains.append(
            {
                "type": "glycan",
                "anchor_chain": "A",
                "anchor_res_1based": int(target_n_pos),
                "glycoform": glycoform,
            }
        )
    msa_cache = {("", t_seq): target_msa_fields} if target_msa_fields is not None else None
    return build_state_af3_json(
        name=name,
        state_spec={"name": "glyco" if glycoform else "bare", "chains": chains},
        protomer_seqs={"B": binder_seq},
        msa_cache=msa_cache,
        seed=seed,
    )


# ---------------- AF3 Runners (Hot-JAX, Subprocess, CPU Mock) ----------------
def _read_af3_outputs(
    out_dir: Path, job_name: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Locate and parse AF3 output CIF, summary_confidences, and confidences JSON."""
    safe_name = job_name.lower()
    for prefix in (job_name, safe_name):
        for base in (out_dir, out_dir / prefix):
            cif = base / f"{prefix}_model.cif"
            sum_p = base / f"{prefix}_summary_confidences.json"
            conf_p = base / f"{prefix}_confidences.json"
            if cif.exists() and sum_p.exists() and conf_p.exists():
                return (
                    cif,
                    json.loads(sum_p.read_text(encoding="utf-8")),
                    json.loads(conf_p.read_text(encoding="utf-8")),
                )
    raise FileNotFoundError(f"AF3 outputs not found in {out_dir} for job {job_name!r}")


_SHARED_HOT_RUNNERS: dict[tuple[Any, ...], AF3HotRunner] = {}


def get_shared_hot_runner(**kwargs: Any) -> AF3HotRunner:
    """Return a cached persistent `AF3HotRunner` instance keyed by model/compile config."""
    buckets = tuple(int(b) for b in kwargs.get("buckets", (256, 384, 512, 768, 1024, 1536, 2048)))
    key = (
        str(Path(kwargs.get("af3_dir") or os.environ.get("AF3_CODE_DIR", "./alphafold3"))),
        str(Path(kwargs.get("model_dir") or os.environ.get("AF3_MODEL_DIR", "./models"))),
        str(Path(kwargs.get("jax_cache_dir") or os.environ.get("AF3_JAX_CACHE_DIR", "./tmp/jax_cache_af3"))),
        int(kwargs.get("num_recycles", 3)),
        int(kwargs.get("num_diffusion_samples", 1)),
        str(kwargs.get("flash_attention", "triton")),
        buckets,
        int(kwargs.get("gpu_device", 0)),
    )
    if key not in _SHARED_HOT_RUNNERS:
        _SHARED_HOT_RUNNERS[key] = AF3HotRunner(**kwargs)
    return _SHARED_HOT_RUNNERS[key]


class AF3HotRunner:
    """Persistent in-memory JAX ModelRunner around an AlphaFold 3 installation.

    Loads model parameters, Haiku-jitted ModelRunner, and CCD once at init, keeping XLA
    executables hot across all MCMC steps, states, and runs without restarting Python.
    """

    def __init__(
        self,
        af3_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        jax_cache_dir: str | Path | None = None,
        num_recycles: int = 3,
        num_diffusion_samples: int = 1,
        flash_attention: str = "triton",
        buckets: Sequence[int] = (256, 384, 512, 768, 1024, 1536, 2048),
        gpu_device: int = 0,
    ) -> None:
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        py_bin = str(Path(sys.executable).parent)
        if py_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{py_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        import importlib.util

        resolved_af3 = Path(af3_dir or os.environ.get("AF3_CODE_DIR", "./alphafold3"))
        resolved_model = Path(model_dir or os.environ.get("AF3_MODEL_DIR", "./models"))
        resolved_cache = Path(jax_cache_dir or os.environ.get("AF3_JAX_CACHE_DIR", "./tmp/jax_cache_af3"))

        if importlib.util.find_spec("alphafold3") is None and resolved_af3.exists():
            if str(resolved_af3) not in sys.path:
                sys.path.append(str(resolved_af3))

        from alphafold3.common import folding_input  # type: ignore[import-not-found]
        from alphafold3.constants import chemical_components  # type: ignore[import-not-found]
        from alphafold3.data import featurisation  # type: ignore[import-not-found]
        from alphafold3.model import model as af3_model  # type: ignore[import-not-found]
        from alphafold3.model import params as af3_params  # type: ignore[import-not-found]
        from alphafold3.model import post_processing  # type: ignore[import-not-found]
        from alphafold3.model.components import utils  # type: ignore[import-not-found]
        import haiku as hk  # type: ignore[import-not-found]
        import jax  # type: ignore[import-not-found]
        from jax import numpy as jnp  # type: ignore[import-not-found]

        resolved_cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(resolved_cache))

        self._folding_input = folding_input
        self._featurisation = featurisation
        self._post_processing = post_processing
        self._jax = jax
        self._buckets = tuple(int(b) for b in buckets)
        self._ccd = chemical_components.Ccd()

        cfg = af3_model.Model.Config()
        cfg.global_config.flash_attention_implementation = flash_attention
        cfg.heads.diffusion.eval.num_samples = int(num_diffusion_samples)
        cfg.num_recycles = int(num_recycles)
        cfg.return_embeddings = False
        cfg.return_distogram = False

        devices = jax.local_devices(backend="gpu")
        if not devices:
            devices = jax.local_devices()
        device = devices[int(gpu_device)]
        model_params = af3_params.get_model_haiku_params(model_dir=pathlib.Path(resolved_model))

        @hk.transform
        def forward_fn(batch: Any) -> Any:
            return af3_model.Model(cfg)(batch)

        jitted_apply = functools.partial(jax.jit(forward_fn.apply, device=device), model_params)
        self._device = device
        self._model_params = model_params
        self._jitted_apply = jitted_apply
        self._af3_model = af3_model
        self._utils = utils
        self._jnp = jnp

    def predict_json(
        self, payload: dict[str, Any], out_dir: Path, job_name: str
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_input = self._folding_input.Input.from_json(json.dumps(payload))
        ccd = self._ccd
        if getattr(fold_input, "user_ccd", None):
            from alphafold3.constants import chemical_components  # type: ignore[import-not-found]

            ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

        examples = self._featurisation.featurise_input(
            fold_input=fold_input,
            buckets=self._buckets,
            ccd=ccd,
            verbose=False,
            ref_max_modified_date=datetime.date(2021, 9, 30),
            resolve_msa_overlaps=False,
        )
        best_ir = None
        best_score: float | None = None
        identifier = self._model_params["__meta__"]["__identifier__"].tobytes()

        for seed, example in zip(fold_input.rng_seeds, examples):
            dev_example = self._jax.device_put(
                self._jax.tree_util.tree_map(
                    self._jnp.asarray, self._utils.remove_invalidly_typed_feats(example)
                ),
                self._device,
            )
            raw_res = self._jitted_apply(self._jax.random.PRNGKey(int(seed)), dev_example)
            res = self._jax.tree.map(
                lambda x: np.asarray(
                    x.astype(self._jnp.float32) if getattr(x, "dtype", None) == self._jnp.bfloat16 else x
                ),
                raw_res,
            )
            res_dict = dict(res)
            res_dict["__identifier__"] = identifier
            inf_results = list(
                self._af3_model.Model.get_inference_result(
                    batch=example, result=res_dict, target_name=fold_input.name
                )
            )
            for ir in inf_results:
                score = float(ir.metadata["ranking_score"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_ir = ir

        if best_ir is None:
            raise RuntimeError(f"AF3 returned 0 inference results for {job_name}")

        self._post_processing.write_output(
            inference_result=best_ir,
            output_dir=str(out_dir),
            name=job_name,
        )
        return _read_af3_outputs(out_dir, job_name)


class AF3SubprocessRunner:
    """Subprocess fallback calling `run_alphafold.py --norun_data_pipeline`."""

    def __init__(
        self,
        af3_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        db_dir: str | Path | None = None,
        jax_cache_dir: str | Path | None = None,
        num_recycles: int = 3,
        num_diffusion_samples: int = 1,
        flash_attention: str = "triton",
        python_bin: str = sys.executable,
        tmp_root: str | Path | None = None,
    ) -> None:
        self.af3_dir = Path(af3_dir or os.environ.get("AF3_CODE_DIR", "./alphafold3"))
        self.model_dir = Path(model_dir or os.environ.get("AF3_MODEL_DIR", "./models"))
        self.db_dir = Path(db_dir or os.environ.get("AF3_DB_DIR", "./public_databases"))
        self.jax_cache_dir = Path(jax_cache_dir or os.environ.get("AF3_JAX_CACHE_DIR", "./tmp/jax_cache_af3"))
        self.num_recycles = int(num_recycles)
        self.num_diffusion_samples = int(num_diffusion_samples)
        self.flash_attention = flash_attention
        self.python_bin = python_bin
        self.tmp_root = Path(tmp_root or os.environ.get("AF3_HALL_TMP_DIR", "./tmp"))

    def predict_json(
        self, payload: dict[str, Any], out_dir: Path, job_name: str
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jax_cache_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{job_name}_input.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env["TMPDIR"] = str(self.tmp_root)

        cmd = [
            self.python_bin,
            str(self.af3_dir / "run_alphafold.py"),
            f"--json_path={json_path}",
            f"--output_dir={out_dir}",
            f"--model_dir={self.model_dir}",
            f"--db_dir={self.db_dir}",
            f"--jax_compilation_cache_dir={self.jax_cache_dir}",
            "--norun_data_pipeline",
            "--resolve_msa_overlaps=false",
            "--force_output_dir=true",
            f"--num_recycles={self.num_recycles}",
            f"--num_diffusion_samples={self.num_diffusion_samples}",
            f"--flash_attention_implementation={self.flash_attention}",
        ]
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        return _read_af3_outputs(out_dir, job_name)


# ---------------- Built-in Deterministic CPU MockAF3Runner ----------------
_AA1_TO_3: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

_MOCK_SUGAR_CHAIRS: dict[str, tuple[str, int]] = {
    "NAG": ("4C1", +1),
    "BMA": ("4C1", +1),
    "GAL": ("4C1", +1),
    "MAN": ("4C1", -1),
    "FUC": ("1C4", +1),
}


def _mock_pyranose_atoms(
    parent_xyz: np.ndarray,
    bond_dir: np.ndarray,
    ccd: str,
    cg_xyz: np.ndarray | None = None,
) -> dict[str, tuple[str, np.ndarray]]:
    """Build stereochemically valid pyranose ring + exocyclic oxygens (no O1) attached at `parent_xyz`."""
    chair, anomer_sign = _MOCK_SUGAR_CHAIRS.get(ccd, ("4C1", +1))
    u = bond_dir / max(1e-9, float(np.linalg.norm(bond_dir)))
    if cg_xyz is not None:
        v1 = cg_xyz - parent_xyz
        n1 = float(np.linalg.norm(v1))
        if n1 > 1e-9:
            v1_u = v1 / n1
            perp = np.cross(v1_u, np.array([0.0, 0.0, 1.0]))
            if float(np.linalg.norm(perp)) < 1e-6:
                perp = np.cross(v1_u, np.array([0.0, 1.0, 0.0]))
            perp = perp / max(1e-9, float(np.linalg.norm(perp)))
            # 120 deg from CG - ND2
            u = -0.5 * v1_u + (math.sqrt(3.0) / 2.0) * perp

    c1 = parent_xyz + 1.44 * u
    # Orthonormal frame tilted 45 deg so `-ex` (C1->C4) extends outward along `+u`
    # while `ez` keeps |ascore| ~ 0.61 >> 0.15 and Cremer-Pople (Q, theta) invariant.
    ref = np.array([0.0, 1.0, 0.0]) if abs(float(u[1])) < 0.8 else np.array([1.0, 0.0, 0.0])
    u_perp = np.cross(ref, u)
    u_perp = u_perp / max(1e-9, float(np.linalg.norm(u_perp)))
    ez_pos = (u + u_perp) / math.sqrt(2.0)
    ex = (-u + u_perp) / math.sqrt(2.0)
    ez = ez_pos if anomer_sign > 0 else -ez_pos
    ey = np.cross(ez, ex)

    # Ideal 6-membered chair ring with Q ~ 0.56 A; alternating z = +/- 0.23 A gives theta ~ 0 (4C1) or ~ 180 (1C4)
    r = 1.40
    z_amp = 0.23 if chair == "4C1" else -0.23
    # Order (O5, C1, C2, C3, C4, C5) placed around hexagon; C1 is at j=1
    ring_names = ("O5", "C1", "C2", "C3", "C4", "C5")
    local_pts: dict[str, np.ndarray] = {}
    for j, name in enumerate(ring_names):
        ang = 2.0 * math.pi * (j - 1) / 6.0
        z = z_amp if (j % 2 == 0) else -z_amp
        local_pts[name] = np.array([r * (math.cos(ang) - 1.0), r * math.sin(ang), z + z_amp], dtype=np.float64)

    world_pts: dict[str, tuple[str, np.ndarray]] = {}
    for name, lp in local_pts.items():
        w = c1 + lp[0] * ex + lp[1] * ey - lp[2] * ez
        el = "O" if name.startswith("O") else "C"
        world_pts[name] = (el, w)

    # Place exocyclic acceptor oxygens (O2, O3, O4, O6) radially outward from C2, C3, C4, C5
    ring_cent = np.mean([world_pts[n][1] for n in ring_names], axis=0)
    for oname, cname in (("O2", "C2"), ("O3", "C3"), ("O4", "C4"), ("O6", "C5")):
        cpt = world_pts[cname][1]
        out_dir = cpt - ring_cent
        out_dir = out_dir / max(1e-9, float(np.linalg.norm(out_dir)))
        world_pts[oname] = ("O", cpt + 1.43 * out_dir)
    return world_pts


class MockAF3Runner:
    """Deterministic CPU AF3 runner emitting synthetic mmCIF + confidence JSONs in milliseconds.

    Designed for offline CI, unit tests, and rapid dry-run prototyping without GPU or AF3 weights.
    """

    def __init__(self, base_plddt: float = 82.0) -> None:
        self.base_plddt = float(base_plddt)

    @staticmethod
    def _seq_quality_score(seq: str, seed: int) -> float:
        digest = hashlib.sha256(f"{seed}:{seq}".encode("utf-8")).digest()
        jitter = (int.from_bytes(digest[:4], "little") / 2**32 - 0.5) * 0.08
        polar_aromatics = sum(seq.count(a) for a in "NQDESTRKYWH") / max(1, len(seq))
        hydrophobes = sum(seq.count(a) for a in "VILMFWA") / max(1, len(seq))
        # Reward balanced amphipathic composition + deterministic hash jitter
        balance = 1.0 - abs(polar_aromatics - 0.45) - 0.5 * abs(hydrophobes - 0.40)
        return float(max(0.15, min(0.95, 0.55 + 0.30 * balance + jitter)))

    def predict_json(
        self, payload: dict[str, Any], out_dir: Path, job_name: str
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = payload.get("modelSeeds") or [1]
        seed = int(seeds[0])
        sequences = payload.get("sequences") or []
        bonded_pairs = payload.get("bondedAtomPairs") or []

        # Map child_cid -> (parent_cid, parent_res, parent_atom)
        parent_map: dict[str, tuple[str, int, str]] = {}
        for p_ref, c_ref in bonded_pairs:
            if len(p_ref) >= 3 and len(c_ref) >= 3:
                parent_map[str(c_ref[0])] = (str(p_ref[0]), int(p_ref[1]), str(p_ref[2]))

        atom_rows: list[tuple[str, str, int, str, str, str, np.ndarray, float]] = []
        chain_atom_lookup: dict[tuple[str, int, str], np.ndarray] = {}
        sugar_ring_centers: dict[str, np.ndarray] = {}
        chain_order: list[str] = []
        chain_qualities: list[float] = []
        token_chain_ids: list[str] = []
        prot_index = 0
        lig_index = 0

        # Track anchor points for protein chains
        protein_cids: list[str] = [
            str(e["protein"]["id"]) for e in sequences if "protein" in e
        ]
        all_prot_seq = "".join(
            str(e["protein"]["sequence"]) for e in sequences if "protein" in e
        )
        seq_hash = hashlib.sha256(f"{seed}:{all_prot_seq}".encode("utf-8")).digest()
        shift_x = ((seq_hash[0] / 255.0) - 0.5) * 1.4
        shift_y = ((seq_hash[1] / 255.0) - 0.5) * 1.4

        for entry in sequences:
            if "protein" in entry:
                prot = entry["protein"]
                cid = str(prot["id"])
                seq = str(prot["sequence"]).strip().upper()
                chain_order.append(cid)
                q = self._seq_quality_score(seq, seed + prot_index * 17)
                chain_qualities.append(q)
                res_plddt_base = max(45.0, min(96.0, self.base_plddt * (0.75 + 0.35 * q)))

                # Place chain 0 near origin; chain 1+ near interface (~4.5 A from chain 0)
                if prot_index == 0:
                    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    radius = 5.2
                else:
                    angle_off = (prot_index - 1) * 1.6
                    center = np.array(
                        [9.5 + shift_x * math.cos(angle_off), shift_y + 3.0 * math.sin(angle_off), 0.0],
                        dtype=np.float64,
                    )
                    radius = 4.6

                n_res = len(seq)
                for r_idx, aa in enumerate(seq):
                    rid = r_idx + 1
                    rname = _AA1_TO_3.get(aa, "ALA")
                    theta = 0.65 * r_idx
                    z_pos = (r_idx - 0.5 * max(0, n_res - 1)) * 1.45
                    # Compact fold with one side facing the partner interface
                    ca = center + np.array(
                        [radius * math.cos(theta), radius * math.sin(theta), z_pos],
                        dtype=np.float64,
                    )
                    n_pt = ca + np.array([-0.9, -0.5, -0.8], dtype=np.float64)
                    c_pt = ca + np.array([0.9, 0.5, 0.8], dtype=np.float64)
                    o_pt = c_pt + np.array([0.4, 1.0, 0.2], dtype=np.float64)
                    inward = center - ca
                    inward_n = inward / max(1e-6, float(np.linalg.norm(inward)))
                    # Point CB toward partner when prot_index > 0 or outward toward partner
                    if prot_index == 0 and len(protein_cids) > 1:
                        cb_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    elif prot_index > 0:
                        cb_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        cb_dir = inward_n
                    cb_pt = ca + 1.52 * cb_dir
                    loc_plddt = max(40.0, min(98.0, res_plddt_base + 4.0 * math.sin(r_idx * 0.7 + q * 5.0)))

                    res_atoms: list[tuple[str, str, np.ndarray]] = [
                        ("N", "N", n_pt),
                        ("CA", "C", ca),
                        ("C", "C", c_pt),
                        ("O", "O", o_pt),
                    ]
                    if aa != "G":
                        res_atoms.append(("CB", "C", cb_pt))
                    if aa == "N":
                        cg_pt = cb_pt + 1.45 * cb_dir
                        nd2_pt = cg_pt + 1.33 * cb_dir
                        res_atoms.extend([("CG", "C", cg_pt), ("ND2", "N", nd2_pt)])
                    elif aa in ("F", "Y", "W"):
                        # Aromatic ring perpendicular to cb_dir so CH-pi can score
                        cg_pt = cb_pt + 1.45 * cb_dir
                        u1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                        u2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                        ring_atoms = [
                            ("CG", cg_pt),
                            ("CD1", cg_pt + 1.2 * u1 + 0.7 * u2),
                            ("CD2", cg_pt + 1.2 * u1 - 0.7 * u2),
                            ("CE1", cg_pt + 2.4 * u1 + 0.7 * u2),
                            ("CE2", cg_pt + 2.4 * u1 - 0.7 * u2),
                            ("CZ", cg_pt + 3.1 * u1),
                        ]
                        res_atoms.extend((an, "C", pt) for an, pt in ring_atoms)
                    elif aa in ("D", "E", "Q", "K", "R", "S", "T"):
                        # Add a polar sidechain oxygen/nitrogen near interface
                        polar_el = "N" if aa in ("K", "R") else "O"
                        polar_an = "NZ" if aa in ("K", "R") else "OG"
                        res_atoms.append((polar_an, polar_el, cb_pt + 1.42 * cb_dir))

                    for aname, el, pt in res_atoms:
                        atom_rows.append(("ATOM", cid, rid, rname, aname, el, pt, loc_plddt))
                        chain_atom_lookup[(cid, rid, aname)] = pt
                    token_chain_ids.append(cid)
                prot_index += 1

            elif "ligand" in entry:
                lig = entry["ligand"]
                cid = str(lig["id"])
                chain_order.append(cid)
                ccd = str((lig.get("ccdCodes") or ["LIG"])[0])
                if cid in parent_map:
                    p_cid, p_rid, p_atom = parent_map[cid]
                    p_xyz = chain_atom_lookup.get((p_cid, p_rid, p_atom))
                    if p_xyz is None:
                        p_xyz = np.array([5.5, 0.0, 0.0], dtype=np.float64)
                    cg_xyz = chain_atom_lookup.get((p_cid, p_rid, "CG")) if p_atom == "ND2" else None
                    # Extend radially outward from parent ring center (or outward toward binder for ASN)
                    if p_cid in sugar_ring_centers:
                        bond_dir = p_xyz - sugar_ring_centers[p_cid]
                    else:
                        bond_dir = np.array([1.0, 0.25 * math.sin(lig_index + 1), 0.25 * math.cos(lig_index + 1)])
                    sugar_atoms = _mock_pyranose_atoms(p_xyz, bond_dir, ccd, cg_xyz=cg_xyz)
                    sugar_ring_centers[cid] = np.mean(
                        [sugar_atoms[k][1] for k in ("O5", "C1", "C2", "C3", "C4", "C5")], axis=0
                    )
                    for aname, (el, pt) in sugar_atoms.items():
                        atom_rows.append(("HETATM", cid, 1, ccd, aname, el, pt, 76.0))
                        chain_atom_lookup[(cid, 1, aname)] = pt
                        token_chain_ids.append(cid)
                else:
                    # Non-glycan small-molecule ligand positioned at binding pocket between chains
                    lig_center = np.array([4.8 + 0.3 * shift_x, 0.2 * shift_y, 0.0], dtype=np.float64)
                    lig_pts = [
                        ("C1", "C", lig_center + np.array([0.0, 0.0, 0.0])),
                        ("C2", "C", lig_center + np.array([1.4, 0.0, 0.0])),
                        ("C3", "C", lig_center + np.array([2.1, 1.2, 0.0])),
                        ("C4", "C", lig_center + np.array([1.4, 2.4, 0.0])),
                        ("C5", "C", lig_center + np.array([0.0, 2.4, 0.0])),
                        ("C6", "C", lig_center + np.array([-0.7, 1.2, 0.0])),
                        ("N1", "N", lig_center + np.array([2.8, 0.0, 0.6])),
                        ("N2", "N", lig_center + np.array([-1.4, 1.2, -0.5])),
                        ("O1", "O", lig_center + np.array([0.0, 3.6, 0.4])),
                        ("O2", "O", lig_center + np.array([1.4, -1.2, 0.3])),
                    ]
                    for aname, el, pt in lig_pts:
                        atom_rows.append(("HETATM", cid, 1, ccd[:3], aname, el, pt, 78.0))
                        chain_atom_lookup[(cid, 1, aname)] = pt
                        token_chain_ids.append(cid)
                lig_index += 1

        # Build deterministic PAE matrix [N_tok, N_tok]
        n_tok = len(token_chain_ids)
        mean_q = float(np.mean(chain_qualities)) if chain_qualities else 0.7
        tok_arr = np.asarray(token_chain_ids)
        same_chain = tok_arr[:, None] == tok_arr[None, :]
        intra_pae = max(1.2, 5.5 - 3.5 * mean_q)
        inter_pae = max(2.2, 11.0 - 7.0 * mean_q)
        pae = np.where(same_chain, intra_pae, inter_pae)
        np.fill_diagonal(pae, 0.6)
        # Mild deterministic asymmetric gradient across binder-partner tokens
        idx_grid = np.arange(n_tok, dtype=np.float64)
        pae = np.clip(pae + 0.4 * np.sin(0.3 * idx_grid[:, None] - 0.2 * idx_grid[None, :]), 0.5, 31.75)

        ptm = float(max(0.25, min(0.94, 0.42 + 0.48 * mean_q)))
        has_multi = len(set(token_chain_ids)) > 1
        iptm = float(max(0.20, min(0.92, 0.35 + 0.50 * mean_q))) if has_multi else 0.0
        chain_iptm = [
            float(max(0.20, min(0.92, iptm + 0.02 * math.sin(i + mean_q))))
            if has_multi
            else 0.0
            for i in range(len(chain_order))
        ]
        ranking_score = float(0.8 * iptm + 0.2 * ptm) if has_multi else ptm

        # Write mmCIF
        safe = job_name.lower()
        cif_path = out_dir / f"{safe}_model.cif"
        cif_lines = [
            f"data_{safe}",
            "#",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_asym_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
        atom_plddts: list[float] = []
        atom_chain_ids: list[str] = []
        for idx, (grp, cid, rid, rname, aname, el, pt, plddt_v) in enumerate(atom_rows, start=1):
            seq_lbl = str(rid) if grp == "ATOM" else "."
            cif_lines.append(
                f"{grp} {idx} {el} {aname} . {rname} {cid} 1 {seq_lbl} ? "
                f"{pt[0]:.3f} {pt[1]:.3f} {pt[2]:.3f} 1.00 {plddt_v:.2f} {rid} {cid} 1"
            )
            atom_plddts.append(round(float(plddt_v), 2))
            atom_chain_ids.append(cid)
        cif_lines.append("#")
        cif_path.write_text("\n".join(cif_lines) + "\n", encoding="utf-8")

        summary = {
            "ptm": round(ptm, 4),
            "iptm": round(iptm, 4),
            "chain_iptm": [round(x, 4) for x in chain_iptm],
            "ranking_score": round(ranking_score, 4),
            "fraction_disordered": 0.05,
            "has_clash": 0.0,
        }
        confidences = {
            "atom_plddts": atom_plddts,
            "atom_chain_ids": atom_chain_ids,
            "token_chain_ids": token_chain_ids,
            "pae": np.round(pae, 3).tolist(),
        }
        (out_dir / f"{safe}_summary_confidences.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (out_dir / f"{safe}_confidences.json").write_text(
            json.dumps(confidences), encoding="utf-8"
        )
        return cif_path, summary, confidences


class AF3Predictor:
    """Persistent multistate AF3 predictor caching fixed target MSAs once in memory."""

    def __init__(
        self,
        state_specs: Sequence[Mapping[str, Any]],
        cfg: Mapping[str, Any],
        runner: AF3RunnerProtocol | None = None,
    ) -> None:
        self.state_specs = list(state_specs)
        self.cfg = dict(cfg)
        self.exclude_aa = str(cfg.get("exclude_aa", "CP"))
        self.forbid_sequon = bool(cfg.get("forbid_sequon", True))
        self.allow_init_forbidden = bool(cfg.get("allow_init_forbidden", False))

        # Pre-load and crop every fixed protein MSA ONCE at startup
        self._msa_cache: dict[tuple[str, str], tuple[str, str, list[Any]]] = {}
        for spec in self.state_specs:
            for ent in spec.get("chains") or []:
                if ent.get("type", "protein") == "protein" and not ent.get("mutable"):
                    seq = str(ent["sequence"]).strip().upper()
                    msa_path = str(ent.get("msa_json") or cfg.get("target_msa_json") or "")
                    key = (msa_path, seq)
                    if key not in self._msa_cache:
                        self._msa_cache[key] = extract_msa_fields(msa_path or None, seq)

        if runner is not None:
            self._runner = runner
        else:
            backend = str(cfg.get("backend", "in_memory")).lower()
            if backend == "mock":
                self._runner = MockAF3Runner()
            elif backend == "subprocess":
                self._runner = AF3SubprocessRunner(
                    af3_dir=cfg.get("af3_dir"),
                    model_dir=cfg.get("model_dir"),
                    db_dir=cfg.get("db_dir"),
                    jax_cache_dir=cfg.get("jax_cache_dir"),
                    num_recycles=int(cfg.get("num_recycles", 3)),
                    num_diffusion_samples=int(cfg.get("num_diffusion_samples", 1)),
                    flash_attention=str(cfg.get("flash_attention", "triton")),
                    tmp_root=cfg.get("tmp_root"),
                )
            else:
                self._runner = get_shared_hot_runner(
                    af3_dir=cfg.get("af3_dir"),
                    model_dir=cfg.get("model_dir"),
                    jax_cache_dir=cfg.get("jax_cache_dir"),
                    num_recycles=int(cfg.get("num_recycles", 3)),
                    num_diffusion_samples=int(cfg.get("num_diffusion_samples", 1)),
                    flash_attention=str(cfg.get("flash_attention", "triton")),
                    buckets=cfg.get("buckets", (256, 384, 512, 768, 1024, 1536, 2048)),
                    gpu_device=int(cfg.get("gpu_device", 0)),
                )

    def predict_state(
        self,
        state_spec: Mapping[str, Any],
        protomer_seqs: Mapping[str, str],
        seed: int,
        work_dir: Path,
        tag: str,
    ) -> AF3Prediction:
        """Predict one AF3 state (validating mutable sequences before folding)."""
        ex_aa = "" if self.allow_init_forbidden else self.exclude_aa
        fb_seq = False if self.allow_init_forbidden else self.forbid_sequon
        for pkey, seq in protomer_seqs.items():
            validate_binder_seq(
                seq,
                f"protomer {pkey}",
                exclude_aa=ex_aa,
                forbid_sequon=fb_seq,
            )

        sname = str(state_spec["name"])
        job_name = f"{tag}_{sname}".lower()
        payload = build_state_af3_json(
            name=job_name,
            state_spec=state_spec,
            protomer_seqs=protomer_seqs,
            msa_cache=self._msa_cache,
            seed=seed,
        )
        t0 = time.time()
        cif_path, summary, conf = self._runner.predict_json(payload, work_dir / sname, job_name)
        wall_s = time.time() - t0

        chain_specs = state_spec.get("chains") or []
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
        glycan_cids = state_glycan_chain_ids(state_spec)

        cx = load_structure(
            cif_path,
            target_chain=target_cid,
            binder_chain=binder_cid,
            glycan_chain_ids=glycan_cids,
        )
        pae = np.asarray(conf["pae"], dtype=np.float64)
        token_chains = list(conf.get("token_chain_ids") or [])
        if not token_chains:
            for cid in prot_chains:
                if cid in cx.all_chains:
                    token_chains.extend([cid] * cx.all_chains[cid].n_res)
            for cid in glycan_cids + ligand_cids:
                if cid in cx.all_chains:
                    token_chains.extend([cid] * len(cx.all_chains[cid].xyz))
        tok_coords = extract_token_coords(cx, token_chains, protein_chains=prot_chains)
        mut_idx = prot_chains.index(binder_cid) if binder_cid in prot_chains else 0
        iptm_val = extract_binder_iptm(summary, chain_idx=mut_idx)

        return AF3Prediction(
            state_name=sname,
            cif_path=Path(cif_path),
            summary=summary,
            confidences=conf,
            complex_coords=cx,
            pae=pae,
            token_chain_ids=token_chains,
            token_coords=tok_coords,
            iptm_binder=iptm_val,
            wall_s=wall_s,
        )

    def predict_all(
        self,
        protomer_seqs: Mapping[str, str],
        seed: int,
        work_dir: Path,
        tag: str,
    ) -> dict[str, AF3Prediction]:
        """Predict all configured states for `protomer_seqs` in one persistent session."""
        return {
            str(spec["name"]): self.predict_state(spec, protomer_seqs, seed=seed, work_dir=work_dir, tag=tag)
            for spec in self.state_specs
        }
