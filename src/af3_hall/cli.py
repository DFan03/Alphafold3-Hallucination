"""Command-line interface supporting AFhall-style `--oligo` shorthands, YAML configs, and `--mock` CPU mode."""
from __future__ import annotations

import argparse
from typing import Sequence

from .config import load_config
from .mcmc import run


def _parse_residue_range_arg(raw: str) -> list[int]:
    cleaned = raw.replace("-", ",")
    parts = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected two 1-based residue numbers (e.g. '10,70' or '10-70'), got {raw!r}"
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="af3-hall",
        description="Multistate all-atom AlphaFold 3 MCMC hallucination & binder evolution engine.",
    )
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML/JSON run config.")
    parser.add_argument(
        "--oligo",
        type=str,
        default=None,
        help="AFhall multistate oligomer shorthand, e.g. 'AB+,AA-,BB-' or 'A+'.",
    )
    parser.add_argument(
        "--L",
        "--length",
        "--binder-length",
        dest="binder_length",
        type=str,
        default=None,
        help="Protomer length(s), e.g. '50,50' or '80'.",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default=None,
        help="Comma-separated loss plugins (e.g. 'dual', 'plddt,iptm,ptm', 'glyco_delta').",
    )
    parser.add_argument(
        "--loss-weights",
        type=str,
        default=None,
        help="Comma-separated weights matching --losses.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of MCMC steps.")
    parser.add_argument("--proposals-per-step", type=int, default=None, help="Proposals evaluated per step.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=str, default=None, help="Output run directory.")
    parser.add_argument("--out-root", type=str, default=None, help="Root directory for timestamped runs.")
    parser.add_argument("--job", "--name", dest="job", type=str, default=None, help="Run / job name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic built-in CPU MockAF3Runner (no GPU or AF3 weights required).",
    )
    parser.add_argument(
        "--backend",
        choices=("in_memory", "subprocess", "mock"),
        default=None,
        help="AF3 execution backend.",
    )
    parser.add_argument(
        "--save-trajectory",
        choices=("accepted", "all"),
        default=None,
        help="Trajectory CIFs in <out>/trajectory/ are always written for accepted steps; 'all' also keeps rejected candidates.",
    )
    parser.add_argument("--fastrelax", action="store_true", default=None, help="tmol FastRelax of every AF3-predicted state BEFORE scoring (see docs/fastrelax.md).")
    parser.add_argument("--relax-scope", choices=("all", "shell"), default=None, help="all (default): whole complex; shell: backbone-fixed interface side chains.")
    parser.add_argument("--relax-repeats", type=int, default=None, help="FastRelax repeats (default 1).")
    parser.add_argument("--mutation-method", type=str, default=None, help="Proposal method (uniform|frequency_adjusted|blosum|carb_blosum|mpnn|ligand_mpnn_biased).")
    parser.add_argument("--mutation-rate", type=str, default=None, help="Mutation schedule (e.g. '3-1', '2', 'rand:1-3').")
    parser.add_argument("--T-init", "--t-init", dest="T_init", type=float, default=None, help="Initial annealing temperature.")
    parser.add_argument("--half-life", type=float, default=None, help="Exponential temperature half-life in steps.")
    parser.add_argument("--temp-schedule", type=str, default=None, help="Temperature schedule (exp|linear|const).")
    parser.add_argument("--recycles", "--num-recycles", dest="num_recycles", type=int, default=None, help="AF3 recycles.")
    parser.add_argument("--num-diffusion-samples", type=int, default=None, help="AF3 diffusion samples per prediction.")
    parser.add_argument("--mask-mode", type=str, default=None, help="Mutable mask mode (auto|interface|all|explicit).")
    parser.add_argument("--mask-refresh-interval", type=int, default=None, help="Steps between interface mask refreshes (0=disabled).")
    parser.add_argument("--p-nonint", type=float, default=None, help="Probability of mutating outside interface mask.")
    parser.add_argument("--lazy-neg-interval", type=int, default=None, help="Lazy negative-state evaluation interval.")
    parser.add_argument("--exclude-aa", type=str, default=None, help="Banned amino acid characters (default 'CP').")
    parser.add_argument("--no-forbid-sequon", action="store_true", help="Allow N-X-S/T sequons in designed chains.")
    parser.add_argument("--allow-init-forbidden", action="store_true", help="Allow starting sequences with forbidden residues/sequons.")
    parser.add_argument("--steric-clash-ref-cif", type=str, default=None, help="Reference CIF/PDB for steric clash guarding.")
    parser.add_argument("--steric-ref-residues", type=_parse_residue_range_arg, default=None, help="1-based alignment residue range 'start,end' in steric ref.")
    parser.add_argument("--steric-pred-residues", type=_parse_residue_range_arg, default=None, help="1-based alignment residue range 'start,end' in predicted target.")
    parser.add_argument("--af3-dir", type=str, default=None, help="Path to AlphaFold 3 repository (or set AF3_CODE_DIR).")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to AlphaFold 3 model weights (or set AF3_MODEL_DIR).")
    parser.add_argument("--db-dir", type=str, default=None, help="Path to AlphaFold 3 public databases (or set AF3_DB_DIR).")
    parser.add_argument("--jax-cache-dir", type=str, default=None, help="Path to JAX XLA compilation cache (or set AF3_JAX_CACHE_DIR).")
    parser.add_argument("--mpnn-dir", type=str, default=None, help="Path to LigandMPNN/ProteinMPNN directory (or set LIGANDMPNN_DIR).")
    parser.add_argument("--buckets", type=str, default=None, help="Comma-separated AF3 compilation token buckets.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in key=value form.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)

    if args.oligo is not None:
        cfg["oligo"] = args.oligo
        cfg["states"] = None
    if args.binder_length is not None:
        cfg["binder_length"] = args.binder_length
    if args.losses is not None:
        cfg["losses"] = args.losses
    if args.loss_weights is not None:
        cfg["loss_weights"] = args.loss_weights
    if args.steps is not None:
        cfg["steps"] = int(args.steps)
    if args.proposals_per_step is not None:
        cfg["proposals_per_step"] = int(args.proposals_per_step)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.out_root is not None:
        cfg["out_root"] = args.out_root
    if args.job is not None:
        cfg["job"] = args.job
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.backend is not None:
        cfg["backend"] = args.backend
    if args.fastrelax:
        cfg["fastrelax"] = True
    if args.relax_scope is not None:
        cfg["relax_scope"] = args.relax_scope
    if args.relax_repeats is not None:
        cfg["relax_repeats"] = int(args.relax_repeats)
    if args.save_trajectory is not None:
        cfg["save_trajectory"] = args.save_trajectory
    if args.mock:
        cfg["backend"] = "mock"
    if args.mutation_method is not None:
        cfg["mutation_method"] = args.mutation_method
    if args.mutation_rate is not None:
        cfg["mutation_rate"] = args.mutation_rate
    if args.T_init is not None:
        cfg["T_init"] = float(args.T_init)
    if args.half_life is not None:
        cfg["half_life"] = float(args.half_life)
    if args.temp_schedule is not None:
        cfg["temp_schedule"] = args.temp_schedule
    if args.num_recycles is not None:
        cfg["num_recycles"] = int(args.num_recycles)
    if args.num_diffusion_samples is not None:
        cfg["num_diffusion_samples"] = int(args.num_diffusion_samples)
    if args.mask_mode is not None:
        cfg["mask_mode"] = args.mask_mode
    if args.mask_refresh_interval is not None:
        cfg["mask_refresh_interval"] = int(args.mask_refresh_interval)
    if args.p_nonint is not None:
        cfg["p_nonint"] = float(args.p_nonint)
    if args.lazy_neg_interval is not None:
        cfg["lazy_neg_interval"] = int(args.lazy_neg_interval)
        cfg["lazy_bare_interval"] = int(args.lazy_neg_interval)
    if args.exclude_aa is not None:
        cfg["exclude_aa"] = args.exclude_aa
    if args.no_forbid_sequon:
        cfg["forbid_sequon"] = False
    if args.allow_init_forbidden:
        cfg["allow_init_forbidden"] = True
    if args.steric_clash_ref_cif is not None:
        cfg["steric_clash_ref_cif"] = args.steric_clash_ref_cif
    if args.steric_ref_residues is not None:
        cfg["steric_ref_residues"] = args.steric_ref_residues
    if args.steric_pred_residues is not None:
        cfg["steric_pred_residues"] = args.steric_pred_residues
    if args.af3_dir is not None:
        cfg["af3_dir"] = args.af3_dir
    if args.model_dir is not None:
        cfg["model_dir"] = args.model_dir
    if args.db_dir is not None:
        cfg["db_dir"] = args.db_dir
    if args.jax_cache_dir is not None:
        cfg["jax_cache_dir"] = args.jax_cache_dir
    if args.mpnn_dir is not None:
        cfg["mpnn_dir"] = args.mpnn_dir
    if args.buckets is not None:
        cfg["buckets"] = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
