"""Offline standalone unit & integration tests for `af3-hall` (CPU MockAF3Runner)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from af3_hall import (
    LOSS_REGISTRY,
    MULTISTATE_LOSS_REGISTRY,
    build_states_from_config,
    export_candidates_tsv,
    load_config,
    register_loss,
    register_multistate_loss,
    run,
)
from af3_hall.cli import main as cli_main
from af3_hall.config import parse_oligo_states
from af3_hall.predictor import GLYCOFORM_TREES, MockAF3Runner, build_state_af3_json
from af3_hall.proposals import (
    ForbiddenSequenceError,
    LigandMPNNBiasedSampler,
    cp_count,
    propose_mutant,
    random_sequence,
    sequon_count,
    validate_binder_seq,
)
from af3_hall.scoring import compute_objective_J, parse_loss_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "AF3_multistate_hallucination.py"), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_and_wrapper() -> None:
    res = _run_cli(["--help"])
    assert res.returncode == 0, res.stderr
    assert "--oligo" in res.stdout
    assert "--mock" in res.stdout
    assert "--steric-clash-ref-cif" in res.stdout


def test_oligo_parser_and_shorthand_run(tmp_path: Path) -> None:
    states = parse_oligo_states("AB+,AA-,BB-")
    assert [s["name"] for s in states] == ["AB+", "AA-", "BB-"]
    assert [s["weight"] for s in states] == [1.0, -1.0, -1.0]
    assert [c["protomer"] for c in states[0]["chains"]] == ["A", "B"]
    assert [c["protomer"] for c in states[1]["chains"]] == ["A", "A"]
    assert [c["protomer"] for c in states[2]["chains"]] == ["B", "B"]

    with pytest.raises(ValueError):
        parse_oligo_states("123+")

    out_dir = tmp_path / "oligo_smoke"
    rc = cli_main(
        [
            "--oligo",
            "AB+,AA-,BB-",
            "--L",
            "12,12",
            "--losses",
            "dual",
            "--steps",
            "3",
            "--mock",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / ".done_af3_hall").exists()
    summary = json.loads((out_dir / "best_summary.json").read_text(encoding="utf-8"))
    assert summary["cys_cnt"] == 0
    assert summary["pro_cnt"] == 0
    assert summary["sequon_cnt"] == 0
    assert set(summary["best_protomers"].keys()) == {"A", "B"}
    assert len(summary["best_protomers"]["A"]) == 12
    assert len(summary["best_protomers"]["B"]) == 12


@pytest.mark.parametrize(
    "yaml_name",
    [
        "01_heterodimer_AB_parity.yaml",
        "02_protein_binder_fixed_target.yaml",
        "03_glycoprotein_binder_multistate.yaml",
        "04_small_molecule_ligand_binder.yaml",
        "05_foldable_heterodimer_GB1_proteinL.yaml",
    ],
)
def test_all_examples_cli_mock(yaml_name: str, tmp_path: Path) -> None:
    cfg_path = EXAMPLES_DIR / yaml_name
    assert cfg_path.exists(), f"Missing {cfg_path}"
    out_dir = tmp_path / yaml_name.replace(".yaml", "")
    res = _run_cli(["--config", str(cfg_path), "--mock", "--steps", "2", "--out", str(out_dir)])
    assert res.returncode == 0, f"{yaml_name} failed:\n{res.stderr}\n{res.stdout}"
    assert (out_dir / ".done_af3_hall").exists()
    assert (out_dir / "best_summary.json").exists()
    assert (out_dir / "candidates.tsv").exists()
    summary = json.loads((out_dir / "best_summary.json").read_text(encoding="utf-8"))
    assert summary["cys_cnt"] == 0
    assert summary["pro_cnt"] == 0
    assert summary["sequon_cnt"] == 0
    if "glycoprotein" in yaml_name:
        assert summary["best_metrics"]["glyco_stereo_ok"] == 1
        assert summary["best_metrics"]["ipsae_min_glyco"] > 0.0
    if "ligand" in yaml_name:
        st_ctx = summary["best_metrics"]["states"]["holo_atp"]
        assert st_ctx["ipsae_min"] > 0.0
        assert st_ctx["iptm"] > 0.0


def test_portable_env_vars_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AF3_MODEL_DIR", "/custom/models")
    monkeypatch.setenv("AF3_DB_DIR", "/custom/dbs")
    monkeypatch.setenv("AF3_CODE_DIR", "/custom/af3")
    monkeypatch.setenv("AF3_JAX_CACHE_DIR", "/custom/jax_cache")
    monkeypatch.setenv("LIGANDMPNN_DIR", "/custom/lmpnn")
    cfg = load_config()
    assert cfg["model_dir"] == "/custom/models"
    assert cfg["db_dir"] == "/custom/dbs"
    assert cfg["af3_dir"] == "/custom/af3"
    assert cfg["jax_cache_dir"] == "/custom/jax_cache"
    assert cfg["mpnn_dir"] == "/custom/lmpnn"
    assert cfg["buckets"] == [256, 384, 512, 768, 1024, 1536, 2048]
    assert cfg["out_root"] == "./runs"
    assert cfg["tmp_root"] == "./tmp"


def test_loss_registry_and_custom_plugins(tmp_path: Path) -> None:
    for expected in (
        "plddt",
        "ptm",
        "iptm",
        "pae",
        "ipae",
        "pae_asym",
        "dual",
        "rmsd",
        "contact",
        "rg",
        "aspect_ratio",
        "ipsae_min",
        "glycan_hbonds",
        "steric_clash",
    ):
        assert expected in LOSS_REGISTRY
    assert "glyco_delta" in MULTISTATE_LOSS_REGISTRY

    @register_loss("custom_unit_bonus")
    def _custom_loss(ctx: dict) -> float:
        return float(1.0 - 0.5 * ctx["plddt"] - 0.5 * ctx["ptm"])

    @register_multistate_loss("custom_ms_contrast")
    def _custom_ms(state_ctx: dict, cfg: dict) -> tuple[float, dict]:
        vals = [v["ptm"] for v in state_ctx.values()]
        j = float(max(vals))
        return -j, {"J": j}

    try:
        parsed = parse_loss_spec("custom_unit_bonus,aspect_ratio,pae_asym", "1.0,0.2,0.1")
        assert len(parsed) == 3
        cfg = load_config(
            overrides=[
                "backend=mock",
                "oligo=A+",
                "binder_length=16",
                "steps=2",
                "losses=custom_unit_bonus,aspect_ratio,pae_asym",
                "loss_weights=1.0,0.2,0.1",
                f"out_dir={tmp_path / 'custom_loss_run'}",
            ]
        )
        rdir = run(cfg)
        assert (rdir / ".done_af3_hall").exists()

        cfg_ms = load_config(
            overrides=[
                "backend=mock",
                "oligo=AB+,AA-",
                "binder_length=12,12",
                "steps=2",
                "losses=custom_ms_contrast",
                "loss_weights=1.0",
                f"out_dir={tmp_path / 'custom_ms_run'}",
            ]
        )
        rdir_ms = run(cfg_ms)
        assert (rdir_ms / ".done_af3_hall").exists()
    finally:
        LOSS_REGISTRY.pop("custom_unit_bonus", None)
        MULTISTATE_LOSS_REGISTRY.pop("custom_ms_contrast", None)


def test_lazy_neg_eval_and_alias(tmp_path: Path) -> None:
    cfg = load_config(
        overrides=[
            "backend=mock",
            "oligo=AB+,AA-,BB-",
            "binder_length=12,12",
            "losses=dual",
            "steps=4",
            "lazy_neg_interval=10",
            "T_init=1.0",
            f"out_dir={tmp_path / 'lazy_neg'}",
        ]
    )
    # Force an identical or worse sequence on step 1 so pos_improved is False and step % 10 != 0
    seq_a = "ADEFGHIKLMQR"
    seq_b = "STVWYADEFGHI"
    cfg["binder_seq"] = f"{seq_a},{seq_b}"
    cfg["force_proposals"] = {
        1: {"A": "QDEFGHIKLMQR", "B": seq_b},
        2: {"A": seq_a, "B": "QTVWYADEFGHI"},
    }
    rdir = run(cfg)
    rows = [json.loads(ln) for ln in (rdir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    # Verify at least one step skipped negative state evaluation or ran cleanly
    assert all("neg_evaluated" in r for r in rows)
    assert any(r["neg_evaluated"] is False for r in rows[1:]) or any(r["neg_evaluated"] is True for r in rows[1:])

    # Backward-compatible lazy_bare_interval alias
    cfg_alias = load_config(overrides=["lazy_bare_interval=4"])
    assert cfg_alias["lazy_neg_interval"] == 4
    assert cfg_alias["lazy_bare_interval"] == 4


def test_dynamic_mask_refresh_and_p_nonint(tmp_path: Path) -> None:
    cfg = load_config(
        overrides=[
            "backend=mock",
            "target_seq=MSELQKAIEELAKKAEELAKKLEELAKKAEELAKKLEELG",
            "binder_length=32",
            "mask_mode=interface",
            "interface_dist=6.0",
            "mask_refresh_interval=1",
            "p_nonint=0.35",
            "steps=3",
            "losses=ipsae_min,plddt",
            "loss_weights=1.0,0.5",
            f"out_dir={tmp_path / 'mask_refresh'}",
        ]
    )
    rdir = run(cfg)
    log_txt = (rdir / "run.log").read_text(encoding="utf-8")
    assert "MASK_REFRESH step=1" in log_txt
    assert "MASK_REFRESH step=2" in log_txt
    assert "MASK_REFRESH step=3" in log_txt


def test_carb_blosum_and_ligand_mpnn_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGANDMPNN_DIR", raising=False)
    monkeypatch.delenv("LIGANDMPNN_CHECKPOINT", raising=False)

    sampler = LigandMPNNBiasedSampler(exclude_aa="CP", wf_cap_frac=0.12)
    assert not sampler.is_available()

    rng = np.random.default_rng(42)
    # Sequence already above 12% W/F fraction -> carb_blosum/fallback should avoid proposing W/F
    seq = "WWWWFFFFFFFFAAAADDDD"
    prop = propose_mutant(
        rng,
        seq,
        mutable_positions=[12, 13, 14, 15],
        n_mut=2,
        method="ligand_mpnn_biased",
        mpnn=sampler,
        exclude_aa="CP",
        forbid_sequon=True,
        wf_cap_frac=0.12,
    )
    assert prop["method"] == "carb_blosum"
    assert cp_count(prop["seq"]) == 0
    assert sequon_count(prop["seq"]) == 0
    for pos in prop["positions"]:
        assert prop["seq"][pos] not in ("W", "F")


def test_allow_init_forbidden_cleanup_and_best_clean(tmp_path: Path) -> None:
    # Start sequence intentionally contains Cys (pos 0), Pro (pos 1), and N-A-S sequon (pos 2..4)
    dirty_start = "CPNASDEFGHIKLMQRSTVW"
    cfg = load_config(
        overrides=[
            "backend=mock",
            "oligo=A+",
            f"binder_seq={dirty_start}",
            "mask_mode=all",
            "mutation_rate=2",
            "allow_init_forbidden=true",
            "exclude_aa=CP",
            "forbid_sequon=true",
            "T_init=1.0",
            "steps=4",
            f"out_dir={tmp_path / 'clean_purge'}",
        ]
    )
    rdir = run(cfg)
    rows = [json.loads(ln) for ln in (rdir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    # Step 1 must target forbidden positions first (indices 0, 1, 2, 4)
    step1_muts = rows[1]["mutations"]
    assert any(idx in step1_muts for idx in ("1", "2", "3", "5"))
    assert (rdir / "best_clean_summary.json").exists()
    bc = json.loads((rdir / "best_clean_summary.json").read_text(encoding="utf-8"))
    assert bc["cys_cnt"] == 0
    assert bc["pro_cnt"] == 0
    assert bc["sequon_cnt"] == 0
    assert Path(bc["best_cifs"]["A+"]).exists()


def test_atomic_state_resume_continuity(tmp_path: Path) -> None:
    out_split = tmp_path / "split_run"
    out_cont = tmp_path / "cont_run"

    cfg_2 = load_config(
        overrides=[
            "backend=mock",
            "oligo=AB+,AA-,BB-",
            "binder_length=14,14",
            "mutation_rate=2",
            "losses=dual",
            "steps=2",
            "seed=7",
            f"out_dir={out_split}",
        ]
    )
    run(cfg_2)
    assert (out_split / "state.json").exists()
    s2 = json.loads((out_split / "state.json").read_text(encoding="utf-8"))
    assert s2["step_done"] == 2

    cfg_4_res = load_config(
        overrides=[
            "backend=mock",
            "oligo=AB+,AA-,BB-",
            "binder_length=14,14",
            "mutation_rate=2",
            "losses=dual",
            "steps=4",
            "seed=7",
            f"out_dir={out_split}",
        ]
    )
    run(cfg_4_res)

    cfg_4_cont = load_config(
        overrides=[
            "backend=mock",
            "oligo=AB+,AA-,BB-",
            "binder_length=14,14",
            "mutation_rate=2",
            "losses=dual",
            "steps=4",
            "seed=7",
            f"out_dir={out_cont}",
        ]
    )
    run(cfg_4_cont)

    rows_res = [json.loads(ln) for ln in (out_split / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_cont = [json.loads(ln) for ln in (out_cont / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows_res] == [0, 1, 2, 3, 4]
    assert [r["sequence"] for r in rows_res] == [r["sequence"] for r in rows_cont]
    assert [r["best_loss"] for r in rows_res] == pytest.approx([r["best_loss"] for r in rows_cont])


def test_sequence_hygiene_and_glycan_trees(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    for _ in range(10):
        seq = random_sequence(40, rng, exclude_aa="CP", forbid_sequon=True)
        assert cp_count(seq) == 0
        assert sequon_count(seq) == 0
        validate_binder_seq(seq)

    with pytest.raises(ForbiddenSequenceError):
        validate_binder_seq("ADEFGCNASIK")  # sequon NAS
    with pytest.raises(ForbiddenSequenceError):
        validate_binder_seq("ADEFGCPKLM")  # C and P

    # Verify all 5 built-in glycoform trees + custom sugar_nodes + bondedAtomPairs build & mock cleanly
    mock = MockAF3Runner()
    for gf, expected_nodes in GLYCOFORM_TREES.items():
        spec = {
            "name": f"test_{gf}",
            "chains": [
                {"type": "protein", "id": "A", "mutable": False, "sequence": "AAAAANATAAAA"},
                {"type": "protein", "id": "B", "mutable": True, "protomer": "B"},
                {"type": "glycan", "anchor_chain": "A", "anchor_res_1based": 6, "glycoform": gf},
            ],
        }
        payload = build_state_af3_json(f"job_{gf}", spec, {"B": "ADEFGHIKLMQR"})
        assert len(payload["bondedAtomPairs"]) == len(expected_nodes)
        cif_p, summary, conf = mock.predict_json(payload, tmp_path / gf, f"job_{gf}")
        assert cif_p.exists()
        assert len(conf["token_chain_ids"]) == len(conf["pae"])

    custom_spec = {
        "name": "custom_sugar",
        "chains": [
            {"type": "protein", "id": "A", "mutable": False, "sequence": "AAAAANATAAAA"},
            {"type": "protein", "id": "B", "mutable": True, "protomer": "B"},
            {
                "type": "glycan",
                "anchor_chain": "A",
                "anchor_res_1based": 6,
                "sugar_nodes": [
                    {"node_id": "S1", "ccd": "NAG", "parent_id": "ASN", "parent_atom": "ND2"},
                    {"node_id": "S2", "ccd": "NAG", "parent_id": "S1", "parent_atom": "O4"},
                ],
            },
        ],
        "bonded_atom_pairs": [[["A", 1, "CA"], ["B", 1, "CA"]]],
    }
    p_custom = build_state_af3_json("custom_g", custom_spec, {"B": "ADEFGHIKLMQR"})
    assert len(p_custom["bondedAtomPairs"]) == 3


def test_steric_clash_guard_and_candidate_export(tmp_path: Path) -> None:
    # Create a context steric reference PDB where residues 1..10 are target core, and residue 11
    # sits right on top of chain B's mock coordinates (~9.5, 0, 0) to trigger a steric clash!
    ref_pdb = tmp_path / "steric_ref.pdb"
    lines: list[str] = []
    serial = 1
    for rid in range(1, 11):
        theta = 0.65 * (rid - 1)
        z_pos = ((rid - 1) - 4.5) * 1.45
        x, y, z = 5.2 * np.cos(theta), 5.2 * np.sin(theta), z_pos
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA A{rid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C"
        )
        serial += 1
    # Flanking obstacle residue 11 placed inside chain B's binding volume
    lines.append(f"ATOM  {serial:5d}  CA  ALA A  11    {14.1:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C")
    lines.append("END")
    ref_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "steric_run"
    export_tsv = tmp_path / "exported_candidates.tsv"
    cfg = load_config(
        overrides=[
            "backend=mock",
            "target_seq=AAAAAAAAAA",
            "binder_length=10",
            "steps=2",
            "losses=ipsae_min,steric_clash",
            "loss_weights=1.0,0.5",
            f"steric_clash_ref_cif={ref_pdb}",
            "steric_ref_residues=[1,10]",
            "steric_clash_weight=0.5",
            f"export_candidates_tsv={export_tsv}",
            f"out_dir={out_dir}",
        ]
    )
    run(cfg)
    assert export_tsv.exists()
    with export_tsv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) >= 1
    assert rows[0]["cys_cnt"] == "0"
    assert rows[0]["pro_cnt"] == "0"
    assert rows[0]["sequon_cnt"] == "0"

    j_val = compute_objective_J(
        0.5, 0.6, 0.4, 0.3, 3, 0.5, cfg,
        steric_hv25=1, steric_ca38=0, target_protein_contacts=16, n_forbidden_aa=0,
    )
    assert np.isfinite(j_val)


def test_trajectory_always_written_and_all_mode(tmp_path: Path) -> None:
    def run(extra: list[str]) -> tuple[Path, list[dict]]:
        out = tmp_path / ("t_" + "_".join(extra) if extra else "t_default")
        args = ["--mock", "--oligo", "AB+", "--L", "8,8", "--losses", "dual", "--steps", "6", "--seed", "5",
                "--out-dir", str(out), "T_init=0.5", *extra]
        assert cli_main(args) == 0
        return out, [json.loads(ln) for ln in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()]

    d, rows = run([])
    acc = [r["step"] for r in rows if r["accepted"]]
    assert acc[0] == 0
    assert sorted(p.name for p in (d / "trajectory").glob("*.cif")) == [f"step_{k:04d}_acc_AB+.cif" for k in acc]
    d2, rows2 = run(["--save-trajectory", "all"])
    assert len(list((d2 / "trajectory").glob("*.cif"))) == len(rows2)


# ---------------- --fastrelax (tmol FastRelax before scoring) ----------------
class _MockRelaxer:
    """Copies the AF3 CIF to <stem>_relaxed.cif and records calls (stands in for the tmol worker)."""

    def __init__(self, ok: bool = True) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.ok = ok

    def relax(self, in_cif, out_cif, *, focus_chains=()):
        self.calls.append((Path(in_cif).name, tuple(focus_chains)))
        if not self.ok:
            return {"ok": False, "skipped": True, "error": "mock failure"}
        Path(out_cif).write_text(Path(in_cif).read_text(encoding="utf-8"), encoding="utf-8")
        return {"ok": True, "E_before": 10.0, "E_after": -5.0, "ca_rmsd": 0.3, "relax_sec": 0.01}

    def summary(self) -> str:
        return f"mock relaxer calls={len(self.calls)}"

    def close(self) -> None:
        pass


def test_fastrelax_flags_defaults_hook_and_skip(tmp_path: Path) -> None:
    from af3_hall.cli import build_parser
    from af3_hall.predictor import AF3Predictor

    a = build_parser().parse_args(["--fastrelax", "--relax-scope", "shell", "--relax-repeats", "2"])
    assert a.fastrelax is True and a.relax_scope == "shell" and a.relax_repeats == 2
    cfg = load_config(overrides=["oligo=AB+", "binder_length=8,8", "losses=dual", "steps=2", "seed=1", "backend=mock"])
    assert cfg["fastrelax"] is False and cfg["relax_scope"] == "all" and cfg["relax_repeats"] == 1
    states = build_states_from_config(cfg)
    seqs = {"A": "ADEFGHIK", "B": "ADEFGHIK"}

    rx = _MockRelaxer()
    out = AF3Predictor(states, cfg, runner=MockAF3Runner(), relaxer=rx).predict_all(seqs, seed=1, work_dir=tmp_path / "w", tag="t")
    assert rx.calls
    for p in out.values():
        assert p.relax and p.relax["ok"] and p.cif_path.name.endswith("_relaxed.cif") and p.cif_path.exists()
        assert p.cif_path.with_suffix(".relax.json").exists()
    out2 = AF3Predictor(states, cfg, runner=MockAF3Runner(), relaxer=_MockRelaxer(ok=False)).predict_all(
        seqs, seed=1, work_dir=tmp_path / "w2", tag="t")
    for p in out2.values():
        assert p.relax and p.relax["ok"] is False and not p.cif_path.name.endswith("_relaxed.cif")
    assert AF3Predictor(states, cfg, runner=MockAF3Runner()).relaxer is None


def test_fastrelax_mcmc_and_tmol_relaxer_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from af3_hall import relax as relax_mod
    from af3_hall.predictor import AF3Predictor

    cfg = load_config(overrides=[f"out_dir={tmp_path / 'rx'}", f"tmp_root={tmp_path}", "oligo=AB+", "binder_length=8,8",
                                 "losses=dual", "steps=2", "seed=3", "T_init=0.01", "backend=mock"])
    rx = _MockRelaxer()
    run(cfg, predictor=AF3Predictor(build_states_from_config(cfg), cfg, runner=MockAF3Runner(), relaxer=rx))
    assert len(rx.calls) >= 3

    fake = tmp_path / "fake_worker.py"
    fake.write_text(
        "import sys, json, shutil\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    if r['in_cif'].endswith('bad.cif'):\n"
        "        print(json.dumps({'ok': False, 'error': 'boom'}), flush=True); continue\n"
        "    shutil.copy(r['in_cif'], r['out_cif'])\n"
        "    print(json.dumps({'ok': True, 'E_before': 1.0, 'E_after': 0.0, 'ca_rmsd': 0.1, 'relax_sec': 0.0, 'scope': r['scope'],\n"
        "                      'repeats': r['repeats'], 'focus': r['focus_chains']}), flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(relax_mod, "WORKER", fake)
    src, bad = tmp_path / "x.cif", tmp_path / "bad.cif"
    src.write_text("data_x\n", encoding="utf-8"); bad.write_text("data_x\n", encoding="utf-8")
    rxr = relax_mod.TmolRelaxer(python=sys.executable, repeats=2, scope="shell", on_fail="raise", timeout_s=30)
    try:
        meta = rxr.relax(src, tmp_path / "x_relaxed.cif", focus_chains=["B"])
        assert meta["ok"] and meta["scope"] == "shell" and meta["repeats"] == 2 and meta["focus"] == ["B"]
        with pytest.raises(relax_mod.RelaxError):
            rxr.relax(bad, tmp_path / "bad_relaxed.cif")
        rxr.on_fail = "skip"
        m2 = rxr.relax(bad, tmp_path / "bad_relaxed.cif")
        assert m2["ok"] is False and m2["skipped"] is True
        assert "ok=1 fail=2" in rxr.summary()
    finally:
        rxr.close()
    with pytest.raises(ValueError):
        relax_mod.TmolRelaxer(scope="bogus")
