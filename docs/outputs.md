# Outputs, Resumption & Sequence Hygiene

Every `af3-hall` run writes deterministic, crash-safe checkpoints to `--out` (or `<out_root>/<YYYY-MM-DD>_af3_hall_<job>` in UTC).

---

## 1. Output Directory Layout

```text
runs/2026-11-25_af3_hall_my_job/
├── config.json               # Resolved configuration snapshot written at startup
├── run.log                   # UTC ISO-8601 progress log (or <log_dir>/af3_hall_<job>.log)
├── steps.jsonl               # Append-only per-step trajectory log (step 0..N)
├── state.json                # Atomic checkpoint enabling seamless crash/stop resume
├── ref_<state>.cif           # Step-0 reference structure (or copied from ref_cif)
├── cur_<state>.cif           # Most recently accepted structure per state
├── best_<state>.cif          # Lowest-loss (highest-J) structure overall per state
├── best_clean_<state>.cif    # Lowest-loss structure passing all hygiene rules per state
├── best_summary.json         # Final summary (best + best_clean sequences, metrics, CIF paths)
├── best_clean_summary.json   # Emitted when at least one hygiene-clean state was reached
├── candidates.tsv            # Single-row summary TSV (prefers best_clean if available)
├── .done_af3_hall            # Empty sentinel file created when all steps finish
└── work/                     # Transient per-step AF3 scratch (auto-cleaned after each step)
```

> **Note:** Setting `export_candidates_tsv: ./top_k.tsv` in YAML (or calling `af3_hall.export_candidates_tsv(run_dir, top_k=10)`) exports up to `top_k` distinct hygiene-clean candidates ranked by `J`.

---

## 2. `steps.jsonl` Schema

Each line in `steps.jsonl` is a standalone JSON object (step `0` for initialization, followed by steps `1..steps`):

| Field | Type | Description |
| :--- | :--- | :--- |
| `step` | `int` | `0` for initial evaluation, `1..steps` for MCMC proposals. |
| `accepted` | `bool` | Whether the Metropolis criterion accepted the best proposal of this step. |
| `reason` | `str` | `"init"`, `"improved"`, `"metropolis"`, or `"worse"`. |
| `temperature` | `float` | Simulated-annealing temperature `T` at this step. |
| `n_mut` | `int` | Number of mutated positions attempted (`0` at step 0). |
| `mutations` | `str` | 1-based mutation string (e.g. `"A12K,S15E"` or `"B:A12K"` for multiprotomer, `"none"` at step 0). |
| `sequence` | `str` | Current accepted sequence signature after this step (for single protomer: sequence; for multi-protomer: `"A=...:B=..."`). |
| `proposed_sequence` | `str` | Candidate sequence evaluated at this step (present for steps `>= 1`). |
| `protomers` | `dict[str, str]` | Mapping `{protomer_id: current_accepted_sequence}`. |
| `cys_cnt`, `pro_cnt`, `sequon_cnt` | `int` | Total Cys, Pro, and `N-X-S/T` (`X != P`) counts across current `protomers`. |
| `loss`, `loss_cur`, `best_loss` | `float` | Proposed `loss`, current accepted `loss` after step, and best `loss` seen so far. |
| `J`, `J_cur`, `best_J`, `delta_J` | `float` | Proposed `J`, current `J` after step, best `J` seen so far, and `J_proposed - J_cur_before`. |
| `p_accept` | `float` | Metropolis acceptance probability (`1.0` when improved, `exp(-(d_loss + margin)/T)` otherwise). |
| `neg_evaluated` | `bool` | `true` if negative states (`weight < 0` or `bare`) were evaluated this step; `false` when lazy-skipped (`bare_evaluated` is an alias). |
| `states` | `dict[str, dict]` | Full per-state context metrics (`plddt`, `ptm`, `iptm`, `ipsae_min`, `hbonds_glycan`, `rmsd_core`, etc.). |
| `t_step_s` | `float` | Wall-clock seconds for this step. |

When present in `evaluate_multistate`, top-level convenience metrics are also mirrored onto each `steps.jsonl` record:
`ipsae_min_glyco`, `ipsae_glycan_min`, `ipsae_min_bare`, `delta_ipsae_min`, `iptm_glyco`, `iptm_bare`, `delta_iptm`, `hbonds_glycan`, `glycan_contacts_45`, `nonstem_sugars_hit`, `ch_pi_contacts`, `binder_plddt`, `glyco_stereo_ok`, `rg_ratio`, `aspect_ratio`, `target_protein_contacts`, `n_forbidden_aa`, `rmsd_core`, `steric_hv25`, `steric_ca38`, `intact_clash_ok`.

---

## 3. Automatic Checkpoint & Resume (`state.json`)

After step 0 and every subsequent step, `mcmc.py` atomically writes `state.json` via write-then-`os.replace`:
- Stores `step_done`, `cur_seqs`, `cur_eval`, `best_seqs`, `best_eval`, `best_clean_seqs`, `best_clean_eval`, `mutable_masks`, `core_positions`, `ref_cifs`, and `cur_cifs`.
- **To resume an interrupted run** (or extend `--steps` from `100` to `200`), re-run `af3-hall` pointing `--out` at the existing run directory:

```bash
af3-hall --config examples/02_protein_binder_fixed_target.yaml \
  --out runs/2026-11-25_af3_hall_protein_binder_fixed_target \
  --steps 200
```

`af3-hall` reads `state.json`, logs `RESUME ... at step <step_done + 1>/<total_steps>`, restores the exact masks, core positions, and reference structures, and continues appending to `steps.jsonl`.

---

## 4. Sequence Hygiene Guarantees (`exclude_aa`, `forbid_sequon`, `best_clean_*`)

By default, `af3-hall` enforces manufacturing and post-translational hygiene on every designed protomer:
1. **`exclude_aa: "CP"` (0 Cys, 0 Pro):** Both `C` (unpaired disulfide risk) and `P` (helix/sheet kinks) are excluded from `random_sequence`, `sample_residue`, `MPNNSampler`, `LigandMPNNBiasedSampler`, and `validate_binder_seq`. Customize via `--exclude-aa` (e.g. `--exclude-aa CPM` to also exclude Met).
2. **`forbid_sequon: true` (0 `N-X-S/T` sequons):** Every candidate substitution checks a local 5-residue window (`creates_sequon`) so no `N-X-S` or `N-X-T` (`X != P`) N-glycosylation motif is ever introduced on the binder. Disable only if desired via `--no-forbid-sequon`.
3. **Purging dirty starting sequences (`--allow-init-forbidden`):** When optimizing an existing seed sequence that initially contains `C`, `P`, or `N-X-S/T`, pass `--allow-init-forbidden`. `propose_mutant` prioritizes mutating violating positions first until clean, while `best_clean_<state>.cif` and `best_clean_summary.json` track the best **strictly clean** (`cys_cnt=0`, `pro_cnt=0`, `sequon_cnt=0`) structure separately from `best_<state>.cif`.

---

## 5. Plotting a Trajectory from `steps.jsonl`

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

run_dir = Path("runs/2026-11-25_af3_hall_protein_binder_fixed_target")
rows = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines() if line.strip()]

steps = [r["step"] for r in rows]
j_cur = [r["J_cur"] for r in rows]
j_best = [r["best_J"] for r in rows]
acc_steps = [r["step"] for r in rows if r["accepted"]]
acc_j = [r["J"] for r in rows if r["accepted"]]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, j_cur, label="Current J", color="#1f77b4", lw=1.6)
ax.plot(steps, j_best, label="Best J", color="#2ca02c", lw=2.0, ls="--")
ax.scatter(acc_steps, acc_j, s=14, color="#ff7f0e", alpha=0.75, label="Accepted")
ax.set_xlabel("MCMC Step")
ax.set_ylabel("Objective J (higher = better)")
ax.set_title(run_dir.name)
ax.legend()
fig.tight_layout()
fig.savefig(run_dir / "trajectory.png", dpi=180)
```
