# `--fastrelax`: tmol relax of every predicted state before scoring

Code: `src/af3_hall/relax.py` (client), `src/af3_hall/relax_tmol_worker.py` (worker), hook in `predictor.py::AF3Predictor.predict_state`. Tests: `tests/test_af3_hall_standalone.py` (`test_fastrelax_*`).

## What it does and when to use it
Off by default. With `--fastrelax`, every AF3-predicted state of every MCMC candidate (accepted or not) is relaxed with tmol 0.2.0 (GPU, beta2016) **before** it is scored. The relaxed CIF replaces the AF3 CIF for everything downstream: scoring, proposals, `cur_*.cif` / `best_*.cif`. The raw AF3 CIF stays next to it (`<name>.cif`, relaxed = `<name>_relaxed.cif`, metadata = `<name>_relaxed.relax.json`).

- **Does NOT change AF3 confidence metrics.** pLDDT, pTM, ipTM, ipAE and ipSAE come from the model outputs (confidence heads, PAE), not from the coordinates. Relaxing the structure cannot move them.
- **Changes structure-derived metrics only:** contacts, H-bonds, clash counts, RMSD to a reference, SASA / shape complementarity, and any energy-based or physics-based binder metric. Use `--fastrelax` when the loss or the ranking uses such terms; for pure-confidence losses (`dual`, ipSAE-only) it only adds cost.
- Currently relax metadata (`E_before`, `E_after`, `ca_rmsd`, `relax_sec`) is saved per state but no energy term is added to the loss or to `steps.jsonl`. An interface-energy metric (complex minus separated chains) is the natural next step and is NOT implemented.

## Options
| CLI flag | Config key | Default | Meaning |
|---|---|---|---|
| `--fastrelax` | `fastrelax` | `false` | Turn the feature on. |
| `--relax-scope all\|shell` | `relax_scope` | `all` | `all`: whole-complex Cartesian FastRelax (tmol `relax_pose_stack`, every atom free). `shell`: tmol `relax_shell`, backbone fixed, repack + minimise side chains of the residues of the mutable chain(s) within `relax_shell` A of the other chains. |
| `--relax-repeats N` | `relax_repeats` | `1` | FastRelax repeats (`all`) / repeats (`shell`). 2 repeats gave no meaningful extra energy gain in a test. |
| | `relax_shell` | `8.0` | Shell radius in A (`shell` scope only). |
| | `relax_cartesian` | `true` | Cartesian (true) or torsion-space (false) minimisation for `all`. |
| | `relax_on_fail` | `raise` | `raise` = stop the run; `skip` = keep the raw AF3 structure for that state and write `ok: false` into the `.relax.json`. |
| | `relax_python` | env `AF3_HALL_RELAX_PYTHON`, else `python` | Interpreter of an env with torch + tmol 0.2.0 + the RFD4-Proteina `rfproteina` package (the AF3 env is JAX-only). |
| | `relax_device` | `cuda:0` | Device for the worker. |
| | `relax_timeout_s` | `900` | Per-call timeout; a dead/hung worker is restarted once, then the failure policy applies. |

Config keys can also be given as `key=value` overrides (e.g. `fastrelax=true relax_scope=shell`).

## Example
```bash
export AF3_HALL_RELAX_PYTHON=/path/to/tmol_env/bin/python   # env with torch + tmol 0.2.0 + rfproteina (RFD4-Proteina)
af3-hall --config examples/05_foldable_heterodimer_GB1_proteinL.yaml --fastrelax --relax-scope shell
```
The run log ends with a summary line: `fastrelax(tmol, scope=all, repeats=1): ok=N fail=M relax_total=S s`.

## Requirements
The worker needs an environment with PyTorch + CUDA, [tmol](https://github.com/uw-ipd/tmol) 0.2.0 and the `rfproteina` package of RFD4-Proteina (it provides the `tmol_ddg_core` helpers used to build the pose and run `relax_pose_stack` / `relax_shell`). `--mock` runs never start it.

## Why tmol and not Rosetta FastRelax
- Rosetta FastRelax is CPU-only and slow to call every MCMC step, and needs a Rosetta/PyRosetta install.
- tmol 0.2.0 is GPU-based, runs in a persistent worker, and has a glycan path (`prepare_ligands=True`).

## Measured cost (100-residue AF3 complex, 2x50 aa, warm persistent worker, GPU shared with other users' jobs)
| Scope | relax time | Result |
|---|---|---|
| `all`, 1 repeat | about 13 s | tmol energy +66 -> -237 (block-pair sum, relative use only), CA shift about 0.47 A |
| `all`, 2 repeats | about 26 s (cold run) | -237, CA shift 0.55 A |
| `shell`, 1 repeat | about 3 s | +66 -> +3.6, CA shift about 1e-4 A (backbone fixed) |

An AF3 prediction takes about 4 s per state, so `all` costs roughly 3x a prediction and `shell` about 0.8x. The worker process and the tmol context are cached between calls (first call about 17 s extra for the build).

## Known limits / open items
- Not yet tested on glycosylated states (an unrelaxed A3G3F GAL residue triggered a tmol `C1` error in earlier tests); the worker builds the context with `prepare_ligands=True` when hetero residues are present, but this path is untested here.
- Energies are tmol block-pair sums, not calibrated kcal/mol: use only for before/after deltas or rankings.
- Only heavy atoms present in the AF3 CIF are written back (atom names, order and chain ids unchanged).
- Relax runs serially in one worker; pipelining it with the next AF3 prediction is a possible speed-up, not implemented.
