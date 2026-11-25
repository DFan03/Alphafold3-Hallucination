# Loss Functions, Objectives & Custom Plugins

`af3-hall` minimizes a scalar `loss` during simulated-annealing Metropolis acceptance while logging `J = -loss` (or the explicit composite `J` in `glyco_delta`, where higher `J` is better). Losses are composable via `--losses` and `--loss-weights` and are defined in [src/af3_hall/scoring.py](../src/af3_hall/scoring.py).

---

## 1. Built-In Per-State Losses (`@register_loss`)

Every per-state loss takes `ctx: dict[str, Any]` (produced by `extract_state_context`) and returns a `float` where **lower is better**.

| Registry Name | Formula | Direction / Range | Notes |
| :--- | :--- | :--- | :--- |
| `plddt` | `1.0 - plddt` | `[0, 1]`, lower = higher pLDDT | Mean `atom_plddts / 100` over mutable protein chain(s). |
| `ptm` | `1.0 - ptm` | `[0, 1]`, lower = higher pTM | Complex `summary["ptm"]`. |
| `iptm` | `1.0 - iptm` | `[0, 1]`, lower = higher ipTM | Uses `chain_iptm[mut_idx]` when available, fallback `summary["iptm"]`. |
| `pae` | `pae_mean / 31.75` | `[0, 1]`, lower = tighter PAE | Mean over the full `[N_tok, N_tok]` PAE matrix (`PAE_MAX = 31.75` Å). |
| `ipae` | `ipae_mean / 31.75` | `[0, 1]`, lower = tighter inter-PAE | Mean inter-chain PAE across tokens on different chains. |
| `pae_asym` | `abs(mean(pae[B,R]) - mean(pae[R,B])) / 31.75` | `[0, 1]`, lower = symmetric PAE | Penalizes directional PAE asymmetry between mutable chain and partner. |
| `dual` | `0.5 * ((1.0 - plddt) + (1.0 - ptm))` | `[0, 1]`, default loss | AFhall-parity monomer/multimer folding + confidence loss. |
| `rmsd` | `rmsd_core` | `>= 0` Å | Kabsch-aligned core Cα RMSD of mutable chain relative to `ref_cif` / step-0. |
| `contact` | `contact_score` | `[0, 1]`, `0` = tight contact | ColabDesign-style mean closest `k=10` Cβ–Cβ distance to partner chains normalized by `14.0` Å. |
| `rg` | `max(0.0, rg_ratio - 1.0)` | `>= 0` | Penalizes radius of gyration exceeding `2.2 * N^0.38` Å. |
| `aspect_ratio` | `max(0.0, aspect_ratio - 1.0)` | `>= 0` | Penalizes elongated chains (`sqrt(lambda_max / lambda_mid) > 1`). |
| `ipsae_min` | `1.0 - ipsae_min` | `[0, 1]`, lower = stronger interface | Dunbrack/Overath `ipSAE_min = min(b2r, r2b)` with `PAE < pae_cutoff` (10.0 Å), `dist < dist_cutoff` (10.0 Å), `d0 = max(1.0, 1.24*(max(n,27)-15)^(1/3) - 1.8)`. |
| `glycan_hbonds` | `-hbonds_glycan` | `<= 0` | Rewards polar heavy-atom H-bonds (`<= hbond_cutoff`, default `3.5` Å) between mutable chain and glycan oxygens. |
| `steric_clash` | `steric_hv_buf + 0.5 * steric_ca_buf` | `>= 0` | Binder clashes against `steric_clash_ref_cif` after rigid target alignment (`hv < steric_hv_cutoff`, `ca < steric_ca_cutoff`). |

### How Multiple Per-State Losses Combine Across States

When `losses` contains per-state names `[(n_1, w_1), ...]`, each state `s` with weight `sw` (e.g. `+1.0` or `-0.5` in `AB+,AA-,BB-`) computes its normalized weighted sum of losses (`raw_state_loss`), and `evaluate_multistate` combines them as:

```text
total_loss = sum_s (sw_s / sum_k |sw_k|) * raw_state_loss_s
```

If `steric_clash_weight > 0` and `"steric_clash"` is not explicitly listed in `losses`, `evaluate_multistate` automatically adds `steric_clash_weight * (steric_hv_buf + 0.5 * steric_ca_buf)` for the primary positive state.

---

## 2. Built-In Multistate Loss: `glyco_delta` (`@register_multistate_loss`)

`glyco_delta` evaluates the 2-state contrast between a glycosylated state (`"glyco"`, or the first state if `"bare"` is omitted) and an aglycosylated control state (`"bare"`), maximizing the composite objective `J` and returning `loss = -J`.

### Exact `compute_objective_J` Formula

```text
delta_iptm  = iptm_glyco - iptm_bare
delta_ipsae = ipsae_min_glyco - ipsae_min_bare
steric_pen  = steric_clash_weight * (steric_hv_buf + 0.5 * steric_ca_buf)

J = w1_ipsae_glyco * ipsae_min_glyco
  + w2_delta_iptm  * delta_iptm
  + w3_delta_ipsae * delta_ipsae
  + w4_hbonds_glycan * hbonds_glycan
  - w5_rmsd_core   * rmsd_core
  - steric_pen
  + w_glycan_contacts        * min(glycan_contacts_45 / 20.0, 1.5)
  + w_nonstem_sugars         * min(nonstem_sugars_hit / 3.0, 2.0)
  + w_ch_pi                  * min(ch_pi_contacts, 4.0)
  + w_binder_plddt           * binder_plddt
  - w_stereo_penalty         * (1 - glyco_stereo_ok)
  - w_rg_penalty             * max(0.0, rg_ratio - 1.0)
  - w_excess_target_contacts * max(0.0, target_protein_contacts - target_max_contacts)
  - w_forbidden_aa           * n_forbidden_aa
```

### Default Weights in `DEFAULTS` (`src/af3_hall/config.py`)

| Weight Key | Default | Description |
| :--- | :--- | :--- |
| `w1_ipsae_glyco` | `1.0` | Weight on `ipsae_min` in the `glyco` state |
| `w2_delta_iptm` | `1.0` | Weight on `(iptm_glyco - iptm_bare)` specificity gap |
| `w3_delta_ipsae` | `1.0` | Weight on `(ipsae_min_glyco - ipsae_min_bare)` specificity gap |
| `w4_hbonds_glycan` | `0.05` | Weight per polar H-bond (`<= hbond_cutoff`) to glycan oxygens |
| `w5_rmsd_core` | `0.10` | Penalty per Å of core Cα RMSD drift vs reference |
| `steric_clash_weight` | `0.0` | Multiplier on `steric_hv_buf + 0.5 * steric_ca_buf` |
| `w_glycan_contacts` | `0.0` | Opt-in reward for binder–glycan heavy-atom contacts (`<= 4.5` Å, capped at 30) |
| `w_nonstem_sugars` | `0.0` | Opt-in reward for contacting distal/non-stem sugars (beyond `R1`/`R2`, capped at 6) |
| `w_ch_pi` | `0.0` | Opt-in reward for Trp/Tyr/Phe CH–π stacking with pyranose rings (`<= 5.0` Å, tilt `<= 35°`) |
| `w_binder_plddt` | `0.0` | Opt-in reward for binder mean pLDDT (`[0, 1]`) |
| `w_stereo_penalty` | `0.0` | Opt-in penalty if any pyranose fails Cremer–Pople / anomeric stereochemistry (`glyco_stereo_ok == 0`) |
| `w_rg_penalty` | `0.0` | Opt-in penalty on non-compact binder radius of gyration (`max(0, rg_ratio - 1)`) |
| `w_excess_target_contacts` | `0.0` | Opt-in penalty per target-protein contact above `target_max_contacts` (`14`) |
| `w_forbidden_aa` | `0.0` | Opt-in penalty per `C`/`P` or `N-X-S/T` sequon violation |

---

## 3. Writing a Custom Loss Plugin

Register custom per-state (`@register_loss`) or cross-state (`@register_multistate_loss`) objectives in any Python script before calling `af3_hall.run(cfg)`:

```python
from af3_hall import load_config, register_loss, register_multistate_loss, run


@register_loss("compact_interface")
def loss_compact_interface(ctx: dict) -> float:
    # Lower is better: reward high ipSAE_min while keeping Rg compact
    return (1.0 - ctx["ipsae_min"]) + 0.25 * max(0.0, ctx["rg_ratio"] - 1.0)


@register_multistate_loss("selective_holo")
def loss_selective_holo(state_ctx: dict, cfg: dict) -> tuple[float, dict]:
    holo = state_ctx["holo"]
    apo = state_ctx.get("apo")
    delta_iptm = holo["iptm"] - (apo["iptm"] if apo else 0.0)
    j_val = holo["ipsae_min"] + 0.5 * delta_iptm
    return -j_val, {"J": j_val, "delta_iptm": delta_iptm}


cfg = load_config("examples/02_protein_binder_fixed_target.yaml")
cfg["losses"] = "compact_interface,plddt"
cfg["loss_weights"] = "1.0,0.5"
cfg["backend"] = "mock"
cfg["steps"] = 10
run(cfg)
```

Available keys in every per-state `ctx` dict (`extract_state_context`):
`plddt`, `binder_plddt`, `ptm`, `iptm`, `pae_mean`, `ipae_mean`, `pae_asym`, `ipsae_min`, `ipsae_max`, `ipsae_glycan_min`, `hbonds_glycan`, `glycan_contacts_45`, `nonstem_sugars_hit`, `ch_pi_contacts`, `glyco_stereo_ok`, `target_protein_contacts`, `n_forbidden_aa`, `rmsd_core`, `rg_ratio`, `aspect_ratio`, `contact_score`, `ranking_score`, `steric_hv25`, `steric_ca38`, `steric_hv_buf`, `steric_ca_buf`, `intact_clash_ok`.
