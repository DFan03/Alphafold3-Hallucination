# Installation & Runtime Setup

`af3-hall` separates lightweight orchestration (`numpy`, `pyyaml`) from AlphaFold 3 and LigandMPNN so you can install, test, and prototype trajectories on any CPU laptop before deploying to GPU.

---

## 1. Requirements

- **Python:** `3.10+` (`3.11+` recommended when pairing with AlphaFold 3 v3.0.1)
- **Core dependencies:** `numpy>=1.24`, `pyyaml>=6.0`
- **Optional `[mpnn]`:** `torch>=2.0` (for `--mutation-method mpnn` / `ligand_mpnn_biased`)
- **Optional `[dev]`:** `pytest>=7.4`
- **Real GPU predictions only:** AlphaFold 3 (`v3.0.1`) codebase, model parameters, public CCD/databases, CUDA GPU (Ampere/Hopper recommended for Triton FlashAttention)

---

## 2. Local Package Installation

```bash
git clone <your-repo-url> af3-hall
cd af3-hall

# Standard editable install with pytest
pip install -e ".[dev]"

# Include PyTorch if using ProteinMPNN / LigandMPNN proposals
pip install -e ".[dev,mpnn]"
```

Verify installation and CLI entry points:

```bash
af3-hall --help
python AF3_multistate_hallucination.py --help
pytest -q tests/
```

---

## 3. CPU `--mock` Mode vs Real AlphaFold 3 GPU Mode

### CPU-Only `--mock` Smoke Mode (Zero GPU / Zero Weights)

Passing `--mock` (or `--backend mock` / `backend: mock`) selects `MockAF3Runner` (`src/af3_hall/predictor.py`), which deterministically synthesizes all-atom mmCIF coordinates (including stereochemically valid `4C1`/`1C4` pyranose N-glycan chairs and small-molecule ligands), `atom_plddts`, `summary_confidences`, and `pae` matrices in milliseconds:

```bash
python AF3_multistate_hallucination.py --oligo AB+,AA-,BB- --L 12,12 --steps 5 --mock
af3-hall --config examples/03_glycoprotein_binder_multistate.yaml --mock --steps 5
```

### Real AlphaFold 3 (v3.0.1) GPU Setup

Configure environment variables (or pass `--af3-dir`, `--model-dir`, `--db-dir`, `--jax-cache-dir` / YAML keys):

| Environment Variable | Config Key | Default (`config.py`) | Purpose |
| :--- | :--- | :--- | :--- |
| `AF3_CODE_DIR` | `af3_dir` | `./alphafold3` | Path to Google DeepMind `alphafold3` repository |
| `AF3_MODEL_DIR` | `model_dir` | `./models` | Directory containing AlphaFold 3 model weights |
| `AF3_DB_DIR` | `db_dir` | `./public_databases` | AlphaFold 3 public databases (used by `subprocess` backend) |
| `AF3_JAX_CACHE_DIR` | `jax_cache_dir` | `./tmp/jax_cache_af3` | Persistent JAX XLA compilation cache directory |
| `LIGANDMPNN_DIR` | `mpnn_dir` | `""` | Optional ProteinMPNN / LigandMPNN repository (`run.py`) |
| `LIGANDMPNN_CHECKPOINT` | `mpnn_checkpoint` | `""` | Checkpoint `.pt` path for `ligand_mpnn_biased` |
| `LIGANDMPNN_PYTHON` | `mpnn_python` | `sys.executable` | Python interpreter used to invoke LigandMPNN |
| `AF3_HALL_OUT_ROOT` | `out_root` | `./runs` | Default root directory for timestamped run folders |
| `AF3_HALL_TMP_DIR` | `tmp_root` | `./tmp` | Scratch directory for subprocess/MPNN intermediate files |

Example GPU environment setup:

```bash
export AF3_CODE_DIR=/opt/alphafold3
export AF3_MODEL_DIR=/opt/af3_models
export AF3_DB_DIR=/opt/public_databases
export AF3_JAX_CACHE_DIR=/scratch/jax_cache_af3

af3-hall --config examples/02_protein_binder_fixed_target.yaml
```

> **Backends:** `backend: in_memory` (default) uses `AF3HotRunner` inside the Python process for minimum per-step latency. `backend: subprocess` invokes `run_alphafold.py --norun_data_pipeline` per prediction as a fallback when isolating processes.

---

## 4. Performance Tuning: FlashAttention, Buckets & Persistent `AF3HotRunner`

### Triton FlashAttention vs XLA Fallback

`AF3HotRunner` defaults to `flash_attention: "triton"` (`cfg.global_config.flash_attention_implementation = "triton"`).
- **Use `"triton"`** on Ampere / Ada / Hopper GPUs (SM 8.0+) for fastest attention kernels.
- **Set `flash_attention=xla`** (via YAML or CLI override `flash_attention=xla`) if running on older GPUs or environments without compatible Triton CUDA kernels.

### Token Compilation Buckets

AlphaFold 3 pads inputs to discrete token buckets to avoid recompiling JAX/XLA graphs on minor token-count differences (e.g. between `+glyco` and `-bare` states, or across binder lengths). The default buckets in `config.py` and `AF3HotRunner` are:

```python
(256, 384, 512, 768, 1024, 1536, 2048)
```

For typical binder designs (e.g., 100–350 total tokens), tighten `--buckets` to the smallest bucket(s) covering your states to reduce first-step XLA compile time and GPU memory:

```bash
af3-hall --config examples/03_glycoprotein_binder_multistate.yaml --buckets 256,384
```

> **Tip:** Both `glyco` (+12 sugar tokens) and `bare` states fit comfortably inside a single `384` bucket when `len(target) + len(binder) <= 372`, so JAX compiles only **one** XLA graph for both states.

### Persistent Shared `AF3HotRunner` (`get_shared_hot_runner`)

When sweeping multiple runs from Python in one process, `AF3Predictor` automatically reuses a cached `AF3HotRunner` via `get_shared_hot_runner(...)` (keyed by `af3_dir`, `model_dir`, `jax_cache_dir`, `num_recycles`, `num_diffusion_samples`, `flash_attention`, `buckets`, and `gpu_device`). Model parameters, CCD dictionary, and jitted XLA executables stay resident across runs without reloading weights.

---

## 5. Troubleshooting

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| **GPU OOM during JAX initialization** | JAX preallocates GPU memory by default | `AF3HotRunner` and `AF3SubprocessRunner` set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` automatically; if importing JAX earlier in a custom script, export both env vars before importing `jax`. |
| **Slow step 1 or repeated XLA recompilation** | Cold `AF3_JAX_CACHE_DIR` or states crossing bucket boundaries | Point `AF3_JAX_CACHE_DIR` to persistent storage and choose `--buckets` so all states map to the minimum necessary bucket(s) (e.g. `--buckets 384,512`). |
| **Triton kernel compilation error** | Unsupported GPU architecture or CUDA/Triton mismatch | Pass `flash_attention=xla` as a config key or CLI override. |
| **`ligand_mpnn_biased` logs `carb_blosum` in `steps.jsonl`** | `LIGANDMPNN_DIR` / `LIGANDMPNN_CHECKPOINT` unset or `run.py` failed | `LigandMPNNBiasedSampler` intentionally falls back cleanly to `carb_blosum` whenever `mpnn_dir`, `mpnn_checkpoint`, `mpnn_python`, or the input CIF is unavailable. Verify `LIGANDMPNN_DIR/run.py` and `LIGANDMPNN_CHECKPOINT` paths and that `LIGANDMPNN_PYTHON` has `torch` installed. |
| **`ModuleNotFoundError: No module named 'alphafold3'`** | `AF3_CODE_DIR` not pointing to AlphaFold 3 root | Set `export AF3_CODE_DIR=/path/to/alphafold3` or pass `--af3-dir /path/to/alphafold3` (or test offline with `--mock`). |
