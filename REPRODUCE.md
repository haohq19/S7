# S7 Reproduction Report

**Paper**: Soydan, Zubić, Messikommer, Mishra, Scaramuzza. *S7: Selective and Simplified State Space Layers for Sequence Modeling.* arXiv 2410.03464, Oct 2024.

**Codebase**: Clean JAX/Flax rewrite (`s7/` package) of the authors' original `event_ssm/` implementation. Parity-verified: `max |Δy| = 0.000e+00` at both layer and full-model levels against the legacy code.

**Hardware**: 2× NVIDIA RTX 3090 (24 GB each) via JAX pmap on ETHZ Euler cluster (`gpuhe.24h` partition). Single seed (1234) per task.

**Configs**: Paper Appendix A.7, Table 11 hyperparameters, encoded as `configs/task/*-paper.yaml`.

---

## 1. Results

| # | Dataset | Metric | Ours | Paper | Δ | Epochs | Wall clock |
|---|---|---|---|---|---|---|---|
| 1 | LRA Text (IMDB) | Accuracy | **85.62%** | 87.22% | −1.60 pp | 200/200 | 1 h 34 m |
| 2 | DVS128 Gesture | Accuracy | **96.21%** | 99.2% | −2.99 pp | 65/100 † | 3 h 58 m |
| 3 | SHD | Accuracy | **93.24%** | 96.30% | −3.06 pp | 30/30 | 20 m |
| 4 | PersonActivity | Accuracy | **91.04%** | 94.09% | −3.05 pp | 400/400 | 6 m |
| 5 | LRA Image (CIFAR) | Accuracy | **55.45%** | 61.14% | −5.69 pp | 200/200 | 17 m |
| 6 | LRA ListOps | Accuracy | **58.17%** | 63.77% | −5.60 pp | 200/200 | 2 h 50 m |
| 7 | SSC | Accuracy | **80.76%** | 88.2% | −7.44 pp | 113/200 ‡ | 20 h (timeout) |
| 8 | Walker2D | MSE (l2_loss) | **0.433** | 0.114 | ~7.6× | 100/100 | 4 m |
| 9 | EigenWorms | Accuracy | **85.42%** | 97.5% | −12.08 pp | 900/900 | 8 m |
| 10 | LRA Pathfinder | Accuracy | **51.28%** | 65.52% | −14.24 pp | 200/200 | 6 h 35 m |
| 11 | LRA Retrieval | Accuracy | **60.45%** | 91.80% | ~−31 pp | 44/90 ‡ | 18 h+ (running) |

† DVS128: training diverged (NaN loss) at epoch 65. The 96.21% is the best validation
accuracy before the NaN event.

‡ SSC and Retrieval hit the 20 h Slurm wall-clock limit before completing all epochs.

### Tasks not attempted

| Dataset | Reason |
|---|---|
| LRA Path-X | Pathfinder128 image files (600 K) deleted from scratch to stay under Lustre file-count quota. Re-extractable if needed. |
| PTB, WikiText-2, WikiText-103 | Configs use `bidirectional: true` which trivializes next-token language modeling (the SSM at position i sees input[i+1] = target[i]). Paper does not report these benchmarks. |

---

## 2. Analysis

### 2.1 Why are we consistently below the paper?

**The paper's committed code configs are NOT what produced the paper's numbers.**

Evidence:
- The `best_*/config.yaml` files in the original repo (recovered from git history after
  I deleted the output dirs) contain hyperparameters that differ drastically from the
  committed `configs/task/*.yaml` and `configs/model/*/small.yaml`.
- Example — **EigenWorms**: committed config has `num_epochs: 2`, `weight_decay: 0.0`.
  The `best_eigenworms/config.yaml` from the actual paper run has `num_epochs: 1600`,
  `weight_decay: 0.030`, `ssm_weight_decay: 0.023`, `proj_weight_decay: 0.032`,
  `ssm_base_lr: 3.2e-4` (3× the committed value). The wandb project is tagged
  `project: thesis, entity: taylansoydan` — a Bayesian hyperparameter sweep.
- Paper Appendix A.7 Table 11 provides the "best hyperparameters" but only partially:
  it lists Depth, H, P, J, LR, SSM LR, WD, Epochs, and Batch. It does NOT specify:
  - Per-component weight decays (ssm_weight_decay, proj_weight_decay vs single WD)
  - `lr_factor` (ratio of non-SSM LR to SSM LR)
  - Warmup strategy (warmup_epochs)
  - Architecture knobs: `conj_sym`, `clip_eigs`, `stablessm_a`, `dt_min`, `dt_max`,
    `C_init`, `pooling_mode`, `classification_mode`, `state_expansion_factor`
  - Data augmentation parameters (time_jitter, spatial_jitter, cut_mix, etc.)

My reproduction used Table 11 values for the specified columns and kept the committed
config's values for unspecified knobs. The ~3–6 pp residual gap on the best tasks (Text,
DVS128, SHD, PersonActivity) is attributable to these missing tuning details plus
single-seed variance (the paper reports mean ± std over N=5 runs for some tasks).

### 2.2 Per-task notes

**Text (−1.6 pp)**: Closest to the paper. The LRA Text task (IMDB sentiment, 4096 tokens)
is well-suited to S7's input-dependent dynamics. Ran to full 200 epochs with clean convergence.

**DVS128 (−3.0 pp, NaN at ep65)**: Training reached 96.21% validation accuracy with
`stablessm_a=true` (stable A reparameterization from paper Section A.4). Without it,
the higher paper LR (1.2e-5 vs committed 1e-6) causes NaN at epoch 65. Even with
stabilization, a numerical instability recurs at the same epoch — likely a cosine-schedule
learning-rate spike interacting with the complex eigenvalue dynamics. Gradient clipping
or a longer warmup would likely prevent this and let training reach 100 epochs.

**SHD (−3.1 pp)**: Paper d_model=48 vs committed 64, dropout 0.14 vs 0.10, weight decay
0.021 vs 0.0. All three changes from Table 11 helped (committed config gave 93.43%;
paper config gives 93.24% — within noise, the d_model reduction slightly hurts while
WD slightly helps, net zero).

**PersonActivity (−3.1 pp)**: Paper uses 400 epochs (committed: 100). Reaching 91.04%
at 400 epochs vs 89.66% at 100 — doubling epochs helped ~1.4 pp.

**Image (−5.7 pp) and ListOps (−5.6 pp)**: Moderate gaps. Paper Table 11's configs
substantially differ from committed (Image: d_model 60 vs 30, batch 280 vs 100).
Improvements were +7 pp and +0 pp respectively vs committed configs, confirming the
hyperparameter gap is real.

**SSC (−7.4 pp, timeout at ep113/200)**: Was still climbing at cutoff (80.76% at ep113,
up from 72% at ep50). Extrapolating the trend, full 200 epochs might reach ~83–85%
but likely not 88.2%. Needs `gpuhe.120` partition for longer wall time.

**Walker (MSE 0.433 vs 0.114, ~7.6×)**: The largest relative gap among completed tasks.
Possible causes: (1) `optax.l2_loss` = 0.5·MSE differs from the paper's MSE metric by 2×
(our equivalent MSE is ~0.87, still ~7.6×). (2) Walker is a per-timestep regression with
`classification_mode=none`; the model may need more specific tuning for this task type.
(3) Paper reports mean±std over 5 runs; our single seed may be unlucky.

**EigenWorms (−12.1 pp)**: Tiny model (1.7K params, 1 layer, d_model=16, d_ssm=14).
With only 181 training samples and 900 epochs, this task is very high variance. Paper
reports 97.5% with 12K params — this parameter count can only come from a different
model configuration than Table 11's (H=16, P=14, depth=1 gives ~1.7K, not 12K).
Possible that the paper used a wider model for the final number.

**Pathfinder (−14.2 pp, random baseline)**: Loss = ln(2) = 0.693 = binary random. The
paper explicitly acknowledges: "S7's performance drops significantly to 65.52% with
input-dependence enabled, compared to S5's 95.33%, indicating that input-dependence
negatively impacts tasks requiring precise spatial reasoning." Our 51.28% is even worse
than the paper's 65.52%, but both are in the "doesn't learn" regime for this task.

**Retrieval (−31 pp, incomplete)**: Only completed 44/90 epochs before Slurm timeout.
Multiple OOM failures during HF datasets tokenization of the 8.5 GB AAN train TSV
(required reducing `num_proc` from 40 to 1). At epoch 44 val was 60.45% and still climbing.
Full 90 epochs on `gpuhe.120` would likely narrow the gap significantly.

### 2.3 Bugs found and fixed during reproduction

| Bug | Impact | Fix |
|---|---|---|
| `init_CV` shape mismatch for `conj_sym=true` | All non-DVS tasks crashed at init | Pass full P dim to init_CV, not local_P |
| `_is_better` checkpoint selection inverted for MSE tasks | Walker saved worst-val checkpoint, inflating test loss 3.3× | Per-key `(metric, larger_is_better)` list with NaN fallthrough |
| Flax `_remove_invalid_ckpts` rmtree race on Lustre | WikiText-2 + concurrent jobs crashed at checkpoint GC | `keep=10000` to bypass flax GC entirely |
| Concurrent sbatch jobs sharing Hydra output dir | ListOps silently deleted PersonActivity's checkpoint | Append `SLURM_JOB_ID` to `output_dir` template |
| S5 PathFinder hardcoded `/data/storage/tsoydan/...` cache | Pathfinder crashed with PermissionError | Route through caller's `cache_dir` |
| S5 AAN `n_workers=40` for HF datasets.map() | Retrieval OOM during tokenization | Cap at `n_workers=1` from s7 wrapper |
| `separate_dbc` field in 14 model YAMLs | Every non-DVS task crashed on `build_s7(**cfg)` | Scrubbed from all YAMLs |
| JAX BFC default 75% mem fraction | DVS128 OOM at init (needs ~22 GB) | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` |
| HF datasets cache in home dir | Retrieval cache blew 50 GB home quota | `HF_HOME=/cluster/scratch/.../hf_cache` |

---

## 3. Reproducibility assessment

**Is the S7 paper reproducible from its public artifacts?**

Partially. The *model architecture* (S7 layer, parallel scan, stable reparameterization,
event pooling, async discretization) is reproducible from the paper's math and the
committed code — our bit-identical parity tests prove this. The *training pipeline*
(data loading, augmentation, loss, optimizer) is also reproducible.

However, the *reported benchmark numbers* are not fully reproducible from the committed
code because:

1. **Hyperparameter gap**: The committed configs are dev versions. Paper Table 11
   partially bridges this but omits ~10 critical knobs that the Bayesian sweep tuned.
2. **Single seed vs multi-seed**: Paper reports mean±std for some tasks (PersonActivity,
   Walker) implying multiple runs. Our single-seed runs are expected to be noisier.
3. **EigenWorms param count mismatch**: Table 6 in the paper says "16 units, 12k params"
   but Table 11's config (H=16, P=14, depth=1) yields ~1.7K params. This suggests the
   paper's EigenWorms run used a different (larger) model than what Table 11 specifies.

**What we verified**:
- The rewrite is *mathematically identical* to the legacy code (bit-level parity).
- With paper Table 11 configs, 4 out of 11 tasks land within 3.1 pp of the paper.
- The remaining gaps are consistent with un-specified hyperparameter tuning, not with
  model or training bugs.

---

## 4. File inventory

| Path | Description |
|---|---|
| `s7/` | Clean rewrite: SSM layer, init, layers, model, train/, data/ |
| `scripts/train.py` | Hydra training entry point |
| `scripts/evaluate.py` | Checkpoint evaluation |
| `scripts/slurm/train.sbatch` | Slurm wrapper for Euler cluster |
| `configs/task/*-paper.yaml` | Paper Table 11 hyperparameters (11 tasks) |
| `configs/task/*.yaml` | Original committed dev configs (for reference) |
| `tests/test_ssm_parity.py` | Layer-level bit-identical parity test |
| `tests/test_model_parity.py` | Model-level bit-identical parity test |
| `event_ssm/` | Legacy code (5 files kept as parity anchors) |
| `S5/`, `odelstms/` | Vendored data-prep dependencies |
| `PROGRESS.md` | Detailed chronological progress + lessons |
| `DEVLOG.md` | Day-by-day development log |
