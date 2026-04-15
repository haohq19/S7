# S7 reimplementation progress

Living status document for the S7 reproduction effort. Pairs with
[DEVLOG.md](DEVLOG.md) (chronological) and [README.md](README.md) (how-to).

## 1. Goal

Reproduce the S7 paper (Soydan et al. 2024, arxiv 2410.03464) with a clean
JAX/Flax codebase. Two deliverables: (a) a rewrite that is bit-identical to the
original author's implementation on forward/backward passes and reads cleanly,
and (b) numerical reproduction of every dataset listed in the paper.

Key design decisions (locked in on 2026-04-14):

- **Framework**: JAX + Flax (not a PyTorch port).
- **Selectivity**: only Δ is input-dependent at the *parameter* level. B and C
  are fixed Flax params; the *discretized* B̄, C̄ inherit input-dependence
  through Δ(u). This is the "Simplified" half of the S7 name — do NOT restore
  full B/C selectivity.
- **Hyperparameters**: the post-bugfix (`c65472d`) values in `configs/` are the
  intended ones. Earlier commits had broken defaults.
- **Dataset order**: DVS128 Gesture first, then the rest.

## Reproduction results

| Dataset | Rewrite | Paper | Δ | Status | Notes |
|---|---|---|---|---|---|
| DVS128 Gesture | — | 99.2 % | — | running | sbatch 63547983 (~3 h in of expected ~7 h). 1-epoch dev-node smoke gave val 0.633. |
| **SHD** | **93.43 %** | 96.30 % | −2.87 pp | **done** | 30 ep, seed 1234, best-val ckpt (epoch 23). Paper gap treated as real. |
| SSC | — | 88.2 % | — | running | sbatch 63547606 (~3 h in of expected ~13 h). 1-epoch smoke val 0.189 (35-class, random 0.029). |
| **EigenWorms** | **81.58 %** | 97.5 % | −15.92 pp | done | 200 ep, sbatch 63564563. Real ~16 pp shortfall — small model (~3K params), config from legacy. Worth investigating but likely needs more careful hyperparam tuning vs paper. |
| LRA ListOps | — | 63.77 % | — | smoked + queued | 1-epoch dev-node smoke val 0.174 (random 0.10). sbatch resubmitted 63572171 with isolated output dir. |
| LRA Text | — | 87.22 % | — | smoke passed | 1-epoch dev-node smoke val 0.583 (random 0.50, binary IMDB). sbatch not yet submitted. |
| LRA Retrieval | — | 91.80 % | — | blocked | AAN tsvs are huge (8.5 GB train); S5 dataset cache build hit Lustre file-count quota mid-rm. Will retry after rm finishes. |
| LRA Image | — | 61.14 % | — | smoke passed | 1-epoch dev-node smoke val 0.358 (random 0.10, 10-class CIFAR-grayscale). Not yet submitted. |
| LRA Pathfinder | — | 65.52 % | — | **unreachable** | OpenDataLab tarball does not contain `pathfinder32/` (only 64/128/256). Can't run without separately preprocessed data. |
| LRA Path-X | — | 61.50 % | — | **unreachable** | Pathfinder128 was 600 K image files. Deleted to free Lustre file-count quota. To attempt Path-X, would need to re-extract just `pathfinder128/` from the tarball and accept the file-count budget. |
| **Walker2D** | **loss 0.5305** | MSE 0.114 | ~9× off | done | 100 ep, sbatch 63569289. optax `l2_loss = 0.5*MSE` so equivalent ≈ 1.06. Real shortfall — needs investigation. |
| PersonActivity | — | 94.09 % | — | running (re) | First sbatch 63564987 hit quota mid-rm; resubmitted as 63572174. 1-epoch dev-node smoke val 0.802. |
| PTB | — | — | — | **skipped** | Config uses `bidirectional: true` which trivializes next-token LM (the SSM at position i sees input[i+1] = target[i] and learns the identity map, collapsing loss to ~0). Paper does not report numbers either. Not a rewrite bug — the legacy config is degenerate for this task. |
| WikiText-2 | — | — | — | **skipped** | Same `bidirectional: true` config issue. Paper does not report. |
| WikiText-103 | — | — | — | **skipped** | Same. Not worth the ~500 MB staging for a degenerate metric. |

## 2. What has been done

### 2.1 Code: full rewrite of the JAX/Flax core

`s7/` is a new package that replaces the `event_ssm/` tree. Internal
structure:

```
s7/
├── __init__.py            # re-exports S7, build_s7, ClassificationModel, ...
├── ssm.py                 # S7 Flax module: Λ init, Δ projection, 3 discretizations, parallel scan
├── init.py                # HiPPO-LegS DPLR decomposition, B/C init, log-step init
├── layers.py              # SequenceLayer, SequenceStage, EventPooling
├── model.py               # StackedEncoder, ClassificationModel, RetrievalModel
├── train/
│   ├── steps.py           # train_step / eval_step (cross-entropy + MSE)
│   ├── optim.py           # Multi-group AdamW (ssm / proj / regular) + cosine+warmup
│   ├── trainer.py         # epoch loop, logging, best-checkpoint tracking
│   └── checkpoint.py      # orbax save/load wrappers
└── data/
    ├── __init__.py        # registry: 15 task names → native builders
    ├── collate.py         # 7 collate variants (event-stream, LRA text/image, retrieval, ...)
    ├── transforms.py      # numpy event augmentations (Roll, Rotate, Scale, CutMix, ...)
    ├── dvs_gesture.py     # DVS128 Gesture (tonic)
    ├── spiking_audio.py   # SHD + SSC (tonic)
    ├── lra.py             # ListOps, Text, Retrieval, Image, Pathfinder, Path-X (wraps S5/)
    ├── irregular.py       # EigenWorms, PersonActivity, Walker2D (wraps odelstms/ + local .pt)
    └── language.py        # PTB, WikiText-2, WikiText-103 (own Dictionary/Corpus tokenizer)
```

Entry scripts:

```
scripts/
├── train.py               # Hydra @configs/base.yaml, supports all 15 tasks
└── evaluate.py            # load checkpoint + run test set
```

Parity tests:

```
tests/
├── test_ssm_parity.py     # S7 layer forward — rewrite vs legacy, bit-identical
└── test_model_parity.py   # ClassificationModel forward — rewrite vs legacy, bit-identical
```

### 2.2 Parity verified

Both parity tests compare the rewrite to the legacy `event_ssm/` code with the
same init PRNG and assert the outputs are bit-identical:

```
test_ssm_parity.py    max |Δy| = 0.000e+00  FULL PARITY OK
test_model_parity.py  max |Δy| = 0.000e+00  MODEL PARITY OK
```

GPU step-time benchmark (B=8, L=524288, DVS128 config, RTX 3090):

| | init | first step (compile) | steady-state step |
|---|---|---|---|
| NEW | 25.2 s | 32.5 s | **0.24 s** |
| OLD | — | 22.1 s | **0.24 s** |

Steady-state throughput is identical. Compile is 10 s slower one-time.

### 2.3 DVS128 smoke training

Single-GPU, 1 epoch, `optimizer.warmup_epochs=0`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`:

```
[*] 2,587,915 model parameters
| epoch 1 | Train | acc 0.3772 | loss 1.7622 | epoch_time 444 s
| epoch 1 | Val   | acc 0.6326 | loss 1.1150
```

~7.4 min/epoch for DVS128, dominated by the complex-valued parallel scan over
L≈524288. 100 epochs ≈ 12 h on 1× RTX 3090. Legacy code has the same throughput.

### 2.4 Repo cleanup

Deleted ~2,500 lines of dead code and ~68 MB of output-artifact dumps:

- `run_training.py`, `run_evaluation.py` — replaced by `scripts/{train,evaluate}.py`.
- `event_ssm/dataloading.py` (2276 lines) — replaced by `s7/data/*`.
- `event_ssm/{transform,trainer,train_utils,layers_old,print,personactivity,worm,physionet,ptb}.py` — dead / duplicate / standalone diagnostic scripts.
- `best_eigenworms/`, `best_image/`, `best_pathfinder/` — stale checkpoint + wandb dumps from the original author's machine.
- `setup.py` — pointed at a nonexistent package.

### 2.5 Dataset staging

- **DVS128 Gesture** — downloaded and verified (MD5 OK) at `/cluster/scratch/haihao/claude/data/dvs128/DVSGesture/`.
- All other datasets: not yet staged — need to be pulled before their respective reproduction runs.

## 3. Lessons learned (non-obvious, worth remembering)

### 3.1 DVS128 download workaround

Tonic's default Figshare URL (`https://figshare.com/ndownloader/files/<id>`) is
gated by an AWS WAF JS challenge that automated clients can't pass. The fix:
use the figshare API instead (`https://api.figshare.com/v2/file/download/<id>`)
which returns a presigned S3 URL with no challenge. File IDs:

- `38022171` → `ibmGestureTrain.tar.gz` (1.8 GB, md5 `3a8f0d41...`)
- `38020584` → `ibmGestureTest.tar.gz` (660 MB, md5 `56070e45...`)

Documented in README.md.

### 3.2 JAX BFC memory fraction

The DVS128 forward at B=8, L=524288 allocates ~22 GB of activations on a
single GPU. JAX's default `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` caps BFC at
18 GB, which is not enough — both legacy and rewrite OOM at init. Fix: set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. `scripts/train.py` and
`scripts/evaluate.py` now set this by default.

Do *not* use `XLA_PYTHON_CLIENT_ALLOCATOR=platform` — it grabs memory on demand
but without pooling, giving ~14× slowdown. BFC at 0.95 is the right fix.

### 3.3 Complex-to-real ComplexWarning in backward

Both legacy and rewrite emit `ComplexWarning: Casting complex values to real
discards the imaginary part` during `jax.grad` through `B @ u_t` where B is
complex and `u_t` is real. The imaginary part is zero by construction, so the
cast is mathematically correct. Harmless. Do not try to "fix" it.

### 3.4 Flax submodule naming affects init RNGs

Flax derives per-submodule init RNGs from the parameter-path hash. Renaming
`S7` → `S7Layer` would change the auto-submodule name (`S7_0` → `S7Layer_0`)
and every param would get a different initial value, breaking bit-exact
comparison. For the same reason, the Δ-projection submodule is kept as
`step_proj` (not `delta_proj`) and its params are `step_proj_kernel` /
`step_proj_bias`. Preserving these names also means legacy checkpoints load
cleanly into the rewrite.

### 3.5 XLA vmap fusion

Splitting discretization and `B @ u_t` into two separate vmaps forces XLA to
materialize a full (L, local_P) `gamma_bar` tensor (~270 MB for DVS128).
Fusing them into one vmap closure lets XLA reuse buffers. Pattern:

```python
def step_fn(u_t, dt, step_t):
    Lambda_bar, gamma_bar = discretize_fn(Lambda, step_rescale * step_t, dt)
    return Lambda_bar, gamma_bar * (B @ u_t)   # all in one vmap closure

Lambda_bar, Bu = jax.vmap(step_fn)(u, integration_timesteps, step)
```

### 3.6 Slurm memcg limits

The Euler compute node has 256 GB physical RAM but each Slurm job step is
memcg-limited to ~32 GB. A PyTorch DataLoader with `num_workers=4` + the
tonic slicer was enough to blow past that on DVS128. Keep `num_workers ≤ 2`
on this hardware.

### 3.7 Legacy `separate_dbc` path

The pre-`c65472d` code had a `separate_dbc=true` mode that routed B and C
through Dense projections (Mamba-style selectivity). That was removed as a
bug fix — S7 is "Simplified" precisely because it does not do this. The
rewrite does not restore it.

### 3.8 Hyperparameters matter for DVS128

The task YAML uses `per_device_batch_size=8` with `accumulation_steps=4` and
`ssm_base_lr=1e-6`. Changing any of these without adjusting the others
produces garbage training curves — stick to the post-bugfix defaults unless
running a deliberate sweep.

`optimizer.warmup_epochs > training.num_epochs` produces a negative `decay_steps`
argument to `optax.warmup_cosine_decay_schedule` and crashes on startup. For
debug runs, set both to the same small number.

## 4. Current code state

### 4.1 What exists and is verified

- `s7/` package — 14 source files, all importable, both parity tests green.
- `scripts/{train,evaluate}.py` — Hydra entry points for training + eval.
- `tests/test_{ssm,model}_parity.py` — bit-identical to legacy.
- `configs/` — Hydra config tree (unchanged from post-bugfix commit).
- `event_ssm/` — shrunk to 5 files (`ssm`, `ssm_init`, `layers`, `seq_model`,
  `__init__`) kept only as parity anchors; will be deleted once reproduction is
  locked.
- `S5/`, `odelstms/` — vendored data-prep deps for LRA and
  PersonActivity/Walker respectively. Not modified.
- `DEVLOG.md` — full chronological record.
- `README.md` — setup + training + dataset status.

### 4.2 What is *not* yet done

- **Reproduction**: DVS128 has only a 1-epoch smoke result (val acc 0.633 vs
  paper 0.992 target). No other dataset has been smoke-tested or run.
- **Data staging**: only DVS128 is downloaded. SHD, SSC, LRA, EigenWorms,
  PersonActivity, Walker, PTB, WikiText still need to be fetched and placed
  under `/cluster/scratch/haihao/claude/data/`.
- **Non-DVS loader validation**: the 14 non-DVS loaders in `s7/data/` import
  cleanly but have not been exercised end-to-end. There may be subtle bugs
  (e.g. path assumptions) that will surface on first use.

## 5. Next-step plan

Ordered by value; each step is independent and can be checkpointed.

1. **Stage SHD** (smallest event dataset, ~a few hundred MB of HDF5 from
   tonic) and smoke-test `scripts/train.py task=spiking-heidelberg-digits
   training.num_epochs=1`. First real validation of `s7/data/spiking_audio.py`.
2. **Run full DVS128 reproduction**: 100 epochs, `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.
   python scripts/train.py task=dvs-gesture logging.wandb=false`. Expected
   wall-clock ~12 h on 1× RTX 3090. Target test acc ≥ 99 % (paper 99.2 %).
3. **Run SHD + SSC reproduction** once they are staged. Paper targets 96.3 %
   and 88.2 %.
4. **Stage LRA tarball** (`lra_release.gz`, ~2 GB from the LRA GitHub release)
   at `/cluster/scratch/haihao/claude/data/long-range-arena/`. Smoke-test
   ListOps (cheapest LRA task) then reproduce ListOps / Text / Retrieval /
   Image / Pathfinder / Path-X.
5. **Stage EigenWorms** (`eigenworms/processed/*.pt`) and **PersonActivity**
   (`person/ConfLongDemo_JSI.txt`) and reproduce.
6. **Stage PTB / WikiText-2 / WikiText-103** and reproduce (paper doesn't
   appear to report these — we may be going beyond the paper here).
7. **Delete `event_ssm/`** entirely once reproduction numbers are locked in.
   This requires deleting the parity tests too (or rewriting them to compare
   against frozen numerical references).
8. **Record final results table** in README.md with paper target, reproduction
   number, delta, and wall-clock per dataset.

Post-reproduction polish (optional, not blocking):

- Multi-GPU pmap timing study — each additional GPU roughly halves wall-clock
  but interacts with `accumulation_steps` and batch size.
- Add a `pyproject.toml` for editable installs (`pip install -e .`).
- Add a minimal GitHub Actions CI that runs the two parity tests on CPU.
