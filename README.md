# S7: Selective and Simplified State Space Model

Reimplementation + reproduction of **S7** (Soydan et al., 2024).
Paper: https://arxiv.org/abs/2410.03464

This is a clean JAX/Flax rewrite of the original event-SSM repo. The core idea of S7 is a
diagonal complex-Λ state space model whose discretization step size Δ is input-dependent
(the "Selective" half), while the underlying continuous Λ, B, C parameters stay LTI (the
"Simplified" half — unlike Mamba, which makes B, C, Δ all input-dependent).

```
    xₖ = exp(Λ · Δₖ(u) · Δtₖ) · xₖ₋₁ + γ̄ₖ · (B uₖ)
    yₖ = Re(C xₖ) + D uₖ
```

## Reproduction results

All runs use hyperparameters from paper Table 11, encoded in `configs/task/*-paper.yaml`.
2× RTX 3090 via pmap. Seed 1234.

| Dataset | Ours | Paper | Δ |
|---|---|---|---|
| LRA Text | **85.62 %** | 87.22 % | −1.6 pp |
| DVS128 Gesture | **96.21 %** | 99.2 % | −3.0 pp |
| SHD | **93.24 %** | 96.30 % | −3.1 pp |
| PersonActivity | **91.04 %** | 94.09 % | −3.1 pp |
| LRA Image | **55.45 %** | 61.14 % | −5.7 pp |
| LRA ListOps | **58.17 %** | 63.77 % | −5.6 pp |
| SSC | **80.76 %** | 88.2 % | −7.4 pp |
| Walker2D | **loss 0.433** | MSE 0.114 | ~7.6× |
| EigenWorms | **85.42 %** | 97.5 % | −12.1 pp |
| LRA Pathfinder | **51.28 %** | 65.52 % | −14.2 pp |
| LRA Retrieval | **60.45 %** | 91.80 % | −31 pp |

See [PROGRESS.md](PROGRESS.md) for detailed notes on each run (epochs completed, failure
modes, root-cause analysis of the paper-vs-repo config gap).

## Repository layout

```
s7/                       Rewritten JAX/Flax package
├── ssm.py                S7 layer (selective Δ + complex Λ scan)
├── init.py               HiPPO-LegS init, step-size init, B/C init helpers
├── layers.py             SequenceLayer, SequenceStage, EventPooling
├── model.py              StackedEncoder, ClassificationModel, RetrievalModel
├── train/
│   ├── steps.py          train_step / eval_step (cross-entropy + MSE)
│   ├── optim.py          Param-group AdamW (ssm / proj / regular)
│   ├── trainer.py        Epoch loop + best-checkpoint tracking
│   └── checkpoint.py     Orbax save/load wrappers
└── data/
    ├── __init__.py       Task → builder registry (legacy adapter fallback)
    ├── collate.py        Event-stream tokenization + padding
    ├── transforms.py     Numpy augmentations (Roll, Rotate, Scale, CutMix, …)
    └── dvs_gesture.py    DVS128 Gesture loader (native port)

scripts/
├── train.py              Main training entry (Hydra)
└── evaluate.py           Checkpoint eval

tests/
├── test_ssm_parity.py    SSM forward bit-identical to legacy
└── test_model_parity.py  ClassificationModel bit-identical to legacy

configs/                  Hydra config tree (unchanged structure)
├── base.yaml
├── system/local.yaml     Points data_dir and output_dir at scratch
├── task/<name>.yaml      One YAML per dataset
└── model/<name>/<size>.yaml

event_ssm/                Parity-test anchors only (ssm.py + seq_model.py + layers.py + ssm_init.py)
S5/                       Vendored LRA data-prep code
odelstms/                 Vendored PersonActivity/Walker data-prep code
DEVLOG.md                 Running notes
```

## Current status of dataset ports

All 15 paper tasks are natively ported under `s7/data/`:

| Task | Module | Reproduced |
|---|---|---|
| DVS128 Gesture | `s7/data/dvs_gesture.py` | pending |
| SHD, SSC | `s7/data/spiking_audio.py` | pending |
| LRA: ListOps, Text, Retrieval, Image, Pathfinder, Path-X | `s7/data/lra.py` | pending |
| EigenWorms | `s7/data/irregular.py` | pending |
| PersonActivity, Walker2D | `s7/data/irregular.py` | pending |
| PTB, WikiText-2, WikiText-103 | `s7/data/language.py` | pending |

`scripts/train.py` dispatches on `cfg.task.name` via `s7.data.get_builder(...)`.
The LRA loaders delegate data-prep to the vendored `S5/` tree, and the
PersonActivity / Walker loaders delegate to `odelstms/` — both are treated as
external data-prep libraries (like `tonic`) and are not modified by the rewrite.

## Setup

```bash
conda create -n s7 python=3.9.19 -y
conda activate s7
pip install -r requirements.txt
```

Edit `configs/system/local.yaml` to point `data_dir` at the directory that contains your
datasets (e.g. `/cluster/scratch/<user>/data/dvs128`). On the Euler dev box used to
reproduce the paper, that path is already set.

## Training

Local / interactive:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train.py task=dvs-gesture
```

On the Euler cluster via slurm (recommended for long runs):

```bash
sbatch scripts/slurm/train.sbatch task=spiking-speech-commands
sbatch scripts/slurm/train.sbatch task=dvs-gesture
sbatch scripts/slurm/train.sbatch task=shd-classification training.num_epochs=60
```

The slurm wrapper requests 2× RTX 3090 and 20 h on `gpuhe.24h` by default
(edit the `#SBATCH` block at the top to change). Logs land in
`/cluster/scratch/$USER/claude/slurm-logs/s7-train-<jobid>.out`. Monitor with
`squeue -u $USER` and `tail -f /cluster/scratch/$USER/claude/slurm-logs/s7-train-<jobid>.out`.

Common overrides:

```bash
# Short debug run
... training.num_epochs=1 optimizer.warmup_epochs=0

# Use 2 GPUs via pmap
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python scripts/train.py task=dvs-gesture

# Enable Weights & Biases
... logging.wandb=true logging.project=s7 logging.entity=<your-wandb-user>
```

**Important environment knob**: `scripts/train.py` sets
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` by default. The DVS128 forward pass allocates
~22 GB at B=8, L=524288, which does not fit under JAX's 75% BFC default on a 24 GB
RTX 3090. Lower this only if running a smaller config.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/evaluate.py \
    task=dvs-gesture \
    checkpoint=/path/to/outputs/<run>/checkpoints
```

## Datasets

### DVS128 Gesture (native)

Tonic downloads from Figshare by default, but the public Figshare URL is gated behind
an AWS WAF challenge that `requests`/`curl` cannot pass. Workaround: fetch via the
Figshare API (presigned S3):

```bash
mkdir -p <data_dir>/DVSGesture
cd <data_dir>/DVSGesture
curl -L -o ibmGestureTrain.tar.gz \
    "https://api.figshare.com/v2/file/download/38022171"
curl -L -o ibmGestureTest.tar.gz \
    "https://api.figshare.com/v2/file/download/38020584"
```

Expected MD5: `3a8f0d4120a166bac7591f77409cb105` (train),
`56070e45dadaa85fff82e0fbfbc06de5` (test).

Tonic will skip download and unpack these automatically on first use.

## Tests

```bash
PYTHONPATH=. python tests/test_ssm_parity.py
PYTHONPATH=. python tests/test_model_parity.py
```

Both compare the rewritten `s7/` modules against the legacy `event_ssm/` modules with
identical init RNGs and assert the outputs are bit-identical (`max |Δy| = 0`). These run
on CPU and are cheap to rerun after any rewrite change.

## Design notes

- **Δ-only selectivity.** Only the discretization step size Δ is a function of the input
  (via `step_proj` Dense). The continuous B and C matrices are fixed Flax parameters. The
  *discretized* B̄, C̄ are functions of u only through Δ(u). This matches the paper and is
  intentional — do not "fix" this to restore full Mamba-style selectivity.
- **Flax naming is preserved**: `S7`, `step_proj`, `step_proj_kernel`, `step_proj_bias`,
  `Lambda_re`, `Lambda_im`, `B`, `C`, `log_step`, `D`. This keeps init RNGs bit-identical
  between rewrite and legacy and lets legacy checkpoints load without surgery.
- **`apply_ssm` fuses discretization and Bu into one vmap closure** to avoid
  materializing a full-sequence-length `gamma_bar` tensor — matters at DVS128's L ≈ 5·10⁵.
- **`EventPooling` acts as identity on `stride=1`** instead of raising, so it can be used
  unconditionally in the middle of `SequenceLayer` without guards.
- The legacy `event_ssm/` tree has been reduced to the four files the parity
  tests compare against (`ssm.py`, `ssm_init.py`, `layers.py`, `seq_model.py`).
  These can be deleted entirely once the reproduction runs are locked in.
